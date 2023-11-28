/*************************************************************************
 * Copyright (C) [2022] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "executor.h"

#include <time.h>

#include <atomic>
#include <cmath>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cndev.h"

#include "core/mlu_env.h"
#include "core/runtime/device.h"
#include "internal_kernel/fill_llc/fill_llc.h"  // mluOpFillLLC
#include "internal_kernel/fill_ram/fill_ram.h"  // mluOpFillRam
#include "kernel_tracing.h"
#include "hardware_monitor.h"

// #if GTEST_ENABLE_GPERFTOOLS
// #include "gperftools/profiler.h"
// #endif

extern std::unordered_map<std::string, std::vector<double>> acc_baseline_map;

namespace mluoptest {

void DataBlock::onlyServeAsInput() {
  dram_tensor_type = DramTensorType::ONLY_INPUT;
}

void DataBlock::alsoServeAsVolatile() {  // currently only for batchnorm_forward
  GTEST_CHECK(dram_tensor_type == DramTensorType::ONLY_INPUT,
              "in hostmalloc it should be set to only_input");
  LOG(WARNING) << "This case is probably ill-formed, make sure all outputs are "
                  "used to compute diff.";
  dram_tensor_type = DramTensorType::BOTH_INPUT_VOLATILE;
}

void DataBlock::alsoServeAsOutput() {
  if (dram_tensor_type == DramTensorType::ONLY_INPUT) {
    dram_tensor_type = DramTensorType::BOTH_INPUT_OUTPUT;
  } else {
    GTEST_CHECK(dram_tensor_type == DramTensorType::ONLY_OUTPUT,
                "in hostmalloc it should be set to only_output");
  }
}

Executor::~Executor() {
  // BUG(zhaolianshui): make sure no throw, exe_config_ might be null and seg
  // fault these function can't throw exception.
  VLOG(4) << "Free all resource.";
  monitor->recordHwtime(eva_res_.mlu.raw_hwtime_list);
  mluOutputFree();
  deviceFree();
  hostFree();
  baselineFree();
  destroyTensors();
// #if GTEST_ENABLE_GPERFTOOLS
//   if (exe_config_ != nullptr && exe_config_->gtest_internal_cpu_profile) {
//     ProfilerStop();
//   }
// #endif
  VLOG(4) << "Executor end.";
}

void Executor::init(const std::shared_ptr<ExecuteContext> ectx) {
  VLOG(4) << "Executor start.";
  exe_context_ = ectx;

  cpu_runtime_.init(exe_context_->cmp);
  mlu_runtime_.init(exe_context_->mmp);

  handle_ = exe_context_->handle;
  queue_ = exe_context_->queue;

  parser_ = std::make_shared<Parser>();
  eva_ = std::make_shared<Evaluator>();
  stride_ = std::make_shared<Stride>(&cpu_runtime_);
}

void Executor::setup(std::string file,
                     const std::shared_ptr<ExecuteConfig> ecfg) {
  exe_config_ = ecfg;

// #if GTEST_ENABLE_GPERFTOOLS
//   if (exe_config_->gtest_internal_cpu_profile) {
//     // TODO(None): make profile filename configurable
//     std::string profile_name = "mluop_gtest_cpu_capture.prof";
//     ProfilerStart(profile_name.c_str());
//   }
// #endif
  // VLOG(5) << __FUNCTION__ << ", " << __LINE__;

  // kernel_tracing_ctx =
  // kernelTracingCtx::instance(exe_config_->kernel_trace_policy.c_str());
  // VLOG(7) << "kernel tracing policy: " << exe_config_->kernel_trace_policy;
  // VLOG(5) << __FUNCTION__ << ", " << __LINE__;

  // if (kernel_tracing_ctx) {
  //   eva_res_.mlu.kernel_tracing_enabled = true;
  // }
  // VLOG(5) << __FUNCTION__ << ", " << __LINE__;

  // if (getFlagHalfInfTo65504()) {
  //   arrayCastFloatAndNormalWrapper = arrayCastFloatAndNormalInvalidInf;
  // }
  // VLOG(5) << __FUNCTION__ << ", " << __LINE__;

  jobLimitCheck();

  clusterLimitCheck();

  recordGtestTimePoint("before_parse");
  parser_->parse(file);
  recordGtestTimePoint("after_parse");
  eva_res_.case_path = file;
  VLOG(4) << "param check.";
  paramCheck();  // op oriented check

  VLOG(4) << "set handle.";
  setHandle();

  VLOG(4) << "Create input(/output) tensors.";
  createTensors();

  VLOG(4) << "Host malloc.";
  hostMalloc();
  recordGtestTimePoint("after_host_malloc");

  getPerfTestMode();

  selectStorageDtype();

  setStorageFuncPtr();

  if (mlu_need_host_data) {
    if (parser_->device() == CPU) {
      VLOG(4) << "Host malloc (for cpu compute).";
      baselineInputMalloc();
      recordGtestTimePoint("after_baseline_input_malloc");
      VLOG(4) << "Init data (random data for cpu compute).";
      initBaselineInput();  // init fp32 cpu data
      recordGtestTimePoint("after_baseline_input_init");
      VLOG(4) << "Cast dtype (host fp32 -> mlu X).";
      castIn();  // init host data(copy to host_data).
    } else {
      flag_quant_mode_ = NO_QUANT;
      VLOG(4) << "Init data from prototxt.";
      initHostData();  // read data to host_data directly
      recordGtestTimePoint("after_init_host_data");
      VLOG(4) << "Set quant param to tensor descs.";
      setQuantizedParam();  // set quant param
    }
  }
  hostReorder();

  VLOG(4) << "Device malloc.";
  setMiscellaneousParam();
  deviceMalloc();
  recordGtestTimePoint("after_device_malloc");
  VLOG(4) << "Copy data from host to device.";
  copyIn();
  recordGtestTimePoint("after_copy_in");

  VLOG(4) << "switch to origin data buffer.";
  switchDataToOrigin();

  VLOG(4) << "select api version.";
  selectApiVersion();

  VLOG(4) << "prepare compute param.";
  prepareComputeParam();

  VLOG(4) << "Device malloc (for workspace).";
  workspaceMalloc();

  deviceRestSpaceMalloc();

  // when get MLUOP_GTEST_FILL_RAM env,
  // fill nram/sram/warm for nan or inf beore compute for each case
  fillRam();

  VLOG(4) << "Algorithm Search.";
  searchAlgo();
}

void Executor::recordKernel(cnrtFunctionType_t, cnrtDim3_t, const char *name) {
  this->eva_res_.mlu.kernel_name_lists.emplace_back(name);
}

void Executor::startInterfaceListening() {
  if (global_var.thread_num_ > 1) {
    LOG_FIRST_N(WARNING, 1) << "multiple thread mode may not support kernel "
                               "name tracing well, so disabled";  // NOLINT
    global_var.kernel_trace_policy_ = KERNEL_TRACING_DISABLED;
    kernel_tracing_ctx = nullptr;
  }
  if (kernel_tracing_ctx == nullptr) return;
  kernel_tracing_ctx->init();
  kernel_tracing_ctx->setCallbackKernelTraced(
      std::bind(&Executor::recordKernel, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3));
}

// just a helper class, to release resource automatically without worrying about
// Exception interruption
template <class T, typename FuncAlloc, typename FuncRelease>
class RAIIHelper {
 public:
  RAIIHelper(T *self, FuncAlloc &&f1, FuncRelease &&f2)
      : self_(self), free_method_(f2) {
    (self_->*f1)();
  }
  ~RAIIHelper() { release(); }
  void release() {
    if (!released_.test_and_set(std::memory_order_acquire)) {
      (self_->*free_method_)();
    }
  }
  RAIIHelper(const RAIIHelper &) = delete;

 private:
  std::atomic_flag released_ = ATOMIC_FLAG_INIT;
  T *self_;
  FuncRelease free_method_;
};

template <class T, typename FuncAlloc, typename FuncRelease>
auto create_raii(T *self, FuncAlloc &&f1, FuncRelease &&f2) {
  return std::make_unique<RAIIHelper<T, FuncAlloc, FuncRelease>>(
      self, std::forward<FuncAlloc>(f1), std::forward<FuncRelease>(f2));
  //  return RAIIHelper<T, FuncAlloc, FuncRelease>(self,
  //      std::forward<FuncAlloc>(f1), std::forward<FuncRelease>(f2));
}

void Executor::stopInterfaceListening() {
  if (kernel_tracing_ctx == nullptr) return;
  kernel_tracing_ctx->release();
}

void Executor::disableKernelTracing() {
  if (kernel_tracing_ctx == nullptr) return;
  // XXX Some op call interface_timer.start/stop outside launch(::compute)
  if (!kernel_tracing_ctx->is_initialized()) return;
  kernel_tracing_ctx->disableKernelTracingImpl();
}

void Executor::enableKernelTracing() {
  if (kernel_tracing_ctx == nullptr) return;
  if (!kernel_tracing_ctx->is_initialized()) return;
  kernel_tracing_ctx->enableKernelTracingImpl();
}

void Executor::launch() {
  // for fusedOp, get layer by layer time
  if (need_compute_by_layer_) {
    launchAndGetTime(BY_LAYER, repeat_val_1);
    // for fusedOp, get layer by layer baseline result
    getBaselineOutputByLayer();
  }
  // comute for warm up
  VLOG(4) << "compute once for warm up.";

  // TODO(None): For kernel symbol name tracing:
  //                   If any mluOp op launch other kernels inside `compute()`,
  //                   kernel recording would have 'noises',
  //                   we may need to consider put 'kernel tracing logic' inside
  //                   `interface_timer_`, or we should ensure that `compute()`
  //                   just call mluOp op itself, and any other preparation
  //                   which launches kernel should be done before `compute`.
  //                   Also, `fillLLC` will be called inside `launchAndGetTime`
  //                   so we may have to inject logic into launchAndGetTime
  //                   iteself. At present, I choose to modify
  //                   `interface_timer_`
  auto interface_listening_handle =
      create_raii(this, &Executor::startInterfaceListening,
                  &Executor::stopInterfaceListening);
  launchAndGetTime(NORMAL, repeat_val_1);
  interface_listening_handle->release();
  recordGtestTimePoint("after_launch");
}

bool Executor::ready() {
  auto ret = cnrtQueueWaitNotifier(exe_context_->hw_notifier->n_stop,
                                   exe_context_->queue, 0);
  if (CNRT_RET_ERR_NOT_READY == ret) {
    return false;
  } else if (CNRT_RET_SUCCESS == ret) {
    return true;
  } else {
    GTEST_CHECK(false,
                "Executor: This kernel call failed because error occurred.");
  }
}

void Executor::sync() {
  GTEST_CHECK(cnrtSuccess == cnrtQueueSync(exe_context_->queue));
  recordGtestTimePoint("after_sync");
}

void Executor::fillLLC() {
  // TODO(None): add conditions of test llc, such as arch
  if (exe_config_->test_llc) {
    /*
    // if need get fill llc time, uncomment here instead
    Func fill_llc_ptr = std::bind(&Executor::fillLLC, this);
    float hw_time_llc = callBackKernelSyncAndGetTime(fill_llc_ptr,
    exe_context_->hw_notifier_llc); VLOG(4) << "fill llc kernel hw_time = " <<
    hw_time_llc;
    */
    GTEST_CHECK(MLUOP_STATUS_SUCCESS ==
                mluOpFillLLC(handle_, (void *)const_addr_, llc_size_));
  }
}

#define DO_NOTHING_VERY_FIRST_TIME() \
  if (repeat == repeat_val_1) {      \
    return;                          \
  }

#define FIVE_EQUAL " ===== "

void Executor::setupForPerfIter(int repeat, int iter, int iter_start) {
  DO_NOTHING_VERY_FIRST_TIME();

  auto getPerfSrcData = [this, &iter](DataBlock *db) {
    if (needDevPerfDataSpace()) {
      VLOG(4) << "[Perf round " << iter
              << "] copy from device_perf_data_ptr space";
      return db->device_perf_data_ptr;
    } else {
      VLOG(4) << "[Perf round " << iter
              << "] copy from device_origin_ptr space";
      return db->device_origin_ptr;
    }
  };

  std::ostringstream oss;
  // get random space for data_vector[i].device_perf_ptr
  if (needDevRandomSpace()) {
    VLOG(4) << FIVE_EQUAL << "[" << iter << "] Setup random MLU device space\n";
    if (unlikely(iter == 0)) {
      mlu_runtime_.mmp->clearBookKeepRandomSpaceBigChunk();
    }
    bool found = false;
    auto setDevicePtr = [&, this](int offset, int i, bool is_input,
                                  bool only_first_iter_random_space) {
      found = false;
      DataBlock *db = &(data_vector_[offset + i]);
      if (skipMallocDevice(db->getMetaTensor())) {
        found = true;
        return;
      }
      MetaTensor *mt = is_input ? parser_->input(i) : parser_->output(i);

      if (likely((!only_first_iter_random_space) ||
                 (only_first_iter_random_space && iter == iter_start))) {
        std::tie(found, db->device_perf_ptr) =
            mlu_runtime_.mmp->getRandomSpaceBigChunk(db->size);
      } else {
        found = true;
      }
      if (found) {
        mt->dev_perf_ptr = db->device_perf_ptr;
        db->device_ptr = db->device_perf_ptr;
        mt->dev_ptr = db->device_perf_ptr;
        oss << "\t\tMLU DRAM space [" << db->device_perf_ptr << ", "
            << (void *)((char *)(db->device_perf_ptr) + db->size) << "); ";
        // some ops need the original data to reproduce the performance or to
        // avoid runtime error.
        // XXX(zhaolianshui): assume that diff is never computed based on perf
        // results (stride?)
        if (is_input) {
          if (perfUseOriginData()) {
            void *src_data = getPerfSrcData(db);
            GTEST_CHECK(cnrtMemcpy(db->device_perf_ptr, src_data, db->size,
                                   CNRT_MEM_TRANS_DIR_DEV2DEV) ==
                        CNRT_RET_SUCCESS);
            oss << "copy data from " << src_data;
          } else {
            oss << "random data is fed to MLU kernel, if it should be real "
                   "data, make sure the op "
                << "name is in test/mluop_gtest/gtest_config/test_list "
                   "[rely_real_data] section.";
          }
        }
        VLOG(4) << oss.str();
        oss.str("");
      }
    };

    // if fail to find unique random allocations, might need to increment the
    // attempts
    int attempts_remaining = 5;
    // poisson may have more than 80k tensors, which require 25s to generate
    // random spaces, and 100 iterations need 40 minutes to finish, so cap it
    const int max_tensor_num_random_space = 1000;
    int input_tensor_num = parser_->inputs().size();
    int output_tensor_num = parser_->outputs().size();
    bool only_first_iter_random_space =
        (input_tensor_num + output_tensor_num > max_tensor_num_random_space);
    // sunyao is okay with the poisson scenario, but still keep the code for
    // future testing
    only_first_iter_random_space = false;
    if (unlikely(only_first_iter_random_space)) {
      LOG(WARNING) << "Random spaces are allocated for the first iteration and "
                      "the rest iterations "
                   << "will just reuse them.";
    }
    while (attempts_remaining--) {
      VLOG(4) << "\tInput" << FIVE_EQUAL;
      for (int i = 0; i < input_tensor_num; ++i) {
        setDevicePtr(0, i, true, only_first_iter_random_space);
        if (!found) {
          break;
        }
      }
      if (!found) {
        mlu_runtime_.mmp->clearRandomSpaceBigChunk();
        continue;
      }
      if (flag_input_reuse_) {
        VLOG(4) << oss.str();
        return;
      }
      VLOG(4) << "\tOutput" << FIVE_EQUAL << "\n";
      for (int i = 0; i < output_tensor_num; ++i) {
        setDevicePtr(input_tensor_num, i, false, only_first_iter_random_space);
        if (!found) {
          break;
        }
      }
      if (!found) {
        mlu_runtime_.mmp->clearRandomSpaceBigChunk();
        continue;
      }
      if (mlu_runtime_.mmp->gotUniqueRandomSpaceAllocations()) {
        break;
      } else {
        int curr_attemp = attempts_remaining + 1;
        if (curr_attemp > 1) {
          mlu_runtime_.mmp->clearRandomSpaceBigChunk();
        } else {
          LOG(WARNING)
              << "Failed to find unique allocations for perf iteration";
        }
      }
    }
    GTEST_CHECK(found, "Failed to allocate random space after 5 attempts.");
  } else if (needDevPerfSpace() && perfUseOriginData()) {
    int input_tensor_num = parser_->inputs().size();
    for (int i = 0; i < input_tensor_num; ++i) {
      auto &db = data_vector_[i];
      if (skipMallocDevice(db.getMetaTensor())) continue;
      void *src_data = getPerfSrcData(&db);
      GTEST_CHECK(cnrtMemcpy(db.device_perf_ptr, src_data, db.size,
                             CNRT_MEM_TRANS_DIR_DEV2DEV) == CNRT_RET_SUCCESS);
    }
  }
}

void Executor::teardownForPerfIter(int repeat, int iter) {
  DO_NOTHING_VERY_FIRST_TIME();

  if (likely(needDevRandomSpace())) {
    VLOG(4) << FIVE_EQUAL << "[" << iter
            << "] Tear down random MLU device space";
    mlu_runtime_.mmp->bookKeepRandomSpaceBigChunk();
    mlu_runtime_.mmp->clearRandomSpaceBigChunk();
  }
}

#undef DO_NOTHING_VERY_FIRST_TIME
#undef FIVE_EQUAL

void Executor::launchAndGetTime(ComputeMode compute_mode, int repeat) {
  size_t time_point;
  double hw_time = 0;
  double hw_time_layer = 0;
  double hw_time_total = 0;
  double hw_time_sum_of_square = 0;
  float hw_time_layer_total = 0;
  Func compute_ptr = std::bind(compute_func, this);
  Func compute_by_layer_ptr = std::bind(&Executor::computeByLayer, this);
  std::vector<double> hw_time_vec;
  int i_start = 0;
  for (int i = i_start; i < repeat; ++i) {
    fillLLC();
    setupForPerfIter(repeat, i, i_start);

    if (compute_mode == BY_LAYER) {
      std::tie(time_point, hw_time_layer) = callBackKernelSyncAndGetTime(
          compute_by_layer_ptr, exe_context_->hw_notifier_layer);
    } else {
      std::tie(time_point, hw_time) =
          callBackKernelSyncAndGetTime(compute_ptr, exe_context_->hw_notifier);
      eva_res_.mlu.raw_hwtime_list.push_back(
          std::make_tuple(time_point, hw_time));
    }
    hw_time_vec.push_back(hw_time);

    teardownForPerfIter(repeat, i);
    VLOG(4) << "repeat iter = " << i << ", hardware time = " << hw_time;
    hw_time_total += hw_time;
    hw_time_sum_of_square += hw_time * hw_time;
    hw_time_layer_total += hw_time_layer;
  }

  if (compute_mode == BY_LAYER) {
    eva_res_.mlu.hardware_time_layer = hw_time_layer_total / repeat;
  } else {
    eva_res_.mlu.hardware_time = hw_time_total / repeat;
    const double hardware_time_variance =
        hw_time_sum_of_square / repeat -
        eva_res_.mlu.hardware_time * eva_res_.mlu.hardware_time;
    eva_res_.mlu.hardware_time_cv =
        sqrt(hardware_time_variance) / eva_res_.mlu.hardware_time;
  }
}

std::tuple<size_t, float> Executor::callBackKernelSyncAndGetTime(
    Func launch_kernel, std::shared_ptr<HardwareTimeNotifier> notifier) {
  float hwtime = 0;
  cnrtNotifier_t n_start = notifier->n_start;
  cnrtNotifier_t n_stop = notifier->n_stop;
  GTEST_CHECK(CNRT_RET_SUCCESS ==
              cnrtPlaceNotifier(n_start, exe_context_->queue));
  launch_kernel();
  GTEST_CHECK(CNRT_RET_SUCCESS ==
              cnrtPlaceNotifier(n_stop, exe_context_->queue));
  GTEST_CHECK(cnrtSuccess == cnrtQueueSync(exe_context_->queue));
  size_t tp = MONITOR_CLOCK::now().time_since_epoch().count();
  GTEST_CHECK(cnrtSuccess == cnrtNotifierDuration(n_start, n_stop, &hwtime));
  // print once kernel time, for debug of repeat mode.
  // VLOG(4) << "call back kernel time = " << hwtime;
  return std::make_tuple(tp, hwtime);
}

void Executor::perfRepeat() {
  mlu_mem_status_.allocated_before = mlu_runtime_.getAllocatedSize();
  if (doPerf()) {
    VLOG(4) << "MLU compute for perf test.";
    if (needDevPerfSpace()) {
      switchDataToPerf();
    }
    // device_ptr is used in opExecutor::compute
    launchAndGetTime(NORMAL, exe_config_->perf_repeat);
    if (need_compute_by_layer_) {
      launchAndGetTime(BY_LAYER, exe_config_->perf_repeat);
    }
    // XXX(zhaolianshui): if device_origin_ptr is used in copyout, there is no
    // need to do
    //                    switchDataToOrigin
    //                    device_ptr: is used in MLU kernel
    //                    device_origin_ptr: used in the very first kernel
    //                    launch device_perf_data_ptr: perf>1 && only used to
    //                    store real data device_perf_ptr: perf>1 && fed to MLU
    //                    kernel
    if (needDevPerfSpace()) {
      switchDataToOrigin();
    }
    VLOG(4) << "End MLU compute.";
  }
  mlu_mem_status_.allocated_after = mlu_runtime_.getAllocatedSize();
}

bool Executor::checkMluMemoryLeak() {
  if (mlu_mem_status_.allocated_before != mlu_mem_status_.allocated_after) {
    LOG(ERROR) << "Duplicated MLU Memory allocated during ::compute, which is "
                  "illegal during perf_repeat. "
                  "You should consider ::workspaceMalloc and ::workspaceFree, "
                  "or setup internal state"
               << "(case: " << eva_res_.case_path << ")";
    return false;
  }
  return true;
}

void Executor::getTestInfo() {
  getMluPerfInfo(&(eva_res_.mlu));
  getGpuPerfInfo(&(eva_res_.gpu));
  getGtestInternalInfo();
}

void Executor::postProcessAfterLaunch() {
  // comupte for perf test
  const char *zero_element = std::getenv("MLUOP_GTEST_BUILD_ZERO_ELEMENT");
  if (zero_element != NULL) {
    std::string env_str = zero_element;
    int env_num = std::stoi(env_str);
    if (env_num == 1) {
      return;
    }
  }

  perfRepeat();

  VLOG(4) << "Device free (for workspace).";
  workspaceFree();

  VLOG(4) << "free compute param.";
  freeComputeParam();
  eva_res_.compute_completed = true;

  if (exe_config_->mlu_only) {
    return;
  }

  // The rest steps are for computing diffs
  VLOG(4) << "Copy data from device to host.";
  // mlu_only should never get here as host space is not allocated
  copyOut();
  recordGtestTimePoint("after_copy_out");

  VLOG(4) << "Host malloc (for baseline output, fp32)";
  baselineOutputMallocFunc(this);
  if (parser_->device() == CPU) {
    VLOG(4) << "Begin cpu compute.";
    cpuCompute();
    // if out dtype is half, cast cpu data from float to half to float,
    // consistent with mlu.
    castHalfOuput();
    VLOG(4) << "End cpu compute.";
  } else {
    // baseline output
    VLOG(4) << "Read in baseline device outputs.";
    getBaselineOutputFunc(this);  // read in baseline output
    recordGtestTimePoint("after_get_baseline_output");
  }

  VLOG(4) << "Host malloc (for mlu output, fp32).";
  mluOutputMallocFunc(this);
  recordGtestTimePoint("after_mlu_output_malloc");

  castOutFunc(this);
  recordGtestTimePoint("after_cast_out");

  diffPreprocess();

  strideOutputFunc(this);

  // save output data to file
  dumpOutputData();
}

// you should modify eva_res_.is_passed only in this function
void Executor::getAllTestResult() {
  // 1.check baseline
  bool baseline_check = true;
  if (exe_config_->perf_baseline) {
    getTestInfo();
    baseline_check = checkBaseline();
  }
  if (exe_config_->mlu_only) {
    eva_res_.is_passed = baseline_check;
    return;
  }

  // 2.check diff
  VLOG(4) << "Calculate diff between mlu and baseline device.";
  bool diff_check = checkDiff();
  recordGtestTimePoint("after_compute_diff");

  // 3.check oveerwritten
  bool overwritten_check = checkMluOverWritten();

  // 4.check accuracy
  bool accuracy_check = true;
  if (exe_config_->acc_baseline) {
    accuracy_check = checkAccuracyBaseline();
  }

  // 5.check mlu memory leak
  auto mlu_memory_leak_check = checkMluMemoryLeak();

  // 6.get final result
  // if need pass on the reason of failed cases,
  // move 5 check below to eva_res_.
  eva_res_.is_passed = diff_check && overwritten_check && baseline_check &&
                       accuracy_check && mlu_memory_leak_check;

  if (::testing::Test::HasFailure()) {
    eva_res_.is_passed = false;
  }
}

EvaluateResult Executor::teardown() {
  postProcessAfterLaunch();
  getAllTestResult();
  getTestInfo();
  return eva_res_;
}

size_t Executor::getProtoApiVersion() { return parser_->getProtoApiVersion(); }

void Executor::selectApiVersion() {
  // default use version in proto
  test_version_ =
      getProtoApiVersion() == -1 ? getLatestApiVersion() : getProtoApiVersion();
  if (exe_config_->compatible_test) {
    test_version_ = getLatestApiVersion();
  }
  VLOG(4) << "test_version = " << test_version_;
  switch (test_version_) {
    case 1:
      compute_func = &Executor::compute;
      break;
    case 2:
      compute_func = &Executor::compute_v2;
      break;
    case 3:
      compute_func = &Executor::compute_v3;
      break;
    case 4:
      compute_func = &Executor::compute_v4;
      break;
    case 5:
      compute_func = &Executor::compute_v5;
      break;
    case 6:
      compute_func = &Executor::compute_v6;
      break;
    case 7:
      compute_func = &Executor::compute_v7;
      break;
    default:
      GTEST_CHECK(false, "Not support api version higher than v7 now.");
  }
}

void Executor::getMluPerfInfo(PerfInfo *res) {
  // interface time
  double time = interface_timer_.duration(exe_config_->perf_repeat);
  res->interface_time = (time != 0) ? time : -1;

  // compute
  res->compute_force = getPeakComputeForce();
  if (parser_->node()->has_theory_compute_ops()) {
    res->theory_ops = parser_->node()->theory_compute_ops();
  } else {
    res->theory_ops = getTheoryOps();
  }

  // op / ( (latency(us) / 1000 / 1000) * PEAK_COMPUTE_FORCE(op/s) )
  res->compute_efficiency = eva_->computeEfficiency(
      res->theory_ops * 1000 * 1000, res->hardware_time, res->compute_force);

  // io
  res->io_bandwidth = getIoBandwidth();
  if (parser_->node()->has_theory_io_size()) {
    res->theory_io = parser_->node()->theory_io_size();
  } else {
    res->theory_io = getTheoryIoSize();
  }

  // io_size(byte) / ( (latency(us) / 1000 / 1000) * IO_BANDWIDTH(GB/s) )
  res->io_efficiency = eva_->computeEfficiency(
      res->theory_io /* * 1000 * 1000*/, res->hardware_time,
      res->io_bandwidth * 1000 /* * 1000 * 1000*/);

  res->workspace_size = eva_->getMluWorkspaceSize();
}

void Executor::getGpuPerfInfo(PerfInfo *res) {
  // compute
  if (parser_->node()->has_efficiency()) {
    res->compute_efficiency = parser_->node()->efficiency();
  }

  // io
  if (parser_->node()->has_io_efficiency()) {
    res->io_efficiency = parser_->node()->io_efficiency();
  }

  // latency
  if (parser_->node()->has_latency()) {
    auto latency_s = parser_->node()->latency();  // sec
    res->hardware_time = latency_s;               // us
  }

  // workspace size of GPU
  if (parser_->node()->has_workspace_size()) {
    res->workspace_size = parser_->node()->workspace_size();
  }
}

void Executor::getGtestInternalInfo() {
  auto &time_cost_ms = eva_res_.gtest.time_costs_ms;
  // wrap into GtestInternal
  for (auto &record : time_point_records_) {
    time_cost_ms.emplace_back(std::make_tuple(
        std::get<0>(record), std::chrono::duration<double, std::milli>(
                                 std::get<1>(record) - time_point_init_)
                                 .count()));
  }
  eva_res_.gtest.parsed_file_size = parser_->getParsedFileSize();
  eva_res_.gtest.parsed_cost_seconds = parser_->getParsedCostSeconds();
  global_var.internal_info_.record_case(eva_res_.case_path, eva_res_.gtest);
}

bool Executor::checkMluOverWritten() { return mlu_runtime_.checkOverWritten(); }

bool Executor::checkAccuracyBaseline() {
  bool accuracy_check = true;
  GTEST_CHECK(eva_res_.op_name != "",
              "Executor: missing op name, didn't set it. We need know it when "
              "get accuracy "
              "baseline threshold");
  std::string case_name = getCaseName(eva_res_.case_path);
  bool in_white_list = false;
  double threshold = 0;
  in_white_list = getAccuracyThreshold(eva_res_.op_name, &threshold);
  if (!in_white_list) {
    auto search = acc_baseline_map.find(case_name);
    if (search != acc_baseline_map.end()) {
      std::vector<double> errors;
      for (const auto &error : eva_res_.errors) {
        errors.push_back(error.error);
      }
      accuracy_check = checkAccuracyBaselineStrategy(case_name, search->second,
                                                     errors, threshold);
    } else {
      LOG(INFO) << "[Accuracy Baseline:" << case_name
                << "]:this case is new and do not have baseline data.";
    }
  }
  if (!accuracy_check) {
    eva_res_.what.emplace_back(
        "The accuracy result exceed baseline threshold.");
  }
  return accuracy_check;
}

// call this func after getMluHardwareTime()
// deal with baseline check of perf test
bool Executor::checkBaseline() {
  GTEST_CHECK(eva_res_.op_name != "",
              "Executor: missing op name, didn't set it. We need know it when "
              "get performance "
              "baseline threshold");

  double hw_time_base = 0;
  bool is_get_base_data = false;
  bool in_white_list = false;
  double hw_time_mean =
      eva_res_.mlu.hardware_time;  // eva_->getMluHardwareTime();
  double scale_bound = 0;
  double threshold_absolute = 0;
  double threshold_relative = 0;
  double workspace_size = 0;
  bool baseline_check = true;

  in_white_list = getThreshold(eva_res_.op_name, &scale_bound,
                               &threshold_absolute, &threshold_relative);

  if (in_white_list) {  // pass if in white list
    hw_time_base = hw_time_mean;
  } else {  // check baseline data in xml file
    std::string case_name = getTestCaseName(eva_res_.case_path);
    is_get_base_data = getTxtData(case_name, &hw_time_base, &workspace_size);
    if (is_get_base_data) {
      LOG(INFO) << "[Baseline:" << case_name
                << "]:hardware time of baseline is " << hw_time_base
                << " (us).";
      LOG(INFO) << "[Baseline:" << case_name
                << "]:workspace size of baseline is " << workspace_size
                << " (Bytes).";
    } else {  // new case
      LOG(INFO) << "[Baseline:" << case_name
                << "]:this case is new and do not have baseline data.";
    }
    if (is_get_base_data) {
      baseline_check =
          updateBaselineStrategy(hw_time_mean, scale_bound, threshold_absolute,
                                 threshold_relative, &hw_time_base);
      if (!baseline_check) {
        LOG(ERROR) << "[Baseline:" << case_name
                   << "]:scale_bound:" << scale_bound
                   << " ,threshold_absolute:" << threshold_absolute
                   << " ,threshold_relative:" << threshold_relative * 100
                   << "%";
        LOG(ERROR) << "[Baseline:" << case_name
                   << "]:hardware time of baseline is " << hw_time_base
                   << " (us).";
        LOG(ERROR) << "[Baseline:" << case_name
                   << "]:hardware time of this test is " << hw_time_mean
                   << " (us).";
        eva_res_.what.emplace_back(
            "The performance result exceed baseline threshold.");
      }
      if (!(workspace_size <= 0 && eva_res_.mlu.workspace_size <= 0) &&
          eva_res_.mlu.workspace_size > workspace_size) {
        LOG(WARNING) << "[Baseline:" << case_name
                     << "]:workspace size of baseline is " << workspace_size
                     << " (Bytes).";
        LOG(WARNING) << "[Baseline:" << case_name
                     << "]:workspace size of this test is "
                     << eva_res_.mlu.workspace_size << " (Bytes).";
      }
    } else {  // pass when new case
      hw_time_base = hw_time_mean;
    }
  }

  eva_res_.mlu.hardware_time_base = hw_time_base;
  return baseline_check;
}

double Executor::getBandWidthByDev() {
  int card = -1;
  GTEST_CHECK(CNRT_RET_SUCCESS == cnrtGetDevice(&card));
  GTEST_CHECK(cndevInit(0) == CNDEV_SUCCESS);
  cndevDDRInfo_t ddrinfo;
  ddrinfo.version = CNDEV_VERSION_5;
  GTEST_CHECK(cndevGetDDRInfo(&ddrinfo, card) == CNDEV_SUCCESS);
  double band_width = ddrinfo.bandWidth;
  double band_width_decimal = ddrinfo.bandWidthDecimal;
  do {
    band_width_decimal /= 10;
  } while (band_width_decimal > 1);
  return band_width + band_width_decimal;
}

// get compute force coefficient
int Executor::getIpuFrequency() {
  int ordinal = -1;
  int ipu_frequency = -1;
  GTEST_CHECK(CNRT_RET_SUCCESS == cnrtGetDevice(&ordinal));
  GTEST_CHECK(cndevInit(0) == CNDEV_SUCCESS);
  cndevFrequencyInfo_t freqInfo;
  freqInfo.version = CNDEV_VERSION_5;
  GTEST_CHECK(CN_SUCCESS ==
              cnDeviceGetAttribute(&ipu_frequency,
                                    CN_DEVICE_ATTRIBUTE_CLUSTER_CLOCK_RATE,
                                    ordinal));
  VLOG(4) << "IPU Frequency = " << (double)ipu_frequency / 1000 / 1000
          << " GHz";
  return ipu_frequency;
}

// return op/cycle
// don't forget * 1GHz to get peak compute force
double Executor::getCtPeakComputeForce() {
  GTEST_CHECK(parser_->inputs().size() >= 1,
              "Executor: when get ct peak force, we need at least 1 input, but "
              "now input num is < 1.");

  // ct peak compute force
  auto cluster_num =
      mluop::runtime::getClusterLimitCapability(exe_context_->handle);
  auto core_num = exe_context_->handle->core_num_per_cluster;
  double rate = (double)getIpuFrequency() / 1000 / 1000;
  switch (parser_->inputs()[0].dtype) {
    default:
      return CT_PEAK_FLOAT32_COMPUTE_FORCE * cluster_num * core_num * rate;
    case MLUOP_DTYPE_HALF:
    case MLUOP_DTYPE_INT16:
      return CT_PEAK_FLOAT16_COMPUTE_FORCE * cluster_num * core_num * rate;
    case MLUOP_DTYPE_FLOAT:
    case MLUOP_DTYPE_INT32:
      return CT_PEAK_FLOAT32_COMPUTE_FORCE * cluster_num * core_num * rate;
  }
}

bool Executor::useTf32ComputeForce() {
  return (useFloatConvInst() && isPlatformSupportTf32() && enableTf32MluEnv() &&
          opParamSupportTf32());
}

bool Executor::opParamSupportTf32() { return parser_->getOpTf32Param(); }

bool Executor::enableTf32MluEnv() { return IS_TF32_OVERRIDE; }

bool Executor::useFloatConvInst() {
  GTEST_CHECK(parser_->inputs().size() >= 2,
              "Executor: when judge whether use float conv,"
              " we need at least 2 input, but now input num is < 2.");
  getLtComputeDtype();
  if (lt_input_dtype_ == MLUOP_DTYPE_FLOAT &&
      lt_weight_dtype_ == MLUOP_DTYPE_FLOAT) {
    return true;
  } else {
    return false;
  }
}

bool Executor::isPlatformSupportTf32() {
  switch (exe_context_->handle->arch) {
    case MLUOP_MLU590: {
      return true;
      break;
    }
    // if a new platform support tf32, do not forget add here.
    default: {
      return false;
    }
  }
}

void Executor::getLtComputeDtype() {
  GTEST_CHECK(parser_->inputs().size() >= 2,
              "Executor: when get lt peak force, we need at least 2 input, but "
              "now input num is < 2.");
  MetaTensor *mt1 = parser_->input(0);
  MetaTensor *mt2 = parser_->input(1);
  lt_input_dtype_ =
      (mt1->oc_dt != MLUOP_DTYPE_INVALID) ? mt1->oc_dt : mt1->dtype;
  lt_weight_dtype_ =
      (mt2->oc_dt != MLUOP_DTYPE_INVALID) ? mt2->oc_dt : mt2->dtype;
}

void Executor::getComputeUnit() {
  if (lt_op_set.find(eva_res_.op_name) != lt_op_set.end()) {
    compute_unit_ = LT_COMPUTE;
  } else {
    compute_unit_ = CT_COMPUTE;
  }
}

// return op/cycle
// don't forget * 1GHz to get peak compute force
double Executor::getLtPeakComputeForce() {
  GTEST_CHECK(parser_->inputs().size() >= 2,
              "Executor: when get lt peak force, we need at least 2 input, but "
              "now input num is < 2.");
  MetaTensor *mt1 = parser_->input(0);
  MetaTensor *mt2 = parser_->input(1);
  auto dtype1 = (mt1->oc_dt != MLUOP_DTYPE_INVALID) ? mt1->oc_dt : mt1->dtype;
  auto dtype2 = (mt2->oc_dt != MLUOP_DTYPE_INVALID) ? mt2->oc_dt : mt2->dtype;
  auto cluster_num =
      mluop::runtime::getClusterLimitCapability(exe_context_->handle);
  auto platform = exe_context_->handle->arch;
  auto core_num = exe_context_->handle->core_num_per_cluster;
  double rate = (double)getIpuFrequency() / 1000 / 1000;
  if (useTf32ComputeForce()) {
    // tf32 + tf32
    VLOG(4) << "use tf32 compute efficiency in mluop_gtest";
    return LT_PEAK_TF32_TF32_COMPUTE_FORCE_590 * cluster_num * core_num * rate;
  }

  if (MLUOP_MLU220 == platform) {
    // dont have int4 + int4
    if (dtype1 == MLUOP_DTYPE_INT8 && dtype2 == MLUOP_DTYPE_INT8) {
      // int8 + int8
      return LT_PEAK_INT8_INT8_COMPUTE_FORCE_220 * cluster_num * core_num *
             rate;
    } else if ((dtype1 == MLUOP_DTYPE_INT8 && dtype2 == MLUOP_DTYPE_INT16) ||
               (dtype1 == MLUOP_DTYPE_INT16 && dtype2 == MLUOP_DTYPE_INT8)) {
      // int16 + int8
      return LT_PEAK_INT16_INT8_COMPUTE_FORCE_220 * cluster_num * core_num *
             rate;
    } else if (dtype1 == MLUOP_DTYPE_INT16 && dtype2 == MLUOP_DTYPE_INT16) {
      // int16 + int16
      return LT_PEAK_INT16_INT16_COMPUTE_FORCE_220 * cluster_num * core_num *
             rate;
    } else if (dtype1 == MLUOP_DTYPE_INT31 || dtype2 == MLUOP_DTYPE_INT31) {
      // int31
      return LT_PEAK_INT16_INT16_COMPUTE_FORCE_220 * cluster_num * core_num *
             rate;
    }
  } else if (MLUOP_MLU270 == platform || MLUOP_MLU290 == platform ||
             MLUOP_MLU370 == platform) {
    if (dtype1 == MLUOP_DTYPE_INT8 && dtype2 == MLUOP_DTYPE_INT8) {
      // int8 + int8
      return LT_PEAK_INT8_INT8_COMPUTE_FORCE_270_290_370 * cluster_num *
             core_num * rate;
    } else if ((dtype1 == MLUOP_DTYPE_INT8 && dtype2 == MLUOP_DTYPE_INT16) ||
               (dtype1 == MLUOP_DTYPE_INT16 && dtype2 == MLUOP_DTYPE_INT8)) {
      // int16 + int8
      return LT_PEAK_INT16_INT8_COMPUTE_FORCE_270_290_370 * cluster_num *
             core_num * rate;
    } else if (dtype1 == MLUOP_DTYPE_INT16 && dtype2 == MLUOP_DTYPE_INT16) {
      // int16 + int16
      return LT_PEAK_INT16_INT16_COMPUTE_FORCE_270_290_370 * cluster_num *
             core_num * rate;
    } else if (dtype1 == MLUOP_DTYPE_INT31 || dtype2 == MLUOP_DTYPE_INT31) {
      // int31
      if (dtype1 == MLUOP_DTYPE_INT31 && dtype2 == MLUOP_DTYPE_INT31) {
        // int31 == int16 * 2   so if int31 * 2, peak force / 4
        return LT_PEAK_INT16_INT16_COMPUTE_FORCE_270_290_370 * cluster_num *
               core_num / 4 * rate;
      } else {
        return LT_PEAK_INT16_INT16_COMPUTE_FORCE_270_290_370 * cluster_num *
               core_num / 2 * rate;
      }
    } else if (dtype1 == MLUOP_DTYPE_HALF && dtype2 == MLUOP_DTYPE_HALF) {
      // fp16 + fp16
      return LT_PEAK_FP16_FP16_COMPUTE_FORCE_370 * cluster_num * core_num *
             rate;
    } else if ((dtype1 == MLUOP_DTYPE_FLOAT && dtype2 == MLUOP_DTYPE_HALF) ||
               (dtype1 == MLUOP_DTYPE_HALF && dtype2 == MLUOP_DTYPE_FLOAT)) {
      // fp16 + fp32
      return LT_PEAK_FP32_FP16_COMPUTE_FORCE_370 * cluster_num * core_num *
             rate;
    } else if (dtype1 == MLUOP_DTYPE_FLOAT && dtype2 == MLUOP_DTYPE_FLOAT) {
      // fp32 + fp32
      return LT_PEAK_FP32_FP32_COMPUTE_FORCE_370 * cluster_num * core_num *
             rate;
    }
  } else if (MLUOP_MLU590 == platform) {
    if (dtype1 == MLUOP_DTYPE_INT8 && dtype2 == MLUOP_DTYPE_INT8) {
      // int8 + int8
      return LT_PEAK_INT8_INT8_COMPUTE_FORCE_590 * cluster_num * core_num *
             rate;
    } else if ((dtype1 == MLUOP_DTYPE_INT8 && dtype2 == MLUOP_DTYPE_INT16) ||
               (dtype1 == MLUOP_DTYPE_INT16 && dtype2 == MLUOP_DTYPE_INT8)) {
      // int16 + int8
      return LT_PEAK_INT16_INT8_COMPUTE_FORCE_590 * cluster_num * core_num *
             rate;
    } else if (dtype1 == MLUOP_DTYPE_INT16 && dtype2 == MLUOP_DTYPE_INT16) {
      // int16 + int16
      return LT_PEAK_INT16_INT16_COMPUTE_FORCE_590 * cluster_num * core_num *
             rate;
    } else if (dtype1 == MLUOP_DTYPE_HALF && dtype2 == MLUOP_DTYPE_HALF) {
      // fp16 + fp16
      return LT_PEAK_FP16_FP16_COMPUTE_FORCE_590 * cluster_num * core_num *
             rate;
    } else if (dtype1 == MLUOP_DTYPE_FLOAT && dtype2 == MLUOP_DTYPE_FLOAT) {
      // fp32 + fp32
      return LT_PEAK_FP32_FP32_COMPUTE_FORCE_590 * cluster_num * core_num *
             rate;
    } else if (dtype1 == MLUOP_DTYPE_INT31 || dtype2 == MLUOP_DTYPE_INT31) {
      // int31
      if (dtype1 == MLUOP_DTYPE_INT31 && dtype2 == MLUOP_DTYPE_INT31) {
        return LT_PEAK_INT16_INT16_COMPUTE_FORCE_590 * cluster_num * core_num /
               4 * rate;
      } else {
        return LT_PEAK_INT16_INT16_COMPUTE_FORCE_590 * cluster_num * core_num /
               2 * rate;
      }
    }
    // BF here
  }

  LOG(WARNING) << "Executor: got unsupported arch when get peak compute force.";
  return -1;
}

// return op/s
double Executor::getPeakComputeForce() {
  GTEST_CHECK(eva_res_.op_name != "",
              "Executor: missing op name, didn't set it. We need know it when "
              "get peak compute force.");

  getComputeUnit();
  if (LT_COMPUTE == compute_unit_) {
    return getLtPeakComputeForce() * 1000 * 1000 * 1000;  // * 1GHz // lt
  } else {
    return getCtPeakComputeForce() * 1000 * 1000 * 1000;  // * 1GHz
  }
}

int64_t Executor::getTheoryIoSize() {
  size_t total_size = 0;
  for (size_t i = 0; i < parser_->inputs().size(); ++i) {
    MetaTensor *ts = parser_->input(i);
    total_size += ts->shape_count * ts->sizeof_dtype;
  }
  for (size_t i = 0; i < parser_->outputs().size(); ++i) {
    MetaTensor *ts = parser_->output(i);
    total_size += ts->shape_count * ts->sizeof_dtype;
  }
  VLOG(4) << "Executor: getTheoryIOs: " << total_size << " bytes";
  return total_size;
}

double Executor::getIoBandwidth() {
  double io_bandwidth = -1;
  auto platform = exe_context_->handle->arch;
  io_bandwidth = getBandWidthByDev();
  auto cluster_num =
      mluop::runtime::getClusterLimitCapability(exe_context_->handle);
  if (cluster_num == 1 && platform == MLUOP_MLU370) {
    io_bandwidth = IO_BANDWIDTH_MLU370_SINGLE_CLUSTER;
  }
  VLOG(4) << "Executor: io bandwidth is " << io_bandwidth << " GB/s";
  return io_bandwidth;
}

// set params to handle
void Executor::setHandle() {
  // origin test case is off zero
  auto round_mode = mluoptest::ROUND_OFF_ZERO;
  if (parser_->getProtoNode()->has_handle_param()) {
    if (parser_->getProtoNode()->handle_param().has_round_mode()) {
      round_mode = parser_->getProtoNode()->handle_param().round_mode();
    }
  }
  if (round_mode == mluoptest::ROUND_TO_EVEN) {
    mluOpSetQuantizeRoundMode(handle_, MLUOP_ROUND_HALF_TO_EVEN);
  } else if (round_mode == mluoptest::ROUND_HALF_UP) {
    mluOpSetQuantizeRoundMode(handle_, MLUOP_ROUND_HALF_UP);
  } else if (round_mode == mluoptest::ROUND_OFF_ZERO) {
    mluOpSetQuantizeRoundMode(handle_, MLUOP_ROUND_HALF_OFF_ZERO);
  }
}

// create tensor desc
// and put them in 1 vector, but output tensor's is_output is true.
// and saved desc in MetaTensor's tensor and ctx->tensors
void Executor::createTensors() {
  auto create_tensor = [&](MetaTensor *mt) {
    if (unlikely((mt->null()))) {
      VLOG(4) << "Executor: skip creating tensor " << mt->name
              << ", set it as nullptr.";
      // if don't have this tensor, set it as nullptr;
      // push an desc as nullptr, and is_output marked as false.
      mt->tensor = nullptr;  //  keep meta tensor == tensor_desc_
      tensor_desc_.emplace_back(nullptr);
      return;
    }

    mluOpTensorDescriptor_t desc = nullptr;
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&desc));

    if (mt->is_cpu_scalar) {
      VLOG(4)
          << "Executor: isCpuScalar is true, start to set tensor attribute.";
      MLUOP_CHECK(
          mluOpSetTensorDescriptorPointerMode(desc, MLUOP_POINTER_MODE_HOST));
    }

    if (mt->stride.empty()) {  // if no stride testing this api.
      MLUOP_CHECK(mluOpSetTensorDescriptor_v2(
          desc, mt->layout, mt->dtype, mt->shape.size(), mt->shape.data()));
    } else {  // if has stride testing this api.
      MLUOP_CHECK(mluOpSetTensorDescriptorEx_v2(
          desc, mt->layout, mt->dtype, mt->shape.size(), mt->shape.data(),
          mt->stride.data()));
    }
    MLUOP_CHECK(mluOpSetTensorDescriptorOnchipDataType(desc, mt->oc_dt));

    if (parser_->device() != CPU) {
      // cpu-mode will set pos/scale when cast dtype
      MLUOP_CHECK(mluOpSetTensorDescriptorPositionAndScale(desc, mt->position,
                                                           mt->scale));
    }

    mt->tensor = desc;  //  keep meta tensor == tensor_desc_
    tensor_desc_.emplace_back(desc);
  };

  for (size_t i = 0; i < parser_->inputs().size(); ++i) {
    create_tensor(parser_->input(i));
  }

  if (flag_input_reuse_) {
    VLOG(4)
        << "Executor: skip creating output tensors, because of tensor reusing.";
    return;
  }

  // for all outputs
  for (size_t i = 0; i < parser_->outputs().size(); ++i) {
    create_tensor(parser_->output(i));
  }
}

void Executor::destroyTensors() noexcept {
  for (int i = 0; i < tensor_desc_.size(); ++i) {
    if (tensor_desc_[i].tensor != nullptr) {
      EXPECT_EQ(mluOpDestroyTensorDescriptor(tensor_desc_[i].tensor),
                MLUOP_STATUS_SUCCESS);
    }
  }
}

// -----------------------------------------------------------------
//   random(with stride)
//         |
//  malloc for cpu_fp32_in/out,mlu_fp32_out (without stride/only shape count)
//         |      (cast dtype and memcpy)
//         | ----------------------------->  host ptr (with strided/total_count)
//         |                                     | (memcpy h2d)
//         |                                  dev ptr
//         |                                     | (load strided if need(in
//         kernel))
//  cpu compute(only shape)                     mlu
//         |                                     | (store strided if need(in
//         kernel) |                                  dev ptr | | (memcpy d2h)
//         |                                  host ptr
//         |                                     | (cast dtype)
//    cpu output                            mlu output
//         | (strided if need)                   |
//         |                                     |
//         | <------------------------------------
//         |
//         |  (so dump input and output are strided, same as kernel)
//         v
//       diff
// -----------------------------------------------------------------

// malloc host ptr
// this ptr is for memcpy to mlu.
// if tensor has stride, this ptr size include stride
void Executor::hostMalloc() {
  auto initHostPtr = [this](MetaTensor *ts) {
    // need host space to store data if compute_diff(both input and output) or
    // rely_real_data_ (at least input, and output if compute_diff)
    if (mlu_need_host_data) {
      ts->host_ptr =
          cpu_runtime_.allocate(ts->total_count * ts->sizeof_dtype, ts->name);
      memset(ts->host_ptr, 0x0, ts->total_count * ts->sizeof_dtype);
      data_vector_.back().host_ptr = ts->host_ptr;
    }
  };
  for (size_t i = 0; i < parser_->inputs().size(); ++i) {
    MetaTensor *ts = parser_->input(i);
    data_vector_.emplace_back(ts, DramTensorType::ONLY_INPUT);

    if (unlikely(ts->empty())) {
      continue;
    }

    initHostPtr(ts);
  }

  // if input reuse, we don't create output.
  if (flag_input_reuse_) {
    // if reuse tensor don't need output, skip
    VLOG(4)
        << "Exeucutor: skip output host ptr malloc, because of tensor reusing.";
    return;
  }

  // create outputs
  for (size_t i = 0; i < parser_->outputs().size(); ++i) {
    MetaTensor *ts = parser_->output(i);
    data_vector_.emplace_back(ts, DramTensorType::ONLY_OUTPUT);

    if (unlikely(ts->empty())) {
      continue;
    }

    initHostPtr(ts);
  }
}

// malloc host ptr
void Executor::hostFree() noexcept {
  for (size_t i = 0; i < data_vector_.size(); ++i) {
    if (data_vector_[i].host_ptr != nullptr) {
      cpu_runtime_.deallocate(data_vector_[i].host_ptr);
      data_vector_[i].host_ptr = nullptr;
    }
  }
}

// call this function after host malloc
// read data from *pb.
// and write data to host ptr
// ONLY FOR NON-CPU MODE
void Executor::initHostData() {
  for (size_t i = 0; i < parser_->inputs().size(); ++i) {
    MetaTensor *ts = parser_->input(i);
    if (unlikely(ts->empty())) {
      cpu_input_.emplace_back(nullptr);
      continue;
    }

    // read data from prototxt.
    parser_->getInputTensorValue(i, data_vector_[i].host_ptr,
                                 data_vector_[i].count);
    cpu_input_.emplace_back(data_vector_[i].host_ptr);
  }
  saveInputWithStrideFunc(this);
}

void Executor::saveInputWithStride() {
  for (size_t i = 0; i < parser_->inputs().size(); ++i) {
    MetaTensor *ts = parser_->input(i);
    if (unlikely(ts->empty())) {
      continue;
    }

    // when tensor has stride, and input is reused, need use input to init
    // baselineOutput.
    if (!ts->stride.empty() && flag_input_reuse_) {
      cpu_fp32_stride_input_.push_back(
          (float *)cpu_runtime_.allocate(ts->total_count * sizeof(float)));
      void *temp_gpu = cpu_runtime_.allocate(
          ts->total_count * mluop::getSizeOfDataType(ts->dtype));
      memcpy(temp_gpu, data_vector_[i].host_ptr,
             ts->total_count * mluop::getSizeOfDataType(ts->dtype));
      // BUG(zhaolianshui): the last allocated cpu_fp32_stride_input may not be
      // i'th; always float?
      castDataOut(temp_gpu, ts->dtype, cpu_fp32_stride_input_[i],
                  getCpuDtype(ts->dtype), ts->total_count, NO_QUANT);
      cpu_runtime_.deallocate(temp_gpu);
    }
  }
}

void Executor::saveInputWithStrideByDtype() {
  for (size_t i = 0; i < parser_->inputs().size(); ++i) {
    MetaTensor *ts = parser_->input(i);
    if (unlikely(ts->empty())) {
      continue;
    }
    // TODO(None): use is_input_and_output, now it is bug
    // TODO(None): move it to class Stride
    if (!ts->stride.empty() && flag_input_reuse_) {
      cpu_stride_input_.push_back((float *)cpu_runtime_.allocate(
          ts->total_count * mluop::getSizeOfDataType(ts->dtype)));
      memcpy(cpu_stride_input_[i], data_vector_[i].host_ptr,
             ts->total_count * mluop::getSizeOfDataType(ts->dtype));
    }
  }
}

// determine dtype of the cpu array for input/output tensor
mluOpDataType_t Executor::getCpuDtype(mluOpDataType_t tensor_dtype) {
  switch (tensor_dtype) {
    // DOUBLE data is still stored as DOUBLE dtype in cpu array
    case MLUOP_DTYPE_DOUBLE: {
      return MLUOP_DTYPE_DOUBLE;
    } break;
    // each complex number is stored as a COMPLEX_FLOAT data in cpu array
    case MLUOP_DTYPE_COMPLEX_HALF:
    case MLUOP_DTYPE_COMPLEX_FLOAT: {
      return MLUOP_DTYPE_COMPLEX_FLOAT;
    } break;
    // the cpu array defaults to FLOAT dtype
    default: {
      return MLUOP_DTYPE_FLOAT;
    }
  }
}

// malloc for baseline input.
// actually only for cpu-mode, so it's fp32.
// and cast to mlu-dtype later, results will write to host_ptr.
// gpu-mode don't need this ptr, data will write to host ptr directly.
void Executor::baselineInputMalloc() {
  for (size_t i = 0; i < parser_->inputs().size(); ++i) {
    MetaTensor *ts = parser_->input(i);
    if (unlikely(ts->empty())) {
      cpu_fp32_input_.push_back(nullptr);
      continue;
    }
    // malloc a ptr with stride, to get random value
    // if this tensor has stride, will stride_in in castIn()
    size_t cpu_dtype_size = mluop::getSizeOfDataType(getCpuDtype(ts->dtype));
    ts->cpu_ptr = (float *)cpu_runtime_.allocate(
        ts->total_count * cpu_dtype_size, ts->name);
    cpu_fp32_input_.push_back(ts->cpu_ptr);
    cpu_fp32_stride_input_.push_back(nullptr);
    if (!ts->stride.empty() && flag_input_reuse_) {
      cpu_fp32_stride_input_[i] =
          (float *)cpu_runtime_.allocate(ts->total_count * cpu_dtype_size);
    }
    memset(ts->cpu_ptr, 0x0, ts->total_count * cpu_dtype_size);
  }
}

// free cpu_fp32_input_ and cpu_fp32_output_.
void Executor::baselineFree() noexcept {
  for (int i = 0; i < cpu_fp32_input_.size(); ++i) {
    cpu_runtime_.deallocate(cpu_fp32_input_[i]);
    cpu_fp32_input_[i] = nullptr;
  }
  for (int i = 0; i < cpu_fp32_output_.size(); ++i) {
    cpu_runtime_.deallocate(cpu_fp32_output_[i]);
    cpu_fp32_output_[i] = nullptr;
  }
  for (int i = 0; i < cpu_fp32_stride_input_.size(); ++i) {
    cpu_runtime_.deallocate(cpu_fp32_stride_input_[i]);
    cpu_fp32_stride_input_[i] = nullptr;
  }
}

// initialize cpu mode's input data.
// if *pb has random param or path, just generate random data or just read them.
// and put them on cpu_fp32_input_, because they are fp32.
// else, (value_i/value_f/value_h), read them in and cast to fp32.(if they are
// not fp32)
void Executor::initBaselineInput() {
  for (size_t i = 0; i < parser_->inputs().size(); ++i) {
    MetaTensor *ts = parser_->input(i);
    if (unlikely(ts->empty())) {
      continue;
    }
    mluOpDataType_t cpu_dtype = getCpuDtype(ts->dtype);
    if (VALUE_RANDOM == ts->value_type) {
      // generate random or read from path
      parser_->getInputTensorValue(i, cpu_fp32_input_[i], ts->total_count);
    } else {
      void *temp = cpu_runtime_.allocate(ts->total_count * ts->sizeof_dtype);
      // read in data and (copy/ cast) to cpu_fp32_input_
      parser_->getInputTensorValue(i, temp, ts->total_count);
      castDataOut(temp, ts->dtype,                // src data and dtype
                  cpu_fp32_input_[i], cpu_dtype,  // dst data and dtype
                  ts->total_count,                // count.
                  NO_QUANT, ts->position, ts->scale,
                  ts->offset);  // quant param.
      cpu_runtime_.deallocate(temp);
    }
  }
}

void Executor::mluOutputFree() noexcept {
  for (int i = 0; i < mlu_fp32_output_.size(); ++i) {
    // delete null is ok
    cpu_runtime_.deallocate(mlu_fp32_output_[i]);
    mlu_fp32_output_[i] = nullptr;
  }
}

void Executor::setLlcSize() {
  if (isSwift()) {
    llc_size_ = 48 * 1024 * 1024;
  } else if (isMagpie()) {
    llc_size_ = 96 * 1024 * 1024;
  }
}

// call this function after host_malloc
// malloc dev ptr, on mlu.
// memcpy data of host_ptr to this ptr later.
void Executor::deviceMalloc() {
  if (exe_config_->test_llc) {
    setLlcSize();
    GTEST_CHECK(CN_SUCCESS == cnMallocConstant(&const_addr_, llc_size_));
    const_dram_.push(const_addr_);
  }
  // TODO(None): once enable_const_dram is removed, move is_const_dram()
  // into DataBlock
  auto is_const_dram = [this](DataBlock *db) {
    return (exe_config_->enable_const_dram) && (db->is_only_input());
  };
  auto malloc_dev = [&is_const_dram, this](MetaTensor *mt, DataBlock *db) {
    if (skipMallocDevice(mt)) {
      VLOG(4) << "Executor: skip " << db->name << " device Malloc.";
      return;
    }

    void *dev_ptr = mlu_runtime_.allocate(db->size, mt->name, mt->sizeof_dtype,
                                          is_const_dram(db));
    mt->dev_origin_ptr = dev_ptr;
    db->device_origin_ptr = dev_ptr;
  };

  auto malloc_dev_perf = [this](MetaTensor *mt, DataBlock *db) {
    if (skipMallocDevice(mt)) {
      VLOG(4) << "Executor: skip " << db->name << " device Malloc.";
      return;
    }
    bool const_dram =
        false;  // constant memory can not be written by cnrtMemcpy_D2D
    void *dev_ptr =
        mlu_runtime_.allocate(db->size, mt->name, mt->sizeof_dtype, const_dram);
    mt->dev_perf_ptr = dev_ptr;
    db->device_perf_ptr = dev_ptr;
  };

  auto malloc_dev_data_perf = [this](MetaTensor *mt, DataBlock *db) {
    if (skipMallocDevice(mt)) {
      return;
    }
    bool const_dram =
        false;  // constant memory can not be written by cnrtMemcpy_D2D
    void *dev_data_ptr =
        mlu_runtime_.allocate(db->size, mt->name, mt->sizeof_dtype, const_dram);
    mt->dev_perf_data_ptr = dev_data_ptr;
    db->device_perf_data_ptr = dev_data_ptr;
  };

  if (flag_input_reuse_) {
    GTEST_CHECK(parser_->inputs().size() == data_vector_.size(),
                "Executor: tensor num in *pb is NOT equal to data_vector size, "
                "they should be equal.");
  } else {
    GTEST_CHECK(parser_->inputs().size() + parser_->outputs().size() ==
                    data_vector_.size(),
                "Executor: tensor num in *pb is NOT equal to data_vector size, "
                "they should be equal.");
  }

  // malloc for input.
  for (size_t i = 0; i < parser_->inputs().size(); ++i) {
    malloc_dev(parser_->input(i), &(data_vector_[i]));
  }
  // space to store real data
  if (needDevPerfDataSpace()) {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
      malloc_dev_data_perf(parser_->input(i), &(data_vector_[i]));
    }
  }

  if (needDevPerfSpace()) {
    for (size_t i = 0; i < parser_->inputs().size(); ++i) {
      malloc_dev_perf(parser_->input(i), &(data_vector_[i]));
    }
  }

  if (flag_input_reuse_) {
    // XXX(zhaolianshui): assume all outputs share input spaces, but this may
    // not be guaranteed for
    //                    some op, i.e some output tensors may need individual
    //                    space, in that case each tensor might need a reuse tag
    //                    in pb/prototxt
    // The following check is not complete
    GTEST_CHECK(parser_->outputs().size() <= parser_->inputs().size(),
                "Assumption failed: when flag_input_reuse_ is true, output "
                "tensor count should be"
                " less than input tensor count.");
    return;
  }

  // malloc for output.
  for (size_t i = 0; i < parser_->outputs().size(); ++i) {
    malloc_dev(parser_->output(i),
               &(data_vector_[i + parser_->inputs().size()]));
  }
  // a copy memory for perf-repeat-mode.
  if (needDevPerfSpace()) {
    for (size_t i = 0; i < parser_->outputs().size(); ++i) {
      malloc_dev_perf(parser_->output(i),
                      &(data_vector_[i + parser_->inputs().size()]));
    }
  }
}

size_t Executor::getTotalDevTensorSize() const {
  size_t sum = 0;
  for (auto &db : data_vector_) {
    sum += getAlignedMLUMemorySize(db.size);
  }
  // XXX(zhaolianshui): should add device workspace as well, but how?
  return sum;
}

void Executor::deviceRestSpaceMalloc() {
  // it should be in deviceMalloc with workspaceMalloc being moved ahead of
  // deviceMalloc, but some ops go awry as op_tensor, so keep the original order
  // big chunk of space for random addr perf
  if (needDevRandomSpace()) {
    // exclude the extra device space required in compute method
    std::vector<size_t> extra_dev_space_compute = getExtraDevSpaceCompute();
    auto alignSize = [](size_t init_sum, size_t allo_bytes) {
      return init_sum + getAlignedMLUMemorySize(allo_bytes);
    };
    size_t reserve_dev_bytes =
        std::accumulate(extra_dev_space_compute.begin(),
                        extra_dev_space_compute.end(), 0, alignSize);
    size_t total_bytes, free_bytes;
    GTEST_CHECK(cnrtMemGetInfo(&free_bytes, &total_bytes) == cnrtSuccess);
    free_bytes -= reserve_dev_bytes;

    GTEST_CHECK(
        free_bytes >= getTotalDevTensorSize(),
        "The rest DRAM space is not sufficient for random space allocation");

    mlu_runtime_.mmp->allocate(free_bytes);
  }
}

bool Executor::needDevRandomSpace() const {
  // get random device_perf_ptr inside the memory pool and assign to device_ptr
  // To make it simple, always use the same data as the very first warmup run.
  // if needDevPerfDataSpace, we need to copy from device_perf_data_ptr to
  // device_ptr; otherwise copy from device_origin_ptr to device_ptr
  return (doPerf() && exe_config_->random_mlu_address);
}

bool Executor::needDevPerfSpace() {
  // If needDevRandomSpace, such space will be inside the memory pool, so no
  // need to allocate individually.
  return (doPerf() && !needDevRandomSpace());
}

bool Executor::needDevPerfDataSpace() {
  // Some ops might have outputs sharing space with the inputs, so the input
  // might be modified after each run, which might lead to performance
  // inconsistency or runtime error. In such case, make a copy of the original
  // input tensors, and store in device_perf_data_ptr
  // XXX(zhaolianshui): Assume that the input won't be modified if input is not
  // reused and
  //                    output is independent of repeat
  return (doPerf() && flag_input_reuse_ && (rely_real_data_ || zero_input_));
}

// free device ptr
void Executor::deviceFree() noexcept {
  // for llc test, free 48MB llc memory
  freeLLC();
  for (int i = 0; i < data_vector_.size(); ++i) {
    // TODO(None): group those device ptrs into a vector?
    if (data_vector_[i].device_origin_ptr != nullptr) {
      EXPECT_EQ(mlu_runtime_.deallocate(data_vector_[i].device_origin_ptr),
                CNRT_RET_SUCCESS);
    }
    if (data_vector_[i].device_perf_ptr != nullptr && needDevPerfSpace()) {
      EXPECT_EQ(mlu_runtime_.deallocate(data_vector_[i].device_perf_ptr),
                CNRT_RET_SUCCESS);
    }
    if (data_vector_[i].device_perf_data_ptr != nullptr) {
      EXPECT_EQ(mlu_runtime_.deallocate(data_vector_[i].device_perf_data_ptr),
                CNRT_RET_SUCCESS);
    }
  }
  if (needDevRandomSpace()) {
    mlu_runtime_.mmp->destroy();
  }
  if (::testing::UnitTest::GetInstance()
          ->current_test_info()
          ->result()
          ->Failed()) {
    return;
  }
  if (eva_res_.compute_completed) {
    EXPECT_EQ(mlu_runtime_.getMemBlocksSize(), 0)
        << "MLU Memory leaked that should be deallocate by user explicitly"
        << "(case: " << eva_res_.case_path << ")";
  }
  EXPECT_EQ(mlu_runtime_.destroy(), CNRT_RET_SUCCESS);
}

void Executor::freeLLC() {
  if (!const_dram_.empty()) {
    // BUG(zhaolianshui): called inside ~executor, so should not throw, replace
    // GTEST_CHECK with
    //                    other proper check
    GTEST_CHECK(CN_SUCCESS == cnFree(const_dram_.top()));
    const_dram_.pop();
  }
}

// get pos/scale/offset of src_data
// if dtype is int8/int16/int31 may get p/s/o by quant_mode.
// else dtype just return 0/1/0.
void Executor::getQuantizedParam(float *src_data, size_t count,
                                 mluOpDataType_t dst_dtype,
                                 QuantMode quant_mode, int *position,
                                 float *scale, int *offset) {
  *position = 0;
  *scale = 1.0f;
  if (nullptr != offset) {
    *offset = 0;
  }
  // dtype may need quant
  // !! same with castDataIn
  if (MLUOP_DTYPE_INT8 == dst_dtype || MLUOP_DTYPE_INT16 == dst_dtype) {
    // get position scale offset
    if (ONLY_POSITION == quant_mode) {
      MLUOP_CHECK(mluop::getPosition(src_data, count, dst_dtype, position));
    } else if (POSITION_SCALE == quant_mode) {
      MLUOP_CHECK(mluop::getPositionAndScale(src_data, count, dst_dtype,
                                             position, scale));
      // only for symmetric quantify int8/16
      if (parser_->negative_scale_) {
        *scale = -*scale;
      }
    } else if (POS_SCALE_OFFSET == quant_mode && nullptr != offset) {
      MLUOP_CHECK(mluop::getPositionScaleAndOffset(src_data, count, dst_dtype,
                                                   position, scale, offset));
    } else if (NO_QUANT == quant_mode) {
      // don't need quant.
      // aka, set pos == 0; scale == 1; offset == 0
    } else {
      GTEST_CHECK(
          false,
          "Executor: when get quant param, found unsupported quant_mode.");
    }
  } else if (MLUOP_DTYPE_INT31 == dst_dtype) {
    // get position scale offset
    if (ONLY_POSITION == quant_mode) {
      MLUOP_CHECK(mluop::getPosition(src_data, count, dst_dtype, position));
    } else if (POSITION_SCALE == quant_mode) {
      MLUOP_CHECK(mluop::getPositionAndScale(src_data, count, dst_dtype,
                                             position, scale));
    } else if (POS_SCALE_OFFSET == quant_mode) {
      MLUOP_CHECK(mluop::getPositionScaleAndOffset(src_data, count, dst_dtype,
                                                   position, scale, offset));
    } else if (NO_QUANT == quant_mode) {
      // don't need quant.
      // aka, set pos == 0; scale == 1
    } else {
      GTEST_CHECK(
          false,
          "Executor: when get quant param, found unsupported quant_mode.");
    }
  } else {
    // aka, set pos == 0; scale == 1; offset == 0
  }
}

// cast data from fp32 -> X
// and return p/s.
// if dequantify is true, reset src_data.
void Executor::castDataIn(float *src_data, mluOpDataType_t src_dtype,
                          void *dst_data, mluOpDataType_t dst_dtype,
                          size_t count, QuantMode quant_mode, int *pos,
                          float *scale, int *offset, bool dequantify,
                          bool online_quantize) {
  if (count == 0) {
    VLOG(4) << "skip castDataIn: count is zero";
    return;
  }
  if (src_dtype == dst_dtype) {
    memcpy(dst_data, src_data, count * mluop::getSizeOfDataType(src_dtype));
  } else if ((src_dtype == MLUOP_DTYPE_FLOAT &&
              dst_dtype == MLUOP_DTYPE_INT8) ||
             (src_dtype == MLUOP_DTYPE_FLOAT &&
              dst_dtype == MLUOP_DTYPE_INT16)) {
    // need quant
    if (online_quantize) {
      getQuantizedParam(src_data, count, dst_dtype, quant_mode, pos, scale,
                        offset);
    }
    VLOG(4) << "skip castDataIn: count is zero" << *scale;
    if (dst_dtype == MLUOP_DTYPE_INT8) {
      MLUOP_CHECK(mluop::castFloat32ToFixed(src_data, (int8_t *)dst_data, count,
                                            *pos, *scale, *offset,
                                            handle_->round_mode));
      if (dequantify) {
        MLUOP_CHECK(mluop::castFixedToFloat32((int8_t *)dst_data, src_data,
                                              count, *pos, *scale, *offset));
      }
    } else if (dst_dtype == MLUOP_DTYPE_INT16) {
      MLUOP_CHECK(mluop::castFloat32ToFixed(src_data, (int16_t *)dst_data,
                                            count, *pos, *scale, *offset,
                                            handle_->round_mode));
      if (dequantify) {
        MLUOP_CHECK(mluop::castFixedToFloat32((int16_t *)dst_data, src_data,
                                              count, *pos, *scale, *offset));
      }
    }
  } else if (src_dtype == MLUOP_DTYPE_FLOAT && dst_dtype == MLUOP_DTYPE_INT31) {
    *pos = 0;
    *scale = 1.0;
    *offset = 0;
    getQuantizedParam(src_data, count, dst_dtype, quant_mode, pos, scale,
                      offset);
    MLUOP_CHECK(mluop::castFloat32ToInt31(src_data, count, dst_data));
    // int31 don't need reset cpu data
  } else if ((src_dtype == MLUOP_DTYPE_FLOAT &&
              (dst_dtype == MLUOP_DTYPE_INT64 ||
               dst_dtype == MLUOP_DTYPE_UINT64 ||
               dst_dtype == MLUOP_DTYPE_INT32 ||
               dst_dtype == MLUOP_DTYPE_UINT32 ||
               dst_dtype == MLUOP_DTYPE_UINT16 ||
               dst_dtype == MLUOP_DTYPE_HALF ||
               dst_dtype == MLUOP_DTYPE_BFLOAT16 ||
               dst_dtype == MLUOP_DTYPE_UINT8 ||
               dst_dtype == MLUOP_DTYPE_BOOL)) ||
             (src_dtype == MLUOP_DTYPE_COMPLEX_FLOAT &&
              dst_dtype == MLUOP_DTYPE_COMPLEX_HALF)) {
    arrayCastFloatAndNormalWrapper(src_data, src_dtype, dst_data, dst_dtype,
                                   count);
    if (dequantify) {
      arrayCastFloatAndNormalWrapper(dst_data, dst_dtype, src_data, src_dtype,
                                     count);
    }
  } else {
    GTEST_CHECK(false,
                "Executor: when cast fp32 to dtype, found unsupported dtype.");
  }
}

void Executor::cnrtCastDataTypeWrap(void *src_data,
                                    const mluOpDataType_t src_dtype,
                                    float *dst_data,
                                    const mluOpDataType_t dst_dtype,
                                    const size_t count,
                                    const cnrtQuantizedParam_t quant_param) {
  auto in_dtype = cvtMluOpDtypeToCnrt_V2(src_dtype);
  auto out_dtype = cvtMluOpDtypeToCnrt_V2(dst_dtype);
  size_t count_repeat = count / INT_MAX;
  size_t count_remain = count % INT_MAX;
  char *src = reinterpret_cast<char *>(src_data);
  char *dst = reinterpret_cast<char *>(dst_data);
  for (size_t i = 0; i < count_repeat; ++i) {
    GTEST_CHECK(CNRT_RET_SUCCESS ==
                cnrtCastDataType_V2(src, in_dtype, dst, out_dtype, INT_MAX,
                                    quant_param, cnrtRounding_rm));
    src += INT_MAX * mluop::getSizeOfDataType(src_dtype);
    dst += INT_MAX * mluop::getSizeOfDataType(dst_dtype);
  }
  if (count_remain) {
    GTEST_CHECK(CNRT_RET_SUCCESS ==
                cnrtCastDataType_V2(src, in_dtype, dst, out_dtype, count_remain,
                                    quant_param, cnrtRounding_rm));
  }
}

// set p/s/o, cast data from fp32 -> X
void Executor::castDataOut(void *src_data, mluOpDataType_t src_dtype,
                           float *dst_data, mluOpDataType_t dst_dtype,
                           size_t count, QuantMode quant_mode, int pos,
                           float scale, int offset) {
  if (count == 0) {
    VLOG(4) << "skip castDataOut: count is zero";
    return;
  }
  if (src_dtype == dst_dtype) {
    memcpy(dst_data, src_data, count * mluop::getSizeOfDataType(src_dtype));
  } else if (src_dtype == MLUOP_DTYPE_COMPLEX_HALF &&
             dst_dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
    arrayCastFloatAndNormalWrapper(src_data, src_dtype, dst_data, dst_dtype,
                                   count);
  } else if ((src_dtype == MLUOP_DTYPE_INT8 &&
              dst_dtype == MLUOP_DTYPE_FLOAT) ||
             (src_dtype == MLUOP_DTYPE_INT16 &&
              dst_dtype == MLUOP_DTYPE_FLOAT)) {
    if (flag_quant_mode_ == NO_QUANT) {
      pos = 0;
      scale = 1;
      offset = 0;
    }
    // need quant
    if (flag_quant_mode_ != POS_SCALE_OFFSET) {
      cnrtQuantizedParam_t quant_param = nullptr;
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtCreateQuantizedParam(&quant_param, pos, scale, offset));
      cnrtCastDataTypeWrap(src_data, src_dtype, dst_data, dst_dtype, count,
                           quant_param);
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtDestroyQuantizedParam(quant_param));
    } else {
      if (src_dtype == MLUOP_DTYPE_INT8) {
        MLUOP_CHECK(mluop::castFixedToFloat32((int8_t *)src_data, dst_data,
                                              count, pos, scale, offset));
      } else if (src_dtype == MLUOP_DTYPE_INT16) {
        MLUOP_CHECK(mluop::castFixedToFloat32((int16_t *)src_data, dst_data,
                                              count, pos, scale, offset));
      }
    }
  } else if (dst_dtype == MLUOP_DTYPE_FLOAT &&
             (src_dtype == MLUOP_DTYPE_HALF ||
              src_dtype == MLUOP_DTYPE_BFLOAT16 ||
              src_dtype == MLUOP_DTYPE_BOOL || src_dtype == MLUOP_DTYPE_INT32 ||
              src_dtype == MLUOP_DTYPE_INT64 ||
              src_dtype == MLUOP_DTYPE_UINT8 ||
              src_dtype == MLUOP_DTYPE_UINT16 ||
              src_dtype == MLUOP_DTYPE_UINT32 ||
              src_dtype == MLUOP_DTYPE_UINT64)) {
    arrayCastFloatAndNormalWrapper(src_data, src_dtype, dst_data, dst_dtype,
                                   count);
  } else if (src_dtype == MLUOP_DTYPE_UINT8 && dst_dtype == MLUOP_DTYPE_HALF) {
    cnrtCastDataTypeWrap(src_data, src_dtype, dst_data, dst_dtype, count,
                         nullptr);
  } else if (src_dtype == MLUOP_DTYPE_INT31 && dst_dtype == MLUOP_DTYPE_FLOAT) {
    MLUOP_CHECK(mluop::castInt31ToFloat32(src_data, dst_data, count, pos));
  } else {
    LOG(WARNING) << "Executor::castDataOut(): Cast "
                 << mluop::getNameOfDataType(src_dtype) << " to "
                 << mluop::getNameOfDataType(dst_dtype) << " is not supported.";
    GTEST_CHECK(false, "Executor: Unsupported dtype cast.");
  }
}

void Executor::quantizeTensorByChannel(
    float *src_data, void *dst_data, mluOpDataType_t dst_dtype, size_t count,
    int tensor_index, mluoptest::ConvolutionCastMode cast_mode) {
  auto shape = parser_->node()->input(1).shape();
  auto layout = parser_->node()->input(1).layout();
  if (layout != mluoptest::LAYOUT_NDHWC || layout != mluoptest::LAYOUT_NHWC) {
    LOG(ERROR) << "unsupport data layput for convolution forward";
    return;
  }
  int co = shape.dims(0);
  int *position = (int *)malloc(co * sizeof(int));
  float *scale = (float *)malloc(co * sizeof(float));
  int *offset = (int *)malloc(co * sizeof(int));
  int deal_count = count / co;
  for (int co_index = 0; co_index < co; ++co_index) {
    castDataIn(
        src_data + deal_count, MLUOP_DTYPE_FLOAT,
        (char *)dst_data + deal_count * mluop::getSizeOfDataType(dst_dtype),
        dst_dtype, deal_count, flag_quant_mode_, position + co_index,
        scale + co_index, offset + co_index, true);
  }
  // MLUOP_CHECK(mluOpSetTensorDescriptorPositionScaleOffsetByChannel(
  //       tensor_desc_[tensor_index].tensor, co, position, scale, offset));
  free(position);
  free(scale);
  free(offset);
}

// only for cpu-mode,
// cast fp32 to dtype (cpu_fp32_input_ -> host_ptr)
// cpu_fp32_input_ add stride and memcpy to host_ptr
void Executor::castIn() {
  for (size_t i = 0; i < parser_->inputs().size(); ++i) {
    MetaTensor *ts = parser_->input(i);
    if (unlikely(ts->empty())) {
      continue;
    }
    auto input_node = parser_->getProtoNode()->input(i);
    float *src_data = cpu_fp32_input_[i];
    void *dst_data = data_vector_[i].host_ptr;
    mluOpDataType_t cpu_dtype = getCpuDtype(ts->dtype);

    // when 1. has quantize param and has real data
    //      2. quant_mode != NO_QUANT
    // use quantize param in prototxt, else init them by 0, 1.0f, 0
    bool online_quantize = (!(input_node.has_position() &&
                              parser_->input(i)->value_type != VALUE_RANDOM)) ||
                           flag_quant_mode_ == NO_QUANT;
    int p = input_node.has_position() ? input_node.position() : 0;
    int o = input_node.has_offset() ? input_node.offset() : 0;
    float s = input_node.has_scale() ? input_node.scale() : 1.0f;

    if (ts->oc_dt == MLUOP_DTYPE_INVALID || ts->oc_dt == ts->dtype ||
        flag_quant_mode_ == NO_QUANT) {
      // if no onchip p/s
      // just cast data from fp32 to dtype
      // then memcpy this to mlu
      castDataIn(src_data, cpu_dtype,  // src data and dtype (fp32)
                 dst_data, ts->dtype,  // dst data and dtype (in *pb)
                 ts->total_count,      // count
                 flag_quant_mode_, &p, &s, &o, true,
                 online_quantize);  // returned p/s
      MLUOP_CHECK(mluOpSetTensorDescriptorPositionAndScale(
          tensor_desc_[i].tensor, p, s));
    } else {
      // if has onchip_dtype
      GTEST_CHECK(
          (ts->dtype != MLUOP_DTYPE_DOUBLE) &&
              (ts->dtype != MLUOP_DTYPE_COMPLEX_HALF) &&
              (ts->dtype != MLUOP_DTYPE_COMPLEX_FLOAT),
          "Executor::castIn(): DOUBLE and COMPLEX dtypes are not supported "
          "when quantization is enabled!");
      // cast fp32 to onchip dtype to get p/s and dequantify fp32 data (let cpu
      // input == mlu input) and cast fp32 to offchip dtype then memcpy this to
      // mlu
      castDataIn(src_data, MLUOP_DTYPE_FLOAT,  // src data
                 dst_data, ts->dtype,  // dst data, memcpy this to mlu later.
                 ts->total_count,      // count
                 flag_quant_mode_, &p, &s, &o, true,
                 online_quantize);  // p/s, discarded.

      // get oc_dt's p/s and set to tensor.
      void *temp = cpu_runtime_.allocate(ts->total_count *
                                         mluop::getSizeOfDataType(ts->oc_dt));
      castDataIn(src_data, MLUOP_DTYPE_FLOAT,  // src data
                 temp, ts->oc_dt,              // dst data
                 ts->total_count,              // count
                 flag_quant_mode_, &p, &s, &o,
                 true,  // returned p/s, set to tensor.
                 online_quantize);
      MLUOP_CHECK(mluOpSetTensorDescriptorPositionAndScale(
          tensor_desc_[i].tensor, p, s));
      cpu_runtime_.deallocate(temp);
    }
    if (!ts->stride.empty()) {
      VLOG(4) << "Executor: " << ts->name << " host ptr been strided_out.";
      size_t cpu_dtype_size = mluop::getSizeOfDataType(getCpuDtype(ts->dtype));
      void *temp = cpu_runtime_.allocate(ts->shape_count * cpu_dtype_size);
      memset(temp, 0x0, ts->shape_count * cpu_dtype_size);
      if (flag_input_reuse_) {
        memcpy(cpu_fp32_stride_input_[i], cpu_fp32_input_[i],
               ts->total_count * cpu_dtype_size);
      }
      tensor_stride_in(temp, cpu_fp32_input_[i], getTensorShapeSizeT(ts),
                       getTensorStrideSizeT(ts), cpu_dtype_size);
      cpu_runtime_.deallocate(cpu_fp32_input_[i]);
      cpu_fp32_input_[i] = (float *)temp;
      ts->cpu_ptr = (float *)temp;
    }

    if (exe_config_->dump_data) {
      saveDataToFile("baseline_" + ts->name, cpu_fp32_input_[i],
                     getCpuDtype(ts->dtype), ts->shape_count);
    }
  }
}

void Executor::switchDataToOrigin() {
  for (int i = 0; i < data_vector_.size(); ++i) {
    if (parser_->getMetaTensor(i).total_count != 0) {
      void *temp_ptr = parser_->getMetaTensor(i).dev_origin_ptr;
      parser_->getMetaTensor(i).dev_ptr = temp_ptr;
    } else {
      // don't have this tensor, it's null
    }
    data_vector_[i].device_ptr = parser_->getMetaTensor(i).dev_ptr;
  }
}

void Executor::switchDataToPerf() {
  for (int i = 0; i < data_vector_.size(); ++i) {
    if (parser_->getMetaTensor(i).total_count != 0) {
      void *temp_ptr = parser_->getMetaTensor(i).dev_perf_ptr;
      parser_->getMetaTensor(i).dev_ptr = temp_ptr;
    } else {
      // don't have this tensor, it's null
    }
    data_vector_[i].device_ptr = parser_->getMetaTensor(i).dev_ptr;
  }
}

void Executor::copyIn() {
  // ONLY_INPUT + BOTH_INPUT_OUTPUT
  auto input_blocks = getInputBlocks();
  for (size_t i = 0; i < input_blocks.size(); ++i) {
    DataBlock *db = input_blocks[i];

    // memcpy only for input
    if (unlikely(skipMallocDevice(db->getMetaTensor()))) {
      VLOG(4) << "Executor: skip " << db->name << " memcpy device => host.";
      continue;
    }

    // TODO(None): should use cnrtmemcpyasync to do host2dev dev2dev
    // copy in the same queue memcpy host to dev
    if (zero_input_) {
      VLOG(4) << "set device_origin_ptr space to 0";
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMemset(db->device_origin_ptr, 0, db->size));
    } else if (mlu_need_host_data) {
      // use_real_data: a) compute diff; b) data_dependent (otherwise maybe
      // runtime error )
      auto t_a = std::chrono::system_clock::now();
      // host to dev for compute
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMemcpy(db->device_origin_ptr,
                                                 db->host_ptr, db->size,
                                                 CNRT_MEM_TRANS_DIR_HOST2DEV));
      auto t_b = std::chrono::system_clock::now();
      auto dur =
          std::chrono::duration_cast<std::chrono::microseconds>(t_b - t_a);
      eva_res_.mlu.h2d_time += dur.count();
    }
    if (needDevPerfDataSpace()) {
      VLOG(4) << "copy from device_origin_ptr to device_perf_data_ptr";
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMemcpy(db->device_perf_data_ptr, db->device_origin_ptr,
                             db->size, CNRT_MEM_TRANS_DIR_DEV2DEV));
    }
    // for debug
    if (exe_config_->dump_data) {
      saveHexDataToFile("hex_" + db->name, db->host_ptr, db->dtype, db->count);
    }
  }
  // if flag_input_reuse_ is true, there should be no output block records up to
  // this point
  // TODO(None): add a gest_check?
  // ONLY_OUTPUT
  auto output_blocks = getOutputBlocks(false);
  for (size_t i = 0; i < output_blocks.size(); ++i) {
    DataBlock *db = output_blocks[i];
    if (!db->stride.empty()) {
      // when output has stride param
      if (unlikely(db->size == 0)) {
        VLOG(4) << "Executor: skip " << db->name << " memcpy device => host.";
        continue;
      }
      // set zeros to dev
      auto t_a = std::chrono::system_clock::now();
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMemset(db->device_origin_ptr, 0, db->size));
      auto t_b = std::chrono::system_clock::now();
      auto dur =
          std::chrono::duration_cast<std::chrono::microseconds>(t_b - t_a);
      eva_res_.mlu.h2d_time += dur.count();

      // FIXME(zhaolianshui): perf_data is not for copute_diff, so no need to
      // set to 0
      if (needDevPerfDataSpace()) {
        // set zeros to dev for perf test
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMemset(db->device_perf_data_ptr, 0, db->size));
      }
    }
  }
}

void Executor::copyOut() {
  // ONLY_OUTPUT + BOTH_INPUT_OUTPUT
  auto output_blocks = getOutputBlocks(true);
  for (int i = 0; i < output_blocks.size(); ++i) {
    DataBlock *db = output_blocks[i];

    // memcpy only for output
    if (unlikely(db->size == 0)) {
      VLOG(4) << "Executor: skip " << db->name << " memcpy device => host.";
      continue;
    }

    // memcpy dev to host
    auto t_a = std::chrono::system_clock::now();
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMemcpy(db->host_ptr,
                                               db->device_origin_ptr, db->size,
                                               CNRT_MEM_TRANS_DIR_DEV2HOST));
    auto t_b = std::chrono::system_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t_b - t_a);
    eva_res_.mlu.d2h_time += dur.count();

    // for debug
    if (exe_config_->dump_data) {
      saveHexDataToFile("hex_" + db->name, db->host_ptr, db->dtype, db->count);
    }
  }
}

std::vector<DataBlock *> Executor::getInputBlocks() {
  std::vector<DataBlock *> temp;
  for (int i = 0; i < data_vector_.size(); ++i) {
    if (data_vector_[i].is_input()) {
      temp.emplace_back(&data_vector_[i]);
    }
  }
  return temp;
}

std::vector<DataBlock *> Executor::getOutputBlocks(bool include_both_inout) {
  std::vector<DataBlock *> temp;
  if (include_both_inout) {
    for (int i = 0; i < data_vector_.size(); ++i) {
      if (data_vector_[i].is_output()) {
        temp.emplace_back(&data_vector_[i]);
      }
    }
  } else {
    for (int i = 0; i < data_vector_.size(); ++i) {
      if (data_vector_[i].is_only_output()) {
        temp.emplace_back(&data_vector_[i]);
      }
    }
  }
  return temp;
}

void Executor::castHalfOuput() {
  // ONLY_OUTPUT + BOTH_INPUT_OUTPUT
  auto output_blocks = getOutputBlocks(true);
  for (int i = 0; i < output_blocks.size(); ++i) {
    if (output_blocks[i]->size == 0) {
      continue;  // null output
    }
    MetaTensor *ts = parser_->output(i);
    // TODO(None): convert complex_half as well
    GTEST_WARNING(ts->dtype != MLUOP_DTYPE_COMPLEX_HALF,
                  "After cpuCompute, "
                  "complex_float->complex_half->complex_float conversion is "
                  "not implemented yet.");
    if (ts->dtype == MLUOP_DTYPE_HALF) {
      int16_t *half_data = (int16_t *)cpu_runtime_.allocate(
          ts->shape_count * mluop::getSizeOfDataType(ts->dtype));
      arrayCastFloatToHalf(half_data, cpu_fp32_output_[i], ts->shape_count);
      if (getFlagHalfInfTo65504()) {
        arrayCastHalfToFloatInvalidInf(cpu_fp32_output_[i], half_data,
                                       ts->shape_count);
      } else {
        arrayCastHalfToFloat(cpu_fp32_output_[i], half_data, ts->shape_count);
      }
      cpu_runtime_.deallocate(half_data);
    } else if (ts->dtype == MLUOP_DTYPE_BFLOAT16) {
      uint16_t *bf16_data = (uint16_t *)cpu_runtime_.allocate(
          ts->shape_count * mluop::getSizeOfDataType(ts->dtype));
      arrayCastFloatToBF16(bf16_data, cpu_fp32_output_[i], ts->shape_count);
      arrayCastBF16ToFloat(cpu_fp32_output_[i], bf16_data, ts->shape_count);
      cpu_runtime_.deallocate(bf16_data);
    }
  }
}

void Executor::fillRam() {
  char *fill_value = getenv("MLUOP_GTEST_FILL_RAM");
  nram_value value = NAN_FLOAT;

  // User defined value
  if (fill_value && strcmp(fill_value, "NAN_HALF") == 0) {
    value = NAN_HALF;
  } else if (fill_value && strcmp(fill_value, "INF_HALF") == 0) {
    value = INF_HALF;
  } else if (fill_value && strcmp(fill_value, "NAN_FLOAT") == 0) {
    value = NAN_FLOAT;
  } else if (fill_value && strcmp(fill_value, "INF_FLOAT") == 0) {
    value = INF_FLOAT;
  } else if (fill_value && strcmp(fill_value, "OFF") == 0) {
    value = NO_FILL;
  } else {
    // Default: randomly fills RAM with nan,inf values
    std::random_device rd;
    std::default_random_engine rand_gen{rd()};
    std::uniform_int_distribution<int> rand_idx(0, 3);
    switch (rand_idx(rand_gen)) {
      default:
      case 0: {
        fill_value = (char *)"NAN_FLOAT";
        value = NAN_FLOAT;
      }; break;
      case 1: {
        fill_value = (char *)"INF_FLOAT";
        value = INF_FLOAT;
      }; break;
      case 2: {
        fill_value = (char *)"NAN_HALF";
        value = NAN_HALF;
      }; break;
      case 3: {
        fill_value = (char *)"INF_HALF";
        value = INF_HALF;
      }; break;
    }
  }

  if (value != NO_FILL) {
    VLOG(4) << "Fill nram/sram/wram with " << fill_value;
    mluOpFillRam(handle_, value);
  }
}

void Executor::jobLimitCheck() {
  char *set_job_limit_ptr = getenv("MLUOP_SET_JOB_LIMIT_CAPABILITY");
  if (set_job_limit_ptr) {
    uint32_t set_job_limit = atoi(set_job_limit_ptr);
    VLOG(4) << "set job limit env successfully " << set_job_limit;
    uint32_t job_limit = mluop::runtime::getJobLimitCapability(handle_);
    VLOG(4) << "job_limit_before = " << job_limit;
    KernelClass cn_kernel_class = CN_KERNEL_CLASS_UNION4;
    switch (set_job_limit) {
      case 1:
        cn_kernel_class = CN_KERNEL_CLASS_UNION;
        break;
      case 2:
        cn_kernel_class = CN_KERNEL_CLASS_UNION2;
        break;
      case 3:
        cn_kernel_class = CN_KERNEL_CLASS_UNION4;
        break;
      case 4:
        cn_kernel_class = CN_KERNEL_CLASS_UNION8;
        break;
      case 5:
        cn_kernel_class = CN_KERNEL_CLASS_UNION16;
        break;
      case 6:
        // not use
        cn_kernel_class = CN_KERNEL_CLASS_BLOCK;
        break;
      case 7:
        // not use
        cn_kernel_class = CN_KERNEL_CLASS_NONE;
        break;
      default:
        LOG(WARNING) << "Executor: got unsupported job limit number."
                     << " Use default CN_KERNEL_CLASS_UNION4.";
    }
    setJobLimitCapability(cn_kernel_class);
    job_limit = mluop::runtime::getJobLimitCapability(handle_);
    VLOG(4) << "job_limit_after = " << job_limit;
  }
}

void Executor::clusterLimitCheck() {
  char *set_cluster_num_ptr = getenv("MLUOP_SET_CLUSTER_LIMIT_CAPABILITY");
  if (set_cluster_num_ptr) {
    uint32_t set_cluster_num = atoi(set_cluster_num_ptr);
    // set_cluster_num is bitmap for cluster index
    // 255 is 000011111111(bin), 8 cluster
    // 127 is 000001111111(bin), 7 cluster
    // ...
    // 3 is 2 cluster
    // 1 is 1 cluster
    VLOG(4) << "cluster which usable is mapped from CLUSTER_LIMIT to cluster "
               "index. "
            << "for example , CLUSTER_LIMIT is 3 (bin:0101), 1st and 3rd "
               "cluster is usable."
            << "now CLUSTER_LIMIT = " << set_cluster_num;
    uint32_t union_number = mluop::runtime::getClusterLimitCapability(handle_);
    VLOG(4) << "union number before set CLUSTER_LIMIT env is " << union_number;
    setClusterLimitCapability(set_cluster_num);
    union_number = mluop::runtime::getClusterLimitCapability(handle_);
    VLOG(4) << "union number after set CLUSTER_LIMIT env is " << union_number;
  }
}

void Executor::setClusterLimitCapability(uint32_t cluster_limit) {
  CNctxConfigParam ctx_conf_param;
  ctx_conf_param.visibleCluster = cluster_limit;
  CNcontext ctx;
  (void)cnCtxGetCurrent(&ctx);
  GTEST_CHECK(
      CN_SUCCESS == cnSetCtxConfigParam(ctx, CN_CTX_CONFIG_VISIBLE_CLUSTER,
                                        &ctx_conf_param),
      "Check the setting of CLUSTER_LIMIT and JOB_LIMIT match or not.");
  cnGetCtxConfigParam(ctx, CN_CTX_CONFIG_VISIBLE_CLUSTER_NUM, &ctx_conf_param);
  handle_->capability_cluster_num = ctx_conf_param.visibleClusterNumber;
  mluOpUpdateContextInformation(handle_);
}

void Executor::setJobLimitCapability(KernelClass kernel_class) {
  CNctxConfigParam ctx_conf_param;
  ctx_conf_param.unionLimit = kernel_class;
  CNcontext ctx;
  (void)cnCtxGetCurrent(&ctx);
  GTEST_CHECK(CN_SUCCESS == cnSetCtxConfigParam(ctx, CN_CTX_CONFIG_UNION_LIMIT,
                                                &ctx_conf_param),
              "Check the setting of CLUSTER_LIMIT and JOB_LIMIT match or not.");
  cnGetCtxConfigParam(ctx, CN_CTX_CONFIG_UNION_LIMIT, &ctx_conf_param);
  handle_->capability_job_limit = (int32_t)ctx_conf_param.unionLimit;
  mluOpUpdateContextInformation(handle_);
}

// XXX(zhaolianshui): should be at a higher level than Executor. SetUpTestSuite?
void Executor::getOpRelyRealData() {
  std::vector<std::string> list_rely_real_data;
  std::string cur_op = parser_->getOpName();
  list_rely_real_data = parser_->getListRelyRealData();
  auto it =
      find(list_rely_real_data.begin(), list_rely_real_data.end(), cur_op);
  if (it == list_rely_real_data.end()) {
    rely_real_data_ = false;
  } else {
    rely_real_data_ = true;
  }
}

void Executor::getPerfTestMode() {
  getOpRelyRealData();
  if (!isComputeDiff()) {
    if (!rely_real_data_) {
      mlu_need_host_data = false;
    }
    if (exe_config_->zero_input && !rely_real_data_ && doPerf()) {
      zero_input_ = true;
    }  // exe_config_->zero_input
  }    // not compute_diff
  printPerfTestInfo();
}

void Executor::printPerfTestInfo() {
  if (!isComputeDiff()) {
    std::ostringstream oss;
    oss << "Not compute diff, ";
    if (zero_input_) {
      oss << "use zero input.";
    } else if (!perfUseOriginData()) {
      oss << "use random input.";
    } else {
      oss << "use input data in pb/prototxt.";
    }
    VLOG(4) << oss.str();
  }  // not compute_diff
}

void Executor::dumpOutputData() {
  if (!exe_config_->dump_data) return;
  auto output_blocks = getOutputBlocks(true);
  for (int i = 0; i < output_blocks.size(); ++i) {
    MetaTensor *ts = parser_->output(i);
    if (VOID == storage_dtype_) {
      saveDataToFile("baseline_" + ts->name, cpu_output_[i], ts->dtype,
                     ts->total_count);
      saveDataToFile("mlu_" + ts->name, mlu_output_[i], ts->dtype,
                     ts->total_count);
    } else {
      saveDataToFile("baseline_" + ts->name, cpu_fp32_output_[i],
                     getCpuDtype(ts->dtype), ts->total_count);
      saveDataToFile("mlu_" + ts->name, mlu_fp32_output_[i],
                     getCpuDtype(ts->dtype), ts->total_count);
    }
  }
}

void Executor::setStorageFuncPtr() {
  if (FLOAT == storage_dtype_) {
    VLOG(4) << "use float* to store output data.";
    castOutFunc = &Executor::castOut;
    getBaselineOutputFunc = &Executor::getBaselineOutput;
    baselineOutputMallocFunc = &Executor::baselineOutputMalloc;
    mluOutputMallocFunc = &Executor::mluOutputMalloc;
    strideOutputFunc = &Executor::strideOutput;
    saveInputWithStrideFunc = &Executor::saveInputWithStride;
    eva_->setStorageDtype(FLOAT);
  } else if (VOID == storage_dtype_) {
    VLOG(4) << "use void* to store output data.";
    castOutFunc = &Executor::castOutByDtype;
    getBaselineOutputFunc = &Executor::getBaselineOutputByDtype;
    baselineOutputMallocFunc = &Executor::baselineOutputMallocByDtype;
    mluOutputMallocFunc = &Executor::mluOutputMallocByDtype;
    strideOutputFunc = &Executor::strideOutputByDtype;
    saveInputWithStrideFunc = &Executor::saveInputWithStrideByDtype;
    eva_->setStorageDtype(VOID);
  } else {
    GTEST_CHECK(false, "set storage dtype failed.");
  }
}

void Executor::castOutByDtype() { return; }

void Executor::getBaselineOutputByDtype() {
  for (size_t i = 0; i < parser_->outputs().size(); ++i) {
    MetaTensor *ts = parser_->output(i);
    if (unlikely(ts->empty())) {
      continue;
    }
    parser_->getOutputTensorValue(i, cpu_output_[i], ts->shape_count);
  }
}

void Executor::baselineOutputMallocByDtype() {
  for (size_t i = 0; i < parser_->outputs().size(); ++i) {
    MetaTensor *ts = parser_->output(i);
    if (unlikely(ts->empty())) {
      cpu_output_.emplace_back(nullptr);
      continue;
    }
    void *temp_ptr = (void *)cpu_runtime_.allocate(
        ts->shape_count * mluop::getSizeOfDataType(ts->dtype), ts->name);
    cpu_output_.emplace_back(temp_ptr);
    memset(temp_ptr, 0x0,
           ts->shape_count * mluop::getSizeOfDataType(ts->dtype));
  }
}

void Executor::mluOutputMallocByDtype() {
  auto output_blocks = getOutputBlocks(true);
  for (size_t i = 0; i < parser_->outputs().size(); ++i) {
    MetaTensor *ts = parser_->output(i);
    DataBlock *db = output_blocks[i];
    // TODO(None): i do not know why, but it in mluOuptutMalloc, let it
    // here for now
    if (unlikely(ts->empty())) {
      mlu_output_.push_back(nullptr);
      continue;
    }
    mlu_output_.push_back(db->host_ptr);
  }
}

void Executor::cpuCompute() {
  GTEST_CHECK(false,
              "not support cpuCompute, please use pt2pb to generate gpu case.");
}

// shape_count to total_count
void Executor::strideOutputByDtype() {
  auto output_blocks = getOutputBlocks(true);
  for (int i = 0; i < output_blocks.size(); ++i) {
    MetaTensor *ts = parser_->output(i);
    // TODO(None): bug, need map output with it's reused input
    bool init_by_input = data_vector_[i].is_output() && flag_input_reuse_;
    void *tensor_copy = nullptr;
    if (init_by_input) {
      tensor_copy = cpu_stride_input_[i];
    }
    stride_->setStrideAttr(cpu_output_[i], tensor_copy, ts, init_by_input);
    cpu_output_[i] = stride_->strideOutputByDtype();
  }
}

bool Executor::checkDiff() {
  auto threshold_use = parser_->threshold_use();
  const auto &criterions_use = getCriterionsUse();

  // get error func and threshold
  // it depends on func saved in pb.
  std::set<Evaluator::Criterion> criterions;
  bool common_threshold = parser_->common_threshold();

  if (common_threshold) {
    criterions = parser_->criterions(-1, criterions_use);
    if (exe_config_->fixed_criterion) {
      // if exe_config_->fixed_criterion, we need compute diff1~diff3
      // if criterions already contain certain func, insert failed
      // set threshold as 0 for new diff func.
      criterions.insert(
          std::move(Evaluator::Criterion(Evaluator::DIFF1, 0.0, false)));
      criterions.insert(
          std::move(Evaluator::Criterion(Evaluator::DIFF2, 0.0, false)));
      criterions.insert(
          std::move(Evaluator::Criterion(Evaluator::DIFF3, 0.0, false)));
    }
  }
  auto output_blocks = getOutputBlocks(true);
  for (int i = 0; i < output_blocks.size(); ++i) {
    if (output_blocks[i]->size == 0 || threshold_use[i] == 0) {
      continue;  // null output
    }
    MetaTensor *ts = parser_->output(i);

    void *baseline_output;
    void *mlu_output;
    if (VOID == storage_dtype_) {
      baseline_output = cpu_output_[i];
      mlu_output = mlu_output_[i];
    } else {
      baseline_output = reinterpret_cast<void *>(cpu_fp32_output_[i]);
      mlu_output = reinterpret_cast<void *>(mlu_fp32_output_[i]);
    }

    if (!common_threshold) {
      criterions = parser_->criterions(i, criterions_use);
    }

    eva_->computeDiff(baseline_output, mlu_output, ts->total_count, criterions,
                      ts->name, ts->dtype);
  }
  eva_res_.errors = eva_->errors();
  eva_res_.what = std::move(eva_->what());
  return eva_->isPassed();
}
}  // namespace mluoptest
