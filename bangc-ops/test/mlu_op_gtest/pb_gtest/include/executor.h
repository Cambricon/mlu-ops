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
#ifndef TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_EXECUTOR_H_
#define TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_EXECUTOR_H_

#include <chrono>  // NOLINT
#include <tuple>
#include <vector>
#include <set>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <memory>
#include <functional>
#include <unordered_set>
#include "gtest/gtest.h"
#include "mlu_op.h"
#include "core/tensor.h"
#include "core/tool.h"
#include "core/type.h"
#include "core/context.h"
#include "core/logging.h"
#include "pb_test_tools.h"
#include "parser.h"
#include "evaluator.h"
#include "runtime.h"
#include "memory_pool.h"
#include "variable.h"
#include "internal_kernel/fill_ram/fill_ram.h"

using namespace std::chrono; // NOLINT
namespace mluoptest {

typedef std::function<void()> Func;
const int interface_time_repeat = 4;

// io bandwidth GB/s
const float IO_BANDWIDTH_MLU220 = 25.6;
const float IO_BANDWIDTH_MLU270 = 102.4;
const float IO_BANDWIDTH_MLU290 = 1024;
const float IO_BANDWIDTH_MLU370 = 307.2;
const float IO_BANDWIDTH_MLU370_SINGLE_CLUSTER = 128;
// 590 add
const float IO_BANDWIDTH_MLU590 = 0.0;

// mlu270 mlu220 mlu290 ct peak compute force is same;
// ref pageid: 42177509
// TODO(tp_520): CT_PEAK_COMPUTE_FORCE is half of these
const int CT_PEAK_FLOAT16_COMPUTE_FORCE = 64;
const int CT_PEAK_FLOAT32_COMPUTE_FORCE = 32;
const int CT_PEAK_ELSE_COMPUTE_FORCE = 64;

const int LT_PEAK_INT4_INT4_COMPUTE_FORCE_270_290_370 = 2 * 8192;
const int LT_PEAK_INT8_INT4_COMPUTE_FORCE_270_290_370 = 2 * 4096;
const int LT_PEAK_INT8_INT8_COMPUTE_FORCE_270_290_370 = 2 * 4096;
const int LT_PEAK_INT16_INT8_COMPUTE_FORCE_270_290_370 = 2 * 2048;
const int LT_PEAK_INT16_INT16_COMPUTE_FORCE_270_290_370 = 2 * 2048;

// ref 1V_User_Guide 20200831, 1.4.5 conv cycle.
// here:comupute force not contain frequency coefficient
const int LT_PEAK_FP16_FP16_COMPUTE_FORCE_370 = 2 * 1536;
const int LT_PEAK_FP32_FP16_COMPUTE_FORCE_370 = 2 * 768;
const int LT_PEAK_FP32_FP32_COMPUTE_FORCE_370 = 2 * 384;

const int LT_PEAK_INT4_INT4_COMPUTE_FORCE_220 = 2 * 4096;
const int LT_PEAK_INT8_INT4_COMPUTE_FORCE_220 = 2 * 2048;
const int LT_PEAK_INT8_INT8_COMPUTE_FORCE_220 = 2 * 2048;
const int LT_PEAK_INT16_INT8_COMPUTE_FORCE_220 = 2 * 1024;
const int LT_PEAK_INT16_INT16_COMPUTE_FORCE_220 = 2 * 1024;

// add 590
const int LT_PEAK_INT8_INT8_COMPUTE_FORCE_590 = 2 * 0.0;
const int LT_PEAK_INT16_INT8_COMPUTE_FORCE_590 = 2 * 0.0;
const int LT_PEAK_INT16_INT16_COMPUTE_FORCE_590 = 2 * 0.0;
const int LT_PEAK_FP16_FP16_COMPUTE_FORCE_590 = 2 * 0.0;
const int LT_PEAK_FP32_FP32_COMPUTE_FORCE_590 = 2 * 0.0;
const int LT_PEAK_TF32_TF32_COMPUTE_FORCE_590 = 2 * 0.0;

// op that use lt peak force to get compute efficiency
const std::unordered_set<std::string> lt_op_set = {};

const std::set<mluOpDevType_t> arch_skip_nan_inf = {MLUOP_MLU220, MLUOP_MLU270,
                                                    MLUOP_MLU290};

// runtime config
struct ExecuteConfig {
  ExecuteConfig() {
    dump_data = mluoptest::getEnv("MLUOP_GTEST_DUMP_DATA", false);
    fixed_criterion = mluoptest::getEnv("MLUOP_GTEST_ALL_CRITERION", false);
    perf_baseline = mluoptest::getEnv("MLUOP_GTEST_PERF_BASELINE", false);
  }

  void print() {
    std::cout << "Execution config:\n";
    std::cout << std::left << std::setw(25)
              << "show diff1~3: " << fixed_criterion << "\n";
    std::cout << std::left << std::setw(25) << "dump data: " << dump_data
              << "\n";
    std::cout << std::left << std::setw(25) << "perf repeat: " << perf_repeat
              << "\n";
    std::cout << std::left << std::setw(25)
              << "check perf baseline: " << perf_baseline << "\n";
  }

  bool mlu_only = false;
  bool zero_input = false;
  bool fixed_criterion = false;
  bool dump_data = false;
  bool perf_baseline = false;
  bool test_llc = false;
  size_t perf_repeat = 1;
};

struct HardwareTimeNotifier {
  HardwareTimeNotifier() { init(); }
  ~HardwareTimeNotifier() { destroy(); }
  void init() {
    ASSERT_EQ(cnrtNotifierCreate(&n_start), cnrtSuccess);
    ASSERT_EQ(cnrtNotifierCreate(&n_stop), cnrtSuccess);
  }
  void destroy() {
    if (n_start != nullptr) {
      ASSERT_EQ(cnrtNotifierDestroy(n_start), cnrtSuccess);
      n_start = nullptr;
    }
    if (n_stop != nullptr) {
      ASSERT_EQ(cnrtNotifierDestroy(n_stop), cnrtSuccess);
      n_stop = nullptr;
    }
  }

  cnrtNotifier_t n_start = nullptr;
  cnrtNotifier_t n_stop = nullptr;
};

// common variable.
// like handle/queue/tensors ...
// create outside (executor) and just create once.
// all case share these variable.
struct ExecuteContext {
  ~ExecuteContext() { destroy(); }
  void init() {
    ASSERT_EQ(mluOpCreate(&handle), MLUOP_STATUS_SUCCESS);
    if (global_var.use_default_queue_) {
      VLOG(1) << "use_default_queue_ set, will use default queue";
      ASSERT_EQ(queue, nullptr);
    } else {
      ASSERT_EQ(cnrtQueueCreate(&queue), cnrtSuccess);
    }
    ASSERT_EQ(mluOpSetQueue(handle, queue), MLUOP_STATUS_SUCCESS);
    hw_notifier = std::make_shared<HardwareTimeNotifier>();
    hw_notifier_layer = std::make_shared<HardwareTimeNotifier>();
  }
  // reserve for memory pool
  std::shared_ptr<CPUMemoryPool> cmp = nullptr;
  std::shared_ptr<MLUMemoryPool> mmp = nullptr;
  void destroy() {
    hw_notifier->destroy();
    hw_notifier_layer->destroy();
    if (queue != nullptr) {
      ASSERT_EQ(cnrtQueueDestroy(queue), cnrtSuccess);
      queue = nullptr;
    }
    if (handle != nullptr) {
      ASSERT_EQ(mluOpDestroy(handle), MLUOP_STATUS_SUCCESS);
      handle = nullptr;
    }
    if (global_var.use_default_queue_) {
      cnrtDeviceReset();
    }
  }
  void reset() {
    LOG(WARNING) << "Executor: reset cnrt and go on running, this may caused "
                    "by cnrt failed.";
    destroy();
    init();
  }

  mluOpHandle_t handle = nullptr;
  cnrtQueue_t queue = nullptr;
  std::shared_ptr<HardwareTimeNotifier> hw_notifier;
  std::shared_ptr<HardwareTimeNotifier> hw_notifier_layer;
};

struct HostTimer {
  struct timespec t0 = {0, 0};
  struct timespec t1 = {0, 0};
  double tv_nsec = 0.0;
  double tv_sec = 0.0;
  double tv_usec = 0.0;
  std::vector<double> durations;
  inline void start() { clock_gettime(CLOCK_MONOTONIC, &t0); }
  inline void stop() {
    clock_gettime(CLOCK_MONOTONIC, &t1);
    tv_nsec = (double)t1.tv_nsec - (double)t0.tv_nsec;
    tv_sec = (double)t1.tv_sec - (double)t0.tv_sec;
    tv_usec = tv_nsec / 1000 + tv_sec * 1000 * 1000;
    durations.push_back(tv_usec);
  }
  double duration(int repeat = 1) {
    if (durations.empty()) {
      LOG(WARNING)
          << "Please add interface_timer_.start() before mluOps op interface.";
      LOG(WARNING)
          << "Please add interface_timer_.stop() after mluOps op interface.";
      return -1;
    }
    double sum = 0;
    if (repeat == 1) {
      return durations[0];
    } else if (repeat < interface_time_repeat) {
      for (int i = 1; i < repeat + 1; ++i) {
        sum += durations[i];
      }
      return sum / repeat;
    } else {
      // get average of the first four times when repeat
      // is bigger than interface_time_repeat.
      for (int i = 1; i <= interface_time_repeat; ++i) {
        sum += durations[i];
      }
      return sum / interface_time_repeat;
    }
  }
};

/* quant mode, used in cast_in().
 * 0:only set position;
 * 1:set position and scale;
 * 2:set posiiton, scale and offset;
 */
enum QuantMode {
  ONLY_POSITION = 0,
  POSITION_SCALE = 1,
  POS_SCALE_OFFSET = 2,
  NO_QUANT = 3
};
enum ComputeMode { NORMAL = 0, BY_LAYER = 1 };

struct DataBlock {
  DataBlock(MetaTensor *ts, bool o) {
    is_null = ts->is_null;
    is_output = o;

    name = ts->name;
    dtype = ts->dtype;
    oc_dt = ts->oc_dt;
    shape = ts->shape;
    stride = ts->stride;
    count = ts->total_count;
    size = count * ts->sizeof_dtype;
    count_without_stride = ts->shape_count;
    size_without_stride = count_without_stride * ts->sizeof_dtype;
  }
  void *host_ptr = nullptr;           // host pointer;
  void *device_ptr = nullptr;         // device pointer
  void *device_origin_ptr = nullptr;  // device pointer of origin
  void *device_perf_ptr = nullptr;    // device pointer for perf test

  size_t size = 0;                  // size in bytes (count * sizeof[dtype])
  size_t count = 0;                 // element count
  size_t count_without_stride = 0;  // not include stride.
  size_t size_without_stride = 0;

  bool is_output = false;
  bool is_null = false;

  mluOpDataType_t dtype = MLUOP_DTYPE_INVALID;
  mluOpDataType_t oc_dt = MLUOP_DTYPE_INVALID;

  std::vector<int> shape;
  std::vector<int> stride;
  std::string name;
};

struct TensorPair {
  TensorPair(mluOpTensorDescriptor_t t, bool i) : tensor(t), is_output(i) {}
  mluOpTensorDescriptor_t tensor = nullptr;
  bool is_output = false;
};

class Executor {
 public:
  Executor() {}
  virtual ~Executor();

  void init(const std::shared_ptr<ExecuteContext>
                ctx);  // set config param by init().
  // set execute variable by setup().
  void setup(std::string file, const std::shared_ptr<ExecuteConfig> ecfg);
  void launch();
  bool ready();
  void sync();
  EvaluateResult teardown();
  inline EvaluateResult *result() { return &eva_res_; }

 protected:
  HostTimer interface_timer_;
  MLURuntime mlu_runtime_;
  CPURuntime cpu_runtime_;
  std::shared_ptr<Parser> parser_ = nullptr;
  std::shared_ptr<Evaluator> eva_ = nullptr;
  std::shared_ptr<ExecuteContext> exe_context_ = nullptr;
  std::shared_ptr<ExecuteConfig> exe_config_ = nullptr;
  // handle pointer point to the handle in ectx_, simplify coding.
  mluOpHandle_t handle_ = nullptr;
  // queue pointer point to the queue in ectx_.
  cnrtQueue_t queue_ = nullptr;

  // true for output, false for input
  // if we have multi input or output
  // their order must consistent with order in prototxt.
  // and consistent with mluOps api
  std::vector<TensorPair> tensor_desc_;  // = delete, same with *->get(i).tensor
  std::vector<DataBlock>
      data_vector_;  // = delete, same with *->get(i).host_ptr

  // allocate by mlu_runtime
  std::vector<void *> workspace_;

  QuantMode flag_quant_mode_ = NO_QUANT;
  bool flag_input_reuse_ = false;

  // baseline data
  // cpu
  std::vector<float *> cpu_fp32_input_;
  std::vector<float *> cpu_fp32_stride_input_;
  // for cpu gpu and cmodel
  std::vector<float *> cpu_fp32_output_;

  // mlu output for evaluation
  std::vector<float *> mlu_fp32_output_;

  virtual void paramCheck() {}  // check op params
  virtual void workspaceMalloc() {}
  virtual void workspaceFree() {}
  virtual void cpuCompute() = 0;
  virtual void compute() = 0;
  virtual void computeByLayer() {}  // for fusedOp
  virtual void initHostData();
  virtual void baselineOutputMalloc();  // malloc cpu input and output
  virtual void getBaselineOutput();
  virtual void getBaselineOutputByLayer() {}
  virtual void setQuantizedParam() {}
  virtual void castIn();
  virtual void castOut();
  virtual void diffPreprocess() {}
  virtual void hostMalloc();
  virtual void hostReorder() {}
  virtual bool useTf32ComputeForce();

  bool isPlatformSupportTf32();
  bool useFloatConvInst();
  bool enableTf32MluEnv();
  bool opParamSupportTf32();

  void castDataIn(float *src_data, mluOpDataType_t src_dtype, void *dst_data,
                  mluOpDataType_t dst_dtype, size_t count, QuantMode quant_mode,
                  int *p = NULL, float *s = NULL, int *o = NULL,
                  bool dequantify = false, bool online_quantize = true);
  void castDataOut(void *src_data, mluOpDataType_t src_dtype, float *dst_data,
                   mluOpDataType_t dst_dtype, size_t count,
                   QuantMode quant_mode, int p = 0, float s = 1.0, int o = 0);

  void getQuantizedParam(float *src_data, size_t count,
                         mluOpDataType_t dst_dtype, QuantMode quant_mode,
                         int *position, float *scale, int *offset = nullptr);
  virtual int64_t getTheoryOps() { return -1; }
  virtual int64_t getTheoryIoSize();
  // placeholder used to identify whether the criterion is used or not
  virtual std::set<Evaluator::Formula> getCriterionsUse() const {
    return {Evaluator::DIFF1, Evaluator::DIFF2, Evaluator::DIFF3,
            Evaluator::DIFF3_2, Evaluator::DIFF4};
  }
  inline void recordGtestTimePoint(const std::string &&name) {
    // XXX At present, this method does not have race condition cuz methods in
    // single executor instance
    //      are always invoked in-ordered even it is in multi-thread mode.
    //      But things may changed in the future, may need to add lock in the
    //      future
    time_point_records_.emplace_back(
        std::make_tuple(name, ::steady_clock::now()));
  }

 private:
  ::time_point<::steady_clock> time_point_init_ = ::steady_clock::now();
  std::vector<std::tuple<std::string, ::time_point<::steady_clock>>>
      time_point_records_;
  void setHandle();
  void createTensors();
  void destroyTensors() noexcept;
  void launchAndGetTime(ComputeMode compute_mode, int repeat);

  void baselineInputMalloc();    // malloc cpu input and output
  void baselineFree() noexcept;  // malloc cpu input and output
  void initBaselineInput();      // read or random data

  // determine dtype of the cpu array for input/output tensor
  mluOpDataType_t getCpuDtype(mluOpDataType_t tensor_dtype);

  // save mlu fp32 output
  void mluOutputMalloc();
  void mluOutputFree() noexcept;

  void hostFree() noexcept;

  void deviceMalloc();
  void deviceFree() noexcept;
  // switch data for perf test
  void switchDataToOrigin();
  void switchDataToPerf();
  // memcpy
  void copyIn();
  void copyOut();
  void isOpRelyRealData();
  void getPerfTestMode();
  void printPerfTestInfo();

  void checkBaseline();
  EvaluateResult evaluate();

  void castHalfOuput();

  std::vector<DataBlock *> getInputBlocks();
  std::vector<DataBlock *> getOutputBlocks();

  // cast mode for conv
  void quantizeTensorByChannel(float *src_data, void *dst_data,
                               mluOpDataType_t dst_dtype, size_t count,
                               int tensor_index);

  // efficiency
  int getIpuFrequency();

  enum ComputeUnit { LT_COMPUTE, CT_COMPUTE };
  ComputeUnit compute_unit_;
  mluOpDataType_t lt_input_dtype_ = MLUOP_DTYPE_INVALID;
  mluOpDataType_t lt_weight_dtype_ = MLUOP_DTYPE_INVALID;
  void getLtComputeDtype();
  void getComputeUnit();
  double getCtPeakComputeForce();
  double getLtPeakComputeForce();
  double getPeakComputeForce();
  double getIoBandwidth();
  double getBandWidthByDev();
  void getMluPerfInfo(PerfInfo *info);
  void getGpuPerfInfo(PerfInfo *info);
  void getGtestInternalInfo();  // will access time_point_records_ and
                                // time_point_init_
  void fillRam();

  EvaluateResult eva_res_;
  // whether op rely on real data
  bool rely_real_data_ = false;
  // fast mode of mlu_only mode, do not malloc host space
  bool mlu_only_fast_ = false;
  // whether input data can be set zero
  bool zero_input_ = false;
  bool need_compute_by_layer_ = false;

  //// set cluster num and set job num
  void jobLimitCheck();
  void clusterLimitCheck();
  void setClusterLimitCapability(uint32_t cluster_limit);
  void setJobLimitCapability(KernelClass kernel_class);

  void cnrtCastDataTypeWrap(void *src_data, const mluOpDataType_t src_dtype,
                            float *dst_data, const mluOpDataType_t dst_dtype,
                            const size_t count,
                            const cnrtQuantizedParam_t quant_param);
  float callBackKernelSyncAndGetTime(
      Func gtest_kernel, std::shared_ptr<HardwareTimeNotifier> notifier);
  void fillLLC();
  void freeLLC();
  // for llc test
  CNaddr const_addr_;

 protected:
  void tensor_stride_in(void *dst, void *src, const std::vector<size_t> &shape,
                        const std::vector<size_t> &dst_stride,
                        size_t sizeof_dtype);
  void tensor_stride_out(void *dst, void *src, const std::vector<size_t> &shape,
                         const std::vector<size_t> &src_stride,
                         size_t sizeof_dtype);

  inline std::vector<size_t> getTensorShapeSizeT(MetaTensor *ts) {
    std::vector<size_t> shape_vec(ts->shape.begin(), ts->shape.end());
    return shape_vec;
  }

  inline std::vector<size_t> getTensorStrideSizeT(MetaTensor *ts) {
    std::vector<size_t> stride_vec(ts->stride.begin(), ts->stride.end());
    return stride_vec;
  }

  inline void needComputeByLayer() { need_compute_by_layer_ = true; }
};

}  // namespace mluoptest

#endif  // TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_EXECUTOR_H_
