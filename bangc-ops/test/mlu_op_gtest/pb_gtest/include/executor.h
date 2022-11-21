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

#include <vector>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <memory>
#include <unordered_set>
#include <set>
#include "core/tensor.h"
#include "core/tool.h"
#include "core/type.h"
#include "core/context.h"
#include "evaluator.h"
#include "gtest/gtest.h"
#include "memory_pool.h"
#include "mlu_op.h"
#include "parser.h"
#include "pb_test_tools.h"
#include "runtime.h"

namespace mluoptest {

const int interface_time_repeat = 4;

// io bandwidth GB/s
const float IO_BANDWIDTH_MLU220 = 25.6;
const float IO_BANDWIDTH_MLU270 = 102.4;
const float IO_BANDWIDTH_MLU290 = 1024;
const float IO_BANDWIDTH_MLU370 = 307.2;

// mlu270 mlu220 mlu290 ct peak compute force is same;
const int CT_PEAK_FLOAT16_COMPUTE_FORCE = 64;
const int CT_PEAK_FLOAT32_COMPUTE_FORCE = 32;
const int CT_PEAK_ELSE_COMPUTE_FORCE = 64;

const int LT_PEAK_INT4_INT4_COMPUTE_FORCE_270_290 = 2 * 8192;
const int LT_PEAK_INT8_INT4_COMPUTE_FORCE_270_290 = 2 * 4096;
const int LT_PEAK_INT8_INT8_COMPUTE_FORCE_270_290 = 2 * 4096;
const int LT_PEAK_INT16_INT8_COMPUTE_FORCE_270_290 = 2 * 2048;
const int LT_PEAK_INT16_INT16_COMPUTE_FORCE_270_290 = 2 * 2048;

const int LT_PEAK_FP16_FP16_COMPUTE_FORCE = 2 * 1.5 * 1024;
const int LT_PEAK_FP32_FP16_COMPUTE_FORCE = 2 * 0.75 * 1024;
const int LT_PEAK_FP32_FP32_COMPUTE_FORCE = 2 * 0.375 * 1024;

const int LT_PEAK_INT4_INT4_COMPUTE_FORCE_220 = 2 * 4096;
const int LT_PEAK_INT8_INT4_COMPUTE_FORCE_220 = 2 * 2048;
const int LT_PEAK_INT8_INT8_COMPUTE_FORCE_220 = 2 * 2048;
const int LT_PEAK_INT16_INT8_COMPUTE_FORCE_220 = 2 * 1024;
const int LT_PEAK_INT16_INT16_COMPUTE_FORCE_220 = 2 * 1024;

// op that use lt peak force to get compute efficiency
const std::unordered_set<std::string> lt_op_set = {};

const std::set<mluOpDevType_t> arch_skip_nan_inf = {MLUOP_MLU220, MLUOP_MLU270,
                                                    MLUOP_MLU290};

// runtime config
struct ExecuteConfig {
  ExecuteConfig() {
    dump_data = getEnv("MLUOP_GTEST_DUMP_DATA", false);
    fixed_criterion = getEnv("MLUOP_GTEST_ALL_CRITERION", false);
    perf_baseline = getEnv("MLUOP_GTEST_PERF_BASELINE", false);
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
  size_t perf_repeat = 1;
};

// common variable.
// like handle/queue/tensors ...
// create outside (executor) and just create once.
// all case share these variable.
struct ExecuteContext {
  ~ExecuteContext() { destroy(); }
  void init() {
    ASSERT_EQ(cnrtQueueCreate(&queue), CNRT_RET_SUCCESS);
    ASSERT_EQ(mluOpCreate(&handle), MLUOP_STATUS_SUCCESS);
    ASSERT_EQ(mluOpSetQueue(handle, queue), MLUOP_STATUS_SUCCESS);
    ASSERT_EQ(cnrtNotifierCreate(&n_start), CNRT_RET_SUCCESS);
    ASSERT_EQ(cnrtNotifierCreate(&n_stop), CNRT_RET_SUCCESS);
  }
  // reserve for memory pool
  std::shared_ptr<CPUMemoryPool> cmp = nullptr;
  std::shared_ptr<MLUMemoryPool> mmp = nullptr;
  void destroy() {
    if (n_start != nullptr) {
      ASSERT_EQ(cnrtNotifierDestroy(n_start), CNRT_RET_SUCCESS);
      n_start = nullptr;
    }
    if (n_stop != nullptr) {
      ASSERT_EQ(cnrtNotifierDestroy(n_stop), CNRT_RET_SUCCESS);
      n_stop = nullptr;
    }
    if (queue != nullptr) {
      ASSERT_EQ(cnrtQueueDestroy(queue), CNRT_RET_SUCCESS);
      queue = nullptr;
    }
    if (handle != nullptr) {
      ASSERT_EQ(mluOpDestroy(handle), MLUOP_STATUS_SUCCESS);
      handle = nullptr;
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
  cnrtNotifier_t n_start = nullptr;
  cnrtNotifier_t n_stop = nullptr;
};

struct HostTimer {
  struct timespec t0 = {0, 0};
  struct timespec t1 = {0, 0};
  double tv_nsec = 0.0;
  double tv_sec = 0.0;
  double tv_usec = 0.0;
  std::vector<double> durations;
  void start() { clock_gettime(CLOCK_MONOTONIC, &t0); }
  void stop() {
    clock_gettime(CLOCK_MONOTONIC, &t1);
    tv_nsec = (double)t1.tv_nsec - (double)t0.tv_nsec;
    tv_sec = (double)t1.tv_sec - (double)t0.tv_sec;
    tv_usec = tv_nsec / 1000 + tv_sec * 1000 * 1000;
    durations.push_back(tv_usec);
  }
  double duration(int repeat = 1) {
    if (durations.empty()) {
      LOG(WARNING) << "Please add interface_timer_.start() before mlu-ops "
                      "interface.";
      LOG(WARNING) << "Please add interface_timer_.stop() after mlu-ops "
                      "interface.";
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
  NO_QUANT = 100,
};

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
  }
  void *host_ptr = nullptr;           // host pointer;
  void *device_ptr = nullptr;         // device pointer
  void *device_origin_ptr = nullptr;  // device pointer of origin
  void *device_perf_ptr = nullptr;    // device pointer for perf test
  size_t size = 0;                    // size in bytes (count * sizeof[dtype])
  size_t count = 0;                   // element count
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

  void init(const std::shared_ptr<ExecuteContext> ctx);  // set config param by
                                                         // init().
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
  // and consistent with mlu-ops api
  std::vector<TensorPair> tensor_desc_;  // = delete, same with
                                         // *->get(i).tensor
  std::vector<DataBlock> data_vector_;   // = delete, same with
                                         // *->get(i).host_ptr

  // allocate by mlu_runtime
  std::vector<void *> workspace_;

  QuantMode flag_quant_mode_ = NO_QUANT;
  bool flag_input_reuse_ = false;

  // baseline data
  // cpu
  std::vector<float *> cpu_fp32_input_;
  std::vector<float *> cpu_fp32_stride_input_;
  // for cpu
  std::vector<float *> cpu_fp32_output_;

  // mlu output for evaluation
  std::vector<float *> mlu_fp32_output_;

  // placeholder used to identify whether the criterion is used or not
  std::vector<int> criterions_use_ = {1, 1, 1, 1};

  virtual void paramCheck() {}  // check op params
  virtual void workspaceMalloc() {}
  virtual void workspaceFree() {}
  virtual void cpuCompute() = 0;
  virtual void compute() = 0;
  virtual void initHostData();
  virtual void baselineOutputMalloc();  // malloc cpu input and output
  virtual void getBaselineOutput();
  virtual void setQuantizedParam() {}
  virtual void castIn();
  virtual void castOut();
  virtual void diffPreprocess() {}
  virtual void hostMalloc();
  void castDataIn(float *src_data, mluOpDataType_t src_dtype, void *dst_data,
                  mluOpDataType_t dst_dtype, size_t count, QuantMode quant_mode,
                  int *pos = nullptr, float *sc = nullptr, int *off = nullptr,
                  bool dequantify = false);

  void castDataOut(void *src_data, mluOpDataType_t src_dtype, float *dst_data,
                   mluOpDataType_t dst_dtype, size_t count,
                   QuantMode quant_mode, int pos = 0, float sc = 1.0,
                   int off = 0);

  virtual int64_t getTheoryOps() { return -1; }
  virtual int64_t getTheoryIoSize();
  virtual std::vector<int> getCriterionsUse() { return criterions_use_; }

 private:
  void createTensors();
  void destroyTensors() noexcept;
  void syncQueueAndGetHardwareTime(int repeat = 1);

  void baselineInputMalloc();    // malloc cpu input and output
  void baselineFree() noexcept;  // malloc cpu input and output
  void initBaselineInput();      // read or random data

  // determine dtype of the cpu array for inout/output tensor
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

  // whether input data can be zero
  bool needZeroInput();

  void checkBaseline();
  EvaluateResult evaluate();

  void castHalfOuput();

  std::vector<DataBlock *> getInputBlocks();
  std::vector<DataBlock *> getOutputBlocks();

  // efficiency
  double getCtPeakComputeForce();
  double getLtPeakComputeForce();
  double getPeakComputeForce();
  double getIoBandwidth();
  void getMluPerfInfo(PerfInfo *info);
  void fillRam();

  EvaluateResult eva_res_;

  //// set cluster num and set job num
  void jobLimitCheck();
  void clusterLimitCheck();
  void setClusterLimitCapability(uint32_t cluster_limit);
  void setJobLimitCapability(KernelClass kernel_class);

 protected:
  void tensor_stride_in(void *dst, void *src, const std::vector<int> &shape,
                        const std::vector<int> &dst_stride,
                        size_t sizeof_dtype);
  void tensor_stride_out(void *dst, void *src, const std::vector<int> &shape,
                         const std::vector<int> &src_stride,
                         size_t sizeof_dtype);
  bool mluOnlyFast();
};

}  // namespace mluoptest

#endif  // TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_EXECUTOR_H_
