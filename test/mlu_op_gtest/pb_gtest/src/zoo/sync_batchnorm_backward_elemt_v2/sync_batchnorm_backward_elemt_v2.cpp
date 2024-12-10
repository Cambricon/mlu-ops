/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
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
#include "sync_batchnorm_backward_elemt_v2.h"

namespace mluoptest {

void cpuSyncBatchnormBackwardElemt(const float *diff_y, const float *x,
                                   const float *mean, const float *invstd,
                                   const float *weight, const float *sum_dy,
                                   const float *sum_dy_xmu, const int32_t sum,
                                   float *diff_x, const int len_x,
                                   const int len_c) {
  int len_nhw = len_x / len_c;
  for (int ci = 0; ci < len_c; ++ci) {
    float sum_dy_temp = sum_dy[ci] / sum;
    float sum_dy_xmu_temp = sum_dy_xmu[ci] / sum;
    for (int i = 0; i < len_nhw; ++i) {
      if (weight == nullptr) {
        diff_x[i * len_c + ci] = (diff_y[i * len_c + ci] - sum_dy_temp -
                                  (x[i * len_c + ci] - mean[ci]) * invstd[ci] *
                                      invstd[ci] * sum_dy_xmu_temp) *
                                 invstd[ci];
      } else {
        diff_x[i * len_c + ci] = (diff_y[i * len_c + ci] - sum_dy_temp -
                                  (x[i * len_c + ci] - mean[ci]) * invstd[ci] *
                                      invstd[ci] * sum_dy_xmu_temp) *
                                 weight[ci] * invstd[ci];
      }
    }
  }
}

void SyncBatchnormBackwardElemtV2Executor::paramCheck() {
  GTEST_CHECK(parser_->getInputNum() == 7 || parser_->getInputNum() == 8,
              "SyncBatchnormBackwardElemtV2Executor: input number is wrong.");
  GTEST_CHECK(parser_->getOutputNum() == 1,
              "SyncBatchnormBackwardElemtV2Executor: output number is wrong.");
}

void SyncBatchnormBackwardElemtV2Executor::compute() {
  mluOpTensorDescriptor_t x_desc, diff_y_desc, diff_x_desc, count_desc;
  mluOpTensorDescriptor_t mean_desc, invstd_desc, weight_desc, sum_dy_desc,
      sum_dy_xmu_desc;

  diff_y_desc = tensor_desc_[0].tensor;
  x_desc = tensor_desc_[1].tensor;
  mean_desc = tensor_desc_[2].tensor;
  invstd_desc = tensor_desc_[3].tensor;
  if (parser_->getInputNum() == 8) {
    weight_desc = tensor_desc_[4].tensor;
    sum_dy_desc = tensor_desc_[5].tensor;
    sum_dy_xmu_desc = tensor_desc_[6].tensor;
    count_desc = tensor_desc_[7].tensor;
    diff_x_desc = tensor_desc_[8].tensor;
  } else {
    weight_desc = nullptr;
    sum_dy_desc = tensor_desc_[4].tensor;
    sum_dy_xmu_desc = tensor_desc_[5].tensor;
    count_desc = tensor_desc_[6].tensor;
    diff_x_desc = tensor_desc_[7].tensor;
  }

  void *dev_diff_y = data_vector_[0].device_ptr;
  void *dev_x = data_vector_[1].device_ptr;
  void *dev_mean = data_vector_[2].device_ptr;
  void *dev_invstd = data_vector_[3].device_ptr;
  void *dev_weight = nullptr;
  void *dev_sum_dy = nullptr;
  void *dev_sum_dy_xmu = nullptr;
  void *dev_count = nullptr;
  void *dev_diff_x = nullptr;
  if (parser_->getInputNum() == 8) {
    dev_weight = data_vector_[4].device_ptr;
    dev_sum_dy = data_vector_[5].device_ptr;
    dev_sum_dy_xmu = data_vector_[6].device_ptr;
    dev_count = data_vector_[7].device_ptr;
    dev_diff_x = data_vector_[8].device_ptr;
  } else {
    dev_sum_dy = data_vector_[4].device_ptr;
    dev_sum_dy_xmu = data_vector_[5].device_ptr;
    dev_count = data_vector_[6].device_ptr;
    dev_diff_x = data_vector_[7].device_ptr;
  }

  VLOG(4) << "Start to run mluOpSyncBatchnormBackwardElemt_v2().";
  interface_timer_.start();
  MLUOP_CHECK(mluOpSyncBatchNormBackwardElemtV2(
      handle_, diff_y_desc, dev_diff_y, x_desc, dev_x, mean_desc, dev_mean,
      invstd_desc, dev_invstd, weight_desc, dev_weight, sum_dy_desc, dev_sum_dy,
      sum_dy_xmu_desc, dev_sum_dy_xmu, count_desc, dev_count, diff_x_desc,
      dev_diff_x));
  interface_timer_.stop();
  VLOG(4) << "mluOpSyncBatchnormBackwardElemt_v2() end";
}

void SyncBatchnormBackwardElemtV2Executor::cpuCompute() {
  int len_x = parser_->getInputDataCount(0);
  int len_c = tensor_desc_[0].tensor->getDimIndex(tensor_desc_[0].tensor->getDim() - 1);
  int len_n = tensor_desc_[0].tensor->getDimIndex(0);

  if (len_x == 0 || len_c == 0) {
    VLOG(4) << "SyncBatchnormBackwardElemtV2Executor: cpu compute zero elemt";
    return;
  }

  VLOG(4) << "Start to run cpuSyncBatchnormBackwardElemt().";

  float *cpu_diff_y = cpu_fp32_input_[0];
  float *cpu_x = cpu_fp32_input_[1];
  float *cpu_mean = cpu_fp32_input_[2];
  float *cpu_invstd = cpu_fp32_input_[3];
  float *cpu_weight = nullptr;
  float *cpu_sum_dy = nullptr;
  float *cpu_sum_dy_xmu = nullptr;
  float *cpu_count = nullptr;
  float *cpu_diff_x = cpu_fp32_output_[0];
  if (parser_->getInputNum() == 8) {
    cpu_weight = cpu_fp32_input_[4];
    cpu_sum_dy = cpu_fp32_input_[5];
    cpu_sum_dy_xmu = cpu_fp32_input_[6];
    cpu_count = cpu_fp32_input_[7];
  } else {
    cpu_sum_dy = cpu_fp32_input_[4];
    cpu_sum_dy_xmu = cpu_fp32_input_[5];
    cpu_count = cpu_fp32_input_[6];
  }
  int sum = 0;
  for (int k = 0; k < len_n; k++) {
    sum += (int32_t)(cpu_count[k]);
  }

  cpuSyncBatchnormBackwardElemt(cpu_diff_y, cpu_x, cpu_mean, cpu_invstd,
                                cpu_weight, cpu_sum_dy, cpu_sum_dy_xmu, sum,
                                cpu_diff_x, len_x, len_c);
  VLOG(4) << "cpuSyncBatchnormBackwardElemt() end";
}

int64_t SyncBatchnormBackwardElemtV2Executor::getTheoryOps() {
  int64_t theory_ops = 0;
  int len_x = parser_->getInputDataCount(0);
  int len_c = tensor_desc_[0].tensor->getDimIndex(tensor_desc_[0].tensor->getDim() - 1);
  if (parser_->getInputNum() == 7) {
    theory_ops = 5 * len_x + 3 * len_c;
  } else {
    theory_ops = 5 * len_x + 2 * len_c;
  }

  VLOG(4) << "SyncBatchnormBackwardElemtV2Executor: getTheoryOps: "
          << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
