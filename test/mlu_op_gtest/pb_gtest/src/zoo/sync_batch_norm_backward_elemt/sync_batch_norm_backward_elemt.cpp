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
#include "sync_batch_norm_backward_elemt.h"

#include <memory>

namespace mluoptest {

void cpuSyncBatchNormBackwardElemt(const float *x, const float *diff_y,
                                   const float *weight, const float *mean,
                                   const float *invstd, const float *mean_dy,
                                   const float *mean_dy_xmu, float *diff_x,
                                   const int len_x, const int len_c) {
  int len_nhw = len_x / len_c;
  for (int ci = 0; ci < len_c; ++ci) {
    for (int i = 0; i < len_nhw; ++i) {
      if (weight == nullptr) {
        diff_x[i * len_c + ci] = (diff_y[i * len_c + ci] - mean_dy[ci] -
                                  (x[i * len_c + ci] - mean[ci]) * invstd[ci] *
                                      invstd[ci] * mean_dy_xmu[ci]) *
                                 invstd[ci];
      } else {
        diff_x[i * len_c + ci] = (diff_y[i * len_c + ci] - mean_dy[ci] -
                                  (x[i * len_c + ci] - mean[ci]) * invstd[ci] *
                                      invstd[ci] * mean_dy_xmu[ci]) *
                                 weight[ci] * invstd[ci];
      }
    }
  }
}

void SyncBatchNormBackwardElemtExecutor::paramCheck() {
  GTEST_CHECK(parser_->getInputNum() == 6 || parser_->getInputNum() == 7,
              "SyncBatchNormBackwardElemtExecutor: input number is wrong.");
  GTEST_CHECK(parser_->getOutputNum() == 1,
              "SyncBatchNormBackwardElemtExecutor: output number is wrong.");
}

void SyncBatchNormBackwardElemtExecutor::compute() {
  mluOpTensorDescriptor_t x_desc, diff_y_desc, diff_x_desc;
  mluOpTensorDescriptor_t mean_desc, invstd_desc, weight_desc, mean_dy_desc,
      mean_dy_xmu_desc;

  diff_y_desc = tensor_desc_[0].tensor;
  x_desc = tensor_desc_[1].tensor;
  mean_desc = tensor_desc_[2].tensor;
  invstd_desc = tensor_desc_[3].tensor;
  if (parser_->getInputNum() == 7) {
    weight_desc = tensor_desc_[4].tensor;
    mean_dy_desc = tensor_desc_[5].tensor;
    mean_dy_xmu_desc = tensor_desc_[6].tensor;
    diff_x_desc = tensor_desc_[7].tensor;
  } else {
    weight_desc = nullptr;
    mean_dy_desc = tensor_desc_[4].tensor;
    mean_dy_xmu_desc = tensor_desc_[5].tensor;
    diff_x_desc = tensor_desc_[6].tensor;
  }

  void *dev_diff_y = data_vector_[0].device_ptr;
  void *dev_x = data_vector_[1].device_ptr;
  void *dev_mean = data_vector_[2].device_ptr;
  void *dev_invstd = data_vector_[3].device_ptr;
  void *dev_weight = nullptr;
  void *dev_mean_dy = nullptr;
  void *dev_mean_dy_xmu = nullptr;
  void *dev_diff_x = nullptr;
  if (parser_->getInputNum() == 7) {
    dev_weight = data_vector_[4].device_ptr;
    dev_mean_dy = data_vector_[5].device_ptr;
    dev_mean_dy_xmu = data_vector_[6].device_ptr;
    dev_diff_x = data_vector_[7].device_ptr;
  } else {
    dev_mean_dy = data_vector_[4].device_ptr;
    dev_mean_dy_xmu = data_vector_[5].device_ptr;
    dev_diff_x = data_vector_[6].device_ptr;
  }

  VLOG(4) << "Start to run mluOpSyncBatchNormBackwardElemt().";
  interface_timer_.start();
  MLUOP_CHECK(mluOpSyncBatchNormBackwardElemt(
      handle_, diff_y_desc, dev_diff_y, x_desc, dev_x, mean_desc, dev_mean,
      invstd_desc, dev_invstd, weight_desc, dev_weight, mean_dy_desc,
      dev_mean_dy, mean_dy_xmu_desc, dev_mean_dy_xmu, diff_x_desc, dev_diff_x));
  interface_timer_.stop();
  VLOG(4) << "mluOpSyncBatchNormBackwardElemt() end";
}

void SyncBatchNormBackwardElemtExecutor::cpuCompute() {
  int len_x = parser_->getInputDataCount(0);
  int len_c =
      tensor_desc_[0].tensor->getDimIndex(tensor_desc_[0].tensor->getDim() - 1);

  if (len_x == 0 || len_c == 0) {
    VLOG(4) << "SyncBatchNormBackwardElemtExecutor: cpu compute zero elemt";
    return;
  }

  VLOG(4) << "Start to run cpuSyncBatchNormBackwardElemt().";

  float *cpu_diff_y = cpu_fp32_input_[0];
  float *cpu_x = cpu_fp32_input_[1];
  float *cpu_mean = cpu_fp32_input_[2];
  float *cpu_invstd = cpu_fp32_input_[3];
  float *cpu_weight = nullptr;
  float *cpu_mean_dy = nullptr;
  float *cpu_mean_dy_xmu = nullptr;
  float *cpu_diff_x = cpu_fp32_output_[0];
  if (parser_->getInputNum() == 7) {
    cpu_weight = cpu_fp32_input_[4];
    cpu_mean_dy = cpu_fp32_input_[5];
    cpu_mean_dy_xmu = cpu_fp32_input_[6];
  } else {
    cpu_mean_dy = cpu_fp32_input_[4];
    cpu_mean_dy_xmu = cpu_fp32_input_[5];
  }

  cpuSyncBatchNormBackwardElemt(cpu_x, cpu_diff_y, cpu_weight, cpu_mean,
                                cpu_invstd, cpu_mean_dy, cpu_mean_dy_xmu,
                                cpu_diff_x, len_x, len_c);
  VLOG(4) << "cpuSyncBatchNormBackwardElemt() end";
}

int64_t SyncBatchNormBackwardElemtExecutor::getTheoryOps() {
  int64_t theory_ops = 0;
  int len_x = parser_->getInputDataCount(0);
  int len_c =
      tensor_desc_[0].tensor->getDimIndex(tensor_desc_[0].tensor->getDim() - 1);
  if (parser_->getInputNum() == 7) {
    theory_ops = 5 * len_x + 3 * len_c;
  } else {
    theory_ops = 5 * len_x + 2 * len_c;
  }

  VLOG(4) << "SyncBatchNormBackwardElemtExecutor: getTheoryOps: " << theory_ops
          << " ops";
  return theory_ops;
}

}  // namespace mluoptest
