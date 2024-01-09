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
#include "three_interpolate_backward.h"

namespace mluoptest {

void ThreeInterpolateBackwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 3,
              "[ThreeInterpolateBackwardExecutor] input number is wrong.");
  GTEST_CHECK(parser_->outputs().size() == 1,
              "[ThreeInterpolateBackwardExecutor] output number is wrong.");
}

void ThreeInterpolateBackwardExecutor::compute() {
  VLOG(4) << "ThreeInterpolateBackwardExecutor call compute begin.";
  auto grad_output_desc = tensor_desc_[0].tensor;
  auto indices_desc = tensor_desc_[1].tensor;
  auto weights_desc = tensor_desc_[2].tensor;
  auto grad_features_desc = tensor_desc_[3].tensor;
  auto grad_output_data_ptr = data_vector_[0].device_ptr;
  auto indices_data_ptr = data_vector_[1].device_ptr;
  auto weights_data_ptr = data_vector_[2].device_ptr;
  auto grad_features_data_ptr = data_vector_[3].device_ptr;
  b_ = grad_output_desc->dims[0];
  c_ = grad_output_desc->dims[1];
  n_ = grad_output_desc->dims[2];
  m_ = grad_features_desc->dims[2];
  VLOG(4) << "call mluOpThreeInterpolateBackward()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpThreeInterpolateBackward(
      handle_, grad_output_desc, grad_output_data_ptr, indices_desc,
      indices_data_ptr, weights_desc, weights_data_ptr, grad_features_desc,
      grad_features_data_ptr));
  interface_timer_.stop();
  VLOG(4) << "ThreeInterpolateBackwardExecutor call compute end.";
}

void ThreeInterpolateBackwardExecutor::cpuCompute() {
  VLOG(4) << "ThreeInterpolateBackwardExecutor call cpuCompute begin.";
  for (int batch = 0; batch < b_; ++batch) {
    for (int channel = 0; channel < c_; ++channel) {
      for (int number = 0; number < n_; ++number) {
        auto grad_output = cpu_fp32_input_[0];
        auto indices = cpu_fp32_input_[1];
        auto weights = cpu_fp32_input_[2];
        auto grad_features = cpu_fp32_output_[0];
        auto grad_output_index = batch * c_ * n_ + channel * n_ + number;
        auto weights_index = batch * n_ * 3 + number * 3;
        auto indices_index = weights_index;
        auto grad_features_index = batch * c_ * m_ + channel * m_;
        grad_features[grad_features_index + (int)indices[indices_index + 0]] +=
            grad_output[grad_output_index] * weights[weights_index + 0];
        grad_features[grad_features_index + (int)indices[indices_index + 1]] +=
            grad_output[grad_output_index] * weights[weights_index + 1];
        grad_features[grad_features_index + (int)indices[indices_index + 2]] +=
            grad_output[grad_output_index] * weights[weights_index + 2];
      }
    }
  }
  VLOG(4) << "ThreeInterpolateBackwardExecutor call cpuCompute end.";
}

int64_t ThreeInterpolateBackwardExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->getInputDataCount(0) * 9;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
