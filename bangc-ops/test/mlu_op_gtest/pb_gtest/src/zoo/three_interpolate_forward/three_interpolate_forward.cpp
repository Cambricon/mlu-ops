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
#include "three_interpolate_forward.h"

namespace mluoptest {

void ThreeInterpolateForwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 3,
              "[ThreeInterpolateForwardExecutor] input number is wrong.");
  GTEST_CHECK(parser_->outputs().size() == 1,
              "[ThreeInterpolateForwardExecutor] output number is wrong.");
}

void ThreeInterpolateForwardExecutor::compute() {
  VLOG(4) << "ThreeInterpolateForwardExecutor call compute begin.";
  auto features_desc = tensor_desc_[0].tensor;
  auto indices_desc = tensor_desc_[1].tensor;
  auto weights_desc = tensor_desc_[2].tensor;
  auto output_desc = tensor_desc_[3].tensor;
  auto features_data_ptr = data_vector_[0].device_ptr;
  auto indices_data_ptr = data_vector_[1].device_ptr;
  auto weights_data_ptr = data_vector_[2].device_ptr;
  auto output_data_ptr = data_vector_[3].device_ptr;
  b_ = features_desc->dims[0];
  c_ = features_desc->dims[1];
  m_ = features_desc->dims[2];
  n_ = output_desc->dims[2];
  VLOG(4) << "call mluOpThreeInterpolateForward()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpThreeInterpolateForward(
      handle_, features_desc, features_data_ptr, indices_desc, indices_data_ptr,
      weights_desc, weights_data_ptr, output_desc, output_data_ptr));
  interface_timer_.stop();
  VLOG(4) << "ThreeInterpolateForwardExecutor call compute end.";
}

void ThreeInterpolateForwardExecutor::cpuCompute() {
  VLOG(4) << "ThreeInterpolateForwardExecutor call cpuCompute begin.";
  for (int batch = 0; batch < b_; ++batch) {
    for (int channel = 0; channel < c_; ++channel) {
      for (int number = 0; number < n_; ++number) {
        auto features = cpu_fp32_input_[0];
        auto indices = cpu_fp32_input_[1];
        auto weights = cpu_fp32_input_[2];
        auto out = cpu_fp32_output_[0];
        auto features_index = batch * c_ * m_ + channel * m_;
        auto weights_index = batch * n_ * 3 + number * 3;
        auto indices_index = weights_index;
        auto out_index = batch * c_ * n_ + channel * n_ + number;
        out[out_index] =
            weights[weights_index + 0] *
                features[features_index + (int)indices[indices_index + 0]] +
            weights[weights_index + 1] *
                features[features_index + (int)indices[indices_index + 1]] +
            weights[weights_index + 2] *
                features[features_index + (int)indices[indices_index + 2]];
      }
    }
  }
  VLOG(4) << "ThreeInterpolateForwardExecutor call cpuCompute end.";
}

int64_t ThreeInterpolateForwardExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->getOutputDataCount(0) * 5;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
