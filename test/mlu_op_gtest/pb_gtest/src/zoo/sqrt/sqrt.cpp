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
#include "sqrt.h"

namespace mluoptest {

void SqrtExecutor::paramCheck() {
  GTEST_CHECK(parser_->getInputNum() == 1,
              "[mluOpSqrt] Input number is wrong.");
}

void SqrtExecutor::compute() {
  VLOG(4) << "SqrtExecutor compute ";

  auto tensor_x = tensor_desc_[0].tensor;
  auto tensor_y = tensor_desc_[1].tensor;
  auto dev_x = data_vector_[0].device_ptr;
  auto dev_y = data_vector_[1].device_ptr;

  mluOpComputationPreference_t prefer =
      (mluOpComputationPreference_t)parser_->getProtoNode()
          ->sqrt_param()
          .prefer();
  VLOG(4) << "call mluOpSqrt()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpSqrt(handle_, prefer, tensor_x, dev_x, tensor_y, dev_y));
  interface_timer_.stop();
}

void SqrtExecutor::cpuCompute() {
  GTEST_CHECK(parser_->getInputNum() == 1);
  GTEST_CHECK(parser_->getOutputNum() == 1);

  auto count1 = parser_->getInputDataCount(0);
  auto count2 = parser_->getOutputDataCount(0);

  for (size_t i = 0; i < count1; ++i) {
    cpu_fp32_output_[0][i] = sqrt(cpu_fp32_input_[0][i]);
  }
}

int64_t SqrtExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->getInputDataCount(0);
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
