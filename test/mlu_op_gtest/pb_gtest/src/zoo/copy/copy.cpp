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
#include <complex>

#include "copy.h"

namespace mluoptest {

void CopyExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 1);
  GTEST_CHECK(parser_->outputs().size() == 1);
}

void CopyExecutor::compute() {
  VLOG(4) << "CopyExecutor compute ";

  auto input_desc = tensor_desc_[0].tensor;
  auto output_desc = tensor_desc_[1].tensor;

  auto dev_input = data_vector_[0].device_ptr;
  auto dev_output = data_vector_[1].device_ptr;

  VLOG(4) << "call mluOpCopy()";
  interface_timer_.start();
  MLUOP_CHECK(
      mluOpCopy(handle_, input_desc, dev_input, output_desc, dev_output));
  interface_timer_.stop();
}

void CopyExecutor::cpuCompute() {
  auto input_num = parser_->input(0)->shape_count;

  if (parser_->input(0)->dtype == MLUOP_DTYPE_DOUBLE) {
    double *host_input = reinterpret_cast<double *>(cpu_fp32_input_[0]);
    double *host_output = reinterpret_cast<double *>(cpu_fp32_output_[0]);
    for (int i = 0; i < input_num; i++) {
      host_output[i] = host_input[i];
    }
  } else if (parser_->input(0)->dtype == MLUOP_DTYPE_COMPLEX_HALF ||
             parser_->input(0)->dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
    std::complex<float> *host_input =
        reinterpret_cast<std::complex<float> *>(cpu_fp32_input_[0]);
    std::complex<float> *host_output =
        reinterpret_cast<std::complex<float> *>(cpu_fp32_output_[0]);
    for (int i = 0; i < input_num; i++) {
      host_output[i] = host_input[i];
    }
  } else {
    float *host_input = cpu_fp32_input_[0];
    float *host_output = cpu_fp32_output_[0];
    for (int i = 0; i < input_num; i++) {
      host_output[i] = host_input[i];
    }
  }
}

int64_t CopyExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->input(0)->total_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
