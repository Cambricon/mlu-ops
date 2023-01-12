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
#include "log.h"

namespace mluoptest {

void LogExecutor::paramCheck() {
  if (parser_->getInputNum() != 1) {
    LOG(ERROR) << "log input number is wrong.";
  }
  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "log output number is wrong.";
  }
}

void LogExecutor::compute() {
  VLOG(4) << "LogExecutor compute ";

  auto input_tensor = tensor_desc_[0].tensor;
  auto input_dev = data_vector_[0].device_ptr;
  auto output_tensor = tensor_desc_[1].tensor;
  auto output_dev = data_vector_[1].device_ptr;
  VLOG(4) << "call mluOpLog()";
  mluOpLogBase_t base =
      (mluOpLogBase_t)(parser_->getProtoNode()->log_param().log_base());
  mluOpComputationPreference_t prefer = (mluOpComputationPreference_t)(
      parser_->getProtoNode()->log_param().prefer());
  interface_timer_.start();
  MLUOP_CHECK(mluOpLog(handle_, prefer, base, input_tensor, input_dev,
                       output_tensor, output_dev));
  interface_timer_.stop();

  data_vector_[1].is_output = true;
}

void LogExecutor::cpuCompute() {
  assert(parser_->getInputNum() == 1);
  assert(parser_->getOutputNum() == 1);
  auto count = parser_->getInputDataCount(0);

  mluOpLogBase_t base =
      (mluOpLogBase_t)(parser_->getProtoNode()->log_param().log_base());
  VLOG(4) << "log base is " << base << " (e -> 0, 2 -> 1, 10 -> 2)";
  if (base == mluOpLogBase_t::MLUOP_LOG_E) {
    for (int i = 0; i < count; ++i) {
      cpu_fp32_output_[0][i] = log(cpu_fp32_input_[0][i]);
    }
  } else if (base == mluOpLogBase_t::MLUOP_LOG_2) {
    for (int i = 0; i < count; ++i) {
      cpu_fp32_output_[0][i] = log2(cpu_fp32_input_[0][i]);
    }
  } else if (base == mluOpLogBase_t::MLUOP_LOG_10) {
    for (int i = 0; i < count; ++i) {
      cpu_fp32_output_[0][i] = log10(cpu_fp32_input_[0][i]);
    }
  } else {
    assert(0);
  }
}

int64_t LogExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->getInputDataCount(0);
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
