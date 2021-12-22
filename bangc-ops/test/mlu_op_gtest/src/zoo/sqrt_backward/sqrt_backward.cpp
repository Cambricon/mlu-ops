/*************************************************************************
 * Copyright (C) 2021 by Cambricon, Inc. All rights reserved.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "sqrt_backward.h"

namespace mluoptest {

void SqrtBackwardExecutor::paramCheck() {
  if (parser_->getInputNum() != 2) {
    LOG(ERROR) << "input number is wrong. ";
  }
}

void SqrtBackwardExecutor::compute() {
  VLOG(4) << "SqrtBackwardExecutor compute ";
  auto tensor_y = tensor_desc_[0].tensor;
  auto tensor_dy = tensor_desc_[1].tensor;
  auto tensor_dx = tensor_desc_[2].tensor;
  auto dev_y = data_vector_[0].device_ptr;
  auto dev_dy = data_vector_[1].device_ptr;
  auto dev_dx = data_vector_[2].device_ptr;
  VLOG(4) << "call mlu-ops SqrtBackward()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpSqrtBackward(handle_, tensor_y, dev_y, tensor_dy, dev_dy, tensor_dx, dev_dx));
  interface_timer_.stop();
}

void SqrtBackwardExecutor::cpuCompute() {
  assert(parser_->getInputNum() == 2);
  assert(parser_->getOutputNum() == 1);

  auto count1 = parser_->getInputDataCount(0);
  auto count2 = parser_->getInputDataCount(1);
  assert(count1 == count2);

  for (int i = 0; i < count1; ++i) {
    cpu_fp32_output_[0][i] = 0.5 * cpu_fp32_input_[1][i] * (1.0 / cpu_fp32_input_[0][i]);
  }
}

int64_t SqrtBackwardExecutor::getTheoryOps() {
  int cp_count = 2;
  int64_t theory_ops = parser_->getInputDataCount(0);
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
