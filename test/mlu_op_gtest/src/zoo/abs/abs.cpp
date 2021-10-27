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
#include "abs.h"

namespace mluoptest {

void AbsExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 1, "abs tensor input number is wrong.");
  GTEST_CHECK(parser_->outputs().size() == 1, "abs tensor output number is wrong.");
}

void AbsExecutor::compute() {
  VLOG(4) << "AbsExecutor compute ";
  auto tensor_x = tensor_desc_[0].tensor;
  auto tensor_y = tensor_desc_[1].tensor;
  auto dev_x = data_vector_[0].device_ptr;
  auto dev_y = data_vector_[1].device_ptr;
  VLOG(4) << "call mluOpAbs()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpAbs(handle_, tensor_x, dev_x, tensor_y, dev_y));
  interface_timer_.stop();
}

void AbsExecutor::cpuCompute() {
  auto count = parser_->input(0)->shape_count;

  for (int i = 0; i < count; ++i) {
    cpu_fp32_output_[0][i] =
        (cpu_fp32_input_[0][i] >= 0) ? cpu_fp32_input_[0][i] : -1 * (cpu_fp32_input_[0][i]);
  }
}

int64_t AbsExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->input(0)->total_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
