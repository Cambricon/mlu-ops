#include "lgamma.h"

namespace mluoptest {

void LgammaExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 1,
              "lgamma tensor input number is wrong.");
  GTEST_CHECK(parser_->outputs().size() == 1,
              "lgamma tensor output number is wrong.");
}

void LgammaExecutor::compute() {
  VLOG(4) << "LgammaExecutor compute ";
  auto tensor_x = tensor_desc_[0].tensor;
  auto tensor_y = tensor_desc_[1].tensor;
  auto dev_x = data_vector_[0].device_ptr;
  auto dev_y = data_vector_[1].device_ptr;
  VLOG(4) << "call mluOpLgamma()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpLgamma(handle_, tensor_x, dev_x, tensor_y, dev_y));
  interface_timer_.stop();
}

void LgammaExecutor::cpuCompute() {
  auto count = parser_->input(0)->shape_count;

  for (int i = 0; i < count; ++i) {
    cpu_fp32_output_[0][i] = ::lgamma(cpu_fp32_input_[0][i]);
  }
}

int64_t LgammaExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->input(0)->total_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
