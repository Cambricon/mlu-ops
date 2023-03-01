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
#include "moe_dispatch_forward.h"

namespace mluoptest {

void MoeDispatchForwardExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_moe_dispatch_forward_param()) {
    LOG(ERROR) << "Lose moe_dispatch_forward_param.";
  }
  GTEST_CHECK(
      parser_->inputs().size() == 5,
      "[MoeDispatchForwardExecutor] tensor input number is wrong.");
  GTEST_CHECK(
      parser_->outputs().size() == 1,
      "[MoeDispatchForwardExecutor] tensor output number is wrong.");
  flag_input_reuse_ = true;
}

void MoeDispatchForwardExecutor::initData() {
  samples_ =
      parser_->getProtoNode()->moe_dispatch_forward_param().samples();
  capacity_ =
      parser_->getProtoNode()->moe_dispatch_forward_param().capacity();
  hidden_ =
      parser_->getProtoNode()->moe_dispatch_forward_param().hidden();
  num_experts_ =
      parser_->getProtoNode()->moe_dispatch_forward_param().num_experts();
}

void MoeDispatchForwardExecutor::compute() {
  VLOG(4) << "[MoeDispatchForwardExecutor] call compute() begin.";
  initData();
  // input tensor
  desc_gates_ = tensor_desc_[0].tensor;
  desc_indices_ = tensor_desc_[1].tensor;
  desc_locations_ = tensor_desc_[2].tensor;
  desc_input_ = tensor_desc_[3].tensor;

  auto dev_gates = data_vector_[0].device_ptr;
  auto dev_indices = data_vector_[1].device_ptr;
  auto dev_locations = data_vector_[2].device_ptr;
  auto dev_input = data_vector_[3].device_ptr;

  // output tensor
  auto dispatch_desc = tensor_desc_[4].tensor;
  auto dev_dispatch = data_vector_[4].device_ptr;

  interface_timer_.start();
  MLUOP_CHECK(mluOpMoeDispatchForward(
      handle_, desc_gates_, dev_gates, desc_indices_, dev_indices,
      desc_locations_, dev_locations, desc_input_, dev_input, samples_,
      capacity_, hidden_, num_experts_, dispatch_desc, dev_dispatch));
  interface_timer_.stop();
  VLOG(4) << "[MoeDispatchForwardExecutor] call compute() end.";
  data_vector_[0].is_output = false;
  data_vector_[1].is_output = false;
  data_vector_[2].is_output = false;
  data_vector_[3].is_output = false;
  data_vector_[4].is_output = true;
}

void MoeDispatchForwardExecutor::cpuCompute() {
  VLOG(4) << "[MoeDispatchForwardExecutor] call cpuCompute() begin.";
  float *gates = cpu_fp32_input_[0];
  float *indices = cpu_fp32_input_[1];
  float *locations = cpu_fp32_input_[2];
  float *input = cpu_fp32_input_[3];
  float *dispatch = cpu_fp32_input_[4];
  float *output = cpu_fp32_output_[0];

  for (int i = 0; i < (num_experts_ * capacity_ * hidden_); ++i) {
    output[i] = dispatch[i];
  }
  for (int i = 0; i < samples_; ++i) {
     if (locations[i] >= 0 && locations[i] < capacity_ &&
         indices[i] >= 0 && indices[i] < num_experts_) {
       for (int j = 0; j < hidden_; ++j) {
         int idx = ((int)indices[i] * capacity_ +
                   (int)locations[i]) * (hidden_) + j;
         output[idx] = gates[i] * input[i * (hidden_) + j];
       }
     }
  }

  VLOG(4) << "[MoeDispatchForwardExecutor] call cpuCompute() end.";
}

int64_t MoeDispatchForwardExecutor::getTheoryOps() {
  int64_t theory_ops = 0;
  theory_ops = samples_ * hidden_;
  VLOG(4) << "[MoeDispatchForwardExecutor] getTheoryOps: " << theory_ops
          << " ops.";
  return theory_ops;
}

int64_t MoeDispatchForwardExecutor::getTheoryIoSize() {
  auto gates_dwidth = mluop::getSizeOfDataType(desc_gates_->dtype);
  auto indices_dwidth = mluop::getSizeOfDataType(desc_indices_->dtype);
  auto locations_dwidth = mluop::getSizeOfDataType(desc_locations_->dtype);
  auto input_dwidth = mluop::getSizeOfDataType(desc_input_->dtype);
  auto dispatch_dwidth = mluop::getSizeOfDataType(desc_input_->dtype);

  int64_t gates_theory_ios = samples_ * gates_dwidth;
  int64_t indices_theory_ios = samples_ * indices_dwidth;
  int64_t locations_theory_ios = samples_ * locations_dwidth;
  int64_t input_theory_ios = samples_ * hidden_ * input_dwidth;
  int64_t dispatch_theory_ios = num_experts_ * capacity_ * hidden_ *
                                dispatch_dwidth;

  int64_t theory_ios = gates_theory_ios + indices_theory_ios +
                       locations_theory_ios + input_theory_ios +
                       dispatch_theory_ios;

  VLOG(4) << "MoeDispatchForwardExecutor::getTheoryIoSize() : "
          << theory_ios << " IoSize";
  return theory_ios;
}

}  // namespace mluoptest
