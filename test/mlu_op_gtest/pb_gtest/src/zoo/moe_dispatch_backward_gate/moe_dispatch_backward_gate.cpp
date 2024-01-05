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
#include "moe_dispatch_backward_gate.h"

namespace mluoptest {

void MoeDispatchBackwardGateExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_moe_dispatch_backward_gate_param()) {
    LOG(ERROR) << "Lose moe_dispatch_backward_gate_param.";
  }
  GTEST_CHECK(
      parser_->inputs().size() == 4,
      "[MoeDispatchBackwardGateExecutor] tensor input number is wrong.");
  GTEST_CHECK(
      parser_->outputs().size() == 1,
      "[MoeDispatchBackwardGateExecutor] tensor output number is wrong.");
}

void MoeDispatchBackwardGateExecutor::initData() {
  samples_ =
      parser_->getProtoNode()->moe_dispatch_backward_gate_param().samples();
  capacity_ =
      parser_->getProtoNode()->moe_dispatch_backward_gate_param().capacity();
  hidden_ =
      parser_->getProtoNode()->moe_dispatch_backward_gate_param().hidden();
  num_experts_ =
      parser_->getProtoNode()->moe_dispatch_backward_gate_param().num_experts();
}

void MoeDispatchBackwardGateExecutor::workspaceMalloc() {
  VLOG(4) << "[MoeDispatchBackwardGateExecutor] call workspaceMalloc() Begin.";
  auto input_desc = tensor_desc_[2].tensor;
  MLUOP_CHECK(mluOpGetMoeDispatchBackwardGateWorkspaceSize(handle_, input_desc,
                                                           &workspace_size_));

  void *workspace_ptr = nullptr;
  if (workspace_size_ > 0) {
    workspace_ptr = mlu_runtime_.allocate(workspace_size_);
    eva_->setMluWorkspaceSize(workspace_size_);
  }
  workspace_.push_back(workspace_ptr);
  VLOG(4) << "[MoeDispatchBackwardGateExecutor] call workspaceMalloc() End.";
}

void MoeDispatchBackwardGateExecutor::workspaceFree() {
  if (workspace_size_ > 0) {
    VLOG(4) << "[MoeDispatchBackwardGateExecutor] Free device workspace space.";
    GTEST_CHECK(CNRT_RET_SUCCESS == mlu_runtime_.deallocate(workspace_[0]));
    workspace_[0] = nullptr;
  }
}

void MoeDispatchBackwardGateExecutor::compute() {
  VLOG(4) << "[MoeDispatchBackwardGateExecutor] call compute() begin.";
  initData();
  // input tensor
  auto indices_desc = tensor_desc_[0].tensor;
  auto locations_desc = tensor_desc_[1].tensor;
  auto input_desc = tensor_desc_[2].tensor;
  auto dispatch_desc = tensor_desc_[3].tensor;
  auto dev_indices = data_vector_[0].device_ptr;
  auto dev_locations = data_vector_[1].device_ptr;
  auto dev_input = data_vector_[2].device_ptr;
  auto dev_dispatch = data_vector_[3].device_ptr;

  // output tensor
  auto grad_gates_desc = tensor_desc_[4].tensor;
  auto dev_grad_gates = data_vector_[4].device_ptr;

  interface_timer_.start();
  MLUOP_CHECK(mluOpMoeDispatchBackwardGate(
      handle_, indices_desc, dev_indices, locations_desc, dev_locations,
      input_desc, dev_input, dispatch_desc, dev_dispatch, samples_, capacity_,
      hidden_, num_experts_, workspace_[0], workspace_size_, grad_gates_desc,
      dev_grad_gates));
  interface_timer_.stop();
  VLOG(4) << "[MoeDispatchBackwardGateExecutor] call compute() end.";
}

void MoeDispatchBackwardGateExecutor::cpuCompute() {
  VLOG(4) << "[MoeDispatchBackwardGateExecutor] call cpuCompute() begin.";
  float *indices = cpu_fp32_input_[0];
  float *locations = cpu_fp32_input_[1];
  float *input = cpu_fp32_input_[2];
  float *dispatch = cpu_fp32_input_[3];
  float *grad_gates = cpu_fp32_output_[0];

  for (int i = 0; i < samples_; ++i) {
    grad_gates[i] = 0.0;
    if (locations[i] < 0 || locations[i] >= capacity_ || indices[i] < 0 ||
        indices[i] >= num_experts_) {
      continue;
    }

    int idx1 = ((int)indices[i] * capacity_ + (int)locations[i]) * hidden_;
    int idx2 = i * hidden_;
    for (int j = 0; j < hidden_; ++j) {
      grad_gates[i] += dispatch[idx1 + j] * input[idx2 + j];
    }
  }
  VLOG(4) << "[MoeDispatchBackwardGateExecutor] call cpuCompute() end.";
}

int64_t MoeDispatchBackwardGateExecutor::getTheoryOps() {
  int64_t theory_ops = 0;
  theory_ops += parser_->getInputDataCount(0) * 3;
  theory_ops += parser_->getInputDataCount(2) * 2;
  VLOG(4) << "[MoeDispatchBackwardGateExecutor] getTheoryOps: " << theory_ops
          << " ops.";
  return theory_ops;
}

}  // namespace mluoptest
