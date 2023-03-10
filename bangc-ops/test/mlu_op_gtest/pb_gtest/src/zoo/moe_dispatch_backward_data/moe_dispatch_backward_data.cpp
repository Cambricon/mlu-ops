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
#include "moe_dispatch_backward_data.h"

#include <algorithm>
#include <string>

#include "mlu_op.h"

namespace mluoptest {

void MoeDispatchBackwardDataExecutor::printDataInfo() {
  VLOG(4) << "############################### printfDataInfo() Begin.";
  VLOG(4) << "# samples:              " << samples_;
  VLOG(4) << "# capacity:             " << capacity_;
  VLOG(4) << "# hidden:               " << hidden_;
  VLOG(4) << "# num_experts:          " << num_experts_;
  VLOG(4) << "############################### printfDataInfo() End.";
}

void MoeDispatchBackwardDataExecutor::initData() {
  VLOG(4) << "MoeDispatchBackwardDataExecutor::initData() Begin.";
  // get params
  desc_gates_ = tensor_desc_[0].tensor;
  desc_indices_ = tensor_desc_[1].tensor;
  desc_locations_ = tensor_desc_[2].tensor;
  desc_dispatch_ = tensor_desc_[3].tensor;
  desc_grad_input_ = tensor_desc_[4].tensor;

  dev_gates_ = data_vector_[0].device_ptr;
  dev_indices_ = data_vector_[1].device_ptr;
  dev_locations_ = data_vector_[2].device_ptr;
  dev_dispatch_ = data_vector_[3].device_ptr;
  dev_grad_input_ = data_vector_[4].device_ptr;

  auto moe_dispatch_backward_data_proto_desc =
      parser_->getProtoNode()->moe_dispatch_backward_data_param();

  samples_ = moe_dispatch_backward_data_proto_desc.samples();
  capacity_ = moe_dispatch_backward_data_proto_desc.capacity();
  hidden_ = moe_dispatch_backward_data_proto_desc.hidden();
  num_experts_ = moe_dispatch_backward_data_proto_desc.num_experts();

  VLOG(4) << "MoeDispatchBackwardDataExecutor::initData() End.";
}

void MoeDispatchBackwardDataExecutor::paramCheck() {
  VLOG(4) << "MoeDispatchBackwardDataExecutor::paramCheck() Begin.";
  GTEST_CHECK(parser_->getInputNum() == 4);
  GTEST_CHECK(parser_->getOutputNum() == 1);
  if (!parser_->getProtoNode()->has_moe_dispatch_backward_data_param()) {
    LOG(ERROR) << "MoeDispatchBackwardDataExecutor::paramCheck() Missing "
                  "moe_dispatch_backward_data param.";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
  VLOG(4) << "MoeDispatchBackwardDataExecutor::paramCheck() End.";
}

void MoeDispatchBackwardDataExecutor::compute() {
  VLOG(4) << "MoeDispatchBackwardDataExecutor::compute() Begin.";
  initData();
  printDataInfo();

  interface_timer_.start();
  MLUOP_CHECK(mluOpMoeDispatchBackwardData(
      handle_, desc_gates_, dev_gates_, desc_indices_, dev_indices_,
      desc_locations_, dev_locations_, desc_dispatch_, dev_dispatch_, samples_,
      capacity_, hidden_, num_experts_, desc_grad_input_, dev_grad_input_));
  interface_timer_.stop();
  VLOG(4) << "MoeDispatchBackwardDataExecutor::compute() End.";
}

void MoeDispatchBackwardDataExecutor::cpuCompute() {
  VLOG(4) << "MoeDispatchBackwardDataExecutor::cpuCompute() Begin.";

  float *gates = cpu_fp32_input_[0];
  int *indices = (int *)(cpu_fp32_input_[1]);
  int *locations = (int *)(cpu_fp32_input_[2]);
  float *dispatch = cpu_fp32_input_[3];
  float *grad_input = cpu_fp32_output_[0];

  // gtest input value need to be float, so float to int
  for (int i = 0; i < samples_; ++i) {
    indices[i] = (int)(((float *)indices)[i]);
    locations[i] = (int)(((float *)locations)[i]);
  }

  // gates: (samples)
  // indices: (samples)
  // locations: (samples)
  // dispatch: (num_experts * capacity, hidden)
  // grad_input: (samples, hidden)
  std::memset((void *)grad_input, 0x00, samples_ * hidden_ * sizeof(float));
  for (int i = 0; i < samples_; ++i) {
    if (locations[i] < 0 || locations[i] >= capacity_ || indices[i] < 0 ||
        indices[i] >= num_experts_) {
      continue;
    }
    int dispatch_offset = (indices[i] * capacity_ + locations[i]) * (hidden_);
    int grad_input_offset = i * hidden_;
    for (int j = 0; j < hidden_; ++j) {
      grad_input[grad_input_offset + j] =
          gates[i] * dispatch[dispatch_offset + j];
    }
    ++samples_mask_num_;
  }

  VLOG(4) << "MoeDispatchBackwardDataExecutor::cpuCompute() End.";
}

int64_t MoeDispatchBackwardDataExecutor::getTheoryOps() {
  int64_t theory_ops = 0;
  theory_ops = samples_ * 4 + samples_ * hidden_ * 11;
  VLOG(4) << "MoeDispatchBackwardDataExecutor::getTheoryOps : " << theory_ops
          << " Ops";
  return theory_ops;
}

int64_t MoeDispatchBackwardDataExecutor::getTheoryIoSize() {
  auto gates_dwidth = mluop::getSizeOfDataType(desc_gates_->dtype);
  auto indices_dwidth = mluop::getSizeOfDataType(desc_indices_->dtype);
  auto locations_dwidth = mluop::getSizeOfDataType(desc_locations_->dtype);
  auto dispatch_dwidth = mluop::getSizeOfDataType(desc_dispatch_->dtype);
  auto grad_input_dwidth = mluop::getSizeOfDataType(desc_grad_input_->dtype);

  int64_t gates_theory_ios = samples_mask_num_ * gates_dwidth;
  int64_t indices_theory_ios = samples_mask_num_ * indices_dwidth;
  int64_t locations_theory_ios = samples_mask_num_ * locations_dwidth;
  int64_t dispatch_theory_ios = samples_mask_num_ * hidden_ * dispatch_dwidth;
  int64_t grad_input_theory_ios = samples_mask_num_ * grad_input_dwidth;

  int64_t theory_ios = gates_theory_ios + indices_theory_ios +
                       locations_theory_ios + dispatch_theory_ios +
                       grad_input_theory_ios;
  VLOG(4) << "MoeDispatchBackwardDataExecutor::getTheoryIoSize() : "
          << theory_ios << " IoSize";
  return theory_ios;
}

}  // namespace mluoptest
