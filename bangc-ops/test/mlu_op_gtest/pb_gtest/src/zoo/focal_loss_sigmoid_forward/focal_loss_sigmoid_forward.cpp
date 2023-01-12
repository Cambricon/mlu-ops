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

#include <algorithm>
#include <vector>
#include <string>
#include <memory>

#include "focal_loss_sigmoid_forward_cpu.h"
#include "focal_loss_sigmoid_forward.h"

namespace mluoptest {

mluOpComputationPreference_t
FocalLossSigmoidForwardExecutor::getComputationPreference() {
  auto focal_proto_desc =
      parser_->getProtoNode()->focal_loss_sigmoid_forward_param();
  auto prefer = focal_proto_desc.prefer();
  if (prefer == ComputationPreference::COMPUTATION_FAST) {
    return mluOpComputationPreference_t::MLUOP_COMPUTATION_FAST;
  }
  return mluOpComputationPreference_t::MLUOP_COMPUTATION_HIGH_PRECISION;
}

mluOpLossReduction_t FocalLossSigmoidForwardExecutor::getLossReduction() {
  auto focal_proto_desc =
      parser_->getProtoNode()->focal_loss_sigmoid_forward_param();
  auto reduction = focal_proto_desc.reduction();
  switch (reduction) {
    case LossReduction::LOSS_REDUCTION_NONE: {
      return mluOpLossReduction_t::MLUOP_LOSS_REDUCTION_NONE;
    } break;
    case LossReduction::LOSS_REDUCTION_MEAN: {
      return mluOpLossReduction_t::MLUOP_LOSS_REDUCTION_MEAN;
    } break;
    case LossReduction::LOSS_REDUCTION_SUM: {
      return mluOpLossReduction_t::MLUOP_LOSS_REDUCTION_SUM;
    } break;
    default:
      break;
      ;
  }
  return mluOpLossReduction_t::MLUOP_LOSS_REDUCTION_NONE;
}

void FocalLossSigmoidForwardExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_focal_loss_sigmoid_forward_param()) {
    LOG(ERROR) << "Missing has_focal_loss_sigmoid_forward param. ";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
}

void FocalLossSigmoidForwardExecutor::compute() {
  // get params
  auto focal_proto_desc =
      parser_->getProtoNode()->focal_loss_sigmoid_forward_param();
  auto alpha = focal_proto_desc.alpha();
  auto gamma = focal_proto_desc.gamma();
  mluOpComputationPreference_t prefer = getComputationPreference();
  mluOpLossReduction_t reduction = getLossReduction();

  // get inputs
  auto input_desc = tensor_desc_[0].tensor;
  auto input = data_vector_[0].device_ptr;
  auto target_desc = tensor_desc_[1].tensor;
  auto target = data_vector_[1].device_ptr;
  void *weight_desc = nullptr;
  void *weight = nullptr;
  void *output_desc = nullptr;
  void *output = nullptr;

  if (parser_->getInputNum() == 2) {
    output_desc = tensor_desc_[2].tensor;
    output = data_vector_[2].device_ptr;
  } else if (parser_->getInputNum() > 2) {
    weight_desc = tensor_desc_[2].tensor;
    weight = data_vector_[2].device_ptr;
    output_desc = tensor_desc_[3].tensor;
    output = data_vector_[3].device_ptr;
  }

  interface_timer_.start();
  VLOG(5) << "[FOCAL_LOSS_SIGMOID_FORWARD] call mluOp FocalLossSigmoidForward.";
  MLUOP_CHECK(mluOpFocalLossSigmoidForward(
      handle_, prefer, reduction, input_desc, input, target_desc, target,
      (mluOpTensorDescriptor_t)weight_desc, weight, alpha, gamma,
      (mluOpTensorDescriptor_t)output_desc, output));
  interface_timer_.stop();
}

void FocalLossSigmoidForwardExecutor::cpuCompute() {
  // get params
  auto focal_proto_desc =
      parser_->getProtoNode()->focal_loss_sigmoid_forward_param();
  auto prefer = focal_proto_desc.prefer();
  float alpha = focal_proto_desc.alpha();
  float gamma = focal_proto_desc.gamma();
  auto reduction = focal_proto_desc.reduction();

  // get inputs
  float *input = cpu_fp32_input_[0];
  float *target = cpu_fp32_input_[1];
  auto input_count = parser_->getInputDataCount(0);
  auto target_count = parser_->getInputDataCount(1);

  float *weight = nullptr;
  int32_t weight_count = 0;
  if (cpu_fp32_input_.size() > 2) {
    weight = cpu_fp32_input_[2];
    weight_count = parser_->getInputDataCount(2);
  }
  float *output = cpu_fp32_output_[0];
  auto output_count = parser_->getOutputDataCount(0);

  VLOG(5) << "[FOCAL_LOSS_SIGMOID_FORWARD] call focalLossSigmoidForwardCpu.";
  focalLossSigmoidForwardCpu(prefer, reduction, input, input_count, target,
                             target_count, weight, weight_count, alpha, gamma,
                             output);
}

int64_t FocalLossSigmoidForwardExecutor::getTheoryOps() {
  // the shape of input is [N,C].
  size_t input_count = parser_->getInputDataCount(0);
  size_t N = parser_->getInputDataCount(1);
  int64_t theory_ops = 10 * input_count + 9 * N;

  // with weight
  if (parser_->getInputNum() > 2) {
    theory_ops += input_count;
  }
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

std::set<Evaluator::Formula> FocalLossSigmoidForwardExecutor::getCriterionsUse()
    const {
  return {Evaluator::DIFF1, Evaluator::DIFF2};
}

}  // namespace mluoptest
