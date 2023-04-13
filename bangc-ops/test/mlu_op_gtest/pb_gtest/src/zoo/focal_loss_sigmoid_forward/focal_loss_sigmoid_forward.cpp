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
#include "focal_loss_sigmoid_forward.h"

#include <string>

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
    default: {
      break;
    }
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

void FocalLossSigmoidForwardExecutor::focalLossSigmoidForwardCpuFast(
    const float *input, const size_t input_num, const float *target,
    const size_t target_num, const float *weight, const size_t weight_num,
    const float alpha, const float gamma, float *output) {
  size_t N = target_num;
  size_t C = input_num / target_num;

  if (weight_num == 0) {
    for (int i = 0; i < input_num; ++i) {
      int32_t row_num = i / C;
      int32_t col_num = i % C;
      int32_t t = target[row_num];
      float p = 1. / (1. + exp(-input[i]));
      float temp_p = pow(1. - p, gamma) * log(fmax(p, FLT_MIN));
      float temp_n = pow(p, gamma) * log(fmax(1. - p, FLT_MIN));
      if (t == col_num) {
        output[i] = -alpha * temp_p;
      } else {
        output[i] = -(1 - alpha) * temp_n;
      }
    }

  } else {
    for (int i = 0; i < input_num; ++i) {
      int32_t row_num = i / C;
      int32_t col_num = i % C;
      int32_t t = target[row_num];
      float p = 1. / (1. + exp(-input[i]));
      float temp_p = pow(1. - p, gamma) * log(fmax(p, FLT_MIN));
      float temp_n = pow(p, gamma) * log(fmax(1. - p, FLT_MIN));
      if (t == col_num) {
        output[i] = -alpha * temp_p;
      } else {
        output[i] = -(1 - alpha) * temp_n;
      }
      output[i] *= weight[t];
    }
  }
}

void FocalLossSigmoidForwardExecutor::
    focalLossSigmoidForwardCpuHighPrecisionDoub(
        const float *input, const size_t input_num, const float *target,
        const size_t target_num, const float *weight, const size_t weight_num,
        const float alpha, const float gamma, float *output) {
  double focal_loss_temp = 0.0;
  size_t N = target_num;
  size_t C = input_num / target_num;
  double alpha_double = double(alpha);
  double gamma_double = double(gamma);

  if (weight_num == 0) {
    for (int i = 0; i < input_num; ++i) {
      int32_t row_num = i / C;
      int32_t col_num = i % C;
      int32_t t = target[row_num];
      double x = double(input[i]);
      double p = 1. / (1. + exp(-x));
      double temp_p = pow(1. - p, gamma_double) * log(fmax(p, DBL_MIN));
      double temp_n = pow(p, gamma_double) * log(fmax(1. - p, DBL_MIN));
      if (t == col_num) {
        focal_loss_temp = -alpha_double * temp_p;
      } else {
        focal_loss_temp = -(1 - alpha_double) * temp_n;
      }
      output[i] = float(focal_loss_temp);
    }

  } else {
    for (int i = 0; i < input_num; ++i) {
      int32_t row_num = i / C;
      int32_t col_num = i % C;
      int32_t t = target[row_num];
      double x = double(input[i]);
      double p = 1. / (1. + exp(-x));
      double temp_p = pow(1. - p, gamma_double) * log(fmax(p, DBL_MIN));
      double temp_n = pow(p, gamma_double) * log(fmax(1. - p, DBL_MIN));
      if (t == col_num) {
        focal_loss_temp = -alpha_double * temp_p;
      } else {
        focal_loss_temp = -(1 - alpha_double) * temp_n;
      }
      output[i] = float(focal_loss_temp);
      output[i] *= weight[t];

      double pow_temp = pow(1. - p, gamma);
      double log_temp = log(fmax(p, DBL_MIN));
    }
  }
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
  auto input_num = parser_->getInputDataCount(0);
  auto target_num = parser_->getInputDataCount(1);

  float *weight = nullptr;
  int32_t weight_num = 0;
  if (cpu_fp32_input_.size() > 2) {
    weight = cpu_fp32_input_[2];
    weight_num = parser_->getInputDataCount(2);
  }
  float *output = cpu_fp32_output_[0];
  auto output_num = parser_->getOutputDataCount(0);

  VLOG(5)
      << "[FOCAL_LOSS_SIGMOID_FORWARD] call focalLossSigmoidForwardCpu....2017";
  {
    switch (prefer) {
      case ComputationPreference::COMPUTATION_FAST: {
        VLOG(5) << "[focalLossSigmoidForwardCpu] call Fast.";
        focalLossSigmoidForwardCpuFast(input, input_num, target, target_num,
                                       weight, weight_num, alpha, gamma,
                                       output);
      }; break;
      case ComputationPreference::COMPUTATION_HIGH_PRECISION: {
        VLOG(5) << "[focalLossSigmoidForwardCpu] call HighPrecision.";
        focalLossSigmoidForwardCpuHighPrecisionDoub(
            input, input_num, target, target_num, weight, weight_num, alpha,
            gamma, output);
      }; break;
      default:
        LOG(ERROR) << "[focalLossSigmoidForwardCpu] not Implemented.";
    }
  }
}

int64_t FocalLossSigmoidForwardExecutor::getTheoryOps() {
  // the shape of input is [N,C].
  size_t input_num = parser_->getInputDataCount(0);
  size_t N = parser_->getInputDataCount(1);
  int64_t theory_ops = 10 * input_num + 9 * N;

  // with weight
  if (parser_->getInputNum() > 2) {
    theory_ops += input_num;
  }
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

std::set<Evaluator::Formula> FocalLossSigmoidForwardExecutor::getCriterionsUse()
    const {
  return {Evaluator::DIFF1, Evaluator::DIFF2};
}

}  // namespace mluoptest
