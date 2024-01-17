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
#include "focal_loss_sigmoid_backward.h"

#include <float.h>
#include <math.h>

#include <algorithm>
#include <iostream>

namespace mluoptest {

void FocalLossSigmoidBackwardExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_focal_loss_sigmoid_backward_param()) {
    LOG(ERROR) << "Lose focal_loss_sigmoid_backward param.";
  }
  if (parser_->getInputNum() != 3 && parser_->getInputNum() != 2) {
    LOG(ERROR) << "focal_loss_sigmoid_backward input number is wrong.";
  }
  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "focal_loss_sigmoid_backward output number is wrong.";
  }
}

void FocalLossSigmoidBackwardExecutor::compute() {
  VLOG(4) << "FocalLossSigmoidBackwardExecutor compute. ";
  mluOpComputationPreference_t prefer =
      (mluOpComputationPreference_t)parser_->getProtoNode()
          ->focal_loss_sigmoid_backward_param()
          .prefer();
  mluOpLossReduction_t reduction = (mluOpLossReduction_t)parser_->getProtoNode()
                                       ->focal_loss_sigmoid_backward_param()
                                       .reduction();
  auto alpha =
      parser_->getProtoNode()->focal_loss_sigmoid_backward_param().alpha();
  auto gamma =
      parser_->getProtoNode()->focal_loss_sigmoid_backward_param().gamma();
  auto input_desc = tensor_desc_[0].tensor;
  auto target_desc = tensor_desc_[1].tensor;
  auto input_mlu = data_vector_[0].device_ptr;
  auto target_mlu = data_vector_[1].device_ptr;
  if (parser_->getInputNum() == 3) {
    auto weight_desc = tensor_desc_[2].tensor;
    auto grad_input_desc = tensor_desc_[3].tensor;
    auto weight_mlu = data_vector_[2].device_ptr;
    auto grad_input_mlu = data_vector_[3].device_ptr;
    interface_timer_.start();
    MLUOP_CHECK(mluOpFocalLossSigmoidBackward(
        handle_, prefer, reduction, input_desc, input_mlu, target_desc,
        target_mlu, weight_desc, weight_mlu,
        alpha, gamma, grad_input_desc, grad_input_mlu));
    interface_timer_.stop();
  } else {
    auto grad_input_desc = tensor_desc_[2].tensor;
    auto grad_input_mlu = data_vector_[2].device_ptr;
    interface_timer_.start();
    MLUOP_CHECK(mluOpFocalLossSigmoidBackward(
        handle_, prefer, reduction, input_desc, input_mlu, target_desc,
        target_mlu, NULL, NULL, alpha, gamma,
        grad_input_desc, grad_input_mlu));
    interface_timer_.stop();
  }
}

void FocalLossSigmoidBackwardExecutor::setMiscellaneousParam() {
  if (parser_->getInputNum() == 3) {
    data_vector_[3].alsoServeAsOutput();
  } else {
    data_vector_[2].alsoServeAsOutput();
  }
}

void FocalLossSigmoidBackwardExecutor::cpuCompute() {
  assert(parser_->getOutputNum() == 1);
  auto alpha =
      parser_->getProtoNode()->focal_loss_sigmoid_backward_param().alpha();
  auto gamma =
      parser_->getProtoNode()->focal_loss_sigmoid_backward_param().gamma();
  auto input_cpu = cpu_fp32_input_[0];
  auto target_cpu = cpu_fp32_input_[1];
  auto grad_input_cpu = cpu_fp32_output_[0];

  float *weight_cpu = NULL;
  if (parser_->getInputNum() == 3) {
    weight_cpu = cpu_fp32_input_[2];
  }
  int count = parser_->getInputDataCount(0);
  auto input_tensor = parser_->getProtoNode()->input(0);
  auto input_shape = input_tensor.shape();
  int N = input_shape.dims(0);
  int C = input_shape.dims(1);
  for (int index = 0; index < count; ++index) {
    int n = index / C;
    int c = index % C;
    int t = target_cpu[n];
    float flag_p = (t == c);
    float flag_n = (t != c);
    // p = sigmoid(x) = 1. / 1. + expf(-x)
    float p = 1.0 / (1.0 + exp(-input_cpu[index]));
    // (1 - p)**gamma * (1 - p - gamma*p*log(p))
    float temp_p =
        pow(1.0 - p, gamma) * (1.0 - p - gamma * p * log(std::max(p, FLT_MIN)));
    // p**gamma * (gamma * (1 - p) * log(1 - p) - p)
    float temp_n =
        pow(p, gamma) *
        (gamma * (1.0 - p) * log(std::max((float)(1.0 - p), FLT_MIN)) - p);
    grad_input_cpu[index] += -flag_p * alpha * temp_p;
    grad_input_cpu[index] += -flag_n * (1.0 - alpha) * temp_n;
    if (weight_cpu != NULL && t != C) {
      grad_input_cpu[index] *= weight_cpu[t];
    }
  }
}

int64_t FocalLossSigmoidBackwardExecutor::getTheoryOps() {
  // the shape of input is [N,C].
  size_t input_count = parser_->getInputDataCount(0);
  size_t N = parser_->getInputDataCount(1);
  int stream_compute_count = 23;
  int scalar_compute_count = 6;
  int64_t theory_ops =
      stream_compute_count * input_count + scalar_compute_count * N;

  // with weight
  if (parser_->getInputNum() > 2) {
    theory_ops += input_count;
  }
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  //  namespace mluoptest
