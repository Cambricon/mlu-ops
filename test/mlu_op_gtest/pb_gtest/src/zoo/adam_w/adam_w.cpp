/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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

#include "adam_w.h"
#include "cn_api.h"

namespace mluoptest {

void AdamWExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_adamw_param()) {
    LOG(ERROR) << "Lose AdamW param. ";
  }
  if (parser_->getInputNum() != 5) {
    LOG(ERROR) << "AdamW input number is wrong. ";
  }
  // set flag
  flag_input_reuse_ = true;
}

void AdamWExecutor::compute() {
  auto desc_param = tensor_desc_[0].tensor;
  auto desc_paramh = tensor_desc_[1].tensor;
  auto desc_momentum = tensor_desc_[2].tensor;
  auto desc_velocity = tensor_desc_[3].tensor;
  auto desc_grad = tensor_desc_[4].tensor;

  auto dev_param = data_vector_[0].device_ptr;
  auto dev_paramh = data_vector_[1].device_ptr;
  auto dev_momentum = data_vector_[2].device_ptr;
  auto dev_velocity = data_vector_[3].device_ptr;
  auto dev_grad = data_vector_[4].device_ptr;

  const float fp32_lr = parser_->getProtoNode()->adamw_param().lr();
  const float fp32_beta1 = parser_->getProtoNode()->adamw_param().beta1();
  const float fp32_beta2 = parser_->getProtoNode()->adamw_param().beta2();
  const float fp32_bias1 = parser_->getProtoNode()->adamw_param().bias1();
  const float fp32_bias2 = parser_->getProtoNode()->adamw_param().bias2();
  const float fp32_epsilon = parser_->getProtoNode()->adamw_param().epsilon();
  const float fp32_weight_decay =
      parser_->getProtoNode()->adamw_param().weight_decay();
  const float fp32_scale = parser_->getProtoNode()->adamw_param().scale();
  bool use_nesterov = parser_->getProtoNode()->adamw_param().use_nesterov();

  if (!exe_config_->enable_lite_interface) {
    VLOG(4) << "call mluOpAdamw. ";
    mluOpAdamWDescriptor_t adamw_desc;
    MLUOP_CHECK(mluOpCreateAdamWDescriptor(&adamw_desc));
    MLUOP_CHECK(mluOpSetAdamWDescAttr(adamw_desc, MLUOP_ADAMW_WEIGHT_DECAY,
                                      &fp32_weight_decay,
                                      sizeof(fp32_weight_decay)));
    MLUOP_CHECK(mluOpSetAdamWDescAttr(adamw_desc, MLUOP_ADAMW_GRAD_SCALE,
                                      &fp32_scale, sizeof(fp32_scale)));
    MLUOP_CHECK(mluOpSetAdamWDescAttr(adamw_desc, MLUOP_ADAMW_USE_NESTEROV,
                                      &use_nesterov, sizeof(use_nesterov)));
    interface_timer_.start();
    MLUOP_CHECK(mluOpAdamW(handle_, adamw_desc, desc_param, dev_param,
                           desc_paramh, dev_paramh, desc_momentum, dev_momentum,
                           desc_velocity, dev_velocity, desc_grad, dev_grad,
                           fp32_lr, fp32_beta1, fp32_beta2, fp32_bias1,
                           fp32_bias2, fp32_epsilon));
    interface_timer_.stop();
    MLUOP_CHECK(mluOpDestroyAdamWDescriptor(adamw_desc));
  } else {
    VLOG(4) << "call mluAdamW. ";
    const int size = mluOpGetTensorElementNum(desc_momentum) * sizeof(float);
    interface_timer_.start();
    const auto adamw_status = bangc_kernels::mluAdamW(
        handle_->queue, fp32_lr, fp32_beta1, fp32_beta2, fp32_bias1, fp32_bias2,
        fp32_epsilon, fp32_weight_decay, fp32_scale, use_nesterov, size,
        BANG_WRAP_T((Eigen::bfloat16 *)dev_paramh),
        BANG_WRAP_T((Eigen::bfloat16 *)dev_grad), (float *)dev_param,
        (float *)dev_momentum, (float *)dev_velocity);
    interface_timer_.stop();
  }
}

void AdamWExecutor::setMiscellaneousParam() {
  data_vector_[0].alsoServeAsOutput();
  data_vector_[1].alsoServeAsOutput();
  data_vector_[2].alsoServeAsOutput();
  data_vector_[3].alsoServeAsOutput();
}

void AdamWExecutor::cpuCompute() {
  GTEST_CHECK(parser_->getInputNum() == 5);
  GTEST_CHECK(parser_->getOutputNum() == 4);
  float lr = parser_->getProtoNode()->adamw_param().lr();
  float beta1 = parser_->getProtoNode()->adamw_param().beta1();
  float beta2 = parser_->getProtoNode()->adamw_param().beta2();
  float bias1 = parser_->getProtoNode()->adamw_param().bias1();
  float bias2 = parser_->getProtoNode()->adamw_param().bias2();
  float epsilon = parser_->getProtoNode()->adamw_param().epsilon();
  float fp32_weight_decay =
      parser_->getProtoNode()->adamw_param().weight_decay();
  float fp32_scale = parser_->getProtoNode()->adamw_param().scale();
  float use_nesterov = parser_->getProtoNode()->adamw_param().use_nesterov();

  auto count1 = parser_->getInputDataCount(0);
  auto count2 = parser_->getInputDataCount(1);
  auto count3 = parser_->getInputDataCount(2);
  auto count4 = parser_->getInputDataCount(3);
  auto count5 = parser_->getInputDataCount(4);

  GTEST_CHECK(count1 == count2);
  GTEST_CHECK(count1 == count3);
  GTEST_CHECK(count1 == count4);
  GTEST_CHECK(count1 == count5);

  auto cpu_tensor_param = cpu_fp32_input_[0];
  auto cpu_tensor_paramh = cpu_fp32_input_[1];
  auto cpu_tensor_momentum = cpu_fp32_input_[2];
  auto cpu_tensor_velocity = cpu_fp32_input_[3];
  auto cpu_tensor_grad = cpu_fp32_input_[4];

  auto cpu_tensor_param_output = cpu_fp32_output_[0];
  auto cpu_tensor_paramh_output = cpu_fp32_output_[1];
  auto cpu_tensor_momentum_output = cpu_fp32_output_[2];
  auto cpu_tensor_velocity_output = cpu_fp32_output_[3];

  for (int i = 0; i < count1; ++i) {
    // output is: momentum velocity param param_h
    cpu_tensor_grad[i] = cpu_tensor_grad[i] / fp32_scale;
    cpu_tensor_momentum_output[i] =
        cpu_tensor_momentum[i] +
        (cpu_tensor_grad[i] - cpu_tensor_momentum[i]) * (1 - beta1);
    cpu_tensor_velocity_output[i] =
        cpu_tensor_velocity[i] +
        (cpu_tensor_grad[i] * cpu_tensor_grad[i] - cpu_tensor_velocity[i]) *
            (1 - beta2);
    cpu_tensor_param_output[i] =
        cpu_tensor_param[i] -
        lr * cpu_tensor_momentum_output[i] / bias1 /
            (sqrt(cpu_tensor_velocity_output[i] / bias2) + epsilon) -
        lr * fp32_weight_decay * cpu_tensor_param[i];
    cpu_tensor_paramh_output[i] = cpu_tensor_param_output[i];
  }
}

}  // namespace mluoptest
