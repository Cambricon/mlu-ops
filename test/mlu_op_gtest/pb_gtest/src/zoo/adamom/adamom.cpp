/*************************************************************************
 * Copyright (C) [2025] by Cambricon, Inc.
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
#include "adamom.h"
#include "memory"

namespace mluoptest {
void AdamomExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_adamom_param()) {
    LOG(ERROR) << "Lose Adamom param. ";
  }

  if (parser_->getInputNum() != 5) {
    LOG(ERROR) << "adamom input number is wrong. ";
  }

  // set flag
  flag_input_reuse_ = true;
}

void AdamomExecutor::workspaceMalloc() {
  auto desc_var = tensor_desc_[1].tensor;
  init_dev_ptr_by_type(desc_var->getDtype());
}

void AdamomExecutor::workspaceFree() {
  destroy_dev_ptr();
}

void AdamomExecutor::compute() {
  VLOG(4) << "AdamomExecutor compute ";

  auto desc_grads = tensor_desc_[0].tensor;
  auto desc_ms = tensor_desc_[1].tensor;
  auto desc_vs = tensor_desc_[2].tensor;
  auto desc_v_bias_corrections = tensor_desc_[3].tensor;
  auto desc_weights = tensor_desc_[4].tensor;
  auto dev_grads = data_vector_[0].device_ptr;
  auto dev_ms = data_vector_[1].device_ptr;
  auto dev_vs = data_vector_[2].device_ptr;
  auto dev_v_bias_corrections = data_vector_[3].device_ptr;
  auto dev_weights = data_vector_[4].device_ptr;

  float fp32_lr = parser_->getProtoNode()->adamom_param().lr();
  float fp32_beta1 = parser_->getProtoNode()->adamom_param().beta1();
  float fp32_beta2 = parser_->getProtoNode()->adamom_param().beta2();
  float fp32_weight_decay = parser_->getProtoNode()->adamom_param().weight_decay();
  float fp32_epsilon = parser_->getProtoNode()->adamom_param().epsilon();
  bool nan_inf_found = parser_->getProtoNode()->adamom_param().nan_inf_found();

  GTEST_CHECK(cnrtSuccess ==
                cnrtMemcpy(dev_lr, &fp32_lr,
                sizeof(float), cnrtMemcpyHostToDev))
  GTEST_CHECK(cnrtSuccess ==
                cnrtMemcpy(dev_beta1, &fp32_beta1,
                sizeof(float), cnrtMemcpyHostToDev))
  GTEST_CHECK(cnrtSuccess ==
                cnrtMemcpy(dev_beta2, &fp32_beta2,
                sizeof(float), cnrtMemcpyHostToDev))
  GTEST_CHECK(cnrtSuccess ==
                cnrtMemcpy(dev_weight_decay, &fp32_weight_decay,
                sizeof(float), cnrtMemcpyHostToDev))
  GTEST_CHECK(cnrtSuccess ==
                cnrtMemcpy(dev_epsilon, &fp32_epsilon,
                sizeof(float), cnrtMemcpyHostToDev))
  GTEST_CHECK(cnrtSuccess ==
                cnrtMemcpy(dev_nan_inf_found, &nan_inf_found,
                sizeof(bool), cnrtMemcpyHostToDev))

  VLOG(4) << "call mluOpAdamom()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpAdamom(handle_, desc_grads,
                        dev_grads, desc_ms, dev_ms, desc_vs, dev_vs, desc_v_bias_corrections,
                        dev_v_bias_corrections, desc_weights, dev_weights, dev_nan_inf_found,
                        dev_lr, dev_beta1, dev_beta2, dev_weight_decay, dev_epsilon));
  interface_timer_.stop();

}

void AdamomExecutor::setMiscellaneousParam() {
  data_vector_[1].alsoServeAsOutput();
  data_vector_[2].alsoServeAsOutput();
  data_vector_[3].alsoServeAsOutput();
  data_vector_[4].alsoServeAsOutput();
}

void AdamomExecutor::cpuCompute() {
  auto count1 = parser_->getInputDataCount(0);

  assert(parser_->getInputNum() == 5);
  assert(parser_->getOutputNum() == 4);
  float lr = parser_->getProtoNode()->adamom_param().lr();
  float beta1 = parser_->getProtoNode()->adamom_param().beta1();
  float beta2 = parser_->getProtoNode()->adamom_param().beta2();
  float weight_decay = parser_->getProtoNode()->adamom_param().weight_decay();
  float epsilon = parser_->getProtoNode()->adamom_param().epsilon();
  for (size_t i = 0; i < count1; i++) {
    float grad = cpu_fp32_input_[0][i];
    float m = cpu_fp32_input_[1][i];
    float v = cpu_fp32_input_[2][i];
    float v_bias_correction = cpu_fp32_input_[3][i];
    float weight = cpu_fp32_input_[4][i];
    float dx = grad + weight_decay * weight;
    float new_v = beta2 * v + dx * dx;
    float new_v_bias_correction = beta2 * v_bias_correction + 1;
    float new_m = m * beta1 + dx * (1 - beta1);
    float eta = lr * std::sqrt(1.0f / (new_v / new_v_bias_correction + epsilon));
    float new_weight = weight - eta * new_m;
    if (std::isfinite(new_m) && std::isfinite(new_v) && std::isfinite(new_weight) && new_v >= 0) {
      cpu_fp32_output_[0][i] = new_m;
      cpu_fp32_output_[1][i] = new_v;
      cpu_fp32_output_[2][i] = new_v_bias_correction;
      cpu_fp32_output_[3][i] = new_weight;
    } else {
      cpu_fp32_output_[0][i] = m;
      cpu_fp32_output_[1][i] = v;
      cpu_fp32_output_[2][i] = v_bias_correction;
      cpu_fp32_output_[3][i] = weight;
    }
  }
}
}  // namespace mluoptest
