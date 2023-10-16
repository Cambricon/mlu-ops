/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "transform.h"

namespace mluoptest {

typedef struct complex {
  float real;
  float imag;
} ComplexType;

void TransformExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_transform_param()) {
    LOG(ERROR) << "Lose transform param. ";
  }
  if (parser_->getInputNum() != 1) {
    LOG(ERROR) << "transform tensor input number is wrong. ";
  }

  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "transform tensor output number is wrong. ";
  }
}

void TransformExecutor::compute() {
  VLOG(4) << "TransformExecutor compute ";
  if (!parser_->getProtoNode()->has_transform_param()) {
    LOG(ERROR) << "Lose transform param. ";
  }
  auto tensor_a = tensor_desc_[0].tensor;
  auto tensor_out = tensor_desc_[1].tensor;
  auto dev_a = data_vector_[0].device_ptr;
  auto dev_c = data_vector_[1].device_ptr;
  float alpha = 0.0;
  float beta = 0.0;
  if (tensor_a->dtype != MLUOP_DTYPE_COMPLEX_FLOAT) {
    alpha = parser_->getProtoNode()->transform_param().alpha();
    beta = parser_->getProtoNode()->transform_param().beta();
  }
  VLOG(4) << "call mluOp transform()";
  interface_timer_.start();
  bool device_interface = true;
  char *env_temp = getenv("MLUOP_GTEST_TRANSFORM_ONCHIP_SCALE_PARAM");
  if (env_temp && strcmp(env_temp, "ON") == 0) {
    device_interface = true;
  } else {
    device_interface = false;
  }
  void *alpha_device = NULL;
  void *beta_device = NULL;
  if (tensor_a->dtype != MLUOP_DTYPE_COMPLEX_FLOAT) {
    alpha_device = mlu_runtime_.allocate(4);
    beta_device = mlu_runtime_.allocate(4);
  } else {
    alpha_device = mlu_runtime_.allocate(sizeof(ComplexType));
    beta_device = mlu_runtime_.allocate(sizeof(ComplexType));
  }
  if (tensor_a->dtype == MLUOP_DTYPE_INT32) {
    int alpha_int = (int)alpha;
    int beta_int = (int)beta;
    if (device_interface) {
      VLOG(6) << "[gtest]->"
              << "[mluOpTransform]"
              << ": enable gtest onchip scale param";
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMemcpy(alpha_device, &alpha_int, 4,
                             CNRT_MEM_TRANS_DIR_HOST2DEV));
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMemcpy(beta_device, &beta_int, 4,
                             CNRT_MEM_TRANS_DIR_HOST2DEV));
      MLUOP_CHECK(mluOpTransform(handle_, MLUOP_POINTER_MODE_DEVICE,
                                  alpha_device, tensor_a, dev_a,
                                  beta_device, tensor_out, dev_c));
    } else {
      VLOG(6) << "[gtest]->"
              << "[mluOpTransform]"
              << ": enable gtest host scale param";
      MLUOP_CHECK(mluOpTransform(handle_, MLUOP_POINTER_MODE_HOST,
                                  &alpha_int, tensor_a, dev_a,
                                  &beta_int, tensor_out, dev_c));
    }
  } else if (tensor_a->dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
    ComplexType alpha_complex, beta_complex;
    alpha_complex.real = parser_->getProtoNode()->transform_param().alpha();
    alpha_complex.imag =
          parser_->getProtoNode()->transform_param().alpha_imag();
    beta_complex.real = parser_->getProtoNode()->transform_param().beta();
    beta_complex.imag =
           parser_->getProtoNode()->transform_param().beta_imag();
    if (device_interface) {
      VLOG(6) << "[gtest]->"
              << "[mluOpTransform]"
              << ": enable gtest onchip scale param";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMemcpy(alpha_device, &alpha_complex,
                                                 sizeof(ComplexType),
                                                 CNRT_MEM_TRANS_DIR_HOST2DEV));
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMemcpy(beta_device, &beta_complex,
                                                 sizeof(ComplexType),
                                                 CNRT_MEM_TRANS_DIR_HOST2DEV));
      MLUOP_CHECK(mluOpTransform(handle_, MLUOP_POINTER_MODE_DEVICE,
                                 &alpha_complex, tensor_a, dev_a,
                                 &beta_complex, tensor_out, dev_c));
    } else {
      VLOG(6) << "[gtest]->"
              << "[mluOpTransform]"
              << ": enable gtest host scale param";
      MLUOP_CHECK(mluOpTransform(handle_, MLUOP_POINTER_MODE_HOST,
                                 &alpha_complex, tensor_a, dev_a,
                                 &beta_complex, tensor_out, dev_c));
    }
  } else {  // float/half/bfloat16
    if (device_interface) {
      VLOG(6) << "[gtest]->"
              << "[mluOpTransform]"
              << ": enable gtest onchip scale param";
      GTEST_CHECK(CNRT_RET_SUCCESS ==
           cnrtMemcpy(alpha_device, &alpha, 4, CNRT_MEM_TRANS_DIR_HOST2DEV));
      GTEST_CHECK(CNRT_RET_SUCCESS ==
           cnrtMemcpy(beta_device, &beta, 4, CNRT_MEM_TRANS_DIR_HOST2DEV));
      MLUOP_CHECK(mluOpTransform(handle_, MLUOP_POINTER_MODE_DEVICE,
                                 alpha_device, tensor_a, dev_a, beta_device,
                                 tensor_out, dev_c));
    } else {
      VLOG(6) << "[gtest]->"
              << "[mluOpTransform]"
              << ": enable gtest host scale param";
      MLUOP_CHECK(mluOpTransform(handle_, MLUOP_POINTER_MODE_HOST, &alpha,
                                 tensor_a, dev_a, &beta, tensor_out, dev_c));
    }
  }
  mlu_runtime_.deallocate(alpha_device);
  mlu_runtime_.deallocate(beta_device);
  interface_timer_.stop();
}


void TransformExecutor::cpuCompute() {
  assert(parser_->getInputNum() == 1);
  assert(parser_->getOutputNum() == 1);
  float alpha = 0.0;
  float beta = 0.0;

  auto tensor_a = tensor_desc_[0].tensor;
  size_t count1 = parser_->getInputDataCount(0);
  if (tensor_a->dtype != MLUOP_DTYPE_COMPLEX_FLOAT) {
    alpha = parser_->getProtoNode()->transform_param().alpha();
    beta = parser_->getProtoNode()->transform_param().beta();
  }

  if (tensor_a->dtype == MLUOP_DTYPE_INT32) {
    for (size_t i = 0; i < count1; ++i) {
      cpu_fp32_output_[0][i] = (int)alpha * cpu_fp32_input_[0][i] + (int)beta;
    }
  } else if (tensor_a->dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
    ComplexType alpha_complex, beta_complex;
    alpha_complex.real = parser_->getProtoNode()->transform_param().alpha();
    alpha_complex.imag =
            parser_->getProtoNode()->transform_param().alpha_imag();
    beta_complex.real = parser_->getProtoNode()->transform_param().beta();
    beta_complex.imag =
            parser_->getProtoNode()->transform_param().beta_imag();
    for (size_t i = 0; i < count1 * 2; i += 2) {
      float a = cpu_fp32_input_[0][i];
      float b = cpu_fp32_input_[0][i + 1];
      float c = alpha_complex.real;
      float d = alpha_complex.imag;
      cpu_fp32_output_[0][i] = a * c - b * d + beta_complex.real;
      cpu_fp32_output_[0][i + 1] = a * d + b * c + beta_complex.imag;
    }
  } else {  // float/half/bfloat16
    for (size_t i = 0; i < count1; ++i) {
      cpu_fp32_output_[0][i] = alpha * cpu_fp32_input_[0][i] + beta;
    }
  }
}

int64_t TransformExecutor::getTheoryOps() {
  int cp_count = 2;
  int64_t theory_ops = parser_->getInputDataCount(0) * cp_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}


}  // namespace mluoptest
