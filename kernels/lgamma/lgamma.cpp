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
#include "lgamma.h"

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/unary_op/unary_op_host.h"

mluOpStatus_t MLUOP_WIN_API mluOpLgamma(mluOpHandle_t handle,
                                        const mluOpTensorDescriptor_t x_desc,
                                        const void *x,
                                        const mluOpTensorDescriptor_t y_desc,
                                        void *y) {
  // param check
  mluOpDataType_t support_type[2] = {MLUOP_DTYPE_HALF, MLUOP_DTYPE_FLOAT};
  bool zero_element = false;
  mluOpStatus_t param_check =
      unaryOpParamCheck("[mluOpLgamma]", handle, x_desc, x, y_desc, y,
                        support_type, 2, zero_element);
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }
  if (zero_element == true) {
    return MLUOP_STATUS_SUCCESS;
  }

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("lgamma", "LGAMMA");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "x", x, x_desc, 100, 0.1);
    GEN_CASE_DATA(false, "y", y, y_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "lgamma", "inplace", (x == y));
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  // policy select
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  unaryOpPolicyFunc(handle, &k_dim, &k_type, x_desc);
  VLOG(5) << "[mluOpLgamma] launch kernel policyFUnc[" << k_dim.x << ", "
          << k_dim.y << ", " << k_dim.z << "]";

  size_t element_num = mluOpGetTensorElementNum(x_desc);
  if (handle->arch < MLUOP_MLU370) {
    LOG(ERROR) << "[mluOpLgamma] now only support ARCH >= <MLU370>\n";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  bool if_stride_kernel = false;
  if (mluop::strideCaseWithNotConsistentDense(2, x_desc, y_desc)) {
    if_stride_kernel = true;
  }
  if (if_stride_kernel) {
    VLOG(5) << "kernel Kernel3StagePipelineWithStrideLgamma";
    PARAM_CHECK("[mluOpLgamma]", x_desc->dim <= MLUOP_DIM_MAX);
    mluop::TensorShape x_shape;
    mluop::TensorShape y_shape;
    mluop::getTensorShape(x_desc, &x_shape);
    mluop::getTensorShape(y_desc, &y_shape);
    CHECK_RETURN("[mluOpLgamma]",
                 Kernel3StagePipelineWithStrideLgamma(
                     k_dim, k_type, handle->queue, x_desc->dtype, x, x_shape, y,
                     y_shape, element_num));
  } else {
    VLOG(5) << "kernel Kernel3StagePipelineLgamma.";
    CHECK_RETURN("[mluOpLgamma]",
                 Kernel3StagePipelineLgamma(k_dim, k_type, handle->queue,
                                            x_desc->dtype, x, y, element_num));
  }
  GEN_CASE_END();

  return MLUOP_STATUS_SUCCESS;
}
