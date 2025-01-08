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
#include "sqrt.h"

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/binary_op/binary_op_host.h"
#include "kernels/unary_op/unary_op_host.h"

#define op_name_forward "[mluOpSqrt]"
#define op_name_backward "[mluOpSqrtBackWard]"
#define ALIGN_SIZE 128

mluOpStatus_t MLUOP_WIN_API mluOpSqrt(mluOpHandle_t handle,
                                      const mluOpComputationPreference_t prefer,
                                      const mluOpTensorDescriptor_t x_desc,
                                      const void *x,
                                      const mluOpTensorDescriptor_t y_desc,
                                      void *y) {
  VLOG(5) << op_name_forward << " begin: ";
  mluOpComputationPreference_t support_prefer_type[2] = {
      MLUOP_COMPUTATION_FAST, MLUOP_COMPUTATION_HIGH_PRECISION};
  mluOpStatus_t param_check_prefer = MLUOP_STATUS_BAD_PARAM;
  for (int i = 0; i < 2; ++i) {
    if (support_prefer_type[i] == prefer) {
      param_check_prefer = MLUOP_STATUS_SUCCESS;
    }
  }
  if (param_check_prefer != MLUOP_STATUS_SUCCESS) {
    LOG(ERROR) << op_name_forward
               << ":prefer's mode is not supported, it should be "
                  "MLUOP_COMPUTATION_FAST "
                  "or MLUOP_COMPUTATION_HIGH_PRECISION.";
    return param_check_prefer;
  }

  // transform int32 to float when input's type is int32
  PARAM_CHECK(op_name_forward, x_desc != NULL);
  PARAM_CHECK(op_name_forward, y_desc != NULL);
  bool x_dtype_transform = false;
  if (x_desc->getDtype() == MLUOP_DTYPE_INT32 &&
      y_desc->getDtype() == MLUOP_DTYPE_FLOAT) {
    x_desc->setDtype(MLUOP_DTYPE_FLOAT);
    x_dtype_transform = true;
  }

  PARAM_CHECK(op_name_forward, handle != NULL);
  bool zero_element = false;
  mluOpStatus_t param_check = MLUOP_STATUS_SUCCESS;
  if (handle->arch >= MLUOP_MLU590) {
    mluOpDataType_t support_type[3] = {MLUOP_DTYPE_HALF, MLUOP_DTYPE_FLOAT,
                                       MLUOP_DTYPE_BFLOAT16};
    param_check = unaryOpParamCheck(op_name_forward, handle, x_desc, x, y_desc,
                                    y, support_type, 3, zero_element);
  } else {
    mluOpDataType_t support_type[2] = {MLUOP_DTYPE_HALF, MLUOP_DTYPE_FLOAT};
    param_check = unaryOpParamCheck(op_name_forward, handle, x_desc, x, y_desc,
                                    y, support_type, 2, zero_element);
  }
  // correct the input's type
  if (x_dtype_transform) {
    x_desc->setDtype(MLUOP_DTYPE_INT32);
  }

  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }
  // check stride
  if (mluop::strideCaseWithNotConsistentDense(2, x_desc, y_desc)) {
    LOG(ERROR) << op_name_forward
               << ": stride case with not consistent dense is not supported.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  if (zero_element == true) {
    return MLUOP_STATUS_SUCCESS;
  }

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("sqrt", "SQRT");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "x", x, x_desc, 100, 0.1);
    GEN_CASE_DATA(false, "y", y, y_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  // Choose the best task dimension.
  // Choose the best task dimension.
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  unaryOpPolicyFuncBlock(handle, &k_dim, &k_type, x_desc);

  size_t dim_x = mluOpGetTensorElementNum(x_desc);

  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  VLOG(5) << "kernel Kernel3StagePipelineSqrt.";
  CHECK_RETURN("[mluOpSqrt] ", Kernel3StagePipelineSqrt(
                                   k_dim, k_type, handle->queue,
                                   x_desc->getDtype(), prefer, x, y, dim_x));

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSqrtBackward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t y_desc, const void *y,
    const mluOpTensorDescriptor_t dy_desc, const void *diff_y,
    const mluOpTensorDescriptor_t dx_desc, void *diff_x) {
  mluOpStatus_t param_check = MLUOP_STATUS_SUCCESS;
  bool zero_element = false;
  mluOpDataType_t support_type[2] = {MLUOP_DTYPE_HALF, MLUOP_DTYPE_FLOAT};
  param_check =
      binaryOpParamCheck(op_name_backward, handle, y_desc, y, dy_desc, diff_y,
                         dx_desc, diff_x, support_type, 2, zero_element, false);
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }

  // check stride
  if (mluop::strideCaseWithNotConsistentDense(3, y_desc, dy_desc, dx_desc)) {
    LOG(ERROR) << op_name_backward
               << " stride case with not consistent dense is not supported.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("sqrt_backward", "SQRT_BACKWARD");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "y", y, y_desc, 10, 0.1);
    GEN_CASE_DATA(true, "diff_y", diff_y, dy_desc, -10, 10);
    GEN_CASE_DATA(false, "diff_x", diff_x, dx_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  binaryOpPolicyFunc(handle, ALIGN_SIZE, &k_dim, &k_type, y_desc);

  size_t num_elem = mluOpGetTensorElementNum(y_desc);
  VLOG(5) << "Kernel Kernel3StagePipelineSqrtBackward.";
  CHECK_RETURN("[mluOpSqrtBackward] ",
               Kernel3StagePipelineSqrtBackward(k_dim, k_type, handle->queue,
                                                y_desc->getDtype(), y, diff_y,
                                                diff_x, num_elem));
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
