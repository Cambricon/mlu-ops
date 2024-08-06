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
#include "kernels/unary_op/unary_op_host.h"
#include "abs.h"

#define op_name "[mluOpAbs]"

static inline bool isAbsSupportType(const mluOpDataType_t check_type,
                                    const mluOpDataType_t support_type[],
                                    const int len) {
  for (int i = 0; i < len; ++i) {
    if (check_type == support_type[i]) {
      return true;
    }
  }
  return false;
}

static mluOpStatus_t mluOpAbsParamCheck(mluOpHandle_t handle,
                                        const mluOpTensorDescriptor_t x_desc,
                                        const void *x,
                                        const mluOpTensorDescriptor_t y_desc,
                                        void *y, bool *zero_element) {
  PARAM_CHECK(op_name, handle != NULL);
  PARAM_CHECK(op_name, x_desc != NULL);
  PARAM_CHECK(op_name, y_desc != NULL);
  // check dim and dtype
  if (x_desc->dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
    PARAM_CHECK_EQ(op_name, y_desc->dtype, MLUOP_DTYPE_FLOAT);
  } else {
    PARAM_CHECK_EQ(op_name, x_desc->dtype, y_desc->dtype);
  }
  PARAM_CHECK_EQ(op_name, x_desc->dim, y_desc->dim);
  // check data type
  mluOpStatus_t param_check;
  if (handle->arch >= MLUOP_MLU590) {
    mluOpDataType_t support_type[1] = {MLUOP_DTYPE_HALF};
    if (!isAbsSupportType(x_desc->dtype, support_type, 1)) {
      LOG(ERROR) << op_name << ":x_desc's data type is not supported.";
      return MLUOP_STATUS_BAD_PARAM;
    }
  } else {
    mluOpDataType_t support_type[1] = {MLUOP_DTYPE_HALF};
    if (!isAbsSupportType(x_desc->dtype, support_type, 4)) {
      LOG(ERROR) << op_name << ":x_desc's data type is not supported.";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  PARAM_CHECK_GT(op_name, x_desc->dim, 0);
  PARAM_CHECK_GT(op_name, y_desc->dim, 0);
  for (int i = 0; i < x_desc->dim; i++) {
    if (x_desc->dims[i] != y_desc->dims[i]) {
      LOG(ERROR) << op_name << ":The shape of x should be equal to y"
                 << ". But now x_desc's shape[" << i << "] is "
                 << x_desc->dims[i] << ", y_desc's shape[" << i << "] is "
                 << y_desc->dims[i] << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  // check 0 element
  if (mluOpGetTensorElementNum(x_desc) == 0) {
    VLOG(5) << op_name << "skip zero element tensor.";
    *zero_element = true;
    return MLUOP_STATUS_SUCCESS;
  }

  // check largetensor
  if (handle->arch < MLUOP_MLU590) {
    uint64_t num_input = mluOpGetTensorElementNum(x_desc);
    TENSOR_NUM_CHECK(op_name, num_input, LARGE_TENSOR_NUM,
                     "input tensor num is too large. ");
  }

  if (needStrideProcess(x_desc, y_desc)) {
    PARAM_CHECK(op_name, x_desc->dim <= MLUOP_DIM_MAX);
    if (handle->arch < MLUOP_MLU590) {
      // num_with_stride affects offset (related with mul op, which cannot
      // exceed 32-bit on MLU300)
      uint64_t num_input_with_stride = shapeStrideCount(x_desc);
      uint64_t num_output_with_stride = shapeStrideCount(y_desc);
      TENSOR_NUM_CHECK(op_name, num_input_with_stride, LARGE_TENSOR_NUM,
                       "input tensor num with stride is too large. ");
      TENSOR_NUM_CHECK(op_name, num_output_with_stride, LARGE_TENSOR_NUM,
                       "output tensor num with stride is too large. ");
    }
  }

  PARAM_CHECK(op_name, x != NULL);
  PARAM_CHECK(op_name, y != NULL);

  return MLUOP_STATUS_SUCCESS;
}
mluOpStatus_t MLUOP_WIN_API mluOpAbs(mluOpHandle_t handle,
                                     const mluOpTensorDescriptor_t x_desc,
                                     const void *x,
                                     const mluOpTensorDescriptor_t y_desc,
                                     void *y) {
  bool zero_element = false;
  mluOpStatus_t param_check =
      mluOpAbsParamCheck(handle, x_desc, x, y_desc, y, &zero_element);
  if (zero_element == true) {
    return MLUOP_STATUS_SUCCESS;
  }
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }

  // generate prototxt
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("abs", "ABS");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "x", x, x_desc, 10, 0);
    GEN_CASE_DATA(false, "y", y, y_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  // Choose the best task dimension.
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  unaryOpPolicyFuncBlock(handle, &k_dim, &k_type, x_desc);

  size_t dim_x = mluOpGetTensorElementNum(x_desc);

  bool if_stride_kernel = false;
  if (mluop::strideCaseWithNotConsistentDense(2, x_desc, y_desc)) {
    if_stride_kernel = true;
  }
  if (if_stride_kernel) {
    VLOG(5) << "kernel Kernel3StagePipelineWithStrideAbs";
    PARAM_CHECK(op_name, x_desc->dim <= MLUOP_DIM_MAX);
    mluop::TensorShape x_shape;
    mluop::TensorShape y_shape;
    mluop::getTensorShape(x_desc, &x_shape);
    mluop::getTensorShape(y_desc, &y_shape);
    CHECK_RETURN(op_name, Kernel3StagePipelineWithStrideAbs(
                              k_dim, k_type, handle->queue, x_desc->dtype, x,
                              x_shape, y, y_shape, dim_x));
  } else {
    VLOG(5) << "kernel Kernel3StagePipelineAbs";
    CHECK_RETURN(op_name, Kernel3StagePipelineAbs(k_dim, k_type, handle->queue,
                                                  x_desc->dtype, x, y, dim_x));
  }
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
