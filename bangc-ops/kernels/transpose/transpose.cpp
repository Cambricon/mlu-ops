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
#include "kernels/utils/cnnl_helper.h"

mluOpStatus_t MLUOP_WIN_API
mluOpCreateTransposeDescriptor(mluOpTransposeDescriptor_t *desc) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpCreateTransposeDescriptor] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpTranspose", desc != NULL);
  CHECK_FUNC_RETURN(cnnlCreateTransposeDescriptor(desc), CNNL_STATUS_SUCCESS,
                    "[mluOpTranspose] Internal error accured in "
                    "mluOpCreateTransposeDescriptor.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpDestroyTransposeDescriptor(mluOpTransposeDescriptor_t desc) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpDestroyTransposeDescriptor] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpTranspose", desc != NULL);
  cnnlDestroyTransposeDescriptor(desc);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetTransposeDescriptor(
    mluOpTransposeDescriptor_t desc, const int dims, const int *permute) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpSetTransposeDescriptor] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpTranspose", desc != NULL);
  CHECK_FUNC_RETURN(cnnlSetTransposeDescriptor(desc, dims, permute),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpTranspose] Internal error accured in "
                    "mluOpSetTransposeDescriptor.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetTransposeWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t x_desc,
    const mluOpTransposeDescriptor_t desc, size_t *size) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpGetTransposeWorkspaceSize] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpTranspose", handle != NULL);
  PARAM_CHECK("mluOpTranspose", x_desc != NULL);
  PARAM_CHECK("mluOpTranspose", desc != NULL);
  PARAM_CHECK("mluOpTranspose", size != NULL);
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(x_desc, cnnl_x_desc);
  CHECK_FUNC_RETURN(
      cnnlGetTransposeWorkspaceSize(cnnl_handle, cnnl_x_desc, desc, size),
      CNNL_STATUS_SUCCESS,
      "[mluOpTranspose] Internal error accured in "
      "mluOpGetTransposeWorkspaceSize.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpTranspose(mluOpHandle_t handle, const mluOpTransposeDescriptor_t desc,
               const mluOpTensorDescriptor_t x_desc, const void *x,
               const mluOpTensorDescriptor_t y_desc, void *y) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpTranspose] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpTranspose", handle != NULL);
  PARAM_CHECK("mluOpTranspose", desc != NULL);
  PARAM_CHECK("mluOpTranspose", x_desc != NULL);
  PARAM_CHECK("mluOpTranspose", x != NULL);
  PARAM_CHECK("mluOpTranspose", y_desc != NULL);
  PARAM_CHECK("mluOpTranspose", y != NULL);
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(x_desc, cnnl_x_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(y_desc, cnnl_y_desc);
  CHECK_FUNC_RETURN(
      cnnlTranspose(cnnl_handle, desc, cnnl_x_desc, x, cnnl_y_desc, y),
      CNNL_STATUS_SUCCESS,
      "[mluOpTranspose] Internal error accured in mluOpTranspose.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_y_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpTranspose_v2(mluOpHandle_t handle, const mluOpTransposeDescriptor_t desc,
                  const mluOpTensorDescriptor_t x_desc, const void *x,
                  const mluOpTensorDescriptor_t y_desc, void *y,
                  void *workspace, size_t workspace_size) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpTranspose_v2] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpTranspose_v2", handle != NULL);
  PARAM_CHECK("mluOpTranspose_v2", desc != NULL);
  PARAM_CHECK("mluOpTranspose_v2", x_desc != NULL);
  PARAM_CHECK("mluOpTranspose_v2", x != NULL);
  PARAM_CHECK("mluOpTranspose_v2", y_desc != NULL);
  PARAM_CHECK("mluOpTranspose_v2", y != NULL);
  if (workspace_size > 0) {
    PARAM_CHECK("mluOpTranspose_v2", workspace != NULL);
  }
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(x_desc, cnnl_x_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(y_desc, cnnl_y_desc);
  CHECK_FUNC_RETURN(cnnlTranspose_v2(cnnl_handle, desc, cnnl_x_desc, x,
                                     cnnl_y_desc, y, workspace, workspace_size),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpTranspose_v2] Internal error accured in "
                    "mluOpTranspose_v2.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_y_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
