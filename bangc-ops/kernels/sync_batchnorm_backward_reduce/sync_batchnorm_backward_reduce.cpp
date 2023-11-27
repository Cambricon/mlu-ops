/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
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

mluOpStatus_t MLUOP_WIN_API mluOpGetSyncBatchnormBackwardReduceWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t desc_x,
    size_t *workspace_size) {
  PARAM_CHECK("mluOpSyncBatchnormBackwardReduce_v2", handle != NULL);
  PARAM_CHECK("mluOpSyncBatchnormBackwardReduce_v2", desc_x != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_x, _desc_x);

  CHECK_FUNC_RETURN(
      cnnlGetSyncBatchnormBackwardReduceWorkspaceSize(_handle, _desc_x,
                                                      workspace_size),
      CNNL_STATUS_SUCCESS,
      "[mluOpSyncBatchnormBackwardReduce_v2] Internal error"
      " accured in mluOpGetSyncBatchnormBackwardReduceWorkspaceSize.",
      MLUOP_STATUS_INTERNAL_ERROR);

  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_x);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSyncBatchnormBackwardReduce(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t desc_dz, const void *dz,
    const mluOpTensorDescriptor_t desc_x, const void *x,
    const mluOpTensorDescriptor_t desc_mean, const void *mean,
    const mluOpTensorDescriptor_t desc_invstd, const void *invstd,
    const mluOpTensorDescriptor_t desc_dfilter, void *dfilter,
    const mluOpTensorDescriptor_t desc_dbias, void *dbias,
    const mluOpTensorDescriptor_t desc_sum_dy, void *sum_dy,
    const mluOpTensorDescriptor_t desc_sum_dy_xmu, void *sum_dy_xmu,
    const bool needs_input_grad0, const bool needs_input_grad1,
    const bool needs_input_grad2) {
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", handle != NULL);
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", desc_dz != NULL);
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", desc_x != NULL);
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", desc_mean != NULL);
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", desc_invstd != NULL);
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", dz != NULL);
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", x != NULL);
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", mean != NULL);
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", invstd != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_dz, _desc_dz);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_x, _desc_x);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_mean, _desc_mean);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_invstd, _desc_invstd);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_dfilter, _desc_dfilter);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_dbias, _desc_dbias);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_sum_dy, _desc_sum_dy);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_sum_dy_xmu,
                                               _desc_sum_dy_xmu);

  CHECK_FUNC_RETURN(
      cnnlSyncBatchnormBackwardReduce(
          _handle, _desc_dz, dz, _desc_x, x, _desc_mean, mean, _desc_invstd,
          invstd, _desc_dfilter, dfilter, _desc_dbias, dbias, _desc_sum_dy,
          sum_dy, _desc_sum_dy_xmu, sum_dy_xmu, needs_input_grad0,
          needs_input_grad1, needs_input_grad2),
      CNNL_STATUS_SUCCESS,
      "[mluOpSyncBatchnormBackwardReduce] Internal error"
      " accured in mluOpSyncBatchnormBackwardReduce.",
      MLUOP_STATUS_INTERNAL_ERROR);

  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_dz);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_x);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_mean);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_invstd);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_dfilter);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_dbias);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_sum_dy);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_sum_dy_xmu);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSyncBatchnormBackwardReduce_v2(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t desc_dz, const void *dz,
    const mluOpTensorDescriptor_t desc_x, const void *x,
    const mluOpTensorDescriptor_t desc_mean, const void *mean,
    const mluOpTensorDescriptor_t desc_invstd, const void *invstd,
    void *workspace, size_t workspace_size,
    const mluOpTensorDescriptor_t desc_dfilter, void *dfilter,
    const mluOpTensorDescriptor_t desc_dbias, void *dbias,
    const mluOpTensorDescriptor_t desc_sum_dy, void *sum_dy,
    const mluOpTensorDescriptor_t desc_sum_dy_xmu, void *sum_dy_xmu,
    const bool needs_input_grad0, const bool needs_input_grad1,
    const bool needs_input_grad2) {
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", handle != NULL);
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", desc_dz != NULL);
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", desc_x != NULL);
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", desc_mean != NULL);
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", desc_invstd != NULL);
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", dz != NULL);
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", x != NULL);
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", mean != NULL);
  PARAM_CHECK("[mluOpSyncBatchnormBackwardReduce]", invstd != NULL);
  if (workspace_size > 0) {
    PARAM_CHECK("mluOpSyncBatchnormBackwardReduce_v2", workspace != NULL);
  }

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_dz, _desc_dz);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_x, _desc_x);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_mean, _desc_mean);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_invstd, _desc_invstd);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_dfilter, _desc_dfilter);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_dbias, _desc_dbias);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_sum_dy, _desc_sum_dy);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_sum_dy_xmu,
                                               _desc_sum_dy_xmu);

  CHECK_FUNC_RETURN(
      cnnlSyncBatchnormBackwardReduce_v2(
          _handle, _desc_dz, dz, _desc_x, x, _desc_mean, mean, _desc_invstd,
          invstd, workspace, workspace_size, _desc_dfilter, dfilter,
          _desc_dbias, dbias, _desc_sum_dy, sum_dy, _desc_sum_dy_xmu,
          sum_dy_xmu, needs_input_grad0, needs_input_grad1, needs_input_grad2),
      CNNL_STATUS_SUCCESS,
      "[mluOpSyncBatchnormBackwardReduce] Internal error"
      " accured in mluOpSyncBatchnormBackwardReduce_v2.",
      MLUOP_STATUS_INTERNAL_ERROR);

  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_dz);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_x);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_mean);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_invstd);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_dfilter);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_dbias);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_sum_dy);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc_sum_dy_xmu);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}
