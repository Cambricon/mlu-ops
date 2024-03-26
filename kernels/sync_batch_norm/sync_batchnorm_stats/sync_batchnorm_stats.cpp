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

mluOpStatus_t MLUOP_WIN_API mluOpGetSyncBatchNormStatsWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t x_desc,
    size_t *workspace_size) {
  PARAM_CHECK("mluOpSyncBatchNormStats_v2", handle != NULL);
  PARAM_CHECK("mluOpSyncBatchNormStats_v2", x_desc != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(x_desc, cnnl_x_desc);

  CALL_CNNL(cnnlGetSyncBatchNormStatsWorkspaceSize(
                cnnl_handle, cnnl_x_desc, workspace_size));

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSyncBatchNormStats(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t x_desc, const void *x,
    const float eps, const mluOpTensorDescriptor_t mean_desc, void *mean,
    const mluOpTensorDescriptor_t invstd_desc, void *invstd) {
  PARAM_CHECK("[mluOpSyncBatchNormStats]", handle != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormStats]", x_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormStats]", mean_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormStats]", invstd_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormStats]", x != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormStats]", mean != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormStats]", invstd != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(x_desc, cnnl_x_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(mean_desc, cnnl_mean_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(invstd_desc, cnnl_invstd_desc);

  CALL_CNNL(
      cnnlSyncBatchNormStats(cnnl_handle, cnnl_x_desc, x, eps, cnnl_mean_desc,
                             mean, cnnl_invstd_desc, invstd));

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_mean_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_invstd_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSyncBatchNormStats_v2(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t x_desc, const void *x,
    void *workspace, size_t workspace_size, const float eps,
    const mluOpTensorDescriptor_t mean_desc, void *mean,
    const mluOpTensorDescriptor_t invstd_desc, void *invstd) {
  PARAM_CHECK("[mluOpSyncBatchNormStats_v2]", handle != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormStats_v2]", x_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormStats_v2]", mean_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormStats_v2]", invstd_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormStats_v2]", x != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormStats_v2]", mean != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormStats_v2]", invstd != NULL);
  if (workspace_size > 0) {
    PARAM_CHECK("mluOpSyncBatchNormStats_v2", workspace != NULL);
  }

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(x_desc, cnnl_x_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(mean_desc, cnnl_mean_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(invstd_desc, cnnl_invstd_desc);

  CALL_CNNL(cnnlSyncBatchNormStats_v2(
                cnnl_handle, cnnl_x_desc, x, workspace, workspace_size,
                eps, cnnl_mean_desc, mean, cnnl_invstd_desc, invstd));

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_mean_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_invstd_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
