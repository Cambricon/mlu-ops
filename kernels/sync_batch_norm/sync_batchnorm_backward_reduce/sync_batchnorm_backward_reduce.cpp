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
#include "core/cnnl_helper.h"

mluOpStatus_t MLUOP_WIN_API mluOpGetSyncBatchNormBackwardReduceWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t desc_x,
    size_t *workspace_size) {
  PARAM_CHECK("mluOpGetSyncBatchNormBackwardReduceWorkspaceSize",
              handle != NULL);
  PARAM_CHECK("mluOpGetSyncBatchNormBackwardReduceWorkspaceSize",
              desc_x != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_x, cnnl_desc_x);

  CALL_CNNL(cnnlGetSyncBatchnormBackwardReduceWorkspaceSize(
      cnnl_handle, cnnl_desc_x, workspace_size));

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_x);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetSyncBatchnormBackwardReduceWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t desc_x,
    size_t *workspace_size) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpGetSyncBatchnormBackwardReduceWorkspaceSize] is deprecated and"
      << " will be removed in the future release, please use "
      << "[mluOpGetSyncBatchNormBackwardReduceWorkspaceSize] instead.";
  return mluOpGetSyncBatchNormBackwardReduceWorkspaceSize(handle, desc_x,
                                                          workspace_size);
}

mluOpStatus_t MLUOP_WIN_API mluOpSyncBatchNormBackwardReduce(
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
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce]", handle != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce]", desc_dz != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce]", desc_x != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce]", desc_mean != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce]", desc_invstd != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce]", dz != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce]", x != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce]", mean != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce]", invstd != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_dz, cnnl_desc_dz);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_x, cnnl_desc_x);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_mean, cnnl_desc_mean);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_invstd, cnnl_desc_invstd);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_dfilter, cnnl_desc_dfilter);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_dbias, cnnl_desc_dbias);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_sum_dy, cnnl_desc_sum_dy);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_sum_dy_xmu,
                                               cnnl_desc_sum_dy_xmu);

  CALL_CNNL(cnnlSyncBatchnormBackwardReduce(
      cnnl_handle, cnnl_desc_dz, dz, cnnl_desc_x, x, cnnl_desc_mean, mean,
      cnnl_desc_invstd, invstd, cnnl_desc_dfilter, dfilter, cnnl_desc_dbias,
      dbias, cnnl_desc_sum_dy, sum_dy, cnnl_desc_sum_dy_xmu, sum_dy_xmu,
      needs_input_grad0, needs_input_grad1, needs_input_grad2));

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_dz);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_x);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_mean);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_invstd);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_dfilter);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_dbias);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_sum_dy);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_sum_dy_xmu);
  DESTROY_CNNL_HANDLE(cnnl_handle);
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
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpSyncBatchnormBackwardReduce] is deprecated and"
      << " will be removed in the future release, please use "
      << "[mluOpSyncBatchNormBackwardReduce] instead.";
  return mluOpSyncBatchNormBackwardReduce(
      handle, desc_dz, dz, desc_x, x, desc_mean, mean, desc_invstd, invstd,
      desc_dfilter, dfilter, desc_dbias, dbias, desc_sum_dy, sum_dy,
      desc_sum_dy_xmu, sum_dy_xmu, needs_input_grad0, needs_input_grad1,
      needs_input_grad2);
}

mluOpStatus_t MLUOP_WIN_API mluOpSyncBatchNormBackwardReduce_v2(
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
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce_v2]", handle != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce_v2]", desc_dz != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce_v2]", desc_x != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce_v2]", desc_mean != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce_v2]", desc_invstd != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce_v2]", dz != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce_v2]", x != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce_v2]", mean != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardReduce_v2]", invstd != NULL);
  if (workspace_size > 0) {
    PARAM_CHECK("mluOpSyncBatchNormBackwardReduce_v2", workspace != NULL);
  }

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_dz, cnnl_desc_dz);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_x, cnnl_desc_x);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_mean, cnnl_desc_mean);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_invstd, cnnl_desc_invstd);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_dfilter, cnnl_desc_dfilter);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_dbias, cnnl_desc_dbias);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_sum_dy, cnnl_desc_sum_dy);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_sum_dy_xmu,
                                               cnnl_desc_sum_dy_xmu);

  CALL_CNNL(cnnlSyncBatchnormBackwardReduce_v2(
      cnnl_handle, cnnl_desc_dz, dz, cnnl_desc_x, x, cnnl_desc_mean, mean,
      cnnl_desc_invstd, invstd, workspace, workspace_size, cnnl_desc_dfilter,
      dfilter, cnnl_desc_dbias, dbias, cnnl_desc_sum_dy, sum_dy,
      cnnl_desc_sum_dy_xmu, sum_dy_xmu, needs_input_grad0, needs_input_grad1,
      needs_input_grad2));

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_dz);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_x);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_mean);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_invstd);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_dfilter);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_dbias);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_sum_dy);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_sum_dy_xmu);
  DESTROY_CNNL_HANDLE(cnnl_handle);
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
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpSyncBatchnormBackwardReduce_v2] is deprecated and"
      << " will be removed in the future release, please use "
      << "[mluOpSyncBatchNormBackwardReduce_v2] instead.";
  return mluOpSyncBatchNormBackwardReduce_v2(
      handle, desc_dz, dz, desc_x, x, desc_mean, mean, desc_invstd, invstd,
      workspace, workspace_size, desc_dfilter, dfilter, desc_dbias, dbias,
      desc_sum_dy, sum_dy, desc_sum_dy_xmu, sum_dy_xmu, needs_input_grad0,
      needs_input_grad1, needs_input_grad2);
}
