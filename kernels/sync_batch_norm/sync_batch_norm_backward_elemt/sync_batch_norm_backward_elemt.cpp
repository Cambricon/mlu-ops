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

mluOpStatus_t MLUOP_WIN_API mluOpSyncBatchNormBackwardElemt(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t diff_y_desc,
    const void *diff_y, const mluOpTensorDescriptor_t x_desc, const void *x,
    const mluOpTensorDescriptor_t mean_desc, const void *mean,
    const mluOpTensorDescriptor_t invstd_desc, const void *invstd,
    const mluOpTensorDescriptor_t filter_desc, const void *filter,
    const mluOpTensorDescriptor_t mean_dy_desc, const void *mean_dy,
    const mluOpTensorDescriptor_t mean_dy_xmu_desc, const void *mean_dy_xmu,
    const mluOpTensorDescriptor_t diffcnnl_x_desc, void *diff_x) {
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", handle != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", diff_y_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", x_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", mean_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", invstd_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", mean_dy_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", mean_dy_xmu_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", diffcnnl_x_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", diff_y != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", x != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", mean != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", invstd != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", mean_dy != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", mean_dy_xmu != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", diff_x != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(diff_y_desc, cnnl_diff_y_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(x_desc, cnnl_x_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(mean_desc, cnnl_mean_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(invstd_desc, cnnl_invstd_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(filter_desc, cnnl_filter_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(mean_dy_desc, cnnl_mean_dy_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(mean_dy_xmu_desc,
                                               cnnl_mean_dy_xmu_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(diffcnnl_x_desc,
                                               cnnl_diffcnnl_x_desc);

  CALL_CNNL(cnnlSyncBatchNormBackwardElemt(
      cnnl_handle, cnnl_diff_y_desc, diff_y, cnnl_x_desc, x, cnnl_mean_desc,
      mean, cnnl_invstd_desc, invstd, cnnl_filter_desc, filter,
      cnnl_mean_dy_desc, mean_dy, cnnl_mean_dy_xmu_desc, mean_dy_xmu,
      cnnl_diffcnnl_x_desc, diff_x));

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_diff_y_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_mean_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_invstd_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_filter_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_mean_dy_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_mean_dy_xmu_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_diffcnnl_x_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
