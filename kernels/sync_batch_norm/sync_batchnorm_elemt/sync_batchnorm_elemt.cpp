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

mluOpStatus_t MLUOP_WIN_API mluOpSyncBatchNormElemt(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t x_desc, const void *x,
    const mluOpTensorDescriptor_t mean_desc, const void *mean,
    const mluOpTensorDescriptor_t invstd_desc, const void *invstd,
    const mluOpTensorDescriptor_t filter_desc, const void *filter,
    const mluOpTensorDescriptor_t bias_desc, const void *bias,
    const mluOpTensorDescriptor_t y_desc, void *y) {
  PARAM_CHECK("[mluOpSyncBatchNormElemt]", handle != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormElemt]", x_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormElemt]", mean_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormElemt]", invstd_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormElemt]",
              (filter_desc != NULL && bias_desc != NULL) ||
                  (filter_desc == NULL && bias_desc == NULL));
  PARAM_CHECK("[mluOpSyncBatchNormElemt]", y_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormElemt]", x != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormElemt]", mean != NULL);
  PARAM_CHECK(
      "[mluOpSyncBatchNormElemt]",
      (filter != NULL && bias != NULL) || (filter == NULL && bias == NULL));
  PARAM_CHECK("[mluOpSyncBatchNormElemt]", invstd != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormElemt]", y != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(x_desc, cnnl_x_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(mean_desc, cnnl_mean_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(invstd_desc, cnnl_invstd_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(filter_desc, cnnl_filter_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(bias_desc, cnnl_bias_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(y_desc, cnnl_y_desc);

  CALL_CNNL(cnnlSyncBatchNormElemt(
      cnnl_handle, cnnl_x_desc, x, cnnl_mean_desc, mean, cnnl_invstd_desc,
      invstd, cnnl_filter_desc, filter, cnnl_bias_desc, bias, cnnl_y_desc, y));

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_mean_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_invstd_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_filter_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_bias_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_y_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
