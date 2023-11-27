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
    const mluOpTensorDescriptor_t diff_x_desc, void *diff_x) {
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", handle != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", diff_y_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", x_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", mean_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", invstd_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", mean_dy_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", mean_dy_xmu_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", diff_x_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", diff_y != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", x != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", mean != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", invstd != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", mean_dy != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", mean_dy_xmu != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormBackwardElemt]", diff_x != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(diff_y_desc, _diff_y_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(x_desc, _x_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(mean_desc, _mean_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(invstd_desc, _invstd_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(filter_desc, _filter_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(mean_dy_desc, _mean_dy_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(mean_dy_xmu_desc,
                                               _mean_dy_xmu_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(diff_x_desc, _diff_x_desc);

  CHECK_FUNC_RETURN(
      cnnlSyncBatchNormBackwardElemt(
          _handle, _diff_y_desc, diff_y, _x_desc, x, _mean_desc, mean,
          _invstd_desc, invstd, _filter_desc, filter, _mean_dy_desc, mean_dy,
          _mean_dy_xmu_desc, mean_dy_xmu, _diff_x_desc, diff_x),
      CNNL_STATUS_SUCCESS,
      "[mluOpSyncBatchNormBackwardElemt] Internal error"
      " accured in mluOpSyncBatchNormBackwardElemt.",
      MLUOP_STATUS_INTERNAL_ERROR);

  DESTROY_CNNL_TENSOR_DESCRIPTOR(_diff_y_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_x_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_mean_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_invstd_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_filter_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_mean_dy_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_mean_dy_xmu_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_diff_x_desc);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}
