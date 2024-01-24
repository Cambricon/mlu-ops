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

mluOpStatus_t MLUOP_WIN_API mluOpSyncBatchNormGatherStatsWithCounts(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t mean_all_desc,
    const void *mean_all, const mluOpTensorDescriptor_t invstd_all_desc,
    const void *invstd_all, const mluOpTensorDescriptor_t movingcnnl_mean_desc,
    void *moving_mean, const mluOpTensorDescriptor_t moving_var_desc,
    void *moving_var, float momentum, float eps,
    const mluOpTensorDescriptor_t count_all_desc, const void *count_all,
    const mluOpTensorDescriptor_t mean_desc, void *mean,
    const mluOpTensorDescriptor_t invstd_desc, void *invstd) {
  PARAM_CHECK("[mluOpSyncBatchNormGatherStatsWithCounts]", handle != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormGatherStatsWithCounts]",
              mean_all_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormGatherStatsWithCounts]",
              invstd_all_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormGatherStatsWithCounts]",
              count_all_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormGatherStatsWithCounts]", mean_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormGatherStatsWithCounts]", invstd_desc != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormGatherStatsWithCounts]",
              (movingcnnl_mean_desc != NULL && moving_var_desc != NULL) ||
                  (movingcnnl_mean_desc == NULL && moving_var_desc == NULL));
  PARAM_CHECK("[mluOpSyncBatchNormGatherStatsWithCounts]", mean_all != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormGatherStatsWithCounts]", invstd_all != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormGatherStatsWithCounts]",
              (moving_mean != NULL && moving_var != NULL) ||
                  (moving_mean == NULL && moving_var == NULL));
  PARAM_CHECK("[mluOpSyncBatchNormGatherStatsWithCounts]", count_all != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormGatherStatsWithCounts]", mean != NULL);
  PARAM_CHECK("[mluOpSyncBatchNormGatherStatsWithCounts]", invstd != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(mean_all_desc,
                                               cnnl_mean_all_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(invstd_all_desc,
                                               cnnl_invstd_all_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(movingcnnl_mean_desc,
                                               cnnl_movingcnnl_mean_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(moving_var_desc,
                                               cnnl_moving_var_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(count_all_desc,
                                               cnnl_count_all_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(mean_desc, cnnl_mean_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(invstd_desc, cnnl_invstd_desc);

  CHECK_FUNC_RETURN(
      cnnlSyncBatchNormGatherStatsWithCounts(
          cnnl_handle, cnnl_mean_all_desc, mean_all, cnnl_invstd_all_desc,
          invstd_all, cnnl_movingcnnl_mean_desc, moving_mean,
          cnnl_moving_var_desc, moving_var, momentum, eps, cnnl_count_all_desc,
          count_all, cnnl_mean_desc, mean, cnnl_invstd_desc, invstd),
      CNNL_STATUS_SUCCESS,
      "[mluOpSyncBatchNormGatherStatsWithCounts] Internal error"
      " accured in mluOpSyncBatchNormGatherStatsWithCounts.",
      MLUOP_STATUS_INTERNAL_ERROR);

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_mean_all_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_invstd_all_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_movingcnnl_mean_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_moving_var_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_count_all_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_mean_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_invstd_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
