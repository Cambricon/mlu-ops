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
#include "kernels/kernel_wrapper/wrapper.h"

mluOpStatus_t MLUOP_WIN_API mluOpSyncBatchNormGatherStatsWithCounts(
    mluOpHandle_t handle,
    const mluOpTensorDescriptor_t mean_all_desc,
    const void *mean_all,
    const mluOpTensorDescriptor_t invstd_all_desc,
    const void *invstd_all,
    const mluOpTensorDescriptor_t moving_mean_desc,
    void *moving_mean,
    const mluOpTensorDescriptor_t moving_var_desc,
    void *moving_var,
    float momentum,
    float eps,
    const mluOpTensorDescriptor_t count_all_desc,
    const void *count_all,
    const mluOpTensorDescriptor_t mean_desc,
    void *mean,
    const mluOpTensorDescriptor_t invstd_desc,
    void *invstd) {
  SyncBatchNormGatherStatsWithCountsWrapper wrapper;
  mluOpStatus_t ret = wrapper.invoke(handle, mean_all_desc, mean_all,
        invstd_all_desc, invstd_all, moving_mean_desc, moving_mean,
        moving_var_desc, moving_var, momentum, eps, count_all_desc,
        count_all, mean_desc, mean, invstd_desc, invstd);
  return ret;
}

