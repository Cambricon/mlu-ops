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

mluOpStatus_t MLUOP_WIN_API mluOpSyncBatchNormBackwardElemt(
    mluOpHandle_t handle,
    const mluOpTensorDescriptor_t diff_y_desc,
    const void *diff_y,
    const mluOpTensorDescriptor_t x_desc,
    const void *x,
    const mluOpTensorDescriptor_t mean_desc,
    const void *mean,
    const mluOpTensorDescriptor_t invstd_desc,
    const void *invstd,
    const mluOpTensorDescriptor_t filter_desc,
    const void *filter,
    const mluOpTensorDescriptor_t mean_dy_desc,
    const void *mean_dy,
    const mluOpTensorDescriptor_t mean_dy_xmu_desc,
    const void *mean_dy_xmu,
    const mluOpTensorDescriptor_t diff_x_desc,
    void *diff_x) {
  SyncBatchNormBackwardElemtWrapper wrapper;
  mluOpStatus_t ret = wrapper.invoke(handle, diff_y_desc, diff_y, x_desc,
        x, mean_desc, mean, invstd_desc, invstd, filter_desc, filter,
        mean_dy_desc, mean_dy, mean_dy_xmu_desc, mean_dy_xmu, diff_x_desc,
        diff_x);
  return ret;
}
