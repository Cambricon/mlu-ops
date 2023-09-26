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

mluOpStatus_t MLUOP_WIN_API mluOpSyncBatchNormElemt(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t x_desc, const void *x,
    const mluOpTensorDescriptor_t mean_desc, const void *mean,
    const mluOpTensorDescriptor_t invstd_desc, const void *invstd,
    const mluOpTensorDescriptor_t filter_desc, const void *filter,
    const mluOpTensorDescriptor_t bias_desc, const void *bias,
    const mluOpTensorDescriptor_t y_desc, void *y) {
  SyncBatchNormElemtWrapper wrapper;
  mluOpStatus_t ret =
      wrapper.invoke(handle, x_desc, x, mean_desc, mean, invstd_desc, invstd,
                     filter_desc, filter, bias_desc, bias, y_desc, y);
  return ret;
}

