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
  SyncBatchnormBackwardReduceWrapper wrapper;
  mluOpStatus_t ret =
      wrapper.invoke(handle, desc_dz, dz, desc_x, x, desc_mean, mean,
                     desc_invstd, invstd, desc_dfilter, dfilter, desc_dbias,
                     dbias, desc_sum_dy, sum_dy, desc_sum_dy_xmu, sum_dy_xmu,
                     needs_input_grad0, needs_input_grad1, needs_input_grad2);
  return ret;
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
  SyncBatchnormBackwardReduceV2Wrapper wrapper;
  mluOpStatus_t ret = wrapper.invoke(
      handle, desc_dz, dz, desc_x, x, desc_mean, mean, desc_invstd, invstd,
      workspace, workspace_size, desc_dfilter, dfilter, desc_dbias, dbias,
      desc_sum_dy, sum_dy, desc_sum_dy_xmu, sum_dy_xmu, needs_input_grad0,
      needs_input_grad1, needs_input_grad2);
  return ret;
}

