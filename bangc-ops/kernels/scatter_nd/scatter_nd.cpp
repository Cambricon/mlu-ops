/*************************************************************************
 * Copyright (C) [2022] by Cambricon, Inc.
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

mluOpStatus_t MLUOP_WIN_API mluOpScatterNd(
    mluOpHandle_t handle,
    const mluOpTensorDescriptor_t indices_desc,
    const void *indices,
    const mluOpTensorDescriptor_t updates_desc,
    const void *updates,
    const mluOpTensorDescriptor_t output_desc,
    void *output) {
  scatterNdWrapper wrapper;
  mluOpStatus_t ret = wrapper.invoke(handle, indices_desc, indices,
                                     updates_desc, updates, output_desc,
                                     output);
  return ret;
}

mluOpStatus_t MLUOP_WIN_API mluOpScatterNd_v2(
    mluOpHandle_t handle,
    mluOpScatterNdMode_t mode,
    const mluOpTensorDescriptor_t indices_desc,
    const void *indices,
    const mluOpTensorDescriptor_t updates_desc,
    const void *updates,
    const mluOpTensorDescriptor_t input_desc,
    const void *input,
    const mluOpTensorDescriptor_t output_desc,
    void *output) {
  scatterNdV2Wrapper wrapper;
  mluOpStatus_t ret = wrapper.invoke(handle, mode, indices_desc, indices,
                                     updates_desc, updates, input_desc, input,
                                     output_desc, output);
  return ret;
}
