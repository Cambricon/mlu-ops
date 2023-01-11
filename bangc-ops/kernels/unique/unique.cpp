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

mluOpStatus_t MLUOP_WIN_API mluOpUnique(
    mluOpHandle_t handle,
    const mluOpUniqueDescriptor_t unique_desc,
    const mluOpTensorDescriptor_t input_desc,
    const void *input,
    const int output_len,
    void *unique_data,
    void *output_data,
    int *output_index,
    int *output_counts) {
  uniqueWrapper wrapper;
  mluOpStatus_t ret = wrapper.invoke(
      handle, unique_desc, input_desc, input, output_len, unique_data,
      output_data, output_index, output_counts);
  return ret;
}

mluOpStatus_t MLUOP_WIN_API mluOpUnique_v2(
    mluOpHandle_t handle,
    const mluOpUniqueDescriptor_t unique_desc,
    const mluOpTensorDescriptor_t input_desc,
    const void *input,
    void *workspace,
    const size_t workspace_size,
    int *output_num,
    const mluOpTensorDescriptor_t output_desc,
    void *output,
    const mluOpTensorDescriptor_t indices_desc,
    void *inverse_indices,
    const mluOpTensorDescriptor_t counts_desc,
    void *counts) {
  uniqueV2Wrapper wrapper;
  mluOpStatus_t ret = wrapper.invoke(
      handle, unique_desc, input_desc, input, workspace, workspace_size,
      output_num, output_desc, output, indices_desc, inverse_indices,
      counts_desc, counts);
  return ret;
}
