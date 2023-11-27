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

mluOpStatus_t MLUOP_WIN_API mluOpStridedSlice(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, const int *begin, const int *end, const int *stride,
    const mluOpTensorDescriptor_t output_desc, void *output) {
  PARAM_CHECK("mluOpStridedSlice", handle != NULL);
  PARAM_CHECK("mluOpStridedSlice", input_desc != NULL);
  PARAM_CHECK("mluOpStridedSlice", input != NULL);
  PARAM_CHECK("mluOpStridedSlice", begin != NULL);
  PARAM_CHECK("mluOpStridedSlice", end != NULL);
  PARAM_CHECK("mluOpStridedSlice", stride != NULL);
  PARAM_CHECK("mluOpStridedSlice", output_desc != NULL);
  PARAM_CHECK("mluOpstridedSlice", output != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, _input_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, _output_desc);

  CHECK_FUNC_RETURN(cnnlStridedSlice(_handle, _input_desc, input, begin, end,
                                     stride, _output_desc, output),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpStridedSlice] Internal error"
                    " accured in mluOpStridedSlice.",
                    MLUOP_STATUS_INTERNAL_ERROR);

  DESTROY_CNNL_TENSOR_DESCRIPTOR(_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_output_desc);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}
