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

mluOpStatus_t MLUOP_WIN_API mluOpPad(mluOpHandle_t handle,
                                     const mluOpTensorDescriptor_t input_desc,
                                     const void *input, const void *paddings,
                                     const void *padding_value,
                                     const mluOpTensorDescriptor_t output_desc,
                                     void *output) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpPad] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpPad", handle != NULL);
  PARAM_CHECK("mluOpPad", input_desc != NULL);
  PARAM_CHECK("mluOpPad", paddings != NULL);
  PARAM_CHECK("mluOpPad", padding_value != NULL);
  PARAM_CHECK("mluOpPad", output_desc != NULL);
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, cnnl_output_desc);
  CHECK_FUNC_RETURN(cnnlPad(cnnl_handle, cnnl_input_desc, input, paddings,
                            padding_value, cnnl_output_desc, output),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpPad] Internal error accured in mluOpPad.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
