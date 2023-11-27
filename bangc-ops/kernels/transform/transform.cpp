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

mluOpStatus_t MLUOP_WIN_API
mluOpTransform(mluOpHandle_t handle, const mluOpPointerMode_t pointer_mode,
               const void *alpha, const mluOpTensorDescriptor_t input_desc,
               const void *input, const void *beta,
               const mluOpTensorDescriptor_t output_desc, void *output) {
  PARAM_CHECK("[mluOpTransform]", handle != NULL);
  PARAM_CHECK("[mluOpTransform]", alpha != NULL);
  PARAM_CHECK("[mluOpTransform]", input_desc != NULL);
  PARAM_CHECK("[mluOpTransform]", input != NULL);
  PARAM_CHECK("[mluOpTransform]", beta != NULL);
  PARAM_CHECK("[mluOpTransform]", output_desc != NULL);
  PARAM_CHECK("[mluOpTransform]", output != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, _input_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, _output_desc);

  CHECK_FUNC_RETURN(cnnlTransform_v2(_handle, cnnlPointerMode_t(pointer_mode),
                                     alpha, _input_desc, input, beta,
                                     _output_desc, output),
                    CNNL_STATUS_SUCCESS,
                    "[cnnlTransform_v2] Internal error accured in "
                    "cnnlTransform_v2.",
                    MLUOP_STATUS_INTERNAL_ERROR);

  DESTROY_CNNL_TENSOR_DESCRIPTOR(_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_output_desc);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}
