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

mluOpStatus_t MLUOP_WIN_API mluOpGetConcatWorkspaceSize(mluOpHandle_t handle,
                                                        const int concat_num,
                                                        size_t *size) {
  PARAM_CHECK("mluOpConcat", handle != NULL);
  PARAM_CHECK("mluOpConcat", concat_num > 0);
  PARAM_CHECK("mluOpConcat", size != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, _handle);

  CHECK_FUNC_RETURN(cnnlGetConcatWorkspaceSize(_handle, concat_num, size),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpConcat] Internal error accured in "
                    "mluOpGetConcatWorkspaceSize.",
                    MLUOP_STATUS_INTERNAL_ERROR);

  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpConcat(mluOpHandle_t handle, const int concat_num, const int axis,
            const mluOpTensorDescriptor_t inputs_desc[],
            const void *const inputs[], void *workspace, size_t workspace_size,
            const mluOpTensorDescriptor_t output_desc, void *output) {
  PARAM_CHECK("mluOpConcat", handle != NULL);
  PARAM_CHECK("mluOpConcat", concat_num > 0);
  PARAM_CHECK("mluOpConcat", inputs_desc != NULL);
  PARAM_CHECK("mluOpConcat", inputs != NULL);
  PARAM_CHECK("mluOpConcat", output_desc != NULL);
  PARAM_CHECK("mluOpConcat", output != NULL);
  if (workspace_size > 0) {
    PARAM_CHECK("mluOpConcat", workspace != NULL);
  }

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  cnnlTensorDescriptor_t *_inputs_desc = (cnnlTensorDescriptor_t *)malloc(
      sizeof(cnnlTensorDescriptor_t) * concat_num);
  for (int i = 0; i < concat_num; i++) {
    CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(inputs_desc[i], _inputs_desc[i]);
  }
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, _output_desc);

  CHECK_FUNC_RETURN(cnnlConcat(_handle, concat_num, axis, _inputs_desc, inputs,
                               workspace, workspace_size, _output_desc, output),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpConcat] Internal error accured in mluOpConcat.",
                    MLUOP_STATUS_INTERNAL_ERROR);

  for (int i = 0; i < concat_num; i++) {
    DESTROY_CNNL_TENSOR_DESCRIPTOR(_inputs_desc[i]);
  }
  free(_inputs_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_output_desc);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}
