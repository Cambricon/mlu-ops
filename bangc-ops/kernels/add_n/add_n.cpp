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
#include "kernels/utils/cnnl_helper.h"

mluOpStatus_t MLUOP_WIN_API mluOpGetAddNWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_descs[],
    const uint32_t input_num, const mluOpTensorDescriptor_t output_desc,
    size_t *workspace_size) {
  PARAM_CHECK("mluOpAddN_v2", handle != NULL);
  PARAM_CHECK("mluOpAddN_v2", input_descs != NULL);
  PARAM_CHECK("mluOpAddN_v2", output_desc != NULL);
  PARAM_CHECK("mluOpAddN_v2", input_num >= 2);
  CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  cnnlTensorDescriptor_t *_input_descs = (cnnlTensorDescriptor_t *)malloc(
      sizeof(cnnlTensorDescriptor_t) * input_num);
  for (int i = 0; i < input_num; i++) {
    CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR_V2(input_descs[i], _input_descs[i]);
  }
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, _output_desc);
  CHECK_FUNC_RETURN(
      cnnlGetAddNWorkspaceSize(_handle, _input_descs, input_num,
                               _output_desc, workspace_size),
      CNNL_STATUS_SUCCESS,
      "[mluOpAddN_v2] Internal error accured in mluOpGetAddNWorkspaceSize.",
      MLUOP_STATUS_INTERNAL_ERROR);
  for (int i = 0; i < input_num; i++) {
    DESTROY_CNNL_TENSOR_DESCRIPTOR(_input_descs[i]);
  }
  free(_input_descs);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_output_desc);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpAddN(mluOpHandle_t handle, const mluOpTensorDescriptor_t input_descs[],
          const void *inputs[], uint32_t input_num,
          const mluOpTensorDescriptor_t output_desc, void *output) {
  PARAM_CHECK("mluOpAddN", handle != NULL);
  PARAM_CHECK("mluOpAddN", input_descs != NULL);
  PARAM_CHECK("mluOpAddN", output_desc != NULL);
  PARAM_CHECK("mluOpAddN", input_num >= 2);
  CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  cnnlTensorDescriptor_t *_input_descs = (cnnlTensorDescriptor_t *)malloc(
      sizeof(cnnlTensorDescriptor_t) * input_num);
  for (int i = 0; i < input_num; i++) {
    CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR_V2(input_descs[i], _input_descs[i]);
  }
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, _output_desc);
  CHECK_FUNC_RETURN(
      cnnlAddN(_handle, _input_descs, inputs, input_num, _output_desc, output),
      CNNL_STATUS_SUCCESS,
      "[mluOpAddN] Internal error accured in mluOpAddN.",  // NOLINT
      MLUOP_STATUS_INTERNAL_ERROR);
  for (int i = 0; i < input_num; i++) {
    DESTROY_CNNL_TENSOR_DESCRIPTOR(_input_descs[i]);
  }
  free(_input_descs);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_output_desc);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpAddN_v2(mluOpHandle_t handle, const mluOpTensorDescriptor_t input_descs[],
             const void *const inputs[], const uint32_t input_num,
             const mluOpTensorDescriptor_t output_desc, void *output,
             void *workspace, size_t workspace_size) {
  PARAM_CHECK("mluOpAddN_v2", handle != NULL);
  PARAM_CHECK("mluOpAddN_v2", input_descs != NULL);
  PARAM_CHECK("mluOpAddN_v2", output_desc != NULL);
  PARAM_CHECK("mluOpAddN_v2", input_num >= 2);
  CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  cnnlTensorDescriptor_t *_input_descs = (cnnlTensorDescriptor_t *)malloc(
      sizeof(cnnlTensorDescriptor_t) * input_num);
  for (int i = 0; i < input_num; i++) {
    CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR_V2(input_descs[i], _input_descs[i]);
  }
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, _output_desc);
  CHECK_FUNC_RETURN(
      cnnlAddN_v2(_handle, _input_descs, inputs, input_num, _output_desc,
                  output, workspace, workspace_size),
      CNNL_STATUS_SUCCESS,
      "[mluOpAddN_v2] Internal error accured in mluOpAddN_v2.",  // NOLINT
      MLUOP_STATUS_INTERNAL_ERROR);
  for (int i = 0; i < input_num; i++) {
    DESTROY_CNNL_TENSOR_DESCRIPTOR(_input_descs[i]);
  }
  free(_input_descs);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_output_desc);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}
