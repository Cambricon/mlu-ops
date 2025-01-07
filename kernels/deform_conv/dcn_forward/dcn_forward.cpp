/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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
#include "core/cnnl_helper.h"

#define DCNFORWARD_API "mluOpDCNForward"

mluOpStatus_t MLUOP_WIN_API mluOpGetDCNForwardWorkspaceSize(
    mluOpHandle_t handle, const mluOpDCNDescriptor_t dcn_desc,
    const mluOpTensorDescriptor_t input_desc,
    const mluOpTensorDescriptor_t offset_desc,
    const mluOpTensorDescriptor_t mask_desc,
    const mluOpTensorDescriptor_t filter_desc,
    const mluOpTensorDescriptor_t bias_desc,
    const mluOpTensorDescriptor_t output_desc, size_t *size) {
  PARAM_CHECK("mluOpDCNForward", handle != NULL);
  PARAM_CHECK("mluOpDCNForward", dcn_desc != NULL);
  PARAM_CHECK("mluOpDCNForward", input_desc != NULL);
  PARAM_CHECK("mluOpDCNForward", offset_desc != NULL);
  PARAM_CHECK("mluOpDCNForward", filter_desc != NULL);
  PARAM_CHECK("mluOpDCNForward", output_desc != NULL);
  PARAM_CHECK("mluOpDCNForward", size != NULL);
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(offset_desc, cnnl_offset_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(mask_desc, cnnl_mask_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(filter_desc, cnnl_filter_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(bias_desc, cnnl_bias_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, cnnl_output_desc);
  CALL_CNNL(cnnlGetDCNForwardWorkspaceSize(
      cnnl_handle, dcn_desc, cnnl_input_desc, cnnl_offset_desc, cnnl_mask_desc,
      cnnl_filter_desc, cnnl_bias_desc, cnnl_output_desc, size));
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_offset_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_mask_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_filter_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_bias_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpDCNForward(mluOpHandle_t handle, const mluOpDCNDescriptor_t dcn_desc,
                const mluOpTensorDescriptor_t input_desc, const void *input,
                const mluOpTensorDescriptor_t offset_desc, const void *offset,
                const mluOpTensorDescriptor_t mask_desc, const void *mask,
                const mluOpTensorDescriptor_t filter_desc, const void *filter,
                const mluOpTensorDescriptor_t bias_desc, const void *bias,
                void *workspace, size_t workspace_size,
                const mluOpTensorDescriptor_t output_desc, void *output) {
  PARAM_CHECK(DCNFORWARD_API, handle != NULL);
  if (workspace_size > 0) {
    PARAM_CHECK(DCNFORWARD_API, workspace != NULL);
  }
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(offset_desc, cnnl_offset_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(mask_desc, cnnl_mask_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(filter_desc, cnnl_filter_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(bias_desc, cnnl_bias_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, cnnl_output_desc);
  CALL_CNNL(cnnlDCNForward(
      cnnl_handle, dcn_desc, cnnl_input_desc, input, cnnl_offset_desc, offset,
      cnnl_mask_desc, mask, cnnl_filter_desc, filter, cnnl_bias_desc, bias,
      workspace, workspace_size, cnnl_output_desc, output));
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_offset_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_mask_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_filter_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_bias_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
