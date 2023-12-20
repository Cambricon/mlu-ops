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
#include <math.h>
#include <limits.h>
#include <cstdio>

#include <vector>

#include "kernels/utils/cnnl_helper.h"

#define DCNBPDATA_API "mluOpDcnBackwardData"

mluOpStatus_t MLUOP_WIN_API
mluOpCreateDCNDescriptor(mluOpDCNDescriptor_t *dcn_desc) {
  PARAM_CHECK(DCNBPDATA_API, dcn_desc != NULL);
  CHECK_FUNC_RETURN(cnnlCreateDCNDescriptor(dcn_desc), CNNL_STATUS_SUCCESS,
                    "[mluOpDcnBackwardData] Internal error accured in "
                    "mluOpCreateDCNDescriptor.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpDestroyDCNDescriptor(mluOpDCNDescriptor_t dcn_desc) {
  PARAM_CHECK(DCNBPDATA_API, dcn_desc != NULL);
  CHECK_FUNC_RETURN(cnnlDestroyDCNDescriptor(dcn_desc), CNNL_STATUS_SUCCESS,
                    "[mluOpDcnBackwardData] Internal error accured in "
                    "mluOpDestroyDCNDescriptor.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetDCNDescriptor(
    mluOpDCNDescriptor_t dcn_desc, int dimNb, const int pad[],
    const int stride[], const int dilation[], int deformable_group,
    int conv_group, int im2col_step, const mluOpDataType_t compute_type) {
  PARAM_CHECK(DCNBPDATA_API, dcn_desc != NULL);
  CHECK_FUNC_RETURN(
      cnnlSetDCNDescriptor(dcn_desc, dimNb, pad, stride, dilation,
                           deformable_group, conv_group, im2col_step,
                           cnnlDataType_t(compute_type)),
      CNNL_STATUS_SUCCESS,
      "[mluOpDcnBackwardData] Internal error accured in mluOpSetDCNDescriptor.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetDCNBakcwardDataWorkspaceSize(
    mluOpHandle_t handle, const mluOpDCNDescriptor_t dcn_desc,
    const mluOpTensorDescriptor_t input_desc,
    const mluOpTensorDescriptor_t offset_desc,
    const mluOpTensorDescriptor_t mask_desc,
    const mluOpTensorDescriptor_t filter_desc,
    const mluOpTensorDescriptor_t grad_output_desc,
    const mluOpTensorDescriptor_t grad_input_desc,
    const mluOpTensorDescriptor_t grad_offset_desc,
    const mluOpTensorDescriptor_t grad_mask_desc, size_t *workspace_size) {
  PARAM_CHECK(DCNBPDATA_API, handle != NULL);
  PARAM_CHECK(DCNBPDATA_API, dcn_desc != NULL);
  PARAM_CHECK(DCNBPDATA_API, input_desc != NULL);
  PARAM_CHECK(DCNBPDATA_API, offset_desc != NULL);
  PARAM_CHECK(DCNBPDATA_API, filter_desc != NULL);
  PARAM_CHECK(DCNBPDATA_API, dcn_desc != NULL);
  PARAM_CHECK(DCNBPDATA_API, grad_output_desc != NULL);
  PARAM_CHECK(DCNBPDATA_API, grad_input_desc != NULL);
  PARAM_CHECK(DCNBPDATA_API, grad_offset_desc != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(offset_desc, cnnl_offset_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(mask_desc, cnnl_mask_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(filter_desc, cnnl_filter_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grad_output_desc,
                                               cnnl_grad_output_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grad_input_desc,
                                               cnnl_grad_input_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grad_offset_desc,
                                               cnnl_grad_offset_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grad_mask_desc,
                                               cnnl_grad_mask_desc);

  CHECK_FUNC_RETURN(
      cnnlGetDCNBakcwardDataWorkspaceSize(
          cnnl_handle, dcn_desc, cnnl_input_desc, cnnl_offset_desc,
          cnnl_mask_desc, cnnl_filter_desc, cnnl_grad_output_desc,
          cnnl_grad_input_desc, cnnl_grad_offset_desc, cnnl_grad_mask_desc,
          workspace_size),
      CNNL_STATUS_SUCCESS,
      "[mluOpDcnBackwardData] Internal error accured in mluOpReduce.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_offset_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_mask_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_filter_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_grad_output_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_grad_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_grad_offset_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_grad_mask_desc);

  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpDCNBackwardData(
    mluOpHandle_t handle, const mluOpDCNDescriptor_t dcn_desc,
    const mluOpTensorDescriptor_t input_desc, const void *input,
    const mluOpTensorDescriptor_t offset_desc, const void *offset,
    const mluOpTensorDescriptor_t mask_desc, const void *mask,
    const mluOpTensorDescriptor_t filter_desc, const void *filter,
    const mluOpTensorDescriptor_t grad_output_desc, const void *grad_output,
    void *workspace, const size_t workspace_size,
    const mluOpTensorDescriptor_t grad_input_desc, void *grad_input,
    const mluOpTensorDescriptor_t grad_offset_desc, void *grad_offset,
    const mluOpTensorDescriptor_t grad_mask_desc, void *grad_mask) {
  PARAM_CHECK(DCNBPDATA_API, handle != NULL);
  if (workspace_size > 0) {
    PARAM_CHECK(DCNBPDATA_API, workspace != NULL);
  }
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(offset_desc, cnnl_offset_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(mask_desc, cnnl_mask_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(filter_desc, cnnl_filter_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grad_output_desc,
                                               cnnl_grad_output_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grad_input_desc,
                                               cnnl_grad_input_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grad_offset_desc,
                                               cnnl_grad_offset_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grad_mask_desc,
                                               cnnl_grad_mask_desc);
  CHECK_FUNC_RETURN(
      cnnlDCNBackwardData(
          cnnl_handle, dcn_desc, cnnl_input_desc, input, cnnl_offset_desc,
          offset, cnnl_mask_desc, mask, cnnl_filter_desc, filter,
          cnnl_grad_output_desc, grad_output, workspace, workspace_size,
          cnnl_grad_input_desc, grad_input, cnnl_grad_offset_desc, grad_offset,
          cnnl_grad_mask_desc, grad_mask),
      CNNL_STATUS_SUCCESS,
      "[mluOpDcnBackwardData] Internal error accured in mluOpDcnBackwardData.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_offset_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_mask_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_filter_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_grad_output_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_grad_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_grad_offset_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_grad_mask_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
