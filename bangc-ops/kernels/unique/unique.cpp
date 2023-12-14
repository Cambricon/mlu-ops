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

mluOpStatus_t MLUOP_WIN_API
mluOpCreateUniqueDescriptor(mluOpUniqueDescriptor_t *unique_desc) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpCreateUniqueDescriptor] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpUnique", unique_desc != NULL);
  CHECK_FUNC_RETURN(
      cnnlCreateUniqueDescriptor(unique_desc), CNNL_STATUS_SUCCESS,
      "[mluOpUnique] Internal error accured in mluOpCreateUniqueDescriptor.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpDestroyUniqueDescriptor(mluOpUniqueDescriptor_t unique_desc) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpDestroyUniqueDescriptor] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpUnique", unique_desc != NULL);
  CHECK_FUNC_RETURN(
      cnnlDestroyUniqueDescriptor(unique_desc), CNNL_STATUS_SUCCESS,
      "[mluOpUnique] Internal error accured in mluOpDestroyUniqueDescriptor.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetUniqueDescriptor(
    mluOpUniqueDescriptor_t unique_desc, mluOpUniqueSort_t mode, int dim,
    bool return_inverse, bool return_counts) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpSetUniqueDescriptor] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpUnique", unique_desc != NULL);
  CHECK_FUNC_RETURN(
      cnnlSetUniqueDescriptor(unique_desc, cnnlUniqueSort_t(mode), dim,
                              return_inverse, return_counts),
      CNNL_STATUS_SUCCESS,
      "[mluOpUnique] Internal error accured in mluOpSetUniqueDescriptor.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetUniqueWorkSpace(
    mluOpHandle_t handle, const mluOpUniqueDescriptor_t unique_desc,
    const mluOpTensorDescriptor_t input_desc, size_t *size) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpGetUniqueWorkSpace] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpUnique", handle != NULL);
  PARAM_CHECK("mluOpUnique", unique_desc != NULL);
  PARAM_CHECK("mluOpUnique", input_desc != NULL);
  PARAM_CHECK("mluOpUnique", size != NULL);
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
  CHECK_FUNC_RETURN(
      cnnlGetUniqueWorkSpace(cnnl_handle, unique_desc, cnnl_input_desc, size),
      CNNL_STATUS_SUCCESS,
      "[mluOpUnique] Internal error accured in mluOpGetUniqueWorkSpace.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpUniqueGetOutLen(
    mluOpHandle_t handle, const mluOpUniqueDescriptor_t unique_desc,
    const mluOpTensorDescriptor_t input_desc, const void *input,
    void *unique_data, int *output_len) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpUniqueGetOutLen] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpUnique", handle != NULL);
  PARAM_CHECK("mluOpUnique", unique_desc != NULL);
  PARAM_CHECK("mluOpUnique", input_desc != NULL);
  PARAM_CHECK("mluOpUnique", input != NULL);
  PARAM_CHECK("mluOpUnique", unique_data != NULL);
  PARAM_CHECK("mluOpUnique", output_len != NULL);
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
  CHECK_FUNC_RETURN(
      cnnlUniqueGetOutLen(cnnl_handle, unique_desc, cnnl_input_desc, input,
                          unique_data, output_len),
      CNNL_STATUS_SUCCESS,
      "[mluOpUnique] Internal error accured in mluOpUniqueGetOutLen.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetUniqueWorkspaceSize(
    mluOpHandle_t handle, const mluOpUniqueDescriptor_t unique_desc,
    const mluOpTensorDescriptor_t input_desc, size_t *workspace_size) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpGetUniqueWorkspaceSize] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpUnique", handle != NULL);
  PARAM_CHECK("mluOpUnique", unique_desc != NULL);
  PARAM_CHECK("mluOpUnique", input_desc != NULL);
  PARAM_CHECK("mluOpUnique", workspace_size != NULL);
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
  CHECK_FUNC_RETURN(
      cnnlGetUniqueWorkspaceSize(cnnl_handle, unique_desc, cnnl_input_desc,
                                 workspace_size),
      CNNL_STATUS_SUCCESS,
      "[mluOpUnique] Internal error accured in mluOpGetUniqueWorkspaceSize.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpUnique(mluOpHandle_t handle, const mluOpUniqueDescriptor_t unique_desc,
            const mluOpTensorDescriptor_t input_desc, const void *input,
            const int output_len, void *unique_data, void *output_data,
            int *output_index, int *output_counts) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpUnique] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpUnique", handle != NULL);
  PARAM_CHECK("mluOpUnique", unique_desc != NULL);
  PARAM_CHECK("mluOpUnique", input_desc != NULL);
  PARAM_CHECK("mluOpUnique", input != NULL);
  PARAM_CHECK("mluOpUnique", unique_data != NULL);
  PARAM_CHECK("mluOpUnique", output_data != NULL);
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
  CHECK_FUNC_RETURN(
      cnnlUnique(cnnl_handle, unique_desc, cnnl_input_desc, input, output_len,
                 unique_data, output_data, output_index, output_counts),
      CNNL_STATUS_SUCCESS,
      "[mluOpUnique] Internal error accured in mluOpUnique.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpUnique_v2(
    mluOpHandle_t handle, const mluOpUniqueDescriptor_t unique_desc,
    const mluOpTensorDescriptor_t input_desc, const void *input,
    void *workspace, const size_t workspace_size, int *output_num,
    const mluOpTensorDescriptor_t output_desc, void *output,
    const mluOpTensorDescriptor_t indices_desc, void *inverse_indices,
    const mluOpTensorDescriptor_t counts_desc, void *counts) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpUnique_v2] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpUnique_v2", handle != NULL);
  PARAM_CHECK("mluOpUnique_v2", unique_desc != NULL);
  PARAM_CHECK("mluOpUnique_v2", input_desc != NULL);
  PARAM_CHECK("mluOpUnique_v2", input != NULL);
  if (workspace_size != 0) {
    PARAM_CHECK("mluOpUnique_v2", workspace != NULL);
  }
  PARAM_CHECK("mluOpUnique_v2", output_num != NULL);
  PARAM_CHECK("mluOpUnique_v2", output_desc != NULL);
  PARAM_CHECK("mluOpUnique_v2", output != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, cnnl_output_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(indices_desc, cnnl_indices_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(counts_desc, cnnl_counts_desc);

  CHECK_FUNC_RETURN(cnnlUnique_v2(cnnl_handle, unique_desc, cnnl_input_desc,
                                  input, workspace, workspace_size, output_num,
                                  cnnl_output_desc, output, cnnl_indices_desc,
                                  inverse_indices, cnnl_counts_desc, counts),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpUnique] Internal error accured in mluOpUnique.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_indices_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_counts_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
