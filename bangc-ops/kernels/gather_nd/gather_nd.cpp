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
mluOpGatherNd(mluOpHandle_t handle, const mluOpTensorDescriptor_t desc_params,
              const void *params, const mluOpTensorDescriptor_t desc_indices,
              const void *indices, const mluOpTensorDescriptor_t desc_output,
              void *output) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpGatherNd] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpGatherNd", handle != NULL);
  PARAM_CHECK("mluOpGatherNd", desc_params != NULL);
  PARAM_CHECK("mluOpGatherNd", params != NULL);
  PARAM_CHECK("mluOpGatherNd", desc_indices != NULL);
  PARAM_CHECK("mluOpGatherNd", indices != NULL);
  PARAM_CHECK("mluOpGatherNd", desc_output != NULL);
  PARAM_CHECK("mluOpGatherNd", output != NULL);
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_params, cnnl_desc_params);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_indices, cnnl_desc_indices);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc_output, cnnl_desc_output);
  CHECK_FUNC_RETURN(
      cnnlGatherNd(cnnl_handle, cnnl_desc_params, params, cnnl_desc_indices,
                   indices, cnnl_desc_output, output),
      CNNL_STATUS_SUCCESS,
      "[mluOpGatherNd] Internal error accured in mluOpGatherNd.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_params);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_indices);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_desc_output);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
