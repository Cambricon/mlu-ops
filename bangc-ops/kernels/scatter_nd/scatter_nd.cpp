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
mluOpScatterNd(mluOpHandle_t handle, const mluOpTensorDescriptor_t indices_desc,
               const void *indices, const mluOpTensorDescriptor_t updates_desc,
               const void *updates, const mluOpTensorDescriptor_t output_desc,
               void *output) {
  PARAM_CHECK("mluOpScatterNd", handle != NULL);
  PARAM_CHECK("mluOpScatterNd", indices_desc != NULL);
  PARAM_CHECK("mluOpScatterNd", indices != NULL);
  PARAM_CHECK("mluOpScatterNd", updates_desc != NULL);
  PARAM_CHECK("mluOpScatterNd", updates != NULL);
  PARAM_CHECK("mluOpScatterNd", output_desc != NULL);
  PARAM_CHECK("mluOpScatterNd", output != NULL);
  CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(indices_desc, _indices_desc);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(updates_desc, _updates_desc);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, _output_desc);

  CHECK_FUNC_RETURN(
      cnnlScatterNd(_handle, _indices_desc, indices, _updates_desc, updates,
                    _output_desc, output),
      CNNL_STATUS_SUCCESS,
      "[mluOpScatterNd] Internal error accured in mluOpScatterNd.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_indices_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_updates_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_output_desc);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpScatterNd_v2(
    mluOpHandle_t handle, mluOpScatterNdMode_t mode,
    const mluOpTensorDescriptor_t indices_desc, const void *indices,
    const mluOpTensorDescriptor_t updates_desc, const void *updates,
    const mluOpTensorDescriptor_t input_desc, const void *input,
    const mluOpTensorDescriptor_t output_desc, void *output) {
  PARAM_CHECK("mluOpScatterNd_v2", handle != NULL);
  PARAM_CHECK_GE("mluOpScatterNd_v2", mode, 0);
  PARAM_CHECK("mluOpScatterNd_v2", indices_desc != NULL);
  PARAM_CHECK("mluOpScatterNd_v2", indices != NULL);
  PARAM_CHECK("mluOpScatterNd_v2", updates_desc != NULL);
  PARAM_CHECK("mluOpScatterNd_v2", updates != NULL);
  PARAM_CHECK("mluOpScatterNd_v2", output != NULL);
  CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(indices_desc, _indices_desc);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(updates_desc, _updates_desc);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, _input_desc);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, _output_desc);

  CHECK_FUNC_RETURN(
      cnnlScatterNd_v2(_handle, cnnlScatterNdMode_t(mode), _indices_desc,
                       indices, _updates_desc, updates, _input_desc, input,
                       _output_desc, output),
      CNNL_STATUS_SUCCESS,
      "[mluOpScatterNd] Internal error accured in mluOpScatterNd_v2.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_indices_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_updates_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_output_desc);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}
