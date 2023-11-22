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
#include "cnnl_helper.h"

// Enumeration convert template
template <typename SRC_ENUM, typename DST_ENUM>
DST_ENUM mluOpConvertEnum(SRC_ENUM param) {
  return (DST_ENUM)((int)param);
}

// TensorDescriptor convert
cnnlStatus_t mluOpConvertDescriptor(mluOpTensorDescriptor_t desc,
                                    cnnlTensorDescriptor_t _desc) {
  if (desc == NULL) {
    return CNNL_STATUS_SUCCESS;
  }
  mluOpDataType_t dtype, onchip_dtype;
  mluOpTensorLayout_t layout;
  int tensor_dim;
  CHECK_FUNC_RETURN(
      mluOpGetTensorDescriptor(desc, &layout, &dtype, &tensor_dim, NULL),
      MLUOP_STATUS_SUCCESS, "MLUOPS get tensor descriptor failed.",
      CNNL_STATUS_INTERNAL_ERROR);
  CHECK_FUNC_RETURN(mluOpGetTensorDescriptorOnchipDataType(desc, &onchip_dtype),
                    MLUOP_STATUS_SUCCESS,
                    "MLUOPS get tensor descriptor onchip type failed.",
                    CNNL_STATUS_INTERNAL_ERROR);
  int *dims = new int[tensor_dim];
  int *strides = new int[tensor_dim];
  CHECK_FUNC_RETURN(mluOpGetTensorDescriptorEx(desc, &layout, &dtype,
                                               &tensor_dim, dims, strides),
                    MLUOP_STATUS_SUCCESS,
                    "MLUOPS get tensor descriptor Ex failed.",
                    CNNL_STATUS_INTERNAL_ERROR);
  CHECK_FUNC_RETURN(
      cnnlSetTensorDescriptor(
          _desc,
          mluOpConvertEnum<mluOpTensorLayout_t, cnnlTensorLayout_t>(layout),
          mluOpConvertEnum<mluOpDataType_t, cnnlDataType_t>(dtype), tensor_dim,
          dims),
      CNNL_STATUS_SUCCESS, "Internal set tensor descriptor failed.",
      CNNL_STATUS_INTERNAL_ERROR);
  CHECK_FUNC_RETURN(
      cnnlSetTensorDescriptorEx(
          _desc,
          mluOpConvertEnum<mluOpTensorLayout_t, cnnlTensorLayout_t>(layout),
          mluOpConvertEnum<mluOpDataType_t, cnnlDataType_t>(dtype), tensor_dim,
          dims, strides),
      CNNL_STATUS_SUCCESS, "Internal set tensor descriptor Ex failed.",
      CNNL_STATUS_INTERNAL_ERROR);
  CHECK_FUNC_RETURN(
      cnnlSetTensorDescriptorOnchipDataType(
          _desc,
          mluOpConvertEnum<mluOpDataType_t, cnnlDataType_t>(onchip_dtype)),
      CNNL_STATUS_SUCCESS, "Internal set tensor descriptor Ex failed.",
      CNNL_STATUS_INTERNAL_ERROR);
  int position;
  float scale;
  int offset;
  CHECK_FUNC_RETURN(
      mluOpGetTensorDescriptorPositionScaleAndOffset(desc, &position, &scale,
                                                     &offset),
      MLUOP_STATUS_SUCCESS,
      "MLUOPS get tensor descriptor position scale and offset failed.",
      CNNL_STATUS_INTERNAL_ERROR);
  CHECK_FUNC_RETURN(
      cnnlSetTensorDescriptorPositionScaleAndOffset(_desc, position, scale,
                                                    offset),
      CNNL_STATUS_SUCCESS,
      "Internal set tensor descriptor position scale and offset failed.",
      CNNL_STATUS_INTERNAL_ERROR);
  delete[] dims;
  delete[] strides;
  return CNNL_STATUS_SUCCESS;
}

cnnlStatus_t mluOpConvertHandle(mluOpHandle_t handle, cnnlHandle_t _handle) {
  cnrtQueue_t queue;
  CHECK_FUNC_RETURN(mluOpGetQueue(handle, &queue), MLUOP_STATUS_SUCCESS,
                    "MLUOPS get queue failed.", CNNL_STATUS_INTERNAL_ERROR);
  CHECK_FUNC_RETURN(cnnlSetQueue(_handle, queue), CNNL_STATUS_SUCCESS,
                    "Internal set queue failed.", CNNL_STATUS_INTERNAL_ERROR);
  return CNNL_STATUS_SUCCESS;
}
