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

mluOpStatus_t MLUOP_WIN_API mluOpRoiAlignBackward(
    mluOpHandle_t handle, const float spatial_scale, const int sampling_ratio,
    const bool aligned, const mluOpTensorDescriptor_t grads_desc,
    const void *grads, const mluOpTensorDescriptor_t boxes_desc,
    const void *boxes, const mluOpTensorDescriptor_t grads_image_desc,
    void *grads_image) {
  PARAM_CHECK("mluOpRoiAlignBackward", handle != NULL);
  PARAM_CHECK("mluOpRoiAlignBackward", grads_desc != NULL);
  PARAM_CHECK("mluOpRoiAlignBackward", grads != NULL);
  PARAM_CHECK("mluOpRoiAlignBackward", boxes_desc != NULL);
  PARAM_CHECK("mluOpRoiAlignBackward", boxes != NULL);
  PARAM_CHECK("mluOpRoiAlignBackward", grads_image_desc != NULL);
  PARAM_CHECK("mluOpRoiAlignBackward", grads_image != NULL);
  CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grads_desc, _grads_desc);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(boxes_desc, _boxes_desc);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grads_image_desc, _grads_image_desc);
  CHECK_FUNC_RETURN(
      cnnlRoiAlignBackward(_handle, spatial_scale, sampling_ratio, aligned,
                           _grads_desc, grads, _boxes_desc, boxes,
                           _grads_image_desc, grads_image),
      CNNL_STATUS_SUCCESS,
      "[mluOpRoiAlignBackward] Internal error accured in "
      "mluOpRoiAlignBackward.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_grads_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_boxes_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_grads_image_desc);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpRoiAlignBackward_v2(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t grads_desc,
    const void *grads, const mluOpTensorDescriptor_t boxes_desc,
    const void *boxes, const mluOpTensorDescriptor_t argmax_x_desc,
    const void *argmax_x, const mluOpTensorDescriptor_t argmax_y_desc,
    const void *argmax_y, const float spatial_scale, const int sampling_ratio,
    const bool aligned, const int pool_mode,
    const mluOpTensorDescriptor_t grads_image_desc, void *grads_image) {
  PARAM_CHECK("mluOpRoiAlignBackward_v2", handle != NULL);
  PARAM_CHECK("mluOpRoiAlignBackward_v2", grads_desc != NULL);
  PARAM_CHECK("mluOpRoiAlignBackward_v2", grads != NULL);
  PARAM_CHECK("mluOpRoiAlignBackward_v2", boxes_desc != NULL);
  PARAM_CHECK("mluOpRoiAlignBackward_v2", boxes != NULL);
  PARAM_CHECK("mluOpRoiAlignBackward_v2", grads_image_desc != NULL);
  PARAM_CHECK("mluOpRoiAlignBackward_v2", grads_image != NULL);

  CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grads_desc, _grads_desc);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(boxes_desc, _boxes_desc);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grads_image_desc, _grads_image_desc);

  cnnlTensorDescriptor_t _argmax_x_desc = NULL;
  cnnlTensorDescriptor_t _argmax_y_desc = NULL;

  if (pool_mode == 0) {
    PARAM_CHECK("mluOpRoiAlignBackward_v2", argmax_x_desc != NULL);
    PARAM_CHECK("mluOpRoiAlignBackward_v2", argmax_x != NULL);
    PARAM_CHECK("mluOpRoiAlignBackward_v2", argmax_y_desc != NULL);
    PARAM_CHECK("mluOpRoiAlignBackward_v2", argmax_y != NULL);
    CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR_V2(argmax_x_desc, _argmax_x_desc);
    CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR_V2(argmax_y_desc, _argmax_y_desc);
  }
  CHECK_FUNC_RETURN(
      cnnlRoiAlignBackward_v2(_handle, _grads_desc, grads, _boxes_desc, boxes,
                              _argmax_x_desc, argmax_x, _argmax_y_desc,
                              argmax_y, spatial_scale, sampling_ratio, aligned,
                              pool_mode, _grads_image_desc, grads_image),
      CNNL_STATUS_SUCCESS,
      "[mluOpRoiAlignBackward_v2] Internal error accured in "
      "mluOpRoiAlignBackward_v2.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_grads_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_boxes_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_grads_image_desc);
  if (pool_mode == 0) {
    DESTROY_CNNL_TENSOR_DESCRIPTOR(_argmax_x_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(_argmax_y_desc);
  }
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}
