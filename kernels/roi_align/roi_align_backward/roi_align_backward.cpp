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
#include "core/cnnl_helper.h"

mluOpStatus_t MLUOP_WIN_API mluOpRoiAlignBackward(
    mluOpHandle_t handle, const float spatial_scale, const int sampling_ratio,
    const bool aligned, const mluOpTensorDescriptor_t grads_desc,
    const void *grads, const mluOpTensorDescriptor_t boxes_desc,
    const void *boxes, const mluOpTensorDescriptor_t grads_image_desc,
    void *grads_image) {
  LOG(ERROR) << "[mluOpRoiAlignBackward] This API is deprecated. Use "
             << "mluOpRoiAlignBackward_v2 instead.";
  return MLUOP_STATUS_NOT_SUPPORTED;
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

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grads_desc, cnnl_grads_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(boxes_desc, cnnl_boxes_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grads_image_desc,
                                               cnnl_grads_image_desc);

  cnnlTensorDescriptor_t cnnl_argmax_x_desc = NULL;
  cnnlTensorDescriptor_t cnnl_argmax_y_desc = NULL;

  if (pool_mode == 0) {
    PARAM_CHECK("mluOpRoiAlignBackward_v2", argmax_x_desc != NULL);
    PARAM_CHECK("mluOpRoiAlignBackward_v2", argmax_x != NULL);
    PARAM_CHECK("mluOpRoiAlignBackward_v2", argmax_y_desc != NULL);
    PARAM_CHECK("mluOpRoiAlignBackward_v2", argmax_y != NULL);
    CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(argmax_x_desc, cnnl_argmax_x_desc);
    CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(argmax_y_desc, cnnl_argmax_y_desc);
  }
  CALL_CNNL(cnnlRoiAlignBackward_v2(
      cnnl_handle, cnnl_grads_desc, grads, cnnl_boxes_desc, boxes,
      cnnl_argmax_x_desc, argmax_x, cnnl_argmax_y_desc, argmax_y, spatial_scale,
      sampling_ratio, aligned, pool_mode, cnnl_grads_image_desc, grads_image));
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_grads_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_boxes_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_grads_image_desc);
  if (pool_mode == 0) {
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_argmax_x_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_argmax_y_desc);
  }
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
