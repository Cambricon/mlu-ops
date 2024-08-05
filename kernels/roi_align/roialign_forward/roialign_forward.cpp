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
mluOpCreateRoiAlignForwardDescriptor(mluOpRoiAlignForwardDescriptor_t *desc) {
  PARAM_CHECK("[mluOpRoiAlignForward_v2]", desc != NULL);
  CALL_CNNL(cnnlCreateRoiAlignDescriptor(desc));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpDestroyRoiAlignForwardDescriptor(mluOpRoiAlignForwardDescriptor_t desc) {
  PARAM_CHECK("[mluOpRoiAlignForward_v2]", desc != NULL);
  CALL_CNNL(cnnlDestroyRoiAlignDescriptor(desc));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetRoiAlignForwardDescriptor_v2(
    mluOpRoiAlignForwardDescriptor_t desc, const int pooled_height,
    const int pooled_width, const int sampling_ratio, const float spatial_scale,
    const int pool_mode, const bool aligned) {
  PARAM_CHECK("[mluOpRoiAlignForward_v2]", desc != NULL);
  CALL_CNNL(cnnlSetRoiAlignDescriptor_v2(desc, pooled_height, pooled_width,
                                         sampling_ratio, spatial_scale,
                                         pool_mode, aligned));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpRoiAlignForward_v2(
    mluOpHandle_t handle, const mluOpRoiAlignForwardDescriptor_t roialign_desc,
    const mluOpTensorDescriptor_t input_desc, const void *input,
    const mluOpTensorDescriptor_t boxes_desc, const void *boxes,
    const mluOpTensorDescriptor_t output_desc, void *output,
    const mluOpTensorDescriptor_t argmax_x_desc, void *argmax_x,
    const mluOpTensorDescriptor_t argmax_y_desc, void *argmax_y) {
  PARAM_CHECK("mluOpRoiAlignForward_v2", handle != NULL);
  PARAM_CHECK("mluOpRoiAlignForward_v2", roialign_desc != NULL);
  PARAM_CHECK("mluOpRoiAlignForward_v2", input_desc != NULL);
  PARAM_CHECK("mluOpRoiAlignForward_v2", boxes_desc != NULL);
  PARAM_CHECK("mluOpRoiAlignForward_v2", output_desc != NULL);
  PARAM_CHECK("mluOpRoiAlignForward_v2", input != NULL);
  PARAM_CHECK("mluOpRoiAlignForward_v2", boxes != NULL);
  PARAM_CHECK("mluOpRoiAlignForward_v2", output != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(boxes_desc, cnnl_boxes_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, cnnl_output_desc);

  cnnlTensorDescriptor_t cnnl_argmax_x_desc = NULL;
  cnnlTensorDescriptor_t cnnl_argmax_y_desc = NULL;
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(argmax_x_desc, cnnl_argmax_x_desc);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(argmax_y_desc, cnnl_argmax_y_desc);
  CALL_CNNL(cnnlRoiAlign_v2(cnnl_handle, roialign_desc, cnnl_input_desc, input,
                            cnnl_boxes_desc, boxes, cnnl_output_desc, output,
                            cnnl_argmax_x_desc, argmax_x, cnnl_argmax_y_desc,
                            argmax_y));

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_boxes_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_argmax_x_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_argmax_y_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
