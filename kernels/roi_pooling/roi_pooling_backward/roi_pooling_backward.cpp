/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
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

mluOpStatus_t MLUOP_WIN_API mluOpRoiPoolingBackward(
    mluOpHandle_t handle, mluOpPoolingMode_t pooling_mode,
    const mluOpTensorDescriptor_t grads_desc, const void *grads,
    const mluOpTensorDescriptor_t rois_desc, const void *rois,
    const mluOpTensorDescriptor_t argmax_desc, const int *argmax,
    const float spatial_scale, const mluOpTensorDescriptor_t grads_image_desc,
    void *grads_image) {
  PARAM_CHECK("[mluOpRoiPoolingBackward]", handle != NULL);
  PARAM_CHECK("[mluOpRoiPoolingBackward]", grads_desc != NULL);
  PARAM_CHECK("[mluOpRoiPoolingBackward]", grads != NULL);
  PARAM_CHECK("[mluOpRoiPoolingBackward]", rois_desc != NULL);
  PARAM_CHECK("[mluOpRoiPoolingBackward]", rois != NULL);
  PARAM_CHECK("[mluOpRoiPoolingBackward]", argmax_desc != NULL);
  PARAM_CHECK("[mluOpRoiPoolingBackward]", argmax != NULL);
  PARAM_CHECK("[mluOpRoiPoolingBackward]", grads_image_desc != NULL);
  PARAM_CHECK("[mluOpRoiPoolingBackward]", grads_image != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grads_desc, cnnl_grads_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(rois_desc, cnnl_rois_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(argmax_desc, cnnl_argmax_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grads_image_desc,
                                               cnnl_grads_image_desc);

  CALL_CNNL(cnnlRoiPoolingBackward(cnnl_handle, cnnlPoolingMode_t(pooling_mode),
                                   cnnl_grads_desc, grads, cnnl_rois_desc, rois,
                                   cnnl_argmax_desc, argmax, spatial_scale,
                                   cnnl_grads_image_desc, grads_image));

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_grads_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_rois_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_argmax_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_grads_image_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
