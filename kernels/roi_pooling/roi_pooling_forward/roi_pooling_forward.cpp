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
#include "kernels/utils/cnnl_helper.h"

mluOpStatus_t MLUOP_WIN_API mluOpRoiPoolingForward(
    mluOpHandle_t handle, mluOpPoolingMode_t pooling_mode,
    const mluOpTensorDescriptor_t input_desc, const void *input,
    const mluOpTensorDescriptor_t rois_desc, const void *rois,
    float spatial_scale, const mluOpTensorDescriptor_t output_desc,
    void *output, int *argmax) {
  PARAM_CHECK("[mluOpRoiPoolingForward]", handle != NULL);
  PARAM_CHECK("[mluOpRoiPoolingForward]", input_desc != NULL);
  PARAM_CHECK("[mluOpRoiPoolingForward]", input != NULL);
  PARAM_CHECK("[mluOpRoiPoolingForward]", rois_desc != NULL);
  PARAM_CHECK("[mluOpRoiPoolingForward]", rois != NULL);
  PARAM_CHECK("[mluOpRoiPoolingForward]", output_desc != NULL);
  PARAM_CHECK("[mluOpRoiPoolingForward]", output != NULL);
  PARAM_CHECK("[mluOpRoiPoolingForward]", argmax != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(rois_desc, cnnl_rois_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, cnnl_output_desc);

  CALL_CNNL(cnnlRoiPoolingForward(
      cnnl_handle, cnnlPoolingMode_t(pooling_mode), cnnl_input_desc, input,
      cnnl_rois_desc, rois, spatial_scale, cnnl_output_desc, output, argmax));

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_rois_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
