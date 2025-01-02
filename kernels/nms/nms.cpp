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

mluOpStatus_t MLUOP_WIN_API
mluOpCreateNmsDescriptor(mluOpNmsDescriptor_t *desc) {
  PARAM_CHECK("mluOpCreateNmsDescriptor", desc != NULL);
  CALL_CNNL(cnnlCreateNmsDescriptor(desc));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpDestroyNmsDescriptor(mluOpNmsDescriptor_t desc) {
  PARAM_CHECK("mluOpDestroyNmsDescriptor", desc != NULL);
  CALL_CNNL(cnnlDestroyNmsDescriptor(desc));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetNmsDescriptor(
    mluOpNmsDescriptor_t nms_desc, const mluOpNmsBoxPointMode_t box_mode,
    const mluOpNmsOutputMode_t output_mode, const mluOpNmsAlgo_t algo,
    const mluOpNmsMethodMode_t method_mode, const float iou_threshold,
    const float soft_nms_sigma, const int max_output_size,
    const float confidence_threshold, const float offset,
    const int input_layout, const bool pad_to_max_output_size) {
  PARAM_CHECK("mluOpSetNmsDescriptor", nms_desc != NULL);
  CALL_CNNL(cnnlSetNmsDescriptor_v5(
      nms_desc, (cnnlNmsBoxPointMode_t)box_mode,
      (cnnlNmsOutputMode_t)output_mode, (cnnlNmsAlgo_t)algo,
      (cnnlNmsMethodMode_t)method_mode, iou_threshold, soft_nms_sigma,
      max_output_size, confidence_threshold, offset, input_layout,
      pad_to_max_output_size));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetNmsWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t boxes_desc,
    const mluOpTensorDescriptor_t confidence_desc, size_t *workspace_size) {
  PARAM_CHECK("mluOpGetNmsWorkspaceSize", handle != NULL);
  PARAM_CHECK("mluOpGetNmsWorkspaceSize", boxes_desc != NULL);
  PARAM_CHECK("mluOpGetNmsWorkspaceSize", workspace_size != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(boxes_desc, cnnl_boxes_desc);
  cnnlTensorDescriptor_t cnnl_confidence_desc = NULL;
  if (confidence_desc != NULL) {
    CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(confidence_desc,
                                          cnnl_confidence_desc);
  }

  CALL_CNNL(cnnlGetNmsWorkspaceSize_v3(cnnl_handle, cnnl_boxes_desc,
                                       cnnl_confidence_desc, workspace_size));
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_boxes_desc);
  if (cnnl_confidence_desc != NULL) {
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_confidence_desc);
  }
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpNms(mluOpHandle_t handle, const mluOpNmsDescriptor_t nms_desc,
         const mluOpTensorDescriptor_t boxes_desc, const void *boxes,
         const mluOpTensorDescriptor_t confidence_desc, const void *confidence,
         void *workspace, size_t workspace_size,
         const mluOpTensorDescriptor_t output_desc, void *output,
         void *output_size) {
  PARAM_CHECK("mluOpNms", handle != NULL);
  PARAM_CHECK("mluOpNms", boxes_desc != NULL);
  PARAM_CHECK("mluOpNms", nms_desc != NULL);
  PARAM_CHECK("mluOpNms", output_desc != NULL);
  PARAM_CHECK("mluOpNms", boxes != NULL);
  PARAM_CHECK("mluOpNms", output != NULL);
  PARAM_CHECK("mluOpNms", output_size != NULL);
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(boxes_desc, cnnl_boxes_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, cnnl_output_desc);

  cnnlTensorDescriptor_t cnnl_confidence_desc = NULL;
  if (confidence_desc != NULL) {
    CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(confidence_desc,
                                          cnnl_confidence_desc);
  }

  CALL_CNNL(cnnlNms_v2(cnnl_handle, nms_desc, cnnl_boxes_desc, boxes,
                       cnnl_confidence_desc, confidence, workspace,
                       workspace_size, cnnl_output_desc, output, output_size));

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_boxes_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
  if (cnnl_confidence_desc != NULL) {
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_confidence_desc);
  }
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
