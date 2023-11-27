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
mluOpCreateNmsDescriptor(mluOpNmsDescriptor_t *desc) {
  PARAM_CHECK("mluOpCreateNmsDescriptor", desc != NULL);
  CHECK_FUNC_RETURN(cnnlCreateNmsDescriptor(desc), CNNL_STATUS_SUCCESS,
                    "[mluOpNms] Internal error accured in "
                    "mluOpCreateNmsDescriptor.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpDestroyNmsDescriptor(mluOpNmsDescriptor_t desc) {
  PARAM_CHECK("mluOpDestroyNmsDescriptor", desc != NULL);
  CHECK_FUNC_RETURN(cnnlDestroyNmsDescriptor(desc), CNNL_STATUS_SUCCESS,
                    "[mluOpNms] Internal error accured in "
                    "mluOpDestroyNmsDescriptor.",
                    MLUOP_STATUS_INTERNAL_ERROR);
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
  CHECK_FUNC_RETURN(cnnlSetNmsDescriptor_v5(
                        nms_desc, (cnnlNmsBoxPointMode_t)box_mode,
                        (cnnlNmsOutputMode_t)output_mode, (cnnlNmsAlgo_t)algo,
                        (cnnlNmsMethodMode_t)method_mode, iou_threshold,
                        soft_nms_sigma, max_output_size, confidence_threshold,
                        offset, input_layout, pad_to_max_output_size),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpNms] Internal error accured in "
                    "mluOpSetNmsDescriptor.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetNmsWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t boxes_desc,
    const mluOpTensorDescriptor_t confidence_desc, size_t *workspace_size) {
  PARAM_CHECK("mluOpGetNmsWorkspaceSize", handle != NULL);
  PARAM_CHECK("mluOpGetNmsWorkspaceSize", boxes_desc != NULL);
  PARAM_CHECK("mluOpGetNmsWorkspaceSize", workspace_size != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(boxes_desc, _boxes_desc);
  cnnlTensorDescriptor_t _confidence_desc = NULL;
  if (confidence_desc != NULL) {
    CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(confidence_desc, _confidence_desc);
  }

  CHECK_FUNC_RETURN(
      cnnlGetNmsWorkspaceSize_v3(_handle, _boxes_desc, _confidence_desc,
                                 workspace_size),
      CNNL_STATUS_SUCCESS,
      "[mluOpNms] Internal error accured in mluOpGetNmsWorkspaceSize.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_boxes_desc);
  if (_confidence_desc != NULL) {
    DESTROY_CNNL_TENSOR_DESCRIPTOR(_confidence_desc);
  }
  DESTROY_CNNL_HANDLE(_handle);
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
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(boxes_desc, _boxes_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, _output_desc);

  cnnlTensorDescriptor_t _confidence_desc = NULL;
  if (confidence_desc != NULL) {
    CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(confidence_desc, _confidence_desc);
  }

  CHECK_FUNC_RETURN(
      cnnlNms_v2(_handle, nms_desc, _boxes_desc, boxes, _confidence_desc,
                 confidence, workspace, workspace_size, _output_desc, output,
                 output_size),
      CNNL_STATUS_SUCCESS,
      "[mluOpNms] Internal error accured in mluOpNms.",
      MLUOP_STATUS_INTERNAL_ERROR);

  DESTROY_CNNL_TENSOR_DESCRIPTOR(_boxes_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_output_desc);
  if (_confidence_desc != NULL) {
    DESTROY_CNNL_TENSOR_DESCRIPTOR(_confidence_desc);
  }
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}
