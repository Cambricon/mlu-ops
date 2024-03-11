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
#include "yolo_box.h"

#include <string>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/utils/cnnl_helper.h"

#define MAX_CLASS_NUM_ARCH_200 1534
#define MAX_CLASS_NUM_ARCH_300 2558

static void policyFunc(const mluOpHandle_t handle, const int kw_num,
                       cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  *k_type = CNRT_FUNC_TYPE_BLOCK;
  uint32_t cluster_num = mluop::runtime::getClusterLimitCapability(handle);
  uint32_t core_num_per_cluster =
      mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  uint32_t core_max = cluster_num * core_num_per_cluster;
  uint32_t core_used = core_max > kw_num ? kw_num : core_max;
  k_dim->x = core_used;
  k_dim->y = 1;
  k_dim->z = 1;
}

static mluOpStatus_t yoloBoxParamCheck(
    const std::string &op_name, const mluOpHandle_t handle,
    const mluOpTensorDescriptor_t x_desc, const void *x,
    const mluOpTensorDescriptor_t img_size_desc, const void *img_size,
    const mluOpTensorDescriptor_t anchors_desc, const void *anchors,
    const mluOpTensorDescriptor_t boxes_desc, const void *boxes,
    const mluOpTensorDescriptor_t scores_desc, const void *scores,
    const int class_num, const bool iou_aware, const float iou_aware_factor,
    bool *zero_element) {
  // check descriptor and data
  PARAM_CHECK(op_name, handle != NULL);
  PARAM_CHECK(op_name, x_desc != NULL);
  PARAM_CHECK(op_name, img_size_desc != NULL);
  PARAM_CHECK(op_name, anchors_desc != NULL);
  PARAM_CHECK(op_name, boxes_desc != NULL);
  PARAM_CHECK(op_name, scores_desc != NULL);

  // check shape
  PARAM_CHECK(op_name, x_desc->dim == 4);
  PARAM_CHECK(op_name, img_size_desc->dim == 2);
  PARAM_CHECK(op_name, anchors_desc->dim == 1);
  PARAM_CHECK(op_name, boxes_desc->dim == 4);
  PARAM_CHECK(op_name, scores_desc->dim == 4);

  // check data type
  PARAM_CHECK(op_name, x_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(op_name, img_size_desc->dtype == MLUOP_DTYPE_INT32);
  PARAM_CHECK(op_name, anchors_desc->dtype == MLUOP_DTYPE_INT32);
  PARAM_CHECK(op_name, boxes_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(op_name, scores_desc->dtype == MLUOP_DTYPE_FLOAT);

  // check param except tensor
  if (iou_aware) {
    if (iou_aware_factor < 0 || iou_aware_factor > 1) {
      LOG(ERROR) << "[mluOpYoloBox]: iou_aware_factor should be"
                 << " between [0, 1].";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  // check dim
  PARAM_CHECK(op_name, (x_desc->dims[0] == img_size_desc->dims[0]));
  PARAM_CHECK(op_name, (x_desc->dims[0] == boxes_desc->dims[0]));
  PARAM_CHECK(op_name, (x_desc->dims[0] == scores_desc->dims[0]));

  const int anchors_num = anchors_desc->dims[0] / 2;
  PARAM_CHECK(op_name, (anchors_desc->dims[0] % 2 == 0));
  PARAM_CHECK(op_name, anchors_num > 0);
  PARAM_CHECK(op_name, class_num > 0);

  const int class_num_thresh = handle->arch >= MLUOP_MLU370
                                   ? MAX_CLASS_NUM_ARCH_300
                                   : MAX_CLASS_NUM_ARCH_200;
  PARAM_CHECK(op_name, class_num <= class_num_thresh);

  const int dim_c =
      iou_aware ? anchors_num * (6 + class_num) : anchors_num * (5 + class_num);
  PARAM_CHECK(op_name, (x_desc->dims[1] == dim_c));
  PARAM_CHECK(op_name, (img_size_desc->dims[1] == 2));
  PARAM_CHECK(op_name, (boxes_desc->dims[1] == anchors_num));
  PARAM_CHECK(op_name, (boxes_desc->dims[2] == 4));
  PARAM_CHECK(op_name,
              (boxes_desc->dims[3] == (x_desc->dims[2] * x_desc->dims[3])));
  PARAM_CHECK(op_name, (scores_desc->dims[1] == anchors_num));
  PARAM_CHECK(op_name, (scores_desc->dims[2] == class_num));
  PARAM_CHECK(op_name,
              (scores_desc->dims[3] == (x_desc->dims[2] * x_desc->dims[3])));

  // large tensor
  if ((mluOpGetTensorElementNum(x_desc) >= LARGE_TENSOR_NUM) ||
      (mluOpGetTensorElementNum(img_size_desc) >= LARGE_TENSOR_NUM) ||
      (mluOpGetTensorElementNum(anchors_desc) >= LARGE_TENSOR_NUM) ||
      (mluOpGetTensorElementNum(boxes_desc) >= LARGE_TENSOR_NUM) ||
      (mluOpGetTensorElementNum(scores_desc) >= LARGE_TENSOR_NUM)) {
    LOG(ERROR) << op_name << " Overflow max tensor num."
               << " Currently, MLU-OPS supports tensor num smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // check zero element
  if (mluOpGetTensorElementNum(x_desc) == 0) {
    *zero_element = true;
    return MLUOP_STATUS_SUCCESS;
  }

  PARAM_CHECK(op_name, x != NULL);
  PARAM_CHECK(op_name, img_size != NULL);
  PARAM_CHECK(op_name, anchors != NULL);
  PARAM_CHECK(op_name, boxes != NULL);
  PARAM_CHECK(op_name, scores != NULL);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpYoloBox(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t x_desc, const void *x,
    const mluOpTensorDescriptor_t img_size_desc, const void *img_size,
    const mluOpTensorDescriptor_t anchors_desc, const void *anchors,
    const int class_num, const float conf_thresh, const int downsample_ratio,
    const bool clip_bbox, const float scale, const bool iou_aware,
    const float iou_aware_factor, const mluOpTensorDescriptor_t boxes_desc,
    void *boxes, const mluOpTensorDescriptor_t scores_desc, void *scores) {
  // check params
  bool zero_element = false;
  mluOpStatus_t param_check = yoloBoxParamCheck(
      "[mluOpYoloBox]", handle, x_desc, x, img_size_desc, img_size,
      anchors_desc, anchors, boxes_desc, boxes, scores_desc, scores, class_num,
      iou_aware, iou_aware_factor, &zero_element);
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }

  // check zero element
  if (zero_element == true) {
    VLOG(5) << "[mluOpYoloBox] Input skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("yolo_box", "YOLO_BOX");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "x", x, x_desc, 10, 0);
    GEN_CASE_DATA(true, "img_size", img_size, img_size_desc, 1000, 100);
    GEN_CASE_DATA(true, "anchors", anchors, anchors_desc, 10, 1);
    GEN_CASE_DATA(false, "boxes", boxes, boxes_desc, 0, 0);
    GEN_CASE_DATA(false, "scores", scores, scores_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "yolo_box", "class_num", class_num);
    GEN_CASE_OP_PARAM_SINGLE(1, "yolo_box", "conf_thresh", conf_thresh);
    GEN_CASE_OP_PARAM_SINGLE(2, "yolo_box", "downsample_ratio",
                             downsample_ratio);
    GEN_CASE_OP_PARAM_SINGLE(3, "yolo_box", "clip_bbox", clip_bbox);
    GEN_CASE_OP_PARAM_SINGLE(4, "yolo_box", "scale_x_y", scale);
    GEN_CASE_OP_PARAM_SINGLE(5, "yolo_box", "iou_aware", iou_aware);
    GEN_CASE_OP_PARAM_SINGLE(6, "yolo_box", "iou_aware_factor",
                             iou_aware_factor);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  const int n_in = x_desc->dims[0];
  const int c_in = x_desc->dims[1];
  const int h_in = x_desc->dims[2];
  const int w_in = x_desc->dims[3];
  const int anchor_s = anchors_desc->dims[0] / 2;
  const int kw_num = h_in * w_in;

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, kw_num, &k_dim, &k_type);
  VLOG(5) << "[mluOpYoloBox] launch kernel policyFunc[" << k_dim.x << ", "
          << k_dim.y << ", " << k_dim.z << "].";

  float fill_value = 0;
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(boxes_desc, cnnl_output_desc);
    CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &fill_value,
                          cnnl_output_desc, boxes));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(scores_desc, cnnl_output_desc);
    CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &fill_value,
                          cnnl_output_desc, scores));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }

  CHECK_RETURN("[mluOpYoloBox]",
               KernelYoloBox(k_dim, k_type, handle->queue, x, img_size, anchors,
                             class_num, conf_thresh, downsample_ratio,
                             clip_bbox, scale, iou_aware, iou_aware_factor,
                             n_in, anchor_s, c_in, h_in, w_in, boxes, scores));

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
