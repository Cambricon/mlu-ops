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
#include <algorithm>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"
#include "kernels/kernel.h"

static void policyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type, const int HWA) {
  int job = mluop::runtime::getJobLimitCapability(handle);

  // Make sure at least 128 data is processed on each core
  int per_core_num = HWA / job;
  const int min_per_core_num = 128;
  while (per_core_num < min_per_core_num && job >= 4) {
    per_core_num *= 2;
    job /= 2;
  }

  k_dim->y = 1;
  k_dim->z = 1;

  if (job < 4) {
    k_dim->x = 1;
    *k_type = CNRT_FUNC_TYPE_BLOCK;
  } else {
    *k_type = CNRT_FUNC_TYPE_UNION1;
    k_dim->x = handle->core_num_per_cluster;
  }
  return;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetGenerateProposalsV2WorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t scores_desc,
    size_t *size) {
  const std::string API = "[mluOpGenerateProposalsV2]";
  PARAM_CHECK(API, handle != NULL);
  PARAM_CHECK(API, scores_desc != NULL);
  PARAM_CHECK(API, size != NULL);

  PARAM_CHECK(API, scores_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, scores_desc->layout == MLUOP_LAYOUT_ARRAY);

  PARAM_CHECK_EQ(API, scores_desc->dim, 4);
  int N = scores_desc->dims[0];
  int H = scores_desc->dims[1];
  int W = scores_desc->dims[2];
  int A = scores_desc->dims[3];

  PARAM_CHECK_NE(API, A, 0);
  PARAM_CHECK_NE(API, H, 0);
  PARAM_CHECK_NE(API, W, 0);
  if ((mluOpGetTensorElementNum(scores_desc) *
           mluop::getSizeOfDataType(scores_desc->dtype) >=
       LARGE_TENSOR_SIZE)) {
    LOG(ERROR) << API << " Overflow max tensor size."
               << " Currently, MLU-OPS supports tensor size smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  *size = 12 * A * H * W * 4 + handle->core_num_per_cluster * 4 * 3;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGenerateProposalsV2(
    mluOpHandle_t handle, const int pre_nms_top_n, const int post_nms_top_n,
    const float nms_thresh, const float min_size, const float eta,
    bool pixel_offset, const mluOpTensorDescriptor_t scores_desc,
    const void *scores, const mluOpTensorDescriptor_t bbox_deltas_desc,
    const void *bbox_deltas, const mluOpTensorDescriptor_t im_shape_desc,
    const void *im_shape, const mluOpTensorDescriptor_t anchors_desc,
    const void *anchors, const mluOpTensorDescriptor_t variances_desc,
    const void *variances, void *workspace, size_t workspace_size,
    const mluOpTensorDescriptor_t rpn_rois_desc, void *rpn_rois,
    const mluOpTensorDescriptor_t rpn_roi_probs_desc, void *rpn_roi_probs,
    const mluOpTensorDescriptor_t rpn_rois_num_desc, void *rpn_rois_num,
    void *rpn_rois_batch_size) {
  const std::string API = "[mluOpGenerateProposalsV2]";
  // check inputs/outputs
  PARAM_CHECK(API, handle != NULL);
  PARAM_CHECK(API, scores_desc != NULL);
  PARAM_CHECK(API, bbox_deltas_desc != NULL);
  PARAM_CHECK(API, im_shape_desc != NULL);
  PARAM_CHECK(API, anchors_desc != NULL);
  PARAM_CHECK(API, variances_desc != NULL);
  PARAM_CHECK(API, rpn_rois_batch_size != NULL);

  PARAM_CHECK(API, rpn_rois_desc != NULL);
  PARAM_CHECK(API, rpn_roi_probs_desc != NULL);
  PARAM_CHECK(API, rpn_rois_num_desc != NULL);

  // check inputs/outputs data type
  PARAM_CHECK(API, scores_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, bbox_deltas_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, im_shape_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, anchors_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, variances_desc->dtype == MLUOP_DTYPE_FLOAT);

  PARAM_CHECK(API, rpn_rois_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, rpn_roi_probs_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, rpn_rois_num_desc->dtype == MLUOP_DTYPE_INT32);

  // check inputs layout
  PARAM_CHECK(API, scores_desc->layout == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(API, bbox_deltas_desc->layout == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(API, im_shape_desc->layout == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(API, anchors_desc->layout == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(API, variances_desc->layout == MLUOP_LAYOUT_ARRAY);

  PARAM_CHECK(API, rpn_rois_desc->layout == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(API, rpn_roi_probs_desc->layout == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(API, rpn_rois_num_desc->layout == MLUOP_LAYOUT_ARRAY);

  // check inputs shape
  PARAM_CHECK_EQ(API, scores_desc->dim, 4);
  int N = scores_desc->dims[0];
  int H = scores_desc->dims[1];
  int W = scores_desc->dims[2];
  int A = scores_desc->dims[3];

  // [N,H,W,A4]
  PARAM_CHECK_EQ(API, bbox_deltas_desc->dim, 4);
  PARAM_CHECK_EQ(API, bbox_deltas_desc->dims[0], N);
  PARAM_CHECK_EQ(API, bbox_deltas_desc->dims[1], H);
  PARAM_CHECK_EQ(API, bbox_deltas_desc->dims[2], W);
  PARAM_CHECK_EQ(API, bbox_deltas_desc->dims[3], 4 * A);

  // [N, 2]
  PARAM_CHECK_EQ(API, im_shape_desc->dim, 2);
  PARAM_CHECK_EQ(API, im_shape_desc->dims[0], N);
  PARAM_CHECK_EQ(API, im_shape_desc->dims[1], 2);

  // [H, W, A, 4]
  PARAM_CHECK_EQ(API, anchors_desc->dim, 4);
  PARAM_CHECK_EQ(API, anchors_desc->dims[0], H);
  PARAM_CHECK_EQ(API, anchors_desc->dims[1], W);
  PARAM_CHECK_EQ(API, anchors_desc->dims[2], A);
  PARAM_CHECK_EQ(API, anchors_desc->dims[3], 4);

  // [H, W, A, 4]
  PARAM_CHECK_EQ(API, variances_desc->dim, 4);
  PARAM_CHECK_EQ(API, variances_desc->dims[0], H);
  PARAM_CHECK_EQ(API, variances_desc->dims[1], W);
  PARAM_CHECK_EQ(API, variances_desc->dims[2], A);
  PARAM_CHECK_EQ(API, variances_desc->dims[3], 4);

  // check output shape
  PARAM_CHECK_EQ(API, rpn_rois_desc->dim, 2);
  PARAM_CHECK_EQ(API, rpn_rois_desc->dims[0], N * post_nms_top_n);
  PARAM_CHECK_EQ(API, rpn_rois_desc->dims[1], 4);

  PARAM_CHECK_EQ(API, rpn_roi_probs_desc->dim, 2);
  PARAM_CHECK_EQ(API, rpn_roi_probs_desc->dims[0], N * post_nms_top_n);
  PARAM_CHECK_EQ(API, rpn_roi_probs_desc->dims[1], 1);

  PARAM_CHECK_EQ(API, rpn_rois_num_desc->dim, 1);
  PARAM_CHECK_EQ(API, rpn_rois_num_desc->dims[0], N);

  PARAM_CHECK_NE(API, A, 0);
  PARAM_CHECK_NE(API, H, 0);
  PARAM_CHECK_NE(API, W, 0);

  if (N == 0) {
    VLOG(5) << API << " skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  PARAM_CHECK(API, scores != NULL);
  PARAM_CHECK(API, bbox_deltas != NULL);
  PARAM_CHECK(API, im_shape != NULL);
  PARAM_CHECK(API, anchors != NULL);
  PARAM_CHECK(API, variances != NULL);

  PARAM_CHECK(API, rpn_rois != NULL);
  PARAM_CHECK(API, rpn_roi_probs != NULL);
  PARAM_CHECK(API, rpn_rois_num != NULL);
  PARAM_CHECK(API, rpn_rois_batch_size != NULL);

  if (workspace_size > 0) {
    PARAM_CHECK(API, workspace != NULL);
  }

  if (eta < 1.0) {
    LOG(ERROR) << API << " Not support adaptive NMS. The attribute 'eta' "
               << "should not less than 1. But received eta=[" << eta << "]";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if (nms_thresh <= 0) {
    LOG(ERROR) << API
               << " nms_thresh should be more than 0. But received nms_thresh=["
               << nms_thresh << "]";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if ((mluOpGetTensorElementNum(scores_desc) *
           mluop::getSizeOfDataType(scores_desc->dtype) >=
       LARGE_TENSOR_SIZE) ||
      (mluOpGetTensorElementNum(bbox_deltas_desc) *
           mluop::getSizeOfDataType(bbox_deltas_desc->dtype) >=
       LARGE_TENSOR_SIZE) ||
      (mluOpGetTensorElementNum(anchors_desc) *
           mluop::getSizeOfDataType(anchors_desc->dtype) >=
       LARGE_TENSOR_SIZE) ||
      (mluOpGetTensorElementNum(variances_desc) *
           mluop::getSizeOfDataType(variances_desc->dtype) >=
       LARGE_TENSOR_SIZE)) {
    LOG(ERROR) << API << " Overflow max tensor size."
               << " Currently, MLU-OPS supports tensor size smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // generate prototxt
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("generate_proposals_v2");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input1", scores, scores_desc, 10, 0);
    GEN_CASE_DATA(true, "input2", bbox_deltas, bbox_deltas_desc, 10, 0);
    GEN_CASE_DATA(true, "input3", anchors, anchors_desc, 10, 0);
    GEN_CASE_DATA(true, "input4", variances, variances_desc, 10, 0);
    GEN_CASE_DATA(true, "input5", im_shape, im_shape_desc, 10, 0);

    GEN_CASE_DATA(false, "output1", rpn_rois, rpn_rois_desc, 0, 0);
    GEN_CASE_DATA(false, "output2", rpn_roi_probs, rpn_roi_probs_desc, 0, 0);
    GEN_CASE_DATA(false, "output3", rpn_rois_num, rpn_rois_num_desc, 0, 0);
    GEN_CASE_DATA_UNFOLD(false, "output4", rpn_rois_batch_size, 1, {1},
                         MLUOP_DTYPE_INT32, MLUOP_LAYOUT_ARRAY, 0, 0);

    GEN_CASE_OP_PARAM_SINGLE(0, "generate_proposals_v2", "pre_nms_top_n",
                             pre_nms_top_n);
    GEN_CASE_OP_PARAM_SINGLE(1, "generate_proposals_v2", "post_nms_top_n",
                             post_nms_top_n);
    GEN_CASE_OP_PARAM_SINGLE(2, "generate_proposals_v2", "nms_thresh",
                             nms_thresh);
    GEN_CASE_OP_PARAM_SINGLE(3, "generate_proposals_v2", "min_size", min_size);
    GEN_CASE_OP_PARAM_SINGLE(4, "generate_proposals_v2", "eta", eta);
    GEN_CASE_OP_PARAM_SINGLE(5, "generate_proposals_v2", "pixel_offset",
                             pixel_offset);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 3e-3, 3e-3, 0);
  }

  int HWA = H * W * A;
  cnrtDim3_t k_dim;
  cnrtJobType_t k_type;
  policyFunc(handle, &k_dim, &k_type, HWA);

  VLOG(5) << "Launch Kernel mluOpUBestKernelGenerateProposalsV2Float <<<k_dim: "
          << k_type << ", " << k_dim.x << ", " << k_dim.y << ", " << k_dim.z
          << ">>>";

  KERNEL_CHECK(mluOpUBestKernelGenerateProposalsV2Float(
      k_dim, k_type, handle->queue, (float *)scores, (float *)bbox_deltas,
      (float *)im_shape, (float *)anchors, (float *)variances,
      (float *)workspace, (float *)rpn_rois, (float *)rpn_roi_probs,
      (int *)rpn_rois_num, (int *)rpn_rois_batch_size, pre_nms_top_n,
      post_nms_top_n, nms_thresh, min_size, eta, pixel_offset, N, A, H, W));
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
