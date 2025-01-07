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
#include "generate_proposals_v2.h"

#include <algorithm>
#include <string>

#include "core/cnnl_helper.h"
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"

#define GDRAM_ALIGN_SIZE 128
static void policyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type, const size_t HWA) {
  int job = mluop::runtime::getJobLimitCapability(handle);

  // Make sure at least 128 data is processed on each core
  size_t per_core_num = HWA / (size_t)job;
  const size_t min_per_core_num = 128;
  while (per_core_num < min_per_core_num && job >= 4) {
    per_core_num *= 2;
    job /= 2;
  }

  k_dim->y = 1;
  k_dim->z = 1;
  if (job < 4) {
    k_dim->x = 1;
    *k_type = cnrtFuncTypeBlock;
  } else {
    *k_type = cnrtFuncTypeUnion1;
    k_dim->x = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  }
  return;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetGenerateProposalsV2WorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t scores_desc,
    size_t *size) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpGetGenerateProposalsV2WorkspaceSize] is deprecated and will be "
      << "removed in the future release,"
      << "please use [mluOpGetGenerateProposalsV2WorkspaceSize_v2] instead.";

  const std::string API = "[mluOpGenerateProposalsV2]";
  PARAM_CHECK(API, handle != NULL);
  PARAM_CHECK(API, scores_desc != NULL);
  PARAM_CHECK(API, size != NULL);

  PARAM_CHECK(API, scores_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, scores_desc->getLayout() == MLUOP_LAYOUT_ARRAY);

  PARAM_CHECK_EQ(API, scores_desc->getDim(), 4);
  PARAM_CHECK_NE(API, scores_desc->getDimIndex(1), 0);
  PARAM_CHECK_NE(API, scores_desc->getDimIndex(2), 0);
  PARAM_CHECK_NE(API, scores_desc->getDimIndex(3), 0);

  const size_t scores_num = mluOpGetTensorElementNum(scores_desc);
  TENSOR_NUM_CHECK(API, scores_num, LARGE_TENSOR_NUM, "");

  const int64_t n = scores_desc->getDimIndex(0);
  const int64_t h = scores_desc->getDimIndex(1);
  const int64_t w = scores_desc->getDimIndex(2);
  const int64_t a = scores_desc->getDimIndex(3);
  const int64_t hwa = h * w * a;
  if (handle->arch >= 592) {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    cnnlTensorDescriptor_t origin_indices_desc;
    cnnlTensorDescriptor_t sorted_score_desc;
    cnnlTensorDescriptor_t sorted_index_desc;
    CALL_CNNL(cnnlCreateTensorDescriptor(&origin_indices_desc));
    CALL_CNNL(cnnlCreateTensorDescriptor(&sorted_score_desc));
    CALL_CNNL(cnnlCreateTensorDescriptor(&sorted_index_desc));

    int64_t origin_indices_shape[2] = {n, hwa};
    int64_t sorted_indices_shape[2] = {n, hwa};
    int64_t sorted_index_shape[2] = {n, hwa};

    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    CALL_CNNL(cnnlSetTensorDescriptor_v2(origin_indices_desc, layout,
                                         CNNL_DTYPE_FLOAT, 2,
                                         origin_indices_shape));
    CALL_CNNL(cnnlSetTensorDescriptor_v2(sorted_score_desc, layout,
                                         CNNL_DTYPE_FLOAT, 2 /*dims:{n,hwa}*/,
                                         sorted_indices_shape));
    CALL_CNNL(cnnlSetTensorDescriptor_v2(sorted_index_desc, layout,
                                         CNNL_DTYPE_INT32, 2 /*dims:{n,hwa}*/,
                                         sorted_index_shape));

    const bool largest = true;  // param for topk, sort from large to small
    const bool sorted = true;   // param for topk, return sorted indices.
    const bool lower_index_first = true;  // param for topk, sort preservation

    size_t topk_workspace_size = 0;
    CALL_CNNL(cnnlGetTopKTensorWorkspaceSize(
        cnnl_handle, origin_indices_desc, hwa, 1, largest, sorted_score_desc,
        sorted_index_desc, &topk_workspace_size));

    CALL_CNNL(cnnlDestroyTensorDescriptor(origin_indices_desc));
    CALL_CNNL(cnnlDestroyTensorDescriptor(sorted_score_desc));
    CALL_CNNL(cnnlDestroyTensorDescriptor(sorted_index_desc));
    DESTROY_CNNL_HANDLE(cnnl_handle);
    size_t data_size = 0;
    mluOpGetSizeOfDataType(scores_desc->getDtype(), &data_size);

    const size_t topk_workspace_size_align =
        PAD_UP(topk_workspace_size, GDRAM_ALIGN_SIZE);
    const size_t nhwa_size_align =
        PAD_UP(n * hwa * data_size, GDRAM_ALIGN_SIZE);
    // topk_workspace_size_align: workspace be used in cnnlTopKTensor_v3.
    // 2 * nhwa_size_align: workspace be used to store scores and indexes after
    // topk.
    // 10 * hwa: workspace be used to store proposals score and box.
    // handle->core_num_per_cluster * data_size: workspace be used to store per
    // core proposals_num.
    *size = topk_workspace_size_align + 2 * nhwa_size_align +
            10 * hwa * data_size +
            handle->core_num_per_cluster * handle->cluster_num * data_size;
  } else {
    // 4 * hwa: workspace be used store proposals_box/scores
    // 8 * hwa: workspace be used store decode boxes/scores
    *size = 12 * hwa * 4 + handle->core_num_per_cluster * 4 * 3;
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetGenerateProposalsV2WorkspaceSize_v2(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t scores_desc,
    const int32_t pre_nms_top_n, size_t *size) {
  const std::string API = "[mluOpGenerateProposalsV2]";
  PARAM_CHECK(API, handle != NULL);
  PARAM_CHECK(API, scores_desc != NULL);
  PARAM_CHECK(API, size != NULL);

  PARAM_CHECK(API, scores_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, scores_desc->getLayout() == MLUOP_LAYOUT_ARRAY);

  PARAM_CHECK_EQ(API, scores_desc->getDim(), 4);
  PARAM_CHECK_NE(API, scores_desc->getDimIndex(1), 0);
  PARAM_CHECK_NE(API, scores_desc->getDimIndex(2), 0);
  PARAM_CHECK_NE(API, scores_desc->getDimIndex(3), 0);

  const size_t scores_num = mluOpGetTensorElementNum(scores_desc);
  TENSOR_NUM_CHECK(API, scores_num, LARGE_TENSOR_NUM, "");

  const int64_t n = scores_desc->getDimIndex(0);
  const int64_t h = scores_desc->getDimIndex(1);
  const int64_t w = scores_desc->getDimIndex(2);
  const int64_t a = scores_desc->getDimIndex(3);
  const int64_t hwa = h * w * a;
  if (handle->arch >= 592) {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    cnnlTensorDescriptor_t origin_indices_desc;
    cnnlTensorDescriptor_t sorted_score_desc;
    cnnlTensorDescriptor_t sorted_index_desc;
    CALL_CNNL(cnnlCreateTensorDescriptor(&origin_indices_desc));
    CALL_CNNL(cnnlCreateTensorDescriptor(&sorted_score_desc));
    CALL_CNNL(cnnlCreateTensorDescriptor(&sorted_index_desc));

    const int64_t origin_indices_shape[2] = {n, (int64_t)hwa};
    const int64_t max_k =
        (pre_nms_top_n <= 0 || pre_nms_top_n > hwa) ? hwa : pre_nms_top_n;
    const int64_t sorted_indices_shape[2] = {n, max_k};
    const int64_t sorted_index_shape[2] = {n, max_k};

    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    CALL_CNNL(cnnlSetTensorDescriptor_v2(origin_indices_desc, layout,
                                         CNNL_DTYPE_FLOAT, 2,
                                         origin_indices_shape));
    CALL_CNNL(cnnlSetTensorDescriptor_v2(sorted_score_desc, layout,
                                         CNNL_DTYPE_FLOAT, 2 /*dims:{n,hwa}*/,
                                         sorted_indices_shape));
    CALL_CNNL(cnnlSetTensorDescriptor_v2(sorted_index_desc, layout,
                                         CNNL_DTYPE_INT32, 2 /*dims:{n,hwa}*/,
                                         sorted_index_shape));

    const bool largest = true;  // param for topk, sort from large to small
    const bool sorted = true;   // param for topk, return sorted indices.
    const bool lower_index_first = true;  // param for topk, sort preservation

    size_t topk_workspace_size = 0;
    CALL_CNNL(cnnlGetTopKTensorWorkspaceSize(
        cnnl_handle, origin_indices_desc, max_k, 1, largest, sorted_score_desc,
        sorted_index_desc, &topk_workspace_size));

    CALL_CNNL(cnnlDestroyTensorDescriptor(origin_indices_desc));
    CALL_CNNL(cnnlDestroyTensorDescriptor(sorted_score_desc));
    CALL_CNNL(cnnlDestroyTensorDescriptor(sorted_index_desc));
    DESTROY_CNNL_HANDLE(cnnl_handle);
    size_t data_size = 0;
    mluOpGetSizeOfDataType(scores_desc->getDtype(), &data_size);

    const size_t topk_workspace_size_align =
        PAD_UP(topk_workspace_size, GDRAM_ALIGN_SIZE);
    const size_t nhwa_size_align =
        PAD_UP(n * max_k * data_size, GDRAM_ALIGN_SIZE);
    // topk_workspace_size_align: workspace be used in cnnlTopKTensor_v3.
    // 2 * nhwa_size_align: workspace be used to store scores and indexes after
    // topk.
    // 10 * hwa: workspace be used to store proposals score and box.
    // handle->core_num_per_cluster * data_size: workspace be used to store per
    // core proposals_num.
    *size = topk_workspace_size_align + 2 * nhwa_size_align +
            10 * hwa * data_size + handle->core_num_per_cluster * 3 * data_size;
  } else {
    // 4 * hwa: workspace be used store proposals_box/scores
    // 8 * hwa: workspace be used store decode boxes/scores
    *size = 12 * hwa * 4 + handle->core_num_per_cluster * 4 * 3;
  }
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
  PARAM_CHECK(API, scores_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, bbox_deltas_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, im_shape_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, anchors_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, variances_desc->getDtype() == MLUOP_DTYPE_FLOAT);

  PARAM_CHECK(API, rpn_rois_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, rpn_roi_probs_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, rpn_rois_num_desc->getDtype() == MLUOP_DTYPE_INT32);

  // check inputs layout
  PARAM_CHECK(API, scores_desc->getLayout() == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(API, bbox_deltas_desc->getLayout() == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(API, im_shape_desc->getLayout() == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(API, anchors_desc->getLayout() == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(API, variances_desc->getLayout() == MLUOP_LAYOUT_ARRAY);

  PARAM_CHECK(API, rpn_rois_desc->getLayout() == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(API, rpn_roi_probs_desc->getLayout() == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(API, rpn_rois_num_desc->getLayout() == MLUOP_LAYOUT_ARRAY);

  // check inputs shape
  PARAM_CHECK_EQ(API, scores_desc->getDim(), 4);
  const int64_t n = scores_desc->getDimIndex(0);
  const int64_t h = scores_desc->getDimIndex(1);
  const int64_t w = scores_desc->getDimIndex(2);
  const int64_t a = scores_desc->getDimIndex(3);

  // [N,H,W,A4]
  PARAM_CHECK_EQ(API, bbox_deltas_desc->getDim(), 4);
  PARAM_CHECK_EQ(API, bbox_deltas_desc->getDimIndex(0),
                 scores_desc->getDimIndex(0));
  PARAM_CHECK_EQ(API, bbox_deltas_desc->getDimIndex(1),
                 scores_desc->getDimIndex(1));
  PARAM_CHECK_EQ(API, bbox_deltas_desc->getDimIndex(2),
                 scores_desc->getDimIndex(2));
  PARAM_CHECK_EQ(API, bbox_deltas_desc->getDimIndex(3),
                 4 * scores_desc->getDimIndex(3));

  // [N, 2]
  PARAM_CHECK_EQ(API, im_shape_desc->getDim(), 2);
  PARAM_CHECK_EQ(API, im_shape_desc->getDimIndex(0),
                 scores_desc->getDimIndex(0));
  PARAM_CHECK_EQ(API, im_shape_desc->getDimIndex(1), 2);

  // [H, W, A, 4]
  PARAM_CHECK_EQ(API, anchors_desc->getDim(), 4);
  PARAM_CHECK_EQ(API, anchors_desc->getDimIndex(0),
                 scores_desc->getDimIndex(1));
  PARAM_CHECK_EQ(API, anchors_desc->getDimIndex(1),
                 scores_desc->getDimIndex(2));
  PARAM_CHECK_EQ(API, anchors_desc->getDimIndex(2),
                 scores_desc->getDimIndex(3));
  PARAM_CHECK_EQ(API, anchors_desc->getDimIndex(3), 4);

  // [H, W, A, 4]
  PARAM_CHECK_EQ(API, variances_desc->getDim(), 4);
  PARAM_CHECK_EQ(API, variances_desc->getDimIndex(0),
                 scores_desc->getDimIndex(1));
  PARAM_CHECK_EQ(API, variances_desc->getDimIndex(1),
                 scores_desc->getDimIndex(2));
  PARAM_CHECK_EQ(API, variances_desc->getDimIndex(2),
                 scores_desc->getDimIndex(3));
  PARAM_CHECK_EQ(API, variances_desc->getDimIndex(3), 4);

  // check output shape
  PARAM_CHECK_EQ(API, rpn_rois_desc->getDim(), 2);
  PARAM_CHECK_EQ(API, rpn_rois_desc->getDimIndex(0),
                 scores_desc->getDimIndex(0) * post_nms_top_n);
  PARAM_CHECK_EQ(API, rpn_rois_desc->getDimIndex(1), 4);

  PARAM_CHECK_EQ(API, rpn_roi_probs_desc->getDim(), 2);
  PARAM_CHECK_EQ(API, rpn_roi_probs_desc->getDimIndex(0),
                 scores_desc->getDimIndex(0) * post_nms_top_n);
  PARAM_CHECK_EQ(API, rpn_roi_probs_desc->getDimIndex(1), 1);

  PARAM_CHECK_EQ(API, rpn_rois_num_desc->getDim(), 1);
  PARAM_CHECK_EQ(API, rpn_rois_num_desc->getDimIndex(0),
                 scores_desc->getDimIndex(0));

  PARAM_CHECK_NE(API, scores_desc->getDimIndex(1), 0);
  PARAM_CHECK_NE(API, scores_desc->getDimIndex(2), 0);
  PARAM_CHECK_NE(API, scores_desc->getDimIndex(3), 0);

  // check stride
  STRIDE_TENSOR_CHECK(API + ":", scores_desc, "scores_desc must be contiguous");
  STRIDE_TENSOR_CHECK(API + ":", bbox_deltas_desc,
                      "bbox_deltas_desc must be contiguous");
  STRIDE_TENSOR_CHECK(API + ":", im_shape_desc,
                      "im_shape_desc must be contiguous");
  STRIDE_TENSOR_CHECK(API + ":", anchors_desc,
                      "anchors_desc must be contiguous");
  STRIDE_TENSOR_CHECK(API + ":", variances_desc,
                      "variances_desc must be contiguous");
  STRIDE_TENSOR_CHECK(API + ":", rpn_rois_desc,
                      "rpn_rois_desc must be contiguous");
  STRIDE_TENSOR_CHECK(API + ":", rpn_roi_probs_desc,
                      "rpn_roi_probs_desc must be contiguous");
  STRIDE_TENSOR_CHECK(API + ":", rpn_rois_num_desc,
                      "rpn_rois_num_desc must be contiguous");

  if (n == 0) {
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

  const size_t scores_num = mluOpGetTensorElementNum(scores_desc);
  const size_t bbox_deltas_num = mluOpGetTensorElementNum(bbox_deltas_desc);
  const size_t anchors_num = mluOpGetTensorElementNum(anchors_desc);
  const size_t variances_num = mluOpGetTensorElementNum(variances_desc);
  TENSOR_NUM_CHECK(API, scores_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, bbox_deltas_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, anchors_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, variances_num, LARGE_TENSOR_NUM, "");

  // generate prototxt
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("generate_proposals_v2", "GENERATE_PROPOSALS_V2");
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

  VLOG(5) << "H : " << h;
  VLOG(5) << "W : " << w;
  VLOG(5) << "A : " << a;
  VLOG(5) << "N : " << n;
  const size_t hwa = h * w * a;
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, &k_dim, &k_type, hwa);
  VLOG(5) << "Launch Kernel KernelGenerateProposalsV2 <<<k_dim: " << k_type
          << ", " << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>";

  if (handle->arch >= 592) {
    // 1. call cnnlTopk to sort indices: +nan > +inf > number  > -inf > -nan
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    cnnlTensorDescriptor_t origin_indices_desc;
    cnnlTensorDescriptor_t sorted_score_desc;
    cnnlTensorDescriptor_t sorted_index_desc;
    CALL_CNNL(cnnlCreateTensorDescriptor(&origin_indices_desc));
    CALL_CNNL(cnnlCreateTensorDescriptor(&sorted_score_desc));
    CALL_CNNL(cnnlCreateTensorDescriptor(&sorted_index_desc));

    const int64_t origin_indices_shape[2] = {n, (int64_t)hwa};
    const int64_t max_k =
        (pre_nms_top_n <= 0 || pre_nms_top_n > hwa) ? hwa : pre_nms_top_n;
    const int64_t sorted_indices_shape[2] = {n, max_k};
    const int64_t sorted_index_shape[2] = {n, max_k};

    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    CALL_CNNL(cnnlSetTensorDescriptor_v2(origin_indices_desc, layout,
                                         CNNL_DTYPE_FLOAT, 2,
                                         origin_indices_shape));
    CALL_CNNL(cnnlSetTensorDescriptor_v2(
        sorted_score_desc, layout, CNNL_DTYPE_FLOAT, 2, sorted_indices_shape));
    CALL_CNNL(cnnlSetTensorDescriptor_v2(
        sorted_index_desc, layout, CNNL_DTYPE_INT32, 2, sorted_index_shape));

    // need topk scores
    const bool largest = true;  // param for topk, sort from large to small
    const bool sorted = true;   // param for topk, return sorted indices.
    const bool lower_index_first = true;  // param for topk, sort preservation

    size_t topk_workspace_size = 0;
    CALL_CNNL(cnnlGetTopKTensorWorkspaceSize(
        cnnl_handle, origin_indices_desc, max_k, 1, largest, sorted_score_desc,
        sorted_index_desc, &topk_workspace_size));

    size_t tok_workspace_align_size =
        PAD_UP(topk_workspace_size, GDRAM_ALIGN_SIZE);
    size_t data_size = 0;
    mluOpGetSizeOfDataType(scores_desc->getDtype(), &data_size);

    const size_t indices_size = PAD_UP(n * max_k * data_size, GDRAM_ALIGN_SIZE);
    void *sorted_score =
        (void *)((int8_t *)workspace + tok_workspace_align_size);
    void *sorted_index = (void *)((int8_t *)sorted_score + indices_size);

    // call cnnlTopK
    CALL_CNNL(cnnlTopKTensor_v3(
        cnnl_handle, origin_indices_desc, scores, max_k, 1, largest, sorted,
        lower_index_first, workspace, topk_workspace_size, sorted_score_desc,
        sorted_score, sorted_index_desc, sorted_index));

    CALL_CNNL(cnnlDestroyTensorDescriptor(origin_indices_desc));
    CALL_CNNL(cnnlDestroyTensorDescriptor(sorted_score_desc));
    CALL_CNNL(cnnlDestroyTensorDescriptor(sorted_index_desc));
    DESTROY_CNNL_HANDLE(cnnl_handle);
    void *workspace_buffer = (void *)((int8_t *)sorted_index + indices_size);
    CHECK_RETURN(
        "[mluOpGenerateProposalsV2]",
        KernelGenerateProposalsV2(
            k_dim, k_type, handle->queue, (float *)sorted_score,
            (int32_t *)sorted_index, (float *)bbox_deltas, (float *)im_shape,
            (float *)anchors, (float *)variances, (float *)workspace_buffer,
            (float *)rpn_rois, (float *)rpn_roi_probs, (int *)rpn_rois_num,
            (int *)rpn_rois_batch_size, pre_nms_top_n, post_nms_top_n,
            nms_thresh, min_size, eta, pixel_offset, n, a, h, w));
  } else {
    CHECK_RETURN("[mluOpGenerateProposalsV2]",
                 KernelGenerateProposalsV2_Default(
                     k_dim, k_type, handle->queue, (float *)scores,
                     (float *)bbox_deltas, (float *)im_shape, (float *)anchors,
                     (float *)variances, (float *)workspace, (float *)rpn_rois,
                     (float *)rpn_roi_probs, (int *)rpn_rois_num,
                     (int *)rpn_rois_batch_size, pre_nms_top_n, post_nms_top_n,
                     nms_thresh, min_size, eta, pixel_offset, n, a, h, w));
  }
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
