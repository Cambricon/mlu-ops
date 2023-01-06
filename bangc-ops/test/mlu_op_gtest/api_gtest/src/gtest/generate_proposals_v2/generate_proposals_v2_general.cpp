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
#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include "api_test_tools.h"
#include "core/context.h"
#include "core/tensor.h"
#include "core/logging.h"
#include "gtest/gtest.h"
#include "mlu_op.h"

namespace mluopapitest {
typedef std::tuple<int, int, float, float, float, bool>
    GenerateProposalsV2Param;
typedef std::tuple<mluOpDevType_t, mluOpStatus_t> PublicParam;

typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, GenerateProposalsV2Param,
                   PublicParam>
    GenerateProposalsV2;
class generate_proposals_v2_general
    : public testing::TestWithParam<GenerateProposalsV2> {
 public:
  void SetUp() {
    MLUOP_CHECK(mluOpCreate(&handle_));
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      return;
    }
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&scores_desc_));
    MLUOpTensorParam scores_params = std::get<0>(GetParam());
    mluOpTensorLayout_t scores_layout = scores_params.get_layout();
    mluOpDataType_t scores_dtype = scores_params.get_dtype();
    int scores_dim = scores_params.get_dim_nb();
    std::vector<int> scores_dim_size = scores_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(scores_desc_, scores_layout,
                                         scores_dtype, scores_dim,
                                         scores_dim_size.data()));
    uint64_t scores_ele_num = mluOpGetTensorElementNum(scores_desc_);
    uint64_t scores_bytes = mluOpDataTypeBytes(scores_dtype) * scores_ele_num;
    if (scores_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&scores_, scores_bytes))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&bbox_deltas_desc_));
    MLUOpTensorParam bbox_deltas_params = std::get<1>(GetParam());
    mluOpTensorLayout_t bbox_deltas_layout = bbox_deltas_params.get_layout();
    mluOpDataType_t bbox_deltas_dtype = bbox_deltas_params.get_dtype();
    int bbox_deltas_dim = bbox_deltas_params.get_dim_nb();
    std::vector<int> bbox_deltas_dim_size = bbox_deltas_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(bbox_deltas_desc_, bbox_deltas_layout,
                                         bbox_deltas_dtype, bbox_deltas_dim,
                                         bbox_deltas_dim_size.data()));
    uint64_t bbox_deltas_ele_num = mluOpGetTensorElementNum(bbox_deltas_desc_);
    uint64_t bbox_deltas_bytes =
        mluOpDataTypeBytes(bbox_deltas_dtype) * bbox_deltas_ele_num;
    if (bbox_deltas_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&bbox_deltas_, bbox_deltas_bytes))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&im_shape_desc_));
    MLUOpTensorParam im_shape_params = std::get<2>(GetParam());
    mluOpTensorLayout_t im_shape_layout = im_shape_params.get_layout();
    mluOpDataType_t im_shape_dtype = im_shape_params.get_dtype();
    int im_shape_dim = im_shape_params.get_dim_nb();
    std::vector<int> im_shape_dim_size = im_shape_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(im_shape_desc_, im_shape_layout,
                                         im_shape_dtype, im_shape_dim,
                                         im_shape_dim_size.data()));
    uint64_t im_shape_ele_num = mluOpGetTensorElementNum(im_shape_desc_);
    uint64_t im_shape_bytes =
        mluOpDataTypeBytes(im_shape_dtype) * im_shape_ele_num;
    if (im_shape_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&im_shape_, im_shape_bytes))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&anchors_desc_));
    MLUOpTensorParam anchors_params = std::get<3>(GetParam());
    mluOpTensorLayout_t anchors_layout = anchors_params.get_layout();
    mluOpDataType_t anchors_dtype = anchors_params.get_dtype();
    int anchors_dim = anchors_params.get_dim_nb();
    std::vector<int> anchors_dim_size = anchors_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(anchors_desc_, anchors_layout,
                                         anchors_dtype, anchors_dim,
                                         anchors_dim_size.data()));
    uint64_t anchors_ele_num = mluOpGetTensorElementNum(anchors_desc_);
    uint64_t anchors_bytes =
        mluOpDataTypeBytes(anchors_dtype) * anchors_ele_num;
    if (anchors_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&anchors_, anchors_bytes))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&variances_desc_));
    MLUOpTensorParam variances_params = std::get<4>(GetParam());
    mluOpTensorLayout_t variances_layout = variances_params.get_layout();
    mluOpDataType_t variances_dtype = variances_params.get_dtype();
    int variances_dim = variances_params.get_dim_nb();
    std::vector<int> variances_dim_size = variances_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(variances_desc_, variances_layout,
                                         variances_dtype, variances_dim,
                                         variances_dim_size.data()));
    uint64_t variances_ele_num = mluOpGetTensorElementNum(variances_desc_);
    uint64_t variances_bytes =
        mluOpDataTypeBytes(variances_dtype) * variances_ele_num;
    if (variances_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&variances_, variances_bytes))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&rpn_rois_desc_));
    MLUOpTensorParam rpn_rois_params = std::get<5>(GetParam());
    mluOpTensorLayout_t rpn_rois_layout = rpn_rois_params.get_layout();
    mluOpDataType_t rpn_rois_dtype = rpn_rois_params.get_dtype();
    int rpn_rois_dim = rpn_rois_params.get_dim_nb();
    std::vector<int> rpn_rois_dim_size = rpn_rois_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(rpn_rois_desc_, rpn_rois_layout,
                                         rpn_rois_dtype, rpn_rois_dim,
                                         rpn_rois_dim_size.data()));
    uint64_t rpn_rois_ele_num = mluOpGetTensorElementNum(rpn_rois_desc_);
    uint64_t rpn_rois_bytes =
        mluOpDataTypeBytes(rpn_rois_dtype) * rpn_rois_ele_num;
    if (rpn_rois_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&rpn_rois_, rpn_rois_bytes))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&rpn_roi_probs_desc_));
    MLUOpTensorParam rpn_roi_probs_params = std::get<6>(GetParam());
    mluOpTensorLayout_t rpn_roi_probs_layout =
        rpn_roi_probs_params.get_layout();
    mluOpDataType_t rpn_roi_probs_dtype = rpn_roi_probs_params.get_dtype();
    int rpn_roi_probs_dim = rpn_roi_probs_params.get_dim_nb();
    std::vector<int> rpn_roi_probs_dim_size =
        rpn_roi_probs_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(
        rpn_roi_probs_desc_, rpn_roi_probs_layout, rpn_roi_probs_dtype,
        rpn_roi_probs_dim, rpn_roi_probs_dim_size.data()));
    uint64_t rpn_roi_probs_ele_num =
        mluOpGetTensorElementNum(rpn_roi_probs_desc_);
    uint64_t rpn_roi_probs_bytes =
        mluOpDataTypeBytes(rpn_roi_probs_dtype) * rpn_roi_probs_ele_num;
    if (rpn_roi_probs_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&rpn_roi_probs_, rpn_roi_probs_bytes))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&rpn_rois_num_desc_));
    MLUOpTensorParam rpn_rois_num_params = std::get<7>(GetParam());
    mluOpTensorLayout_t rpn_rois_num_layout = rpn_rois_num_params.get_layout();
    mluOpDataType_t rpn_rois_num_dtype = rpn_rois_num_params.get_dtype();
    int rpn_rois_num_dim = rpn_rois_num_params.get_dim_nb();
    std::vector<int> rpn_rois_num_dim_size = rpn_rois_num_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(
        rpn_rois_num_desc_, rpn_rois_num_layout, rpn_rois_num_dtype,
        rpn_rois_num_dim, rpn_rois_num_dim_size.data()));
    uint64_t rpn_rois_num_ele_num =
        mluOpGetTensorElementNum(rpn_rois_num_desc_);
    uint64_t rpn_rois_num_bytes =
        mluOpDataTypeBytes(rpn_rois_num_dtype) * rpn_rois_num_ele_num;
    if (rpn_rois_num_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&rpn_rois_num_, rpn_rois_num_bytes))
    }

    GTEST_CHECK(CNRT_RET_SUCCESS ==
                cnrtMalloc(&rpn_rois_batch_size_,
                           mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));

    GenerateProposalsV2Param generateProposalsV2Param = std::get<8>(GetParam());
    std::tie(pre_nms_top_n_, post_nms_top_n_, nms_thresh_, min_size_, eta_,
             pixel_offset_) = generateProposalsV2Param;

    PublicParam publicParam = std::get<9>(GetParam());
    std::tie(device_, expected_status_) = publicParam;
  }

  bool compute() {
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpGetGenerateProposalsV2WorkspaceSize(
        handle_, scores_desc_, &workspace_size_);
    if (MLUOP_STATUS_SUCCESS != status) {
      destroy();
      return status == expected_status_;
    }
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_));
    status = mluOpGenerateProposalsV2(
        handle_, pre_nms_top_n_, post_nms_top_n_, nms_thresh_, min_size_, eta_,
        pixel_offset_, scores_desc_, scores_, bbox_deltas_desc_, bbox_deltas_,
        im_shape_desc_, im_shape_, anchors_desc_, anchors_, variances_desc_,
        variances_, workspace_, workspace_size_, rpn_rois_desc_, rpn_rois_,
        rpn_roi_probs_desc_, rpn_roi_probs_, rpn_rois_num_desc_, rpn_rois_num_,
        rpn_rois_batch_size_);
    destroy();
    return status == expected_status_;
  }

 protected:
  void destroy() {
    if (handle_) {
      CNRT_CHECK(cnrtQueueSync(handle_->queue));
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = NULL;
    }
    if (scores_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(scores_desc_));
      scores_desc_ = NULL;
    }
    if (scores_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(scores_));
      scores_ = NULL;
    }
    if (bbox_deltas_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(bbox_deltas_desc_));
      bbox_deltas_desc_ = NULL;
    }
    if (bbox_deltas_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(bbox_deltas_));
      bbox_deltas_ = NULL;
    }
    if (im_shape_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(im_shape_desc_));
      im_shape_desc_ = NULL;
    }
    if (im_shape_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(im_shape_));
      im_shape_ = NULL;
    }
    if (anchors_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(anchors_desc_));
      anchors_desc_ = NULL;
    }
    if (anchors_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(anchors_));
      anchors_ = NULL;
    }
    if (variances_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(variances_desc_));
      variances_desc_ = NULL;
    }
    if (variances_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(variances_));
      variances_ = NULL;
    }
    if (workspace_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = NULL;
    }
    if (rpn_rois_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(rpn_rois_desc_));
      rpn_rois_desc_ = NULL;
    }
    if (rpn_rois_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(rpn_rois_));
      rpn_rois_ = NULL;
    }
    if (rpn_roi_probs_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(rpn_roi_probs_desc_));
      rpn_roi_probs_desc_ = NULL;
    }
    if (rpn_roi_probs_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(rpn_roi_probs_));
      rpn_roi_probs_ = NULL;
    }
    if (rpn_rois_num_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(rpn_rois_num_desc_));
      rpn_rois_num_desc_ = NULL;
    }
    if (rpn_rois_num_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(rpn_rois_num_));
      rpn_rois_num_ = NULL;
    }
    if (rpn_rois_batch_size_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(rpn_rois_batch_size_));
      rpn_rois_batch_size_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  int pre_nms_top_n_ = 15;
  int post_nms_top_n_ = 5;
  float nms_thresh_ = 0.8;
  float min_size_ = 4;
  float eta_ = 3;
  bool pixel_offset_ = false;
  mluOpTensorDescriptor_t scores_desc_ = NULL;
  void* scores_ = NULL;
  mluOpTensorDescriptor_t bbox_deltas_desc_ = NULL;
  void* bbox_deltas_ = NULL;
  mluOpTensorDescriptor_t im_shape_desc_ = NULL;
  void* im_shape_ = NULL;
  mluOpTensorDescriptor_t anchors_desc_ = NULL;
  void* anchors_ = NULL;
  mluOpTensorDescriptor_t variances_desc_ = NULL;
  void* variances_ = NULL;
  void* workspace_ = NULL;
  size_t workspace_size_ = 64;
  mluOpTensorDescriptor_t rpn_rois_desc_ = NULL;
  void* rpn_rois_ = NULL;
  mluOpTensorDescriptor_t rpn_roi_probs_desc_ = NULL;
  void* rpn_roi_probs_ = NULL;
  mluOpTensorDescriptor_t rpn_rois_num_desc_ = NULL;
  void* rpn_rois_num_ = NULL;
  void* rpn_rois_batch_size_ = NULL;
  mluOpDevType_t device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(generate_proposals_v2_general, api_test) {
  try {
    EXPECT_TRUE(compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

INSTANTIATE_TEST_CASE_P(
    zero_element_1, generate_proposals_v2_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 16, 16, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 16, 16, 32}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0}))),
        testing::Values(GenerateProposalsV2Param{10, 5, 0.5, 4, 3, 0}),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_SUCCESS})));

INSTANTIATE_TEST_CASE_P(
    zero_element_2, generate_proposals_v2_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 0, 0, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 0, 0, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 0, 0, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 0, 0, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({2}))),
        testing::Values(GenerateProposalsV2Param{10, 5, 0.5, 4, 3, 0}),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_rpn_rois_num_dtype, generate_proposals_v2_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 32}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({2}))),
        testing::Values(GenerateProposalsV2Param{10, 5, 0.5, 4, 3, 0}),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_scores_shape, generate_proposals_v2_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 16, 16, 8})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 13})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 17, 16, 8})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 17, 8})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5,
                                         std::vector<int>({2, 16, 16, 8, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 16, 16}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 32}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({2}))),
        testing::Values(GenerateProposalsV2Param{10, 5, 0.5, 4, 3, 0}),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_bbox_deltas_shape, generate_proposals_v2_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 16, 16, 32})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 33})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 17, 16, 32})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 17, 32})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5,
                                         std::vector<int>({2, 16, 16, 32, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 16, 16}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({2}))),
        testing::Values(GenerateProposalsV2Param{10, 5, 0.5, 4, 3, 0}),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_im_shape_shape, generate_proposals_v2_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 32}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 2, 6})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({2}))),
        testing::Values(GenerateProposalsV2Param{10, 5, 0.5, 4, 3, 0}),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_anchors_shape, generate_proposals_v2_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 32}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 9, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({17, 16, 8, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 17, 8, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5,
                                         std::vector<int>({16, 16, 8, 4, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({16, 16, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({2}))),
        testing::Values(GenerateProposalsV2Param{10, 5, 0.5, 4, 3, 0}),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_variances_shape, generate_proposals_v2_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 32}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 9, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({17, 16, 8, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 17, 8, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5,
                                         std::vector<int>({16, 16, 4, 8, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({16, 16, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({2}))),
        testing::Values(GenerateProposalsV2Param{10, 5, 0.5, 4, 3, 0}),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_rpn_rois_shape, generate_proposals_v2_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 32}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({10, 4, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({2}))),
        testing::Values(GenerateProposalsV2Param{10, 5, 0.5, 4, 3, 0}),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_input_shape, generate_proposals_v2_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 32}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({2}))),
        testing::Values(GenerateProposalsV2Param{10, 5, 0.5, 4, 3, 0}),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_eta_value, generate_proposals_v2_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 32}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({2}))),
        testing::Values(GenerateProposalsV2Param{10, 5, 0.5, 4, 0, 0}),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_nms_thresh_value, generate_proposals_v2_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 16, 32}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({16, 16, 8, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({2}))),
        testing::Values(GenerateProposalsV2Param{10, 5, -0.5, 4, 3, 0}),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

}  // namespace mluopapitest
