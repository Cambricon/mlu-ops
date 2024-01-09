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

typedef std::tuple<int, float, int, bool, float, bool, float> YoloBoxDescParam;
typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, YoloBoxDescParam,
                   mluOpDevType_t, mluOpStatus_t>
    YoloBoxParam;
class yolo_box_general : public testing::TestWithParam<YoloBoxParam> {
 public:
  void SetUp() {
    device_ = std::get<6>(GetParam());
    expected_status_ = std::get<7>(GetParam());
    MLUOP_CHECK(mluOpCreate(&handle_));
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      return;
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&x_desc_));
    MLUOpTensorParam x_params = std::get<0>(GetParam());
    mluOpTensorLayout_t x_layout = x_params.get_layout();
    mluOpDataType_t x_dtype = x_params.get_dtype();
    int x_dim = x_params.get_dim_nb();
    std::vector<int> x_dim_size = x_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(x_desc_, x_layout, x_dtype, x_dim,
                                         x_dim_size.data()));
    uint64_t x_ele_num = mluOpGetTensorElementNum(x_desc_);
    uint64_t x_bytes = mluOpDataTypeBytes(x_dtype) * x_ele_num;
    if (x_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&x_, x_bytes))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&img_size_desc_));
    MLUOpTensorParam img_params = std::get<1>(GetParam());
    mluOpTensorLayout_t img_layout = img_params.get_layout();
    mluOpDataType_t img_dtype = img_params.get_dtype();
    int img_dim = img_params.get_dim_nb();
    std::vector<int> img_dim_size = img_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(img_size_desc_, img_layout, img_dtype,
                                         img_dim, img_dim_size.data()));
    uint64_t img_ele_num = mluOpGetTensorElementNum(img_size_desc_);
    uint64_t img_bytes = mluOpDataTypeBytes(img_dtype) * img_ele_num;
    if (img_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&img_size_, img_bytes));
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&anchors_desc_));
    MLUOpTensorParam an_params = std::get<2>(GetParam());
    mluOpTensorLayout_t an_layout = an_params.get_layout();
    mluOpDataType_t an_dtype = an_params.get_dtype();
    int an_dim = an_params.get_dim_nb();
    std::vector<int> an_dim_size = an_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(anchors_desc_, an_layout, an_dtype,
                                         an_dim, an_dim_size.data()));
    uint64_t an_ele_num = mluOpGetTensorElementNum(anchors_desc_);
    uint64_t an_bytes = mluOpDataTypeBytes(an_dtype) * an_ele_num;
    if (an_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&anchors_, an_bytes));
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&boxes_desc_));
    MLUOpTensorParam b_params = std::get<3>(GetParam());
    mluOpTensorLayout_t b_layout = b_params.get_layout();
    mluOpDataType_t b_dtype = b_params.get_dtype();
    int b_dim = b_params.get_dim_nb();
    std::vector<int> b_dim_size = b_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(boxes_desc_, b_layout, b_dtype, b_dim,
                                         b_dim_size.data()));
    uint64_t b_ele_num = mluOpGetTensorElementNum(boxes_desc_);
    uint64_t b_bytes = mluOpDataTypeBytes(b_dtype) * b_ele_num;
    if (b_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&boxes_, b_bytes));
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&scores_desc_));
    MLUOpTensorParam s_params = std::get<4>(GetParam());
    mluOpTensorLayout_t s_layout = s_params.get_layout();
    mluOpDataType_t s_dtype = s_params.get_dtype();
    int s_dim = s_params.get_dim_nb();
    std::vector<int> s_dim_size = s_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(scores_desc_, s_layout, s_dtype, s_dim,
                                         s_dim_size.data()));
    uint64_t s_ele_num = mluOpGetTensorElementNum(scores_desc_);
    uint64_t s_bytes = mluOpDataTypeBytes(s_dtype) * s_ele_num;
    if (s_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&scores_, s_bytes));
    }

    YoloBoxDescParam params = std::get<5>(GetParam());
    std::tie(class_num_, conf_thresh_, downsample_ratio_, clip_bbox_, scale_,
             iou_aware_, iou_aware_factor_) = params;
  }

  bool compute() {
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpYoloBox(
        handle_, x_desc_, x_, img_size_desc_, img_size_, anchors_desc_,
        anchors_, class_num_, conf_thresh_, downsample_ratio_, clip_bbox_,
        scale_, iou_aware_, iou_aware_factor_, boxes_desc_, boxes_,
        scores_desc_, scores_);
    destroy();
    return expected_status_ == status;
  }

 protected:
  void destroy() {
    VLOG(4) << "Destroy parameters.";
    if (handle_) {
      CNRT_CHECK(cnrtQueueSync(handle_->queue));
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = NULL;
    }
    if (x_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(x_desc_));
      x_desc_ = NULL;
    }
    if (img_size_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(img_size_desc_));
      img_size_desc_ = NULL;
    }
    if (anchors_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(anchors_desc_));
      anchors_desc_ = NULL;
    }
    if (boxes_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(boxes_desc_));
      boxes_desc_ = NULL;
    }
    if (scores_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(scores_desc_));
      scores_desc_ = NULL;
    }
    if (x_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(x_));
      x_ = NULL;
    }
    if (img_size_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(img_size_));
      img_size_ = NULL;
    }
    if (anchors_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(anchors_));
      anchors_ = NULL;
    }
    if (boxes_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(boxes_));
      boxes_ = NULL;
    }
    if (scores_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(scores_));
      scores_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t x_desc_ = NULL;
  mluOpTensorDescriptor_t img_size_desc_ = NULL;
  mluOpTensorDescriptor_t anchors_desc_ = NULL;
  mluOpTensorDescriptor_t boxes_desc_ = NULL;
  mluOpTensorDescriptor_t scores_desc_ = NULL;
  void* x_ = NULL;
  void* img_size_ = NULL;
  void* anchors_ = NULL;
  void* boxes_ = NULL;
  void* scores_ = NULL;
  int class_num_ = 10;
  float conf_thresh_ = 0.1;
  int downsample_ratio_ = 16;
  bool clip_bbox_ = true;
  float scale_ = 0.5;
  bool iou_aware_ = true;
  float iou_aware_factor_ = 0.5;
  mluOpDevType_t device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(yolo_box_general, api_test) { EXPECT_TRUE(compute()); }

INSTANTIATE_TEST_CASE_P(
    zero_element_0, yolo_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 160, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({0, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({20}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 10, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 10, 10, 9}))),
        testing::Values(YoloBoxDescParam{10, 0.1, 16, true, 0.5, true, 0.5}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_1, yolo_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 160, 0, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({20}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 4, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 0}))),
        testing::Values(YoloBoxDescParam{10, 0.1, 16, true, 0.5, true, 0.5}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_2, yolo_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 160, 3, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({20}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 4, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 0}))),
        testing::Values(YoloBoxDescParam{10, 0.1, 16, true, 0.5, true, 0.5}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_3, yolo_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 0, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 0, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 0, 10, 9}))),
        testing::Values(YoloBoxDescParam{10, 0.1, 16, true, 0.5, true, 0.5}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_x_shape_0, yolo_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 150, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({20}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 9}))),
        testing::Values(YoloBoxDescParam{10, 0.1, 16, true, 0.5, true, 0.5}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_x_shape_1, yolo_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 160, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 9}))),
        testing::Values(YoloBoxDescParam{10, 0.1, 16, true, 0.5, false, 0.5}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_class_num_0, yolo_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 60, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 0, 9}))),
        testing::Values(YoloBoxDescParam{0, 0.1, 16, true, 0.5, true, 0.5}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_x_dtype_shape_0, yolo_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         4, std::vector<int>({2, 160, 3, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 160, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({20}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 9}))),
        testing::Values(YoloBoxDescParam{10, 0.1, 16, true, 0.5, true, 0.5}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_img_size_dtype_shape_0, yolo_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 160, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         2, std::vector<int>({2, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({20}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 9}))),
        testing::Values(YoloBoxDescParam{10, 0.1, 16, true, 0.5, true, 0.5}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_anchors_dtype_shape_0, yolo_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 160, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         1, std::vector<int>({20})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({19}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 9}))),
        testing::Values(YoloBoxDescParam{10, 0.1, 16, true, 0.5, true, 0.5}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_boxes_dtype_shape_0, yolo_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 160, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({20}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         4, std::vector<int>({2, 10, 4, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 10, 4, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 9, 4, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 5, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 4, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 9}))),
        testing::Values(YoloBoxDescParam{10, 0.1, 16, true, 0.5, true, 0.5}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_scores_dtype_shape_0, yolo_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 160, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({20}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         4, std::vector<int>({2, 10, 10, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 10, 10, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 9, 10, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 11, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 10}))),
        testing::Values(YoloBoxDescParam{10, 0.1, 16, true, 0.5, true, 0.5}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));
}  // namespace mluopapitest
