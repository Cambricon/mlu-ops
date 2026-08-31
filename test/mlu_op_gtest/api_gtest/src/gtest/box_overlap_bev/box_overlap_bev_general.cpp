/*************************************************************************
 * Copyright (C) [2026] by Cambricon, Inc.
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

#include "gtest/gtest.h"
#include "mlu_op.h"
#include "core/context.h"
#include "core/logging.h"
#include "api_test_tools.h"
#include "core/tensor.h"

namespace mluopapitest {

// boxes1_desc, boxes2_desc, overlaps_desc
typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   mluOpDevType_t, mluOpStatus_t>
    BoxOverlapBevParam;

class box_overlap_bev_general
    : public testing::TestWithParam<BoxOverlapBevParam> {
 public:
  void SetUp() {
    try {
      target_device_ = std::get<3>(GetParam());
      expected_status_ = std::get<4>(GetParam());
      MLUOP_CHECK(mluOpCreate(&handle_));

      mluOpTensorLayout_t boxes1_layout;
      mluOpDataType_t boxes1_dtype;
      int boxes1_dim_nb;
      std::vector<int> boxes1_dims;
      MLUOpTensorParam boxes1_params = std::get<0>(GetParam());
      //   std::tie(boxes1_layout, boxes1_dtype, boxes1_dim_nb, boxes1_dims) =
      //   boxes1_params;
      boxes1_layout = boxes1_params.get_layout();
      boxes1_dtype = boxes1_params.get_dtype();
      boxes1_dim_nb = boxes1_params.get_dim_nb();
      boxes1_dims = boxes1_params.get_dim_size();
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&boxes1_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(boxes1_desc_, boxes1_layout,
                                           boxes1_dtype, boxes1_dim_nb,
                                           boxes1_dims.data()));
      GTEST_CHECK(
          cnrtSuccess ==
          cnrtMalloc(&boxes1_, mluOpDataTypeBytes(boxes1_dtype) *
                                   mluOpGetTensorElementNum(boxes1_desc_)));

      mluOpTensorLayout_t boxes2_layout;
      mluOpDataType_t boxes2_dtype;
      int boxes2_dim_nb;
      std::vector<int> boxes2_dims;
      MLUOpTensorParam boxes2_params = std::get<1>(GetParam());
      //   std::tie(boxes2_layout, boxes2_dtype, boxes2_dim_nb, boxes2_dims) =
      //   boxes2_params;
      boxes2_layout = boxes2_params.get_layout();
      boxes2_dtype = boxes2_params.get_dtype();
      boxes2_dim_nb = boxes2_params.get_dim_nb();
      boxes2_dims = boxes2_params.get_dim_size();
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&boxes2_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(boxes2_desc_, boxes2_layout,
                                           boxes2_dtype, boxes2_dim_nb,
                                           boxes2_dims.data()));
      GTEST_CHECK(
          cnrtSuccess ==
          cnrtMalloc(&boxes2_, mluOpDataTypeBytes(boxes2_dtype) *
                                   mluOpGetTensorElementNum(boxes2_desc_)));

      mluOpTensorLayout_t overlaps_layout;
      mluOpDataType_t overlaps_dtype;
      int overlaps_dim_nb;
      std::vector<int> overlaps_dims;
      MLUOpTensorParam overlapsDescParam = std::get<2>(GetParam());
      //   std::tie(overlaps_layout, overlaps_dtype, overlaps_dim_nb,
      //   overlaps_dims) = overlapsDescParam;
      overlaps_layout = overlapsDescParam.get_layout();
      overlaps_dtype = overlapsDescParam.get_dtype();
      overlaps_dim_nb = overlapsDescParam.get_dim_nb();
      overlaps_dims = overlapsDescParam.get_dim_size();
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&overlaps_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(overlaps_desc_, overlaps_layout,
                                           overlaps_dtype, overlaps_dim_nb,
                                           overlaps_dims.data()));
      GTEST_CHECK(
          cnrtSuccess ==
          cnrtMalloc(&overlaps_, mluOpDataTypeBytes(boxes1_dtype) *
                                     mluOpGetTensorElementNum(overlaps_desc_)));
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in box_overlap_bev_general";
    }
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status =
        mluOpBoxOverlapBev(handle_, boxes1_desc_, boxes1_, boxes2_desc_,
                           boxes2_, overlaps_desc_, overlaps_);
    destroy();

    return expected_status_ == status;
  }

  void destroy() {
    try {
      if (handle_) {
        CNRT_CHECK(cnrtQueueSync(handle_->queue));
        MLUOP_CHECK(mluOpDestroy(handle_));
        handle_ = nullptr;
      }
      if (boxes1_desc_) {
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(boxes1_desc_));
        boxes1_desc_ = nullptr;
      }
      if (boxes2_desc_) {
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(boxes2_desc_));
        boxes2_desc_ = nullptr;
      }
      if (overlaps_desc_) {
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(overlaps_desc_));
        overlaps_desc_ = nullptr;
      }
      if (boxes1_) {
        GTEST_CHECK(cnrtSuccess == cnrtFree(boxes1_));
        boxes1_ = nullptr;
      }
      if (boxes2_) {
        GTEST_CHECK(cnrtSuccess == cnrtFree(boxes2_));
        boxes2_ = nullptr;
      }
      if (overlaps_) {
        GTEST_CHECK(cnrtSuccess == cnrtFree(overlaps_));
        overlaps_ = nullptr;
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in box_overlap_bev_general";
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t boxes1_desc_ = nullptr;
  void *boxes1_ = nullptr;
  mluOpTensorDescriptor_t boxes2_desc_ = nullptr;
  void *boxes2_ = nullptr;
  mluOpTensorDescriptor_t overlaps_desc_ = nullptr;
  void *overlaps_ = nullptr;
  mluOpDevType_t target_device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(box_overlap_bev_general, negative) { EXPECT_TRUE(compute()); }

INSTANTIATE_TEST_CASE_P(
    zero_element, box_overlap_bev_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 7})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 7})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 0})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    negative_dtype_boxes1, box_overlap_bev_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({2, 7})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 7})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 2})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_dtype_boxes2, box_overlap_bev_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({2, 7})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({2, 7})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 2})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_dtype_overlap, box_overlap_bev_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({2, 2})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_UINT32,
                                         2, std::vector<int>({2, 2})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_shape_boxes1, box_overlap_bev_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 6})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 2})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_shape_boxes2, box_overlap_bev_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 1})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 2})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_shape_overlap, box_overlap_bev_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 7})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 2})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 3})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 2, 2})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({2})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));
}  // namespace mluopapitest
