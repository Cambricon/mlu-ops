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
#include "core/logging.h"
#include "core/tensor.h"
#include "gtest/gtest.h"
#include "mlu_op.h"
#include "core/context.h"

#define LARGE_TENSOR_NUM ((uint64_t)2147483648)

namespace mluopapitest {
typedef std::tuple<int, int, int, int, int> RoiPointPool3dDescParam;
typedef std::tuple<RoiPointPool3dDescParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   mluOpDevType_t, mluOpStatus_t>
    RoiPointPool3dParam;
class roipoint_pool3d_general
    : public testing::TestWithParam<RoiPointPool3dParam> {
 public:
  void SetUp() {
    try {
      MLUOP_CHECK(mluOpCreate(&handle_));

      target_device_ = std::get<6>(GetParam());
      expected_status_ = std::get<7>(GetParam());
      if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
            target_device_ == handle_->arch)) {
        VLOG(4) << "Device does not match, skip testing.";
        return;
      }
      RoiPointPool3dDescParam op_param = std::get<0>(GetParam());
      std::tie(batch_size_, pts_num_, boxes_num_, feature_in_len_,
               sampled_pts_num_) = op_param;

      MLUOpTensorParam points_params = std::get<1>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&points_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          points_desc_, points_params.get_layout(), points_params.get_dtype(),
          points_params.get_dim_nb(), points_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(points_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&points_,
                       mluOpDataTypeBytes(points_params.get_dtype()) * 10));
      } else {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&points_, mluOpDataTypeBytes(points_params.get_dtype()) *
                                     mluOpGetTensorElementNum(points_desc_)));
      }

      MLUOpTensorParam point_features_params = std::get<2>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&point_features_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          point_features_desc_, point_features_params.get_layout(),
          point_features_params.get_dtype(), point_features_params.get_dim_nb(),
          point_features_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(point_features_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(
                &point_features_,
                mluOpDataTypeBytes(point_features_params.get_dtype()) * 10));
      } else {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&point_features_,
                       mluOpDataTypeBytes(point_features_params.get_dtype()) *
                           mluOpGetTensorElementNum(point_features_desc_)));
      }

      MLUOpTensorParam boxes3d_params = std::get<3>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&boxes3d_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          boxes3d_desc_, boxes3d_params.get_layout(),
          boxes3d_params.get_dtype(), boxes3d_params.get_dim_nb(),
          boxes3d_params.get_dim_size().data()));

      if (mluOpGetTensorElementNum(boxes3d_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&boxes3d_,
                       mluOpDataTypeBytes(boxes3d_params.get_dtype()) * 10));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&boxes3d_,
                               mluOpDataTypeBytes(boxes3d_params.get_dtype()) *
                                   mluOpGetTensorElementNum(boxes3d_desc_)));
      }

      MLUOpTensorParam pooled_features_params = std::get<4>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&pooled_features_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          pooled_features_desc_, pooled_features_params.get_layout(),
          pooled_features_params.get_dtype(),
          pooled_features_params.get_dim_nb(),
          pooled_features_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(pooled_features_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(
                &pooled_features_,
                mluOpDataTypeBytes(pooled_features_params.get_dtype()) * 10));
      } else {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&pooled_features_,
                       mluOpDataTypeBytes(pooled_features_params.get_dtype()) *
                           mluOpGetTensorElementNum(pooled_features_desc_)));
      }

      MLUOpTensorParam pooled_empty_flag_params = std::get<5>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&pooled_empty_flag_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          pooled_empty_flag_desc_, pooled_empty_flag_params.get_layout(),
          pooled_empty_flag_params.get_dtype(),
          pooled_empty_flag_params.get_dim_nb(),
          pooled_empty_flag_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(pooled_empty_flag_desc_) >=
          LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(
                &pooled_empty_flag_,
                mluOpDataTypeBytes(pooled_empty_flag_params.get_dtype()) * 10));
      } else {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(
                &pooled_empty_flag_,
                mluOpDataTypeBytes(pooled_empty_flag_params.get_dtype()) *
                    mluOpGetTensorElementNum(pooled_empty_flag_desc_)));
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in roipoint_pool3d_general.";
    }
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status;
    status = mluOpGetRoiPointPool3dWorkspaceSize(
        handle_, batch_size_, pts_num_, boxes_num_, feature_in_len_,
        sampled_pts_num_, points_desc_, point_features_desc_, boxes3d_desc_,
        pooled_features_desc_, pooled_empty_flag_desc_, &workspace_size_);
    if (status != MLUOP_STATUS_SUCCESS) {
      destroy();
      return expected_status_ == status;
    }
    GTEST_CHECK(cnrtSuccess == cnrtMalloc(&workspace_, workspace_size_));

    status = mluOpRoiPointPool3d(
        handle_, batch_size_, pts_num_, boxes_num_, feature_in_len_,
        sampled_pts_num_, points_desc_, points_, point_features_desc_,
        point_features_, boxes3d_desc_, boxes3d_, workspace_, workspace_size_,
        pooled_features_desc_, pooled_features_, pooled_empty_flag_desc_,
        pooled_empty_flag_);
    destroy();
    return status == expected_status_;
  }

  void destroy() {
    if (handle_) {
      CNRT_CHECK(cnrtQueueSync(handle_->queue));
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = NULL;
    }
    if (points_desc_) {
      VLOG(4) << "Destroy points_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(points_desc_));
      points_desc_ = nullptr;
    }

    if (points_) {
      VLOG(4) << "Destroy points_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(points_));
      points_ = nullptr;
    }

    if (point_features_desc_) {
      VLOG(4) << "Destroy point_features_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(point_features_desc_));
      point_features_desc_ = nullptr;
    }

    if (point_features_) {
      VLOG(4) << "Destroy point_features_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(point_features_));
      point_features_ = nullptr;
    }

    if (boxes3d_desc_) {
      VLOG(4) << "Destroy boxes3d_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(boxes3d_desc_));
      boxes3d_desc_ = nullptr;
    }

    if (boxes3d_) {
      VLOG(4) << "Destroy boxes3d_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(boxes3d_));
      boxes3d_ = nullptr;
    }

    if (pooled_features_desc_) {
      VLOG(4) << "Destroy pooled_features_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(pooled_features_desc_));
      pooled_features_desc_ = nullptr;
    }

    if (pooled_features_) {
      VLOG(4) << "Destroy pooled_features_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(pooled_features_));
      pooled_features_ = nullptr;
    }

    if (pooled_empty_flag_desc_) {
      VLOG(4) << "Destroy pooled_empty_flag_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(pooled_empty_flag_desc_));
      pooled_empty_flag_desc_ = nullptr;
    }

    if (pooled_empty_flag_) {
      VLOG(4) << "Destroy pooled_empty_flag_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(pooled_empty_flag_));
      pooled_empty_flag_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  int batch_size_ = 1;
  int pts_num_ = 1;
  int boxes_num_ = 1;
  int feature_in_len_ = 1;
  int sampled_pts_num_ = 1;
  mluOpTensorDescriptor_t points_desc_ = nullptr;
  void *points_ = nullptr;
  mluOpTensorDescriptor_t point_features_desc_ = nullptr;
  void *point_features_ = nullptr;
  mluOpTensorDescriptor_t boxes3d_desc_ = nullptr;
  void *boxes3d_ = nullptr;
  mluOpTensorDescriptor_t pooled_features_desc_ = nullptr;
  void *pooled_features_ = nullptr;
  void *workspace_ = nullptr;
  size_t workspace_size_ = 64;
  mluOpTensorDescriptor_t pooled_empty_flag_desc_ = nullptr;
  void *pooled_empty_flag_ = nullptr;
  mluOpDevType_t target_device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(roipoint_pool3d_general, api_test) {
  try {
    EXPECT_TRUE(compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in roipoint_pool3d_general";
  }
}

INSTANTIATE_TEST_CASE_P(
    zero_element_1, roipoint_pool3d_general,
    testing::Combine(
        testing::Values(RoiPointPool3dDescParam{0, 1, 1, 1, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({0, 1, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({0, 1, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({0, 1, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 1, 1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({0, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_2, roipoint_pool3d_general,
    testing::Combine(
        testing::Values(RoiPointPool3dDescParam{1, 0, 1, 1, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 0, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 0, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 1, 1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_3, roipoint_pool3d_general,
    testing::Combine(
        testing::Values(RoiPointPool3dDescParam{1, 1, 0, 1, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 0, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 0, 1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 0}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_4, roipoint_pool3d_general,
    testing::Combine(
        testing::Values(RoiPointPool3dDescParam{1, 1, 1, 0, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 1, 1, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_5, roipoint_pool3d_general,
    testing::Combine(
        testing::Values(RoiPointPool3dDescParam{1, 1, 1, 1, 0}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 1, 0, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_points_dtype_dim_shape, roipoint_pool3d_general,
    testing::Combine(
        testing::Values(RoiPointPool3dDescParam{1, 1, 1, 1, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 1, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 1, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 1, 1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_point_features_dtype_dim_shape, roipoint_pool3d_general,
    testing::Combine(
        testing::Values(RoiPointPool3dDescParam{1, 1, 1, 1, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 1, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 1, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 1, 1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_boxes3d_dtype_dim_shape, roipoint_pool3d_general,
    testing::Combine(
        testing::Values(RoiPointPool3dDescParam{1, 1, 1, 1, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 1, 7})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 1, 7})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 7})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 1, 1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_pooled_features_dtype_dim_shape, roipoint_pool3d_general,
    testing::Combine(
        testing::Values(RoiPointPool3dDescParam{1, 1, 1, 1, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({1, 1, 1, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 1, 1, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 2, 1, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 1, 2, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 1, 1, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_pooled_empty_flag_dtype_dim_shape, roipoint_pool3d_general,
    testing::Combine(
        testing::Values(RoiPointPool3dDescParam{1, 1, 1, 1, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 1, 1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({1, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 2}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_large_tensor_1, roipoint_pool3d_general,
    testing::Combine(
        testing::Values(RoiPointPool3dDescParam{14800, 48367, 1, 1, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3,
                                         std::vector<int>({14800, 48367, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3,
                                         std::vector<int>({14800, 48367, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({14800, 1, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4,
                                         std::vector<int>({14800, 1, 1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({14800, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    bad_large_tensor_2, roipoint_pool3d_general,
    testing::Combine(
        testing::Values(RoiPointPool3dDescParam{14800, 1, 48367, 1, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({14800, 1, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({14800, 1, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3,
                                         std::vector<int>({14800, 48367, 7}))),
        testing::Values(
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 4,
                             std::vector<int>({14800, 48367, 1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({14800, 48367}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    bad_large_tensor_3, roipoint_pool3d_general,
    testing::Combine(
        testing::Values(RoiPointPool3dDescParam{14800, 1, 1, 1451001, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({14800, 1, 3}))),
        testing::Values(
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 3,
                             std::vector<int>({14800, 1, 1451001}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({14800, 1, 7}))),
        testing::Values(
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 4,
                             std::vector<int>({14800, 1, 1, 1451004}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({14800, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    bad_large_tensor_4, roipoint_pool3d_general,
    testing::Combine(
        testing::Values(RoiPointPool3dDescParam{14800, 1, 1, 1, 48367}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({14800, 1, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({14800, 1, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({14800, 1, 7}))),
        testing::Values(
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 4,
                             std::vector<int>({14800, 1, 48367, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({14800, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

}  // namespace mluopapitest
