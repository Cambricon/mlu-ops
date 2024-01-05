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

typedef std::tuple<int, int, int, int, int, int, int, int>
    RoiawarePool3dForwardParam;

typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   RoiawarePool3dForwardParam, mluOpDevType_t, mluOpStatus_t>
    RoiawarePool3dForward;
class roiaware_pool3d_forward_general
    : public testing::TestWithParam<RoiawarePool3dForward> {
 public:
  void SetUp() {
    try {
      MLUOP_CHECK(mluOpCreate(&handle_));

      MLUOpTensorParam rois_params = std::get<0>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&rois_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          rois_desc_, rois_params.get_layout(), rois_params.get_dtype(),
          rois_params.get_dim_nb(), rois_params.get_dim_size().data()));
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(&rois_, mluOpDataTypeBytes(rois_params.get_dtype()) * 10));

      MLUOpTensorParam pts_params = std::get<1>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&pts_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          pts_desc_, pts_params.get_layout(), pts_params.get_dtype(),
          pts_params.get_dim_nb(), pts_params.get_dim_size().data()));
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(&pts_, mluOpDataTypeBytes(pts_params.get_dtype()) * 10));

      MLUOpTensorParam pts_feature_params = std::get<2>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&pts_feature_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          pts_feature_desc_, pts_feature_params.get_layout(),
          pts_feature_params.get_dtype(), pts_feature_params.get_dim_nb(),
          pts_feature_params.get_dim_size().data()));
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(&pts_feature_,
                     mluOpDataTypeBytes(pts_feature_params.get_dtype()) * 10));

      MLUOpTensorParam pooled_features_params = std::get<3>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&pooled_features_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          pooled_features_desc_, pooled_features_params.get_layout(),
          pooled_features_params.get_dtype(),
          pooled_features_params.get_dim_nb(),
          pooled_features_params.get_dim_size().data()));
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(
              &pooled_features_,
              mluOpDataTypeBytes(pooled_features_params.get_dtype()) * 10));

      MLUOpTensorParam argmax_params = std::get<4>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&argmax_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          argmax_desc_, argmax_params.get_layout(), argmax_params.get_dtype(),
          argmax_params.get_dim_nb(), argmax_params.get_dim_size().data()));
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(&argmax_,
                     mluOpDataTypeBytes(argmax_params.get_dtype()) * 10));

      MLUOpTensorParam pts_idx_of_voxels_params = std::get<5>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&pts_idx_of_voxels_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          pts_idx_of_voxels_desc_, pts_idx_of_voxels_params.get_layout(),
          pts_idx_of_voxels_params.get_dtype(),
          pts_idx_of_voxels_params.get_dim_nb(),
          pts_idx_of_voxels_params.get_dim_size().data()));
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(
              &pts_idx_of_voxels_,
              mluOpDataTypeBytes(pts_idx_of_voxels_params.get_dtype()) * 10));

      RoiawarePool3dForwardParam roiawarePool3dForwardParam =
          std::get<6>(GetParam());
      std::tie(boxes_num_, pts_num_, channels_, max_pts_each_voxel_, out_x_,
               out_y_, out_z_, pool_method_) = roiawarePool3dForwardParam;
      target_device_ = std::get<7>(GetParam());
      expected_status_ = std::get<8>(GetParam());
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in roiaware_pool3d_forward general.";
    }
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status;
    status = mluOpGetRoiawarePool3dForwardWorkspaceSize(
        handle_, rois_desc_, pts_desc_, pts_feature_desc_, &workspace_size_);
    if (status != MLUOP_STATUS_SUCCESS) {
      destroy();
      return expected_status_ == status;
    }
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_));
    status = mluOpRoiawarePool3dForward(
        handle_, pool_method_, boxes_num_, pts_num_, channels_, rois_desc_,
        rois_, pts_desc_, pts_, pts_feature_desc_, pts_feature_, workspace_,
        workspace_size_, max_pts_each_voxel_, out_x_, out_y_, out_z_,
        argmax_desc_, argmax_, pts_idx_of_voxels_desc_, pts_idx_of_voxels_,
        pooled_features_desc_, pooled_features_);
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

      if (rois_desc_) {
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(rois_desc_));
        rois_desc_ = nullptr;
      }

      if (rois_) {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(rois_));
        rois_ = nullptr;
      }

      if (pts_desc_) {
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(pts_desc_));
        pts_desc_ = nullptr;
      }

      if (pts_) {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(pts_));
        pts_ = nullptr;
      }

      if (pts_feature_desc_) {
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(pts_feature_desc_));
        pts_feature_desc_ = nullptr;
      }

      if (pts_feature_) {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(pts_feature_));
        pts_feature_ = nullptr;
      }

      if (workspace_) {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
        workspace_ = nullptr;
      }

      if (argmax_desc_) {
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(argmax_desc_));
        argmax_desc_ = nullptr;
      }

      if (argmax_) {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(argmax_));
        argmax_ = nullptr;
      }

      if (pts_idx_of_voxels_desc_) {
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(pts_idx_of_voxels_desc_));
        pts_idx_of_voxels_desc_ = nullptr;
      }

      if (pts_idx_of_voxels_) {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(pts_idx_of_voxels_));
        pts_idx_of_voxels_ = nullptr;
      }

      if (pooled_features_desc_) {
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(pooled_features_desc_));
        pooled_features_desc_ = nullptr;
      }

      if (pooled_features_) {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(pooled_features_));
        pooled_features_ = nullptr;
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in roiaware_pool3d_forward_general";
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  int pool_method_ = 0;
  int boxes_num_ = 0;
  int pts_num_ = 0;
  int channels_ = 0;
  mluOpTensorDescriptor_t rois_desc_ = nullptr;
  void *rois_ = nullptr;
  mluOpTensorDescriptor_t pts_desc_ = nullptr;
  void *pts_ = nullptr;
  mluOpTensorDescriptor_t pts_feature_desc_ = nullptr;
  void *pts_feature_ = nullptr;
  void *workspace_ = nullptr;
  size_t workspace_size_ = 64;
  int max_pts_each_voxel_ = 0;
  int out_x_ = 0;
  int out_y_ = 0;
  int out_z_ = 0;
  mluOpTensorDescriptor_t argmax_desc_ = nullptr;
  void *argmax_ = nullptr;
  mluOpTensorDescriptor_t pts_idx_of_voxels_desc_ = nullptr;
  void *pts_idx_of_voxels_ = nullptr;
  mluOpTensorDescriptor_t pooled_features_desc_ = nullptr;
  void *pooled_features_ = nullptr;
  mluOpDevType_t target_device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(roiaware_pool3d_forward_general, negative) { EXPECT_TRUE(compute()); }

INSTANTIATE_TEST_CASE_P(
    zero_element_1, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({0, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({0, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({0, 1, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{0, 3, 4, 5, 1, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_2, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{2, 0, 4, 5, 1, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_3, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 3, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{2, 3, 0, 5, 1, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_4, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 0, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 0, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 0, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{2, 3, 4, 5, 0, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_rois_dtype, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({2, 7})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{2, 3, 4, 5, 1, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_pts_dtype, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({3, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{2, 3, 4, 5, 1, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_pts_feature_dtype, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({3, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{2, 3, 4, 5, 1, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_pooled_features_dtype, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         5, std::vector<int>({2, 1, 2, 3, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{2, 3, 4, 5, 1, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_input_dtype, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{2, 3, 4, 5, 1, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_argmax_dtype, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{2, 3, 4, 5, 1, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_pts_idx_of_voxels_dtype, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{2, 3, 4, 5, 1, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_rois_shape, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 7})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 6})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 7, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{2, 3, 4, 5, 1, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_pts_shape, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({3, 3, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{2, 3, 4, 5, 1, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_pts_feature_shape, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({3, 4, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{2, 3, 4, 5, 1, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_pooled_features_shape, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({3, 1, 2, 3, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 2, 2, 3, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 3, 3, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 4, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 3, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 1, 2, 3, 4, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 1, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{2, 3, 4, 5, 1, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_argmax_shape, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({3, 1, 2, 3, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 2, 2, 3, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 3, 3, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 4, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         6,
                                         std::vector<int>({2, 1, 2, 3, 4, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 1, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{2, 3, 4, 5, 1, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_pts_idx_of_voxels_shape, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({3, 1, 2, 3, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 2, 2, 3, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 3, 3, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 4, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 6})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         6,
                                         std::vector<int>({2, 1, 2, 3, 5, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 1, 2, 3}))),
        testing::Values(RoiawarePool3dForwardParam{2, 3, 4, 5, 1, 2, 3, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_pool_method_value, roiaware_pool3d_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 1, 2, 3, 5}))),
        testing::Values(RoiawarePool3dForwardParam{2, 3, 4, 5, 1, 2, 3, 2}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

}  // namespace mluopapitest
