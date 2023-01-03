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
typedef std::tuple<int, int, int, bool> VoxelizationParam;
typedef std::tuple<mluOpDevType_t, mluOpStatus_t> PublicParam;

typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   VoxelizationParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, PublicParam>
    Voxelization;
class voxelization_general : public testing::TestWithParam<Voxelization> {
 public:
  void SetUp() {
    MLUOP_CHECK(mluOpCreate(&handle_));
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      return;
    }
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&points_desc_));
    MLUOpTensorParam points_params = std::get<0>(GetParam());
    mluOpTensorLayout_t points_layout = points_params.get_layout();
    mluOpDataType_t points_dtype = points_params.get_dtype();
    int points_dim = points_params.get_dim_nb();
    std::vector<int> points_dim_size = points_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(points_desc_, points_layout,
                                         points_dtype, points_dim,
                                         points_dim_size.data()));
    uint64_t points_ele_num = mluOpGetTensorElementNum(points_desc_);
    uint64_t points_bytes = mluOpDataTypeBytes(points_dtype) * points_ele_num;
    if (points_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&points_, points_bytes))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&voxel_size_desc_));
    MLUOpTensorParam voxel_size_params = std::get<1>(GetParam());
    mluOpTensorLayout_t voxel_size_layout = voxel_size_params.get_layout();
    mluOpDataType_t voxel_size_dtype = voxel_size_params.get_dtype();
    int voxel_size_dim = voxel_size_params.get_dim_nb();
    std::vector<int> voxel_size_dim_size = voxel_size_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(voxel_size_desc_, voxel_size_layout,
                                         voxel_size_dtype, voxel_size_dim,
                                         voxel_size_dim_size.data()));
    uint64_t voxel_size_ele_num = mluOpGetTensorElementNum(voxel_size_desc_);
    uint64_t voxel_size_bytes =
        mluOpDataTypeBytes(voxel_size_dtype) * voxel_size_ele_num;
    if (voxel_size_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&voxel_size_, voxel_size_bytes))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&coors_range_desc_));
    MLUOpTensorParam coors_range_params = std::get<2>(GetParam());
    mluOpTensorLayout_t coors_range_layout = coors_range_params.get_layout();
    mluOpDataType_t coors_range_dtype = coors_range_params.get_dtype();
    int coors_range_dim = coors_range_params.get_dim_nb();
    std::vector<int> coors_range_dim_size = coors_range_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(coors_range_desc_, coors_range_layout,
                                         coors_range_dtype, coors_range_dim,
                                         coors_range_dim_size.data()));
    uint64_t coors_range_ele_num = mluOpGetTensorElementNum(coors_range_desc_);
    uint64_t coors_range_bytes =
        mluOpDataTypeBytes(coors_range_dtype) * coors_range_ele_num;
    if (coors_range_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&coors_range_, coors_range_bytes))
    }

    VoxelizationParam voxelizationParam = std::get<3>(GetParam());
    std::tie(max_points_, max_voxels_, NDim_, deterministic_) =
        voxelizationParam;

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&voxels_desc_));
    MLUOpTensorParam voxels_params = std::get<4>(GetParam());
    mluOpTensorLayout_t voxels_layout = voxels_params.get_layout();
    mluOpDataType_t voxels_dtype = voxels_params.get_dtype();
    int voxels_dim = voxels_params.get_dim_nb();
    std::vector<int> voxels_dim_size = voxels_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(voxels_desc_, voxels_layout,
                                         voxels_dtype, voxels_dim,
                                         voxels_dim_size.data()));
    uint64_t voxels_ele_num = mluOpGetTensorElementNum(voxels_desc_);
    uint64_t voxels_bytes = mluOpDataTypeBytes(voxels_dtype) * voxels_ele_num;
    if (voxels_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&voxels_, voxels_bytes))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&coors_desc_));
    MLUOpTensorParam coors_params = std::get<5>(GetParam());
    mluOpTensorLayout_t coors_layout = coors_params.get_layout();
    mluOpDataType_t coors_dtype = coors_params.get_dtype();
    int coors_dim = coors_params.get_dim_nb();
    std::vector<int> coors_dim_size = coors_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(coors_desc_, coors_layout, coors_dtype,
                                         coors_dim, coors_dim_size.data()));
    uint64_t coors_ele_num = mluOpGetTensorElementNum(coors_desc_);
    uint64_t coors_bytes = mluOpDataTypeBytes(coors_dtype) * coors_ele_num;
    if (coors_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&coors_, coors_bytes))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&num_points_per_voxel_desc_));
    MLUOpTensorParam num_points_per_voxel_params = std::get<6>(GetParam());
    mluOpTensorLayout_t num_points_per_voxel_layout =
        num_points_per_voxel_params.get_layout();
    mluOpDataType_t num_points_per_voxel_dtype =
        num_points_per_voxel_params.get_dtype();
    int num_points_per_voxel_dim = num_points_per_voxel_params.get_dim_nb();
    std::vector<int> num_points_per_voxel_dim_size =
        num_points_per_voxel_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(
        num_points_per_voxel_desc_, num_points_per_voxel_layout,
        num_points_per_voxel_dtype, num_points_per_voxel_dim,
        num_points_per_voxel_dim_size.data()));
    uint64_t num_points_per_voxel_ele_num =
        mluOpGetTensorElementNum(num_points_per_voxel_desc_);
    uint64_t num_points_per_voxel_bytes =
        mluOpDataTypeBytes(num_points_per_voxel_dtype) *
        num_points_per_voxel_ele_num;
    if (num_points_per_voxel_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&num_points_per_voxel_,
                                                 num_points_per_voxel_bytes))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&voxel_num_desc_));
    MLUOpTensorParam voxel_num_params = std::get<7>(GetParam());
    mluOpTensorLayout_t voxel_num_layout = voxel_num_params.get_layout();
    mluOpDataType_t voxel_num_dtype = voxel_num_params.get_dtype();
    int voxel_num_dim = voxel_num_params.get_dim_nb();
    std::vector<int> voxel_num_dim_size = voxel_num_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(voxel_num_desc_, voxel_num_layout,
                                         voxel_num_dtype, voxel_num_dim,
                                         voxel_num_dim_size.data()));
    uint64_t voxel_num_ele_num = mluOpGetTensorElementNum(voxel_num_desc_);
    uint64_t voxel_num_bytes =
        mluOpDataTypeBytes(voxel_num_dtype) * voxel_num_ele_num;
    if (voxel_num_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&voxel_num_, voxel_num_bytes))
    }

    PublicParam publicParam = std::get<8>(GetParam());
    std::tie(device_, expected_status_) = publicParam;
  }

  bool compute() {
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpGetVoxelizationWorkspaceSize(
        handle_, points_desc_, voxel_size_desc_, coors_range_desc_, max_points_,
        max_voxels_, NDim_, deterministic_, voxels_desc_, coors_desc_,
        num_points_per_voxel_desc_, voxel_num_desc_, &workspace_size_);
    if (MLUOP_STATUS_SUCCESS != status) {
      destroy();
      return status == expected_status_;
    }
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_));
    status = mluOpVoxelization(
        handle_, points_desc_, points_, voxel_size_desc_, voxel_size_,
        coors_range_desc_, coors_range_, max_points_, max_voxels_, NDim_,
        deterministic_, workspace_, workspace_size_, voxels_desc_, voxels_,
        coors_desc_, coors_, num_points_per_voxel_desc_, num_points_per_voxel_,
        voxel_num_desc_, voxel_num_);
    destroy();
    return status == expected_status_;
  }

 protected:
  void destroy() {
    if (handle_) {
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = NULL;
    }
    if (points_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(points_desc_));
      points_desc_ = NULL;
    }
    if (points_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(points_));
      points_ = NULL;
    }
    if (voxel_size_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(voxel_size_desc_));
      voxel_size_desc_ = NULL;
    }
    if (voxel_size_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(voxel_size_));
      voxel_size_ = NULL;
    }
    if (coors_range_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(coors_range_desc_));
      coors_range_desc_ = NULL;
    }
    if (coors_range_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(coors_range_));
      coors_range_ = NULL;
    }
    if (workspace_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = NULL;
    }
    if (voxels_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(voxels_desc_));
      voxels_desc_ = NULL;
    }
    if (voxels_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(voxels_));
      voxels_ = NULL;
    }
    if (coors_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(coors_desc_));
      coors_desc_ = NULL;
    }
    if (coors_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(coors_));
      coors_ = NULL;
    }
    if (num_points_per_voxel_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(num_points_per_voxel_desc_));
      num_points_per_voxel_desc_ = NULL;
    }
    if (num_points_per_voxel_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(num_points_per_voxel_));
      num_points_per_voxel_ = NULL;
    }
    if (voxel_num_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(voxel_num_desc_));
      voxel_num_desc_ = NULL;
    }
    if (voxel_num_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(voxel_num_));
      voxel_num_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t points_desc_ = NULL;
  void* points_ = NULL;
  mluOpTensorDescriptor_t voxel_size_desc_ = NULL;
  void* voxel_size_ = NULL;
  mluOpTensorDescriptor_t coors_range_desc_ = NULL;
  void* coors_range_ = NULL;
  int max_points_ = 4;
  int max_voxels_ = 5;
  int NDim_ = 3;
  bool deterministic_ = true;
  void* workspace_ = NULL;
  size_t workspace_size_ = 64;
  mluOpTensorDescriptor_t voxels_desc_ = NULL;
  void* voxels_ = NULL;
  mluOpTensorDescriptor_t coors_desc_ = NULL;
  void* coors_ = NULL;
  mluOpTensorDescriptor_t num_points_per_voxel_desc_ = NULL;
  void* num_points_per_voxel_ = NULL;
  mluOpTensorDescriptor_t voxel_num_desc_ = NULL;
  void* voxel_num_ = NULL;
  mluOpDevType_t device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(voxelization_general, api_test) {
  try {
    EXPECT_TRUE(compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in voxelization";
  }
}

INSTANTIATE_TEST_CASE_P(
    zero_element_max_points, voxelization_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({6}))),
        testing::Values(VoxelizationParam{0, 5, 3, true}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({5, 0, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({5, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_SUCCESS})));

INSTANTIATE_TEST_CASE_P(
    zero_element_max_voxels, voxelization_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({6}))),
        testing::Values(VoxelizationParam{4, 0, 3, true}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({0, 4, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({0, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_SUCCESS})));

INSTANTIATE_TEST_CASE_P(
    zero_element_points_1, voxelization_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({6}))),
        testing::Values(VoxelizationParam{4, 5, 3, true}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({5, 4, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({5, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    zero_element_points_2, voxelization_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({6}))),
        testing::Values(VoxelizationParam{4, 5, 3, true}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({5, 4, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({5, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_points, voxelization_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({6}))),
        testing::Values(VoxelizationParam{4, 5, 3, true}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({5, 4, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({5, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_voxel_size, voxelization_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({0})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({6}))),
        testing::Values(VoxelizationParam{4, 5, 3, true}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({5, 4, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({5, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_coors_range, voxelization_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({0})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({6, 1}))),
        testing::Values(VoxelizationParam{4, 5, 3, true}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({5, 4, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({5, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_max_NDim, voxelization_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({6}))),
        testing::Values(VoxelizationParam{4, 5, 1, true}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({5, 4, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({5, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_voxels, voxelization_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({6}))),
        testing::Values(VoxelizationParam{4, 5, 3, true}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({6, 4, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({5, 3, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({5, 4, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({5, 4, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({5, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({5, 4, 2, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({5, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_coors, voxelization_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({6}))),
        testing::Values(VoxelizationParam{4, 5, 3, true}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({5, 4, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({5, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({5, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({5, 3, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_num_points_per_voxel, voxelization_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({6}))),
        testing::Values(VoxelizationParam{4, 5, 3, true}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({5, 4, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({5, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({5, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_voxel_num, voxelization_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({6}))),
        testing::Values(VoxelizationParam{4, 5, 3, true}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({5, 4, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({5, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 1}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

}  // namespace mluopapitest
