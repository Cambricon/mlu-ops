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
typedef std::tuple<int, int, int, int, int, int> VoxelPoolingForwardDescParam;
typedef std::tuple<VoxelPoolingForwardDescParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   mluOpDevType_t, mluOpStatus_t>
    VoxelPoolingForwardParam;
class voxel_pooling_forward_general
    : public testing::TestWithParam<VoxelPoolingForwardParam> {
 public:
  void SetUp() {
    MLUOP_CHECK(mluOpCreate(&handle_));
    device_ = std::get<5>(GetParam());
    expected_status_ = std::get<6>(GetParam());
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      return;
    }
    VoxelPoolingForwardDescParam op_param = std::get<0>(GetParam());
    std::tie(batch_size_, num_points_, num_channels_, num_voxel_x_,
             num_voxel_y_, num_voxel_z_) = op_param;

    MLUOpTensorParam g_params = std::get<1>(GetParam());
    mluOpTensorLayout_t g_layout = g_params.get_layout();
    mluOpDataType_t g_dtype = g_params.get_dtype();
    int g_dim = g_params.get_dim_nb();
    std::vector<int> g_dim_size = g_params.get_dim_size();
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&geom_xyz_desc_));
    MLUOP_CHECK(mluOpSetTensorDescriptor(geom_xyz_desc_, g_layout, g_dtype,
                                         g_dim, g_dim_size.data()));
    uint64_t g_ele_num = mluOpGetTensorElementNum(geom_xyz_desc_);
    uint64_t g_bytes;
    if (g_ele_num < LARGE_TENSOR_NUM) {
      g_bytes = mluOpDataTypeBytes(g_dtype) * g_ele_num;
    } else {
      g_bytes = mluOpDataTypeBytes(g_dtype) * 36;
    }
    if (g_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&geom_xyz_, g_bytes))
    }

    MLUOpTensorParam i_params = std::get<2>(GetParam());
    mluOpTensorLayout_t i_layout = i_params.get_layout();
    mluOpDataType_t i_dtype = i_params.get_dtype();
    int i_dim = i_params.get_dim_nb();
    std::vector<int> i_dim_size = i_params.get_dim_size();
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_features_desc_));
    MLUOP_CHECK(mluOpSetTensorDescriptor(input_features_desc_, i_layout,
                                         i_dtype, i_dim, i_dim_size.data()));
    uint64_t i_ele_num = mluOpGetTensorElementNum(input_features_desc_);
    uint64_t i_bytes;
    if (i_ele_num < LARGE_TENSOR_NUM) {
      i_bytes = mluOpDataTypeBytes(i_dtype) * i_ele_num;
    } else {
      i_bytes = mluOpDataTypeBytes(g_dtype) * 80;
    }
    if (i_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&input_features_, i_bytes))
    }

    MLUOpTensorParam o_params = std::get<3>(GetParam());
    mluOpTensorLayout_t o_layout = o_params.get_layout();
    mluOpDataType_t o_dtype = o_params.get_dtype();
    int o_dim = o_params.get_dim_nb();
    std::vector<int> o_dim_size = o_params.get_dim_size();
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_features_desc_));
    MLUOP_CHECK(mluOpSetTensorDescriptor(output_features_desc_, o_layout,
                                         o_dtype, o_dim, o_dim_size.data()));
    uint64_t o_ele_num = mluOpGetTensorElementNum(output_features_desc_);
    uint64_t o_bytes;
    if (o_ele_num < LARGE_TENSOR_NUM) {
      o_bytes = mluOpDataTypeBytes(o_dtype) * o_ele_num;
    } else {
      o_bytes = mluOpDataTypeBytes(g_dtype) * 600;
    }
    if (o_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&output_features_, o_bytes))
    }

    MLUOpTensorParam p_params = std::get<4>(GetParam());
    mluOpTensorLayout_t p_layout = p_params.get_layout();
    mluOpDataType_t p_dtype = p_params.get_dtype();
    int p_dim = p_params.get_dim_nb();
    std::vector<int> p_dim_size = p_params.get_dim_size();
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&pos_memo_desc_));
    MLUOP_CHECK(mluOpSetTensorDescriptor(pos_memo_desc_, p_layout, p_dtype,
                                         p_dim, p_dim_size.data()));
    uint64_t p_ele_num = mluOpGetTensorElementNum(pos_memo_desc_);
    uint64_t p_bytes;
    if (p_ele_num < LARGE_TENSOR_NUM) {
      p_bytes = mluOpDataTypeBytes(p_dtype) * p_ele_num;
    } else {
      p_bytes = mluOpDataTypeBytes(p_dtype) * 600;
    }
    if (p_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&pos_memo_, p_bytes))
    }
  }

  bool compute() {
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpVoxelPoolingForward(
        handle_, batch_size_, num_points_, num_channels_, num_voxel_x_,
        num_voxel_y_, num_voxel_z_, geom_xyz_desc_, geom_xyz_,
        input_features_desc_, input_features_, output_features_desc_,
        output_features_, pos_memo_desc_, pos_memo_);
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
    if (geom_xyz_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(geom_xyz_desc_));
      geom_xyz_desc_ = NULL;
    }
    if (input_features_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_features_desc_));
      input_features_desc_ = NULL;
    }
    if (output_features_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_features_desc_));
      output_features_desc_ = NULL;
    }
    if (pos_memo_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(pos_memo_desc_));
      pos_memo_desc_ = NULL;
    }
    if (geom_xyz_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(geom_xyz_));
      geom_xyz_ = NULL;
    }
    if (input_features_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_features_));
      input_features_ = NULL;
    }
    if (output_features_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_features_));
      output_features_ = NULL;
    }
    if (pos_memo_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(pos_memo_));
      output_features_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t geom_xyz_desc_ = NULL;
  mluOpTensorDescriptor_t input_features_desc_ = NULL;
  mluOpTensorDescriptor_t output_features_desc_ = NULL;
  mluOpTensorDescriptor_t pos_memo_desc_ = NULL;
  void* geom_xyz_ = NULL;
  void* input_features_ = NULL;
  void* output_features_ = NULL;
  void* pos_memo_ = NULL;
  int batch_size_ = 2;
  int num_points_ = 4;
  int num_channels_ = 10;
  int num_voxel_x_ = 6;
  int num_voxel_y_ = 5;
  int num_voxel_z_ = 1;
  mluOpDevType_t device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(voxel_pooling_forward_general, api_test) {
  try {
    EXPECT_TRUE(compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in Voxelpool_forward";
  }
}

INSTANTIATE_TEST_CASE_P(
    bad_geom_xyz_dtype_dim, voxel_pooling_forward_general,
    testing::Combine(
        testing::Values(VoxelPoolingForwardDescParam{2, 4, 10, 6, 5, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 4, 3, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 5, 6, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_input_features_dtype_dim, voxel_pooling_forward_general,
    testing::Combine(
        testing::Values(VoxelPoolingForwardDescParam{2, 4, 10, 6, 5, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         3, std::vector<int>({2, 4, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 4, 10, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 5, 6, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_output_features_dtype_dim, voxel_pooling_forward_general,
    testing::Combine(
        testing::Values(VoxelPoolingForwardDescParam{2, 4, 10, 6, 5, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         4, std::vector<int>({2, 5, 6, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 5, 6, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_pos_memo_dtype_dim, voxel_pooling_forward_general,
    testing::Combine(
        testing::Values(VoxelPoolingForwardDescParam{2, 4, 10, 6, 5, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 5, 6, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 4, 3, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_shape_geom_xyz, voxel_pooling_forward_general,
    testing::Combine(
        testing::Values(VoxelPoolingForwardDescParam{2, 4, 10, 6, 5, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 4, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 1, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 5, 6, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_shape_input_features, voxel_pooling_forward_general,
    testing::Combine(
        testing::Values(VoxelPoolingForwardDescParam{2, 4, 10, 6, 5, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 4, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 1, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 5, 6, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_shape_output_features, voxel_pooling_forward_general,
    testing::Combine(
        testing::Values(VoxelPoolingForwardDescParam{2, 4, 10, 6, 5, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 5, 6, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 1, 6, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 5, 1, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 5, 6, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_shape_pos_memo, voxel_pooling_forward_general,
    testing::Combine(
        testing::Values(VoxelPoolingForwardDescParam{2, 4, 10, 6, 5, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 5, 6, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 4, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 1, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_batch_size, voxel_pooling_forward_general,
    testing::Combine(
        testing::Values(VoxelPoolingForwardDescParam{-2, 4, 10, 6, 5, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({-2, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({-2, 4, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({-2, 5, 6, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({-2, 4, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_num_points, voxel_pooling_forward_general,
    testing::Combine(
        testing::Values(VoxelPoolingForwardDescParam{2, -4, 10, 6, 5, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, -4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, -4, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 5, 6, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, -4, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_num_channels, voxel_pooling_forward_general,
    testing::Combine(
        testing::Values(VoxelPoolingForwardDescParam{2, 4, -10, 6, 5, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, -10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 5, 6, -10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_num_voxel_x, voxel_pooling_forward_general,
    testing::Combine(
        testing::Values(VoxelPoolingForwardDescParam{2, 4, 10, -6, 5, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 5, -6, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_num_voxel_y, voxel_pooling_forward_general,
    testing::Combine(
        testing::Values(VoxelPoolingForwardDescParam{2, 4, 10, 6, -5, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, -5, 6, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_large_tensor, voxel_pooling_forward_general,
    testing::Combine(
        testing::Values(VoxelPoolingForwardDescParam{699051, 1024, 10, 6, 5,
                                                     1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3,
                                         std::vector<int>({699051, 1024, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3,
                                         std::vector<int>({699051, 1024, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4,
                                         std::vector<int>({699051, 5, 6, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3,
                                         std::vector<int>({699051, 1024, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));
}  // namespace mluopapitest
