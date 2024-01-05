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

namespace mluopapitest {

typedef std::tuple<mluOpReduceMode_t, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, mluOpDevType_t,
                   mluOpStatus_t>
    DynamicPointToVoxelBackward;
class dynamic_point_to_voxel_backward_general
    : public testing::TestWithParam<DynamicPointToVoxelBackward> {
 public:
  void SetUp() {
    try {
      MLUOP_CHECK(mluOpCreate(&handle_));

      reduce_type_ = std::get<0>(GetParam());
      MLUOpTensorParam grad_voxel_feats_params = std::get<1>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_voxel_feats_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_voxel_feats_desc_, grad_voxel_feats_params.get_layout(),
          grad_voxel_feats_params.get_dtype(),
          grad_voxel_feats_params.get_dim_nb(),
          grad_voxel_feats_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(grad_voxel_feats_desc_) >=
          LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &grad_voxel_feats_,
                mluOpDataTypeBytes(grad_voxel_feats_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&grad_voxel_feats_,
                       mluOpDataTypeBytes(grad_voxel_feats_params.get_dtype()) *
                           mluOpGetTensorElementNum(grad_voxel_feats_desc_)));
      }
      MLUOpTensorParam feats_params = std::get<2>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&feats_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          feats_desc_, feats_params.get_layout(), feats_params.get_dtype(),
          feats_params.get_dim_nb(), feats_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(feats_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&feats_,
                       mluOpDataTypeBytes(feats_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&feats_, mluOpDataTypeBytes(feats_params.get_dtype()) *
                                    mluOpGetTensorElementNum(feats_desc_)));
      }

      MLUOpTensorParam voxel_feats_params = std::get<3>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&voxel_feats_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          voxel_feats_desc_, voxel_feats_params.get_layout(),
          voxel_feats_params.get_dtype(), voxel_feats_params.get_dim_nb(),
          voxel_feats_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(voxel_feats_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&voxel_feats_,
                       mluOpDataTypeBytes(voxel_feats_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&voxel_feats_,
                       mluOpDataTypeBytes(voxel_feats_params.get_dtype()) *
                           mluOpGetTensorElementNum(voxel_feats_desc_)));
      }

      MLUOpTensorParam point2voxel_map_params = std::get<4>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&point2voxel_map_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          point2voxel_map_desc_, point2voxel_map_params.get_layout(),
          point2voxel_map_params.get_dtype(),
          point2voxel_map_params.get_dim_nb(),
          point2voxel_map_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(point2voxel_map_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &point2voxel_map_,
                mluOpDataTypeBytes(point2voxel_map_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&point2voxel_map_,
                       mluOpDataTypeBytes(point2voxel_map_params.get_dtype()) *
                           mluOpGetTensorElementNum(point2voxel_map_desc_)));
      }

      MLUOpTensorParam voxel_points_count_params = std::get<5>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&voxel_points_count_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          voxel_points_count_desc_, voxel_points_count_params.get_layout(),
          voxel_points_count_params.get_dtype(),
          voxel_points_count_params.get_dim_nb(),
          voxel_points_count_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(voxel_points_count_desc_) >=
          LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &voxel_points_count_,
                mluOpDataTypeBytes(voxel_points_count_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &voxel_points_count_,
                mluOpDataTypeBytes(voxel_points_count_params.get_dtype()) *
                    mluOpGetTensorElementNum(voxel_points_count_desc_)));
      }

      MLUOpTensorParam voxel_num_params = std::get<6>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&voxel_num_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          voxel_num_desc_, voxel_num_params.get_layout(),
          voxel_num_params.get_dtype(), voxel_num_params.get_dim_nb(),
          voxel_num_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(voxel_num_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&voxel_num_,
                       mluOpDataTypeBytes(voxel_num_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&voxel_num_,
                       mluOpDataTypeBytes(voxel_num_params.get_dtype()) *
                           mluOpGetTensorElementNum(voxel_num_desc_)));
      }

      MLUOpTensorParam grad_feats_params = std::get<7>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_feats_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_feats_desc_, grad_feats_params.get_layout(),
          grad_feats_params.get_dtype(), grad_feats_params.get_dim_nb(),
          grad_feats_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(grad_feats_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&grad_feats_,
                       mluOpDataTypeBytes(grad_feats_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&grad_feats_,
                       mluOpDataTypeBytes(grad_feats_params.get_dtype()) *
                           mluOpGetTensorElementNum(grad_feats_desc_)));
      }
      target_device_ = std::get<8>(GetParam());
      expected_status_ = std::get<9>(GetParam());

      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&workspace_, MLUOP_DTYPE_FLOAT * workspace_size_));
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in ms_deform_attn_backward general.";
    }
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpDynamicPointToVoxelBackward(
        handle_, reduce_type_, grad_voxel_feats_desc_, grad_voxel_feats_,
        feats_desc_, feats_, voxel_feats_desc_, voxel_feats_,
        point2voxel_map_desc_, point2voxel_map_, voxel_points_count_desc_,
        voxel_points_count_, voxel_num_desc_, voxel_num_, workspace_,
        workspace_size_, grad_feats_desc_, grad_feats_);
    destroy();
    return expected_status_ == status;
  }

  void destroy() {
    if (handle_) {
      CNRT_CHECK(cnrtQueueSync(handle_->queue));
      VLOG(4) << "Destroy handle";
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = nullptr;
    }

    if (grad_voxel_feats_desc_) {
      VLOG(4) << "Destroy grad_voxel_feats_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_voxel_feats_desc_));
      grad_voxel_feats_desc_ = nullptr;
    }

    if (grad_voxel_feats_) {
      VLOG(4) << "Destroy grad_voxel_feats_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_voxel_feats_));
      grad_voxel_feats_ = nullptr;
    }

    if (feats_desc_) {
      VLOG(4) << "Destroy feats_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(feats_desc_));
      feats_desc_ = nullptr;
    }

    if (feats_) {
      VLOG(4) << "Destroy feats_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(feats_));
      feats_ = nullptr;
    }

    if (voxel_feats_desc_) {
      VLOG(4) << "Destroy voxel_feats_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(voxel_feats_desc_));
      voxel_feats_desc_ = nullptr;
    }

    if (voxel_feats_) {
      VLOG(4) << "Destroy voxel_feats_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(voxel_feats_));
      voxel_feats_ = nullptr;
    }

    if (point2voxel_map_desc_) {
      VLOG(4) << "Destroy point2voxel_map_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(point2voxel_map_desc_));
      point2voxel_map_desc_ = nullptr;
    }

    if (point2voxel_map_) {
      VLOG(4) << "Destroy point2voxel_map_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(point2voxel_map_));
      point2voxel_map_ = nullptr;
    }

    if (voxel_points_count_desc_) {
      VLOG(4) << "Destroy data_col_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(voxel_points_count_desc_));
      voxel_points_count_desc_ = nullptr;
    }

    if (voxel_points_count_) {
      VLOG(4) << "Destroy voxel_points_count_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(voxel_points_count_));
      voxel_points_count_ = nullptr;
    }

    if (voxel_num_desc_) {
      VLOG(4) << "Destroy voxel_num_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(voxel_num_desc_));
      voxel_num_desc_ = nullptr;
    }

    if (voxel_num_) {
      VLOG(4) << "Destroy voxel_num_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(voxel_num_));
      voxel_num_ = nullptr;
    }

    if (workspace_) {
      VLOG(4) << "Destroy workspace_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = nullptr;
    }

    if (grad_feats_desc_) {
      VLOG(4) << "Destroy grad_feats_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_feats_desc_));
      grad_feats_desc_ = nullptr;
    }

    if (grad_feats_) {
      VLOG(4) << "Destroy grad_feats_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_feats_));
      grad_feats_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t grad_voxel_feats_desc_ = nullptr;
  void *grad_voxel_feats_ = nullptr;
  mluOpTensorDescriptor_t feats_desc_ = nullptr;
  void *feats_ = nullptr;
  mluOpTensorDescriptor_t voxel_feats_desc_ = nullptr;
  void *voxel_feats_ = nullptr;
  mluOpTensorDescriptor_t point2voxel_map_desc_ = nullptr;
  void *point2voxel_map_ = nullptr;
  mluOpTensorDescriptor_t voxel_points_count_desc_ = nullptr;
  void *voxel_points_count_ = nullptr;
  mluOpTensorDescriptor_t voxel_num_desc_ = nullptr;
  void *voxel_num_ = nullptr;
  size_t workspace_size_ = 10;
  void *workspace_ = nullptr;
  mluOpReduceMode_t reduce_type_ = MLUOP_REDUCE_DMAX;
  mluOpTensorDescriptor_t grad_feats_desc_ = nullptr;
  void *grad_feats_ = nullptr;
  mluOpDevType_t target_device_;
  mluOpStatus_t expected_status_;
};

TEST_P(dynamic_point_to_voxel_backward_general, negative) {
  EXPECT_TRUE(compute());
}

INSTANTIATE_TEST_CASE_P(
    zero_element_1, dynamic_point_to_voxel_backward_general,
    testing::Combine(
        testing::Values(MLUOP_REDUCE_DMAX),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 0})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_2, dynamic_point_to_voxel_backward_general,
    testing::Combine(
        testing::Values(MLUOP_REDUCE_DMAX),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    wrong_grad_voxel_feats_dtype_shape_dims,
    dynamic_point_to_voxel_backward_general,
    testing::Combine(
        testing::Values(MLUOP_REDUCE_DMAX),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({4, 3})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 5})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({8, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    wrong_feats_dtype_dims_shape, dynamic_point_to_voxel_backward_general,
    testing::Combine(
        testing::Values(MLUOP_REDUCE_DMAX),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({4, 3})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 6})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({8, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    wrong_voxel_feats_dtype_dims_shape, dynamic_point_to_voxel_backward_general,
    testing::Combine(
        testing::Values(MLUOP_REDUCE_DMAX),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         2, std::vector<int>({4, 3})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 13})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({24, 3})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    wrong_point2voxel_map_dtype_dims_shape,
    dynamic_point_to_voxel_backward_general,
    testing::Combine(
        testing::Values(MLUOP_REDUCE_DMAX),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({4, 3})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         1, std::vector<int>({4})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({18})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    wrong_voxel_points_count_dtype_dims_shape,
    dynamic_point_to_voxel_backward_general,
    testing::Combine(
        testing::Values(MLUOP_REDUCE_DMAX),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         1, std::vector<int>({4})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({16})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 2, 2})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    wrong_voxel_num_dtype_dims_shape, dynamic_point_to_voxel_backward_general,
    testing::Combine(
        testing::Values(MLUOP_REDUCE_DMAX),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         1, std::vector<int>({1})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({21})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({5, 4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    wrong_grad_feats_dtype_dims_shape, dynamic_point_to_voxel_backward_general,
    testing::Combine(
        testing::Values(MLUOP_REDUCE_DMAX),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({4, 3})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 16})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    wrong_reduce_type, dynamic_point_to_voxel_backward_general,
    testing::Combine(
        testing::Values(MLUOP_REDUCE_DMEAN),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    M_greater_than_N, dynamic_point_to_voxel_backward_general,
    testing::Combine(
        testing::Values(MLUOP_REDUCE_DMAX),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({8, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({8, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({8})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));
}  // namespace mluopapitest
