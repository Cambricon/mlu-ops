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
typedef std::tuple<mluOpDevType_t, mluOpStatus_t> PublicParam;

typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, float, float, int,
                   MLUOpTensorParam, PublicParam>
    BallQuery;
class ball_query_general : public testing::TestWithParam<BallQuery> {
 public:
  void SetUp() {
    MLUOP_CHECK(mluOpCreate(&handle_));
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      return;
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&xyz_desc_));
    MLUOpTensorParam xyz_params = std::get<0>(GetParam());
    mluOpTensorLayout_t xyz_layout = xyz_params.get_layout();
    mluOpDataType_t xyz_dtype = xyz_params.get_dtype();
    int xyz_dim = xyz_params.get_dim_nb();
    std::vector<int> xyz_dim_size = xyz_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(xyz_desc_, xyz_layout, xyz_dtype,
                                         xyz_dim, xyz_dim_size.data()));
    uint64_t xyz_ele_num = mluOpGetTensorElementNum(xyz_desc_);
    uint64_t xyz_bytes = mluOpDataTypeBytes(xyz_dtype) * xyz_ele_num;
    if (xyz_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&xyz_, xyz_bytes))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&new_xyz_desc_));
    MLUOpTensorParam new_xyz_params = std::get<1>(GetParam());
    mluOpTensorLayout_t new_xyz_layout = new_xyz_params.get_layout();
    mluOpDataType_t new_xyz_dtype = new_xyz_params.get_dtype();
    int new_xyz_dim = new_xyz_params.get_dim_nb();
    std::vector<int> new_xyz_dim_size = new_xyz_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(new_xyz_desc_, new_xyz_layout,
                                         new_xyz_dtype, new_xyz_dim,
                                         new_xyz_dim_size.data()));
    uint64_t new_xyz_ele_num = mluOpGetTensorElementNum(new_xyz_desc_);
    uint64_t new_xyz_bytes =
        mluOpDataTypeBytes(new_xyz_dtype) * new_xyz_ele_num;
    if (new_xyz_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&new_xyz_, new_xyz_bytes))
    }

    min_radius_ = std::get<2>(GetParam());
    max_radius_ = std::get<3>(GetParam());
    nsample_ = std::get<4>(GetParam());

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&idx_desc_));
    MLUOpTensorParam idx_params = std::get<5>(GetParam());
    mluOpTensorLayout_t idx_layout = idx_params.get_layout();
    mluOpDataType_t idx_dtype = idx_params.get_dtype();
    int idx_dim = idx_params.get_dim_nb();
    std::vector<int> idx_dim_size = idx_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(idx_desc_, idx_layout, idx_dtype,
                                         idx_dim, idx_dim_size.data()));
    uint64_t idx_ele_num = mluOpGetTensorElementNum(idx_desc_);
    uint64_t idx_bytes = mluOpDataTypeBytes(idx_dtype) * idx_ele_num;
    if (idx_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&idx_, idx_bytes))
    }

    PublicParam publicParam = std::get<6>(GetParam());
    std::tie(device_, expected_status_) = publicParam;
  }

  bool compute() {
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      destroy();
      return true;
    }
    mluOpStatus_t status =
        mluOpBallQuery(handle_, new_xyz_desc_, new_xyz_, xyz_desc_, xyz_,
                       min_radius_, max_radius_, nsample_, idx_desc_, idx_);
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
    if (new_xyz_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(new_xyz_desc_));
      new_xyz_desc_ = NULL;
    }
    if (new_xyz_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(new_xyz_));
      new_xyz_ = NULL;
    }
    if (xyz_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(xyz_desc_));
      xyz_desc_ = NULL;
    }
    if (xyz_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(xyz_));
      xyz_ = NULL;
    }
    if (idx_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(idx_desc_));
      idx_desc_ = NULL;
    }
    if (idx_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(idx_));
      idx_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  float min_radius_ = 0;
  float max_radius_ = 0.2;
  int nsample_ = 32;
  mluOpTensorDescriptor_t new_xyz_desc_ = NULL;
  void* new_xyz_ = NULL;
  mluOpTensorDescriptor_t xyz_desc_ = NULL;
  void* xyz_ = NULL;
  mluOpTensorDescriptor_t idx_desc_ = NULL;
  void* idx_ = NULL;
  mluOpDevType_t device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(ball_query_general, api_test) {
  try {
    EXPECT_TRUE(compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in ball_query";
  }
}

INSTANTIATE_TEST_CASE_P(
    zero_element_xyz, ball_query_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 0, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(0), testing::Values(0.2), testing::Values(32),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 32}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_SUCCESS})));

INSTANTIATE_TEST_CASE_P(
    zero_element_idx, ball_query_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 16, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(0), testing::Values(0.2), testing::Values(0),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 0}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_SUCCESS})));

INSTANTIATE_TEST_CASE_P(
    zero_element_1, ball_query_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({0, 16, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({0, 4, 3}))),
        testing::Values(0), testing::Values(0.2), testing::Values(32),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({0, 4, 32}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    zero_element_2, ball_query_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 16, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 0, 3}))),
        testing::Values(0), testing::Values(0.2), testing::Values(32),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 0, 32}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    zero_element_3, ball_query_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 16, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 0}))),
        testing::Values(0), testing::Values(0.2), testing::Values(32),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 32}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_xyz, ball_query_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         3, std::vector<int>({2, 16, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 16, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({3, 16, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 16, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 16, 3, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 16}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(0), testing::Values(0.2), testing::Values(32),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 32}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_new_xyz, ball_query_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 16, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         3, std::vector<int>({2, 4, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({3, 4, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 1, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 4, 3, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 4}))),
        testing::Values(0), testing::Values(0.2), testing::Values(32),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 32}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_input_dtype, ball_query_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 16, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(0), testing::Values(0.2), testing::Values(32),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 32}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_idx, ball_query_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 16, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(0), testing::Values(0.2), testing::Values(32),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         3, std::vector<int>({2, 4, 32})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({3, 4, 32})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 5, 32})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 6})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 4, 32, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 4}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_min_radius_value, ball_query_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 16, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(-2), testing::Values(0.2), testing::Values(32),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 32}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_max_radius_value, ball_query_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 16, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(0), testing::Values(-0.2), testing::Values(32),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 32}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    bad_nsample_value, ball_query_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 16, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 3}))),
        testing::Values(0), testing::Values(0.2), testing::Values(-32),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 32}))),
        testing::Values(PublicParam{MLUOP_UNKNOWN_DEVICE,
                                    MLUOP_STATUS_BAD_PARAM})));

}  // namespace mluopapitest
