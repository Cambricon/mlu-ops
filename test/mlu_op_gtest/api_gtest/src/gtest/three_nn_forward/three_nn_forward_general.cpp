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
typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, mluOpDevType_t, mluOpStatus_t>
    ThreeNNForwardParams;

class three_nn_forward_general
    : public testing::TestWithParam<ThreeNNForwardParams> {
 public:
  void SetUp() {
    target_device_ = std::get<4>(GetParam());
    expected_status_ = std::get<5>(GetParam());
    MLUOP_CHECK(mluOpCreate(&handle_));
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      return;
    }

    MLUOpTensorParam u_params = std::get<0>(GetParam());
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&unknown_desc_));
    mluOpDataType_t u_dtype = u_params.get_dtype();
    MLUOP_CHECK(mluOpSetTensorDescriptor(unknown_desc_, u_params.get_layout(),
                                         u_dtype, u_params.get_dim_nb(),
                                         u_params.get_dim_size().data()));
    if (mluOpGetTensorElementNum(unknown_desc_) > 0) {
      size_t u_dtype_size;
      MLUOP_CHECK(mluOpGetSizeOfDataType(u_dtype, &u_dtype_size));
      uint64_t u_bytes = u_dtype_size * mluOpGetTensorElementNum(unknown_desc_);
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&unknown_, u_bytes))
    }

    MLUOpTensorParam k_params = std::get<1>(GetParam());
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&known_desc_));
    mluOpDataType_t k_dtype = k_params.get_dtype();
    MLUOP_CHECK(mluOpSetTensorDescriptor(known_desc_, k_params.get_layout(),
                                         k_dtype, k_params.get_dim_nb(),
                                         k_params.get_dim_size().data()));
    if (mluOpGetTensorElementNum(known_desc_) > 0) {
      size_t k_dtype_size;
      MLUOP_CHECK(mluOpGetSizeOfDataType(k_dtype, &k_dtype_size));
      uint64_t k_bytes = k_dtype_size * mluOpGetTensorElementNum(known_desc_);
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&known_, k_bytes))
    }

    MLUOpTensorParam d_params = std::get<2>(GetParam());
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&dist2_desc_));
    mluOpDataType_t d_dtype = d_params.get_dtype();
    MLUOP_CHECK(mluOpSetTensorDescriptor(dist2_desc_, d_params.get_layout(),
                                         d_dtype, d_params.get_dim_nb(),
                                         d_params.get_dim_size().data()));
    if (mluOpGetTensorElementNum(dist2_desc_) > 0) {
      size_t d_dtype_size;
      MLUOP_CHECK(mluOpGetSizeOfDataType(d_dtype, &d_dtype_size));
      uint64_t d_bytes = d_dtype_size * mluOpGetTensorElementNum(dist2_desc_);
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&dist2_, d_bytes))
    }

    MLUOpTensorParam id_params = std::get<3>(GetParam());
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&idx_desc_));
    mluOpDataType_t id_dtype = id_params.get_dtype();
    MLUOP_CHECK(mluOpSetTensorDescriptor(idx_desc_, id_params.get_layout(),
                                         id_dtype, id_params.get_dim_nb(),
                                         id_params.get_dim_size().data()));
    if (mluOpGetTensorElementNum(idx_desc_) > 0) {
      size_t id_dtype_size;
      MLUOP_CHECK(mluOpGetSizeOfDataType(id_dtype, &id_dtype_size));
      uint64_t id_bytes = id_dtype_size * mluOpGetTensorElementNum(dist2_desc_);
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&idx_, id_bytes))
    }
  }
  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpGetThreeNNForwardWorkspaceSize(
        handle_, known_desc_, &workspace_size_);
    if (MLUOP_STATUS_SUCCESS != status) {
      destroy();
      return expected_status_ == status;
    }
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_))
    status = mluOpThreeNNForward(handle_, unknown_desc_, unknown_, known_desc_,
                                 known_, workspace_, workspace_size_,
                                 dist2_desc_, dist2_, idx_desc_, idx_);
    destroy();
    return expected_status_ == status;
  }

 protected:
  void destroy() {
    if (handle_) {
      CNRT_CHECK(cnrtQueueSync(handle_->queue));
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = NULL;
    }
    if (unknown_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(unknown_desc_));
      unknown_desc_ = NULL;
    }
    if (unknown_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(unknown_));
      unknown_ = NULL;
    }
    if (known_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(known_desc_));
      known_desc_ = NULL;
    }
    if (known_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(known_));
      known_ = NULL;
    }
    if (workspace_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = NULL;
    }
    if (dist2_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(dist2_desc_));
      dist2_desc_ = NULL;
    }
    if (dist2_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(dist2_))
      dist2_ = NULL;
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
  mluOpTensorDescriptor_t unknown_desc_ = NULL;
  mluOpTensorDescriptor_t known_desc_ = NULL;
  mluOpTensorDescriptor_t dist2_desc_ = NULL;
  mluOpTensorDescriptor_t idx_desc_ = NULL;
  void *unknown_ = NULL;
  void *known_ = NULL;
  void *dist2_ = NULL;
  void *idx_ = NULL;
  size_t workspace_size_ = 100;
  void *workspace_ = NULL;
  mluOpDevType_t target_device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(three_nn_forward_general, negative) {
  try {
    EXPECT_TRUE(compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in matmul";
  }
}

INSTANTIATE_TEST_CASE_P(
    zero_element_0, three_nn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({0, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({0, 10, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({0, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({0, 2, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_1, three_nn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 0, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 10, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 0, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 0, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_unknown_dtype_shape_0, three_nn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         3, std::vector<int>({1, 2, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 2, 3, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 2, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 3, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 10, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 2, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_known_dtype_shape_0, three_nn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         3, std::vector<int>({1, 10, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 10, 3, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 10, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 10, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 2, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_dist2_dtype_shape_0, three_nn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 10, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         3, std::vector<int>({1, 2, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 2, 3, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 2, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 2, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_idx_dtype_shape_0, three_nn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 10, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({1, 2, 3, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 2, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 2, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 3, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    unsupported_dtype_0, three_nn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 10, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 2, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    unsupported_shape_0, three_nn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 10, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 2, 3}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    unsupported_shape_1, three_nn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 10, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));
}  // namespace mluopapitest
