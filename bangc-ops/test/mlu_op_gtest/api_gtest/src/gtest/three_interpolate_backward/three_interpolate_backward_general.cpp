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
typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, mluOpDevType_t, mluOpStatus_t>
    ThreeInterprolateBackwardParam;

class three_interprolate_backward_general
    : public testing::TestWithParam<ThreeInterprolateBackwardParam> {
 public:
  void SetUp() {
    device_ = std::get<4>(GetParam());
    expected_status_ = std::get<5>(GetParam());
    MLUOP_CHECK(mluOpCreate(&handle_));
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      return;
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_output_desc_));
    MLUOpTensorParam grad_output_params = std::get<0>(GetParam());
    mluOpTensorLayout_t grad_output_layout = grad_output_params.get_layout();
    mluOpDataType_t grad_output_dtype = grad_output_params.get_dtype();
    int grad_output_dim = grad_output_params.get_dim_nb();
    std::vector<int> grad_output_shape = grad_output_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(grad_output_desc_, grad_output_layout,
                                         grad_output_dtype, grad_output_dim,
                                         grad_output_shape.data()));
    uint64_t grad_output_ele_num = mluOpGetTensorElementNum(grad_output_desc_);
    if (grad_output_ele_num > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&grad_output_, 8))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&indices_desc_));
    MLUOpTensorParam indices_params = std::get<1>(GetParam());
    mluOpTensorLayout_t indices_layout = indices_params.get_layout();
    mluOpDataType_t indices_dtype = indices_params.get_dtype();
    int indices_dim = indices_params.get_dim_nb();
    std::vector<int> indices_shape = indices_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(indices_desc_, indices_layout,
                                         indices_dtype, indices_dim,
                                         indices_shape.data()));
    uint64_t indices_ele_num = mluOpGetTensorElementNum(indices_desc_);
    if (indices_ele_num > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&indices_, 8));
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&weight_desc_));
    MLUOpTensorParam weight_params = std::get<2>(GetParam());
    mluOpTensorLayout_t weight_layout = weight_params.get_layout();
    mluOpDataType_t weight_dtype = weight_params.get_dtype();
    int weight_dim = weight_params.get_dim_nb();
    std::vector<int> weight_shape = weight_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(weight_desc_, weight_layout,
                                         weight_dtype, weight_dim,
                                         weight_shape.data()));
    uint64_t weight_ele_num = mluOpGetTensorElementNum(weight_desc_);
    if (weight_ele_num > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&weight_, 8));
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_features_desc_));
    MLUOpTensorParam grad_features_params = std::get<3>(GetParam());
    mluOpTensorLayout_t grad_features_layout =
        grad_features_params.get_layout();
    mluOpDataType_t grad_features_dtype = grad_features_params.get_dtype();
    int grad_features_dim = grad_features_params.get_dim_nb();
    std::vector<int> grad_features_shape = grad_features_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(
        grad_features_desc_, grad_features_layout, grad_features_dtype,
        grad_features_dim, grad_features_shape.data()));
    uint64_t grad_features_ele_num =
        mluOpGetTensorElementNum(grad_features_desc_);
    if (grad_features_ele_num > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&grad_features_, 8));
    }
  }

  bool compute() {
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpThreeInterpolateBackward(
        handle_, grad_output_desc_, grad_output_, indices_desc_, indices_,
        weight_desc_, weight_, grad_features_desc_, grad_features_);
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
    if (grad_output_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_output_desc_));
      grad_output_desc_ = NULL;
    }
    if (indices_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indices_desc_));
      indices_desc_ = NULL;
    }
    if (weight_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(weight_desc_));
      weight_desc_ = NULL;
    }
    if (grad_features_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_features_desc_));
      grad_features_desc_ = NULL;
    }
    if (grad_output_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_output_));
      grad_output_ = NULL;
    }
    if (indices_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(indices_));
      indices_ = NULL;
    }
    if (weight_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(weight_));
      weight_ = NULL;
    }
    if (grad_features_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_features_));
      grad_features_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t grad_output_desc_ = NULL;
  mluOpTensorDescriptor_t indices_desc_ = NULL;
  mluOpTensorDescriptor_t weight_desc_ = NULL;
  mluOpTensorDescriptor_t grad_features_desc_ = NULL;
  void *grad_output_ = NULL;
  void *indices_ = NULL;
  void *weight_ = NULL;
  void *grad_features_ = NULL;
  mluOpDevType_t device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(three_interprolate_backward_general, api_test) {
  EXPECT_TRUE(compute());
}

INSTANTIATE_TEST_CASE_P(
    zero_element_B_0, three_interprolate_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({0, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({0, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({0, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({0, 2, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_C_1, three_interprolate_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 0, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 0, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_C_0, three_interprolate_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 0, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 0, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_N_0, three_interprolate_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 0, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 0, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_M_0, three_interprolate_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 0}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_dtype_0, three_interprolate_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_indices_weight_shape_0, three_interprolate_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 4, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 4, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_grad_features_shape_dtype_0, three_interprolate_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 2, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 2, 4, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_weight_shape_dtype_0, three_interprolate_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 4, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 1, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 4, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 4, 3, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         3, std::vector<int>({1, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_indices_shape_dtype_0, three_interprolate_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 4, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 1, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 4, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({1, 4, 3, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         3, std::vector<int>({1, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 4, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_grad_output_shape_dtype_0, three_interprolate_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 2, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 3, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 2, 5, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         3, std::vector<int>({1, 2, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 5, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 5, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 2, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));
}  // namespace mluopapitest
