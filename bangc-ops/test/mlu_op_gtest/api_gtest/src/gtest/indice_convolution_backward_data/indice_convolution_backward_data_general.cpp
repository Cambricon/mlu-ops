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

typedef std::tuple<std::vector<int64_t>, int64_t, int64_t>
    IndiceConvolutionBackwardDataParam;

typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, IndiceConvolutionBackwardDataParam,
                   mluOpDevType_t, mluOpStatus_t>
    IndiceConvolutionBackwardData;

class indice_convolution_backward_data_general
    : public testing::TestWithParam<IndiceConvolutionBackwardData> {
 public:
  void SetUp() {
    device_ = std::get<5>(GetParam());
    expected_status_ = std::get<6>(GetParam());
    MLUOP_CHECK(mluOpCreate(&handle_));
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      return;
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_grad_desc_));
    MLUOpTensorParam output_grad_params = std::get<0>(GetParam());
    mluOpTensorLayout_t output_grad_layout = output_grad_params.get_layout();
    mluOpDataType_t output_grad_dtype = output_grad_params.get_dtype();
    int output_grad_dim = output_grad_params.get_dim_nb();
    std::vector<int> output_grad_shape = output_grad_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(output_grad_desc_, output_grad_layout,
                                         output_grad_dtype, output_grad_dim,
                                         output_grad_shape.data()));
    uint64_t output_grad_ele_num = mluOpGetTensorElementNum(output_grad_desc_);
    if (output_grad_ele_num > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&output_grad_,
                             mluOpGetTensorElementNum(output_grad_desc_) *
                                 mluOpDataTypeBytes(output_grad_dtype)))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&filters_desc_));
    MLUOpTensorParam filters_params = std::get<1>(GetParam());
    mluOpTensorLayout_t filters_layout = filters_params.get_layout();
    mluOpDataType_t filters_dtype = filters_params.get_dtype();
    int filters_dim = filters_params.get_dim_nb();
    std::vector<int> filters_shape = filters_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(filters_desc_, filters_layout,
                                         filters_dtype, filters_dim,
                                         filters_shape.data()));
    uint64_t filters_ele_num = mluOpGetTensorElementNum(filters_desc_);
    if (filters_ele_num > 0) {
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(&filters_, mluOpGetTensorElementNum(filters_desc_) *
                                    mluOpDataTypeBytes(filters_dtype)));
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&indice_pairs_desc_));
    MLUOpTensorParam indice_pairs_params = std::get<2>(GetParam());
    mluOpTensorLayout_t indice_pairs_layout = indice_pairs_params.get_layout();
    mluOpDataType_t indice_pairs_dtype = indice_pairs_params.get_dtype();
    int indice_pairs_dim = indice_pairs_params.get_dim_nb();
    std::vector<int> indice_pairs_shape = indice_pairs_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(
        indice_pairs_desc_, indice_pairs_layout, indice_pairs_dtype,
        indice_pairs_dim, indice_pairs_shape.data()));
    uint64_t indice_pairs_ele_num =
        mluOpGetTensorElementNum(indice_pairs_desc_);
    if (indice_pairs_ele_num > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&indice_pairs_,
                             mluOpGetTensorElementNum(indice_pairs_desc_) *
                                 mluOpDataTypeBytes(indice_pairs_dtype)));
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_grad_desc_));
    MLUOpTensorParam input_grad_params = std::get<3>(GetParam());
    mluOpTensorLayout_t input_grad_layout = input_grad_params.get_layout();
    mluOpDataType_t input_grad_dtype = input_grad_params.get_dtype();
    int input_grad_dim = input_grad_params.get_dim_nb();
    std::vector<int> input_grad_shape = input_grad_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(input_grad_desc_, input_grad_layout,
                                         input_grad_dtype, input_grad_dim,
                                         input_grad_shape.data()));
    uint64_t input_grad_ele_num = mluOpGetTensorElementNum(input_grad_desc_);
    if (input_grad_ele_num > 0) {
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(&input_grad_, mluOpGetTensorElementNum(input_grad_desc_) *
                                       mluOpDataTypeBytes(input_grad_dtype)));
    }

    IndiceConvolutionBackwardDataParam IndiceConvolutionBackwardData =
        std::get<4>(GetParam());
    std::tie(indice_num_, inverse_, sub_m_) = IndiceConvolutionBackwardData;
  }

  bool compute() {
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpGetIndiceConvolutionBackwardDataWorkspaceSize(
        handle_, output_grad_desc_, filters_desc_, indice_pairs_desc_,
        input_grad_desc_, indice_num_.data(), inverse_, &workspace_size_);
    if (MLUOP_STATUS_SUCCESS != status) {
      destroy();
      return expected_status_ == status;
    }
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_));
    status = mluOpIndiceConvolutionBackwardData(
        handle_, output_grad_desc_, output_grad_, filters_desc_, filters_,
        indice_pairs_desc_, indice_pairs_, indice_num_.data(), inverse_, sub_m_,
        workspace_, workspace_size_, input_grad_desc_, input_grad_);
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
    if (output_grad_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_grad_desc_));
      output_grad_desc_ = NULL;
    }
    if (output_grad_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_grad_));
      output_grad_ = NULL;
    }
    if (filters_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(filters_desc_));
      filters_desc_ = NULL;
    }
    if (filters_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(filters_));
      filters_ = NULL;
    }
    if (indice_pairs_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indice_pairs_desc_));
      indice_pairs_desc_ = NULL;
    }
    if (indice_pairs_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(indice_pairs_));
      indice_pairs_ = NULL;
    }
    if (workspace_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = NULL;
    }
    if (input_grad_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_grad_desc_));
      input_grad_desc_ = NULL;
    }
    if (input_grad_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_grad_));
      input_grad_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t output_grad_desc_ = NULL;
  void* output_grad_ = NULL;
  mluOpTensorDescriptor_t filters_desc_ = NULL;
  void* filters_ = NULL;
  mluOpTensorDescriptor_t indice_pairs_desc_ = NULL;
  void* indice_pairs_ = NULL;
  std::vector<int64_t> indice_num_;
  int64_t inverse_;
  int64_t sub_m_;
  void* workspace_ = NULL;
  size_t workspace_size_ = 64;
  mluOpTensorDescriptor_t input_grad_desc_ = NULL;
  void* input_grad_ = NULL;
  mluOpDevType_t device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(indice_convolution_backward_data_general, api_test) {
  EXPECT_TRUE(compute());
}

INSTANTIATE_TEST_CASE_P(
    zero_element_Y_0, indice_convolution_backward_data_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 21, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 21}))),
        testing::Values(IndiceConvolutionBackwardDataParam{
            std::vector<int64_t>({0, 0, 0, 0, 0, 0, 0, 0, 0}), 0, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_CO_0, indice_convolution_backward_data_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 21, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 21}))),
        testing::Values(IndiceConvolutionBackwardDataParam{
            std::vector<int64_t>({10, 10, 10, 10, 10, 10, 10, 10, 10}), 0, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_CI_0, indice_convolution_backward_data_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 0, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 0}))),
        testing::Values(IndiceConvolutionBackwardDataParam{
            std::vector<int64_t>({10, 10, 10, 10, 10, 10, 10, 10, 10}), 0, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_K_0, indice_convolution_backward_data_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 3, 21, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({0, 2, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 21}))),
        testing::Values(IndiceConvolutionBackwardDataParam{
            std::vector<int64_t>({10, 10, 10, 10, 10, 10, 10, 10, 10}), 0, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_L_0, indice_convolution_backward_data_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 21, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 21}))),
        testing::Values(IndiceConvolutionBackwardDataParam{
            std::vector<int64_t>({0, 0, 0, 0, 0, 0, 0, 0, 0}), 0, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    bad_dtype, indice_convolution_backward_data_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({10, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({3, 3, 21, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({10, 21}))),
        testing::Values(IndiceConvolutionBackwardDataParam{
            std::vector<int64_t>({10, 10, 10, 10, 10, 10, 10, 10, 10}), 0, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_output_grad_dtype_shape, indice_convolution_backward_data_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({10, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({10, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({10, 10, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 21, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 21}))),
        testing::Values(IndiceConvolutionBackwardDataParam{
            std::vector<int64_t>({10, 10, 10, 10, 10, 10, 10, 10, 10}), 0, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_filter_dtype_shape_layout, indice_convolution_backward_data_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 10}))),
        testing::Values(
            MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_HALF, 4,
                             std::vector<int>({3, 3, 21, 10})),
            MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_INT32, 4,
                             std::vector<int>({3, 3, 21, 10})),
            MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT, 4,
                             std::vector<int>({2, 3, 21, 10})),
            MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT, 4,
                             std::vector<int>({3, 2, 21, 10})),
            MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT, 4,
                             std::vector<int>({3, 3, 20, 10})),
            MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT, 4,
                             std::vector<int>({3, 3, 21, 9})),
            MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT, 3,
                             std::vector<int>({3, 21, 10})),
            MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({3, 3, 3, 3, 21, 10})),
            MLUOpTensorParam(MLUOP_LAYOUT_TNC, MLUOP_DTYPE_FLOAT, 4,
                             std::vector<int>({3, 3, 21, 10})),
            MLUOpTensorParam(MLUOP_LAYOUT_NTC, MLUOP_DTYPE_FLOAT, 4,
                             std::vector<int>({3, 3, 21, 10})),
            MLUOpTensorParam(MLUOP_LAYOUT_NC, MLUOP_DTYPE_FLOAT, 4,
                             std::vector<int>({3, 3, 21, 10})),
            MLUOpTensorParam(MLUOP_LAYOUT_NLC, MLUOP_DTYPE_FLOAT, 4,
                             std::vector<int>({3, 3, 21, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 21}))),
        testing::Values(IndiceConvolutionBackwardDataParam{
            std::vector<int64_t>({10, 10, 10, 10, 10, 10, 10, 10, 10}), 0, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_indices_pairs_dtype_shape, indice_convolution_backward_data_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 21, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({9, 2, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         3, std::vector<int>({9, 2, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({8, 2, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 1, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({9, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({9, 2, 10, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 21}))),
        testing::Values(IndiceConvolutionBackwardDataParam{
            std::vector<int64_t>({10, 10, 10, 10, 10, 10, 10, 10, 10}), 0, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_input_grad_dtype_shape, indice_convolution_backward_data_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 21, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({10, 21})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({10, 21})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({9, 21})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 20})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({10, 21, 1}))),
        testing::Values(IndiceConvolutionBackwardDataParam{
            std::vector<int64_t>({10, 10, 10, 10, 10, 10, 10, 10, 10}), 0, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_param, indice_convolution_backward_data_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 21, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 21}))),
        testing::Values(
            IndiceConvolutionBackwardDataParam{
                std::vector<int64_t>({-1, 10, 10, 10, 10, 10, 10, 10, 10}), 0,
                0},
            IndiceConvolutionBackwardDataParam{
                std::vector<int64_t>({20, 10, 10, 10, 10, 10, 10, 10, 10}), 0,
                0},
            IndiceConvolutionBackwardDataParam{
                std::vector<int64_t>({10, 10, 10, 10, 10, 10, 10, 10, 10}), -1,
                0},
            IndiceConvolutionBackwardDataParam{
                std::vector<int64_t>({10, 10, 10, 10, 10, 10, 10, 10, 10}), 0,
                -1},
            IndiceConvolutionBackwardDataParam{
                std::vector<int64_t>({10, 10, 10, 10, 10, 10, 10, 10, 10}), 0,
                2}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_param_inverse, indice_convolution_backward_data_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 21, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 21}))),
        testing::Values(IndiceConvolutionBackwardDataParam{
            std::vector<int64_t>({10, 10, 10, 10, 10, 10, 10, 10, 10}), 1, 0}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    bad_L_Y, indice_convolution_backward_data_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({9, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 21, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 21}))),
        testing::Values(IndiceConvolutionBackwardDataParam{
            std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1}), 0, 1}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_kh_kw, indice_convolution_backward_data_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 4, 21, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 2, 21, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({8, 2, 10}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 21}))),
        testing::Values(IndiceConvolutionBackwardDataParam{
            std::vector<int64_t>({10, 10, 10, 10, 10, 10, 10, 10}), 0, 1}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

}  // namespace mluopapitest
