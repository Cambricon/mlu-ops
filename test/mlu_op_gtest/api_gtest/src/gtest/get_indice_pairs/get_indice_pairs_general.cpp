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

/*mluOpSetSparseConvolutionDescriptor(mluOpSparseConvolutionDescriptor_t desc,
                                    int dimNb,
                                    int batch_size,
                                    const int pad[],
                                    const int stride[],
                                    const int dilation[],
                                    const int input_space[],
                                    const int filter_space[],
                                    const int output_space[],
                                    const int subm,
                                    const int transpose,
                                    const int inverse)*/

typedef std::tuple<int, int, std::vector<int>, std::vector<int>,
                   std::vector<int>, std::vector<int>, std::vector<int>,
                   std::vector<int>, int, int, int>
    GetIndicePairsParam;

typedef std::tuple<GetIndicePairsParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, mluOpDevType_t,
                   mluOpStatus_t>
    GetIndicePairs;

class get_indice_pairs_general : public testing::TestWithParam<GetIndicePairs> {
 public:
  void SetUp() {
    device_ = std::get<5>(GetParam());
    expected_status_ = std::get<6>(GetParam());
    MLUOP_CHECK(mluOpCreate(&handle_));
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      return;
    }
    GetIndicePairsParam GetIndicePairs = std::get<0>(GetParam());
    std::tie(dimNb_, batch_size_, pad_, stride_, dilation_, input_space_,
             filter_space_, output_space_, subm_, transpose_, inverse_) =
        GetIndicePairs;
    MLUOP_CHECK(mluOpCreateSparseConvolutionDescriptor(&sparse_conv_desc_));

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
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(&indices_, mluOpGetTensorElementNum(indices_desc_) *
                                    mluOpDataTypeBytes(indices_dtype)))
    } else {
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&indices_, 4 * mluOpDataTypeBytes(indices_dtype)));
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
    } else {
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&indice_pairs_,
                             54 * mluOpDataTypeBytes(indice_pairs_dtype)));
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&out_indices_desc_));
    MLUOpTensorParam out_indices_params = std::get<3>(GetParam());
    mluOpTensorLayout_t out_indices_layout = out_indices_params.get_layout();
    mluOpDataType_t out_indices_dtype = out_indices_params.get_dtype();
    int out_indices_dim = out_indices_params.get_dim_nb();
    std::vector<int> out_indices_shape = out_indices_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(out_indices_desc_, out_indices_layout,
                                         out_indices_dtype, out_indices_dim,
                                         out_indices_shape.data()));
    uint64_t out_indices_ele_num = mluOpGetTensorElementNum(out_indices_desc_);
    if (out_indices_ele_num > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&out_indices_,
                             mluOpGetTensorElementNum(out_indices_desc_) *
                                 mluOpDataTypeBytes(out_indices_dtype)));
    } else {
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&out_indices_,
                             108 * mluOpDataTypeBytes(out_indices_dtype)));
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&indice_num_desc_));
    MLUOpTensorParam indice_num_params = std::get<4>(GetParam());
    mluOpTensorLayout_t indice_num_layout = indice_num_params.get_layout();
    mluOpDataType_t indice_num_dtype = indice_num_params.get_dtype();
    int indice_num_dim = indice_num_params.get_dim_nb();
    std::vector<int> indice_num_shape = indice_num_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(indice_num_desc_, indice_num_layout,
                                         indice_num_dtype, indice_num_dim,
                                         indice_num_shape.data()));
    uint64_t indice_num_ele_num = mluOpGetTensorElementNum(indice_num_desc_);
    if (indice_num_ele_num > 0) {
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(&indice_num_, mluOpGetTensorElementNum(indice_num_desc_) *
                                       mluOpDataTypeBytes(indice_num_dtype)));
    } else {
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(&indice_num_, 27 * mluOpDataTypeBytes(indice_num_dtype)));
    }
  }

  bool compute() {
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      destroy();
      return true;
    }

    mluOpStatus_t status = mluOpSetSparseConvolutionDescriptor(
        sparse_conv_desc_, dimNb_, batch_size_, pad_.data(), stride_.data(),
        dilation_.data(), input_space_.data(), filter_space_.data(),
        output_space_.data(), subm_, transpose_, inverse_);

    if (MLUOP_STATUS_SUCCESS != status) {
      destroy();
      return expected_status_ == status;
    }
    MLUOP_CHECK(mluOpSetSparseConvolutionDescriptor(
        sparse_conv_desc_, dimNb_, batch_size_, pad_.data(), stride_.data(),
        dilation_.data(), input_space_.data(), filter_space_.data(),
        output_space_.data(), subm_, transpose_, inverse_));

    status = mluOpGetIndicePairsWorkspaceSize(
        handle_, sparse_conv_desc_, indices_desc_, indice_pairs_desc_,
        out_indices_desc_, indice_num_desc_, &workspace_size_);
    if (MLUOP_STATUS_SUCCESS != status) {
      destroy();
      return expected_status_ == status;
    }
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_));

    status = mluOpGetIndicePairs(
        handle_, sparse_conv_desc_, indices_desc_, indices_, workspace_,
        workspace_size_, indice_pairs_desc_, indice_pairs_, out_indices_desc_,
        out_indices_, indice_num_desc_, indice_num_);
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
    if (sparse_conv_desc_) {
      MLUOP_CHECK(mluOpDestroySparseConvolutionDescriptor(sparse_conv_desc_));
      sparse_conv_desc_ = nullptr;
    }
    if (indices_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indices_desc_));
      indices_desc_ = NULL;
    }
    if (indices_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(indices_));
      indices_ = NULL;
    }
    if (workspace_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = nullptr;
    }
    if (indice_pairs_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indice_pairs_desc_));
      indice_pairs_desc_ = NULL;
    }
    if (indice_pairs_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(indice_pairs_));
      indice_pairs_ = NULL;
    }
    if (out_indices_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(out_indices_desc_));
      out_indices_desc_ = NULL;
    }
    if (out_indices_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(out_indices_));
      out_indices_ = NULL;
    }
    if (indice_num_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indice_num_desc_));
      indice_num_desc_ = NULL;
    }
    if (indice_num_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(indice_num_));
      indice_num_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpSparseConvolutionDescriptor_t sparse_conv_desc_ = NULL;
  mluOpTensorDescriptor_t indices_desc_ = NULL;
  void* indices_ = NULL;
  void* workspace_ = NULL;
  size_t workspace_size_ = 64;
  mluOpTensorDescriptor_t indice_pairs_desc_ = NULL;
  void* indice_pairs_ = NULL;
  mluOpTensorDescriptor_t out_indices_desc_ = NULL;
  void* out_indices_ = NULL;
  mluOpTensorDescriptor_t indice_num_desc_ = NULL;
  void* indice_num_ = NULL;
  int dimNb_;
  int batch_size_;
  std::vector<int> pad_;
  std::vector<int> stride_;
  std::vector<int> dilation_;
  std::vector<int> input_space_;
  std::vector<int> filter_space_;
  std::vector<int> output_space_;
  int subm_;
  int transpose_;
  int inverse_;
  mluOpDevType_t device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(get_indice_pairs_general, api_test) { EXPECT_TRUE(compute()); }

INSTANTIATE_TEST_CASE_P(
    zero_element_input_active_in_0, get_indice_pairs_general,
    testing::Combine(
        testing::Values(GetIndicePairsParam{
            5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({3, 3, 3}), std::vector<int>({1, 1, 1}), 0, 0, 0}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({0, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({27, 2, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({27}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_output_active_num_0, get_indice_pairs_general,
    testing::Combine(
        testing::Values(GetIndicePairsParam{
            5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 0, 0}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({27, 2, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({0, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({27}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    bad_dtype, get_indice_pairs_general,
    testing::Combine(
        testing::Values(GetIndicePairsParam{
            5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 0, 0}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         2, std::vector<int>({1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         3, std::vector<int>({27, 2, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         2, std::vector<int>({27, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         1, std::vector<int>({27}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_indices_dtype_shape, get_indice_pairs_general,
    testing::Combine(
        testing::Values(GetIndicePairsParam{
            5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 0, 0}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 4, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({27, 2, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({27, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({27}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_indice_pairs_dtype_shape, get_indice_pairs_general,
    testing::Combine(
        testing::Values(GetIndicePairsParam{
            5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 0, 0}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         3, std::vector<int>({27, 2, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({28, 2, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({27, 1, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({27, 2, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({27, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({27, 2, 1, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({27, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({27}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_out_indices_dtype_shape, get_indice_pairs_general,
    testing::Combine(
        testing::Values(GetIndicePairsParam{
            5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 0, 0}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({27, 2, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({27, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({28, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({27, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({27})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({27, 4, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({27}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_indice_num_dtype_shape, get_indice_pairs_general,
    testing::Combine(
        testing::Values(GetIndicePairsParam{
            5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 0, 0}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({27, 2, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({27, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         1, std::vector<int>({27})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({28})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({27, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_param, get_indice_pairs_general,
    testing::Combine(
        testing::Values(
            GetIndicePairsParam{
                4, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 0,
                0},
            GetIndicePairsParam{
                5, 0, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 0,
                0},
            GetIndicePairsParam{
                5, -1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 0,
                0},
            GetIndicePairsParam{
                5, 1, std::vector<int>({-1, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 0,
                0},
            GetIndicePairsParam{
                5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 0, -1}),
                std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 0,
                0},
            GetIndicePairsParam{
                5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({1, 0, -1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 0,
                0},
            GetIndicePairsParam{
                5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 1,
                0},
            GetIndicePairsParam{
                5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 0,
                1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({27, 2, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({27}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_input_active_in, get_indice_pairs_general,
    testing::Combine(
        testing::Values(GetIndicePairsParam{
            5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({1, 1, 1}), std::vector<int>({3, 3, 3}), 0, 0, 0}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({27, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_K, get_indice_pairs_general,
    testing::Combine(
        testing::Values(GetIndicePairsParam{
            5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({1, 1, 1}), std::vector<int>({3, 3, 3}), 0, 0, 0}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 2, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({27, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({2}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_sub_m_1_param_pad_stride_dilation, get_indice_pairs_general,
    testing::Combine(
        testing::Values(
            GetIndicePairsParam{
                5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({2, 1, 1}),
                std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({3, 3, 3}), std::vector<int>({1, 1, 1}), 1, 0,
                0},
            GetIndicePairsParam{
                5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({1, 1, 2}), std::vector<int>({1, 1, 1}),
                std::vector<int>({3, 3, 3}), std::vector<int>({1, 1, 1}), 1, 0,
                0}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({27, 2, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({27}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_sub_m_0_param_pad_stride_dilation, get_indice_pairs_general,
    testing::Combine(
        testing::Values(
            GetIndicePairsParam{
                5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({2, 1, 1}),
                std::vector<int>({2, 1, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 0,
                0},
            GetIndicePairsParam{
                5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 2, 1}),
                std::vector<int>({1, 2, 1}), std::vector<int>({1, 1, 1}),
                std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 0,
                0},
            GetIndicePairsParam{
                5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 2}),
                std::vector<int>({1, 1, 2}), std::vector<int>({1, 1, 1}),
                std::vector<int>({3, 3, 3}), std::vector<int>({3, 3, 3}), 0, 0,
                0}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({27, 2, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({27, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({27}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_sub_m_1_param_space, get_indice_pairs_general,
    testing::Combine(
        testing::Values(GetIndicePairsParam{
            5, 1, std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({1, 1, 1}), std::vector<int>({1, 1, 1}),
            std::vector<int>({1, 1, 1}), std::vector<int>({3, 3, 3}), 1, 0, 0}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({1, 2, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({27, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));
}  // namespace mluopapitest

