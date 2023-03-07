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
typedef std::tuple<std::vector<int64_t>, int64_t, int64_t, int64_t>
    IndiceConvForwardAdditionalParam;

typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, IndiceConvForwardAdditionalParam,
                   mluOpDevType_t, mluOpStatus_t>
    IndiceConvolutionForwardParam;
class indice_convolution_forward_general
    : public testing::TestWithParam<IndiceConvolutionForwardParam> {
 public:
  void SetUp() {
    try {
      MLUOP_CHECK(mluOpCreate(&handle_));

      MLUOpTensorParam features_params = std::get<0>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&features_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          features_desc_, features_params.get_layout(),
          features_params.get_dtype(), features_params.get_dim_nb(),
          features_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(features_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&features_,
                       mluOpDataTypeBytes(features_params.get_dtype()) * 10));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&features_,
                               mluOpDataTypeBytes(features_params.get_dtype()) *
                                   mluOpGetTensorElementNum(features_desc_)));
      }

      MLUOpTensorParam filters_params = std::get<1>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&filters_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          filters_desc_, filters_params.get_layout(),
          filters_params.get_dtype(), filters_params.get_dim_nb(),
          filters_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(filters_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&filters_,
                       mluOpDataTypeBytes(filters_params.get_dtype()) * 10));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&filters_,
                               mluOpDataTypeBytes(filters_params.get_dtype()) *
                                   mluOpGetTensorElementNum(filters_desc_)));
      }

      MLUOpTensorParam indice_pairs_params = std::get<2>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&indice_pairs_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          indice_pairs_desc_, indice_pairs_params.get_layout(),
          indice_pairs_params.get_dtype(), indice_pairs_params.get_dim_nb(),
          indice_pairs_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(indice_pairs_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &indice_pairs_,
                mluOpDataTypeBytes(indice_pairs_params.get_dtype()) * 10));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&indice_pairs_,
                       mluOpDataTypeBytes(indice_pairs_params.get_dtype()) *
                           mluOpGetTensorElementNum(indice_pairs_desc_)));
      }

      MLUOpTensorParam features_out_params = std::get<3>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&features_out_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          features_out_desc_, features_out_params.get_layout(),
          features_out_params.get_dtype(), features_out_params.get_dim_nb(),
          features_out_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(features_out_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &features_out_,
                mluOpDataTypeBytes(features_out_params.get_dtype()) * 10));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&features_out_,
                       mluOpDataTypeBytes(features_out_params.get_dtype()) *
                           mluOpGetTensorElementNum(features_out_desc_)));
      }

      IndiceConvForwardAdditionalParam additoinal_param_ =
          std::get<4>(GetParam());
      std::tie(indice_num_, num_act_out_, inverse_, sub_m_) = additoinal_param_;

      target_device_ = std::get<5>(GetParam());
      expected_status_ = std::get<6>(GetParam());
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in indice_convolution_forward_general.";
    }
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status;
    status = mluOpGetIndiceConvolutionForwardWorkspaceSize(
        handle_, features_desc_, filters_desc_, indice_pairs_desc_,
        features_out_desc_, indice_num_.data(), num_act_out_, inverse_, sub_m_,
        &workspace_size_);
    if (status != MLUOP_STATUS_SUCCESS) {
      destroy();
      return expected_status_ == status;
    }
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_));

    status = mluOpIndiceConvolutionForward(
        handle_, features_desc_, features_, filters_desc_, filters_,
        indice_pairs_desc_, indice_pairs_, indice_num_.data(), num_act_out_,
        inverse_, sub_m_, workspace_, workspace_size_, features_out_desc_,
        features_out_);
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

      if (features_desc_) {
        VLOG(4) << "Destroy features_desc";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(features_desc_));
        features_desc_ = nullptr;
      }

      if (features_) {
        VLOG(4) << "Destroy features";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(features_));
        features_ = nullptr;
      }

      if (filters_desc_) {
        VLOG(4) << "Destroy filters_desc";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(filters_desc_));
        filters_desc_ = nullptr;
      }

      if (filters_) {
        VLOG(4) << "Destroy filters";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(filters_));
        filters_ = nullptr;
      }

      if (indice_pairs_desc_) {
        VLOG(4) << "Destroy indice_pairs_desc";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(indice_pairs_desc_));
        indice_pairs_desc_ = nullptr;
      }

      if (indice_pairs_) {
        VLOG(4) << "Destroy indice_pairs";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(indice_pairs_));
        indice_pairs_ = nullptr;
      }

      if (features_out_desc_) {
        VLOG(4) << "Destroy features_out_desc";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(features_out_desc_));
        features_out_desc_ = nullptr;
      }

      if (features_out_) {
        VLOG(4) << "Destroy features_out";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(features_out_));
        features_out_ = nullptr;
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in indice_convolution_forward_general";
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t features_desc_ = nullptr;
  void *features_ = nullptr;
  mluOpTensorDescriptor_t filters_desc_ = nullptr;
  void *filters_ = nullptr;
  mluOpTensorDescriptor_t indice_pairs_desc_ = nullptr;
  void *indice_pairs_ = nullptr;
  std::vector<int64_t> indice_num_;
  int64_t num_act_out_ = 10;
  int64_t inverse_ = 0;
  int64_t sub_m_ = 0;
  void *workspace_ = nullptr;
  size_t workspace_size_ = 64;
  mluOpTensorDescriptor_t features_out_desc_ = nullptr;
  void *features_out_ = nullptr;
  mluOpDevType_t target_device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(indice_convolution_forward_general, negative) { EXPECT_TRUE(compute()); }

INSTANTIATE_TEST_CASE_P(
    zero_element_1, indice_convolution_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NDHWC, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({5, 1, 2, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 2, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({9, 5}))),
        testing::Values(IndiceConvForwardAdditionalParam(
            std::vector<int64_t>({0, 0, 0, 0}), 9, 0, 0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_2, indice_convolution_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NDHWC, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({5, 1, 2, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 5}))),
        testing::Values(IndiceConvForwardAdditionalParam(
            std::vector<int64_t>({1, 1, 1, 1}), 0, 0, 0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    bad_features_dtype_dim_shape, indice_convolution_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({2, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({3, 2, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NDHWC, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({5, 1, 2, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({9, 5}))),
        testing::Values(IndiceConvForwardAdditionalParam(
            std::vector<int64_t>({1, 1, 1, 1}), 9, 0, 0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_filters_dtype, indice_convolution_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NDHWC, MLUOP_DTYPE_HALF,
                                         5, std::vector<int>({5, 1, 2, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({9, 5}))),
        testing::Values(IndiceConvForwardAdditionalParam(
            std::vector<int64_t>({1, 1, 1, 1}), 9, 0, 0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_filters_dim_layout, indice_convolution_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NDHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({5, 1, 2, 2, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({5, 1, 2, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({9, 5}))),
        testing::Values(IndiceConvForwardAdditionalParam(
            std::vector<int64_t>({1, 1, 1, 1}), 9, 0, 0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    bad_indice_pairs_dtype_dim_shape, indice_convolution_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NDHWC, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({5, 1, 2, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         3, std::vector<int>({4, 2, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({4, 2, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({5, 2, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 3, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({9, 5}))),
        testing::Values(IndiceConvForwardAdditionalParam(
            std::vector<int64_t>({1, 1, 1, 1}), 9, 0, 0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_features_output_dtype_dim_shape, indice_convolution_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NDHWC, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({5, 1, 2, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({9, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({9, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({10, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({9, 6}))),
        testing::Values(IndiceConvForwardAdditionalParam(
            std::vector<int64_t>({1, 1, 1, 1}), 9, 0, 0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_params, indice_convolution_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NDHWC, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({5, 1, 2, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({9, 5}))),
        testing::Values(
            IndiceConvForwardAdditionalParam(
                std::vector<int64_t>({-1, 1, 1, 1}), 9, 0, 0),
            IndiceConvForwardAdditionalParam(std::vector<int64_t>({1, 1, 4, 1}),
                                             9, 0, 0),
            IndiceConvForwardAdditionalParam(std::vector<int64_t>({1, 1, 1, 1}),
                                             19, 0, 0),
            IndiceConvForwardAdditionalParam(std::vector<int64_t>({1, 1, 1, 1}),
                                             9, 2, 0),
            IndiceConvForwardAdditionalParam(std::vector<int64_t>({1, 1, 1, 1}),
                                             9, 0, 2)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_params_inverse, indice_convolution_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NDHWC, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({5, 1, 2, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({9, 5}))),
        testing::Values(IndiceConvForwardAdditionalParam(
            std::vector<int64_t>({1, 1, 1, 1}), 9, 1, 0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    bad_large_tensor_features, indice_convolution_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({21474837, 100}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NDHWC, MLUOP_DTYPE_FLOAT,
                                         5,
                                         std::vector<int>({5, 1, 2, 2, 100}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3,
                                         std::vector<int>({4, 2, 21474837}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({9, 5}))),
        testing::Values(IndiceConvForwardAdditionalParam(
            std::vector<int64_t>({1, 1, 1, 1}), 9, 0, 0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    bad_large_tensor_filters, indice_convolution_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 3000}))),
        testing::Values(
            MLUOpTensorParam(MLUOP_LAYOUT_NDHWC, MLUOP_DTYPE_FLOAT, 5,
                             std::vector<int>({89479, 2, 2, 2, 3000}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({9, 89479}))),
        testing::Values(IndiceConvForwardAdditionalParam(
            std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1}), 9, 0, 0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    bad_large_tensor_indice_pairs, indice_convolution_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({89478486, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NDHWC, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({5, 3, 2, 2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3,
                                         std::vector<int>({12, 2, 89478486}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({9, 5}))),
        testing::Values(IndiceConvForwardAdditionalParam(
            std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}), 9, 0,
            0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    bad_large_tensor_features_output, indice_convolution_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NDHWC, MLUOP_DTYPE_FLOAT,
                                         5,
                                         std::vector<int>({500, 1, 2, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 2, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4294968, 500}))),
        testing::Values(IndiceConvForwardAdditionalParam(
            std::vector<int64_t>({1, 1, 1, 1}), 4294968, 0, 0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));
}  // namespace mluopapitest
