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
                   MLUOpTensorParam, int64_t, mluOpDevType_t, mluOpStatus_t>
    IndiceConvBpFilterParam;
class indice_convolution_backward_filter_general
    : public testing::TestWithParam<IndiceConvBpFilterParam> {
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

      MLUOpTensorParam output_grad_params = std::get<1>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_grad_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          output_grad_desc_, output_grad_params.get_layout(),
          output_grad_params.get_dtype(), output_grad_params.get_dim_nb(),
          output_grad_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(output_grad_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &output_grad_,
                mluOpDataTypeBytes(output_grad_params.get_dtype()) * 10));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&output_grad_,
                       mluOpDataTypeBytes(output_grad_params.get_dtype()) *
                           mluOpGetTensorElementNum(output_grad_desc_)));
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

      MLUOpTensorParam filters_grad_params = std::get<3>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&filters_grad_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          filters_grad_desc_, filters_grad_params.get_layout(),
          filters_grad_params.get_dtype(), filters_grad_params.get_dim_nb(),
          filters_grad_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(filters_grad_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &filters_grad_,
                mluOpDataTypeBytes(filters_grad_params.get_dtype()) * 10));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&filters_grad_,
                       mluOpDataTypeBytes(filters_grad_params.get_dtype()) *
                           mluOpGetTensorElementNum(filters_grad_desc_)));
      }

      std::vector<int64_t> num = {1, 2, 3, 4, 5, 6, 7, 8, 9};
      for (int i = 0; i < num.size(); i++) {
        indice_num_.push_back(num[i]);
      }
      inverse_ = std::get<4>(GetParam());
      target_device_ = std::get<5>(GetParam());
      expected_status_ = std::get<6>(GetParam());
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in indice_convolution_backward_filter_general.";
    }
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status;
    status = mluOpGetIndiceConvolutionBackwardFilterWorkspaceSize(
        handle_, features_desc_, output_grad_desc_, indice_pairs_desc_,
        filters_grad_desc_, indice_num_.data(), inverse_, sub_m_,
        &workspace_size_);
    if (status != MLUOP_STATUS_SUCCESS) {
      destroy();
      return expected_status_ == status;
    }
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_));

    status = mluOpIndiceConvolutionBackwardFilter(
        handle_, features_desc_, features_, output_grad_desc_, output_grad_,
        indice_pairs_desc_, indice_pairs_, indice_num_.data(), inverse_, sub_m_,
        workspace_, workspace_size_, filters_grad_desc_, filters_grad_);
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

      if (output_grad_desc_) {
        VLOG(4) << "Destroy output_grad_desc";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_grad_desc_));
        output_grad_desc_ = nullptr;
      }

      if (output_grad_) {
        VLOG(4) << "Destroy output_grad";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_grad_));
        output_grad_ = nullptr;
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

      if (filters_grad_desc_) {
        VLOG(4) << "Destroy filters_grad_desc";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(filters_grad_desc_));
        filters_grad_desc_ = nullptr;
      }

      if (filters_grad_) {
        VLOG(4) << "Destroy filters_grad";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(filters_grad_));
        filters_grad_ = nullptr;
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in indice_convolution_backward_filter_general";
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t features_desc_ = nullptr;
  void *features_ = nullptr;
  mluOpTensorDescriptor_t output_grad_desc_ = nullptr;
  void *output_grad_ = nullptr;
  mluOpTensorDescriptor_t indice_pairs_desc_ = nullptr;
  void *indice_pairs_ = nullptr;
  std::vector<int64_t> indice_num_;
  int64_t inverse_ = 0;
  int64_t sub_m_ = 0;
  void *workspace_ = nullptr;
  size_t workspace_size_ = 1;
  mluOpTensorDescriptor_t filters_grad_desc_ = nullptr;
  void *filters_grad_ = nullptr;
  mluOpDevType_t target_device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(indice_convolution_backward_filter_general, negative) {
  EXPECT_TRUE(compute());
}

INSTANTIATE_TEST_CASE_P(
    zero_element_1, indice_convolution_backward_filter_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 5, 3}))),
        testing::Values(0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_2, indice_convolution_backward_filter_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 5, 3}))),
        testing::Values(0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_3, indice_convolution_backward_filter_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({0, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 0, 5, 3}))),
        testing::Values(0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    bad_unsupport_dtype, indice_convolution_backward_filter_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY,
                                         MLUOP_DTYPE_COMPLEX_FLOAT, 2,
                                         std::vector<int>({1, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY,
                                         MLUOP_DTYPE_COMPLEX_FLOAT, 2,
                                         std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY,
                                         MLUOP_DTYPE_COMPLEX_FLOAT, 4,
                                         std::vector<int>({3, 3, 5, 3}))),
        testing::Values(0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_features_dim_shape_dtype, indice_convolution_backward_filter_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({3, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({3, 5, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 15})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({13, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 5, 3}))),
        testing::Values(0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_indice_pairs_dim_shape_dtype,
    indice_convolution_backward_filter_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         3, std::vector<int>({9, 2, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({11, 2, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 1, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({9, 2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 5, 3}))),
        testing::Values(0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_output_grad_dim_shape_dtype, indice_convolution_backward_filter_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({3, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({3, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 5, 3}))),
        testing::Values(0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_filters_grad_dim_shape_dtype,
    indice_convolution_backward_filter_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         4, std::vector<int>({3, 3, 5, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({3, 3, 5, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 15, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 5, 13}))),
        testing::Values(0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_inverse, indice_convolution_backward_filter_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 5, 3}))),
        testing::Values(1), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_large_tensor_features, indice_convolution_backward_filter_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1147483648, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 5, 3}))),
        testing::Values(0), testing::Values(MLUOP_MLU590),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    bad_large_tensor_output_grad, indice_convolution_backward_filter_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({715827883, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 5, 3}))),
        testing::Values(0), testing::Values(MLUOP_MLU590),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    bad_large_tensor_indice_pairs, indice_convolution_backward_filter_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3,
                                         std::vector<int>({715827883, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 5, 3}))),
        testing::Values(0), testing::Values(MLUOP_MLU590),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    bad_large_tensor_filters_grad, indice_convolution_backward_filter_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({9, 2, 3}))),
        testing::Values(
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 4,
                             std::vector<int>({47721859, 3, 5, 3}))),
        testing::Values(0), testing::Values(MLUOP_MLU590),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));
}  // namespace mluopapitest
