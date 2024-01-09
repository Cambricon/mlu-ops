/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
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

typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, int, int, mluOpDevType_t, mluOpStatus_t>
    MaskedIm2colForward;
class masked_im2col_forward_general
    : public testing::TestWithParam<MaskedIm2colForward> {
 public:
  void SetUp() {
    try {
      MLUOP_CHECK(mluOpCreate(&handle_));

      MLUOpTensorParam feature_params = std::get<0>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&feature_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          feature_desc_, feature_params.get_layout(),
          feature_params.get_dtype(), feature_params.get_dim_nb(),
          feature_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(feature_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&feature_,
                       mluOpDataTypeBytes(feature_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&feature_,
                               mluOpDataTypeBytes(feature_params.get_dtype()) *
                                   mluOpGetTensorElementNum(feature_desc_)));
      }

      MLUOpTensorParam mask_h_idx_params = std::get<1>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&mask_h_idx_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          mask_h_idx_desc_, mask_h_idx_params.get_layout(),
          mask_h_idx_params.get_dtype(), mask_h_idx_params.get_dim_nb(),
          mask_h_idx_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(mask_h_idx_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&mask_h_idx_,
                       mluOpDataTypeBytes(mask_h_idx_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&mask_h_idx_,
                       mluOpDataTypeBytes(mask_h_idx_params.get_dtype()) *
                           mluOpGetTensorElementNum(mask_h_idx_desc_)));
      }

      MLUOpTensorParam mask_w_idx_params = std::get<2>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&mask_w_idx_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          mask_w_idx_desc_, mask_w_idx_params.get_layout(),
          mask_w_idx_params.get_dtype(), mask_w_idx_params.get_dim_nb(),
          mask_w_idx_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(mask_w_idx_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&mask_w_idx_,
                       mluOpDataTypeBytes(mask_w_idx_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&mask_w_idx_,
                       mluOpDataTypeBytes(mask_w_idx_params.get_dtype()) *
                           mluOpGetTensorElementNum(mask_w_idx_desc_)));
      }

      MLUOpTensorParam data_col_params = std::get<3>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_col_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          data_col_desc_, data_col_params.get_layout(),
          data_col_params.get_dtype(), data_col_params.get_dim_nb(),
          data_col_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(data_col_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&data_col_,
                       mluOpDataTypeBytes(data_col_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&data_col_,
                               mluOpDataTypeBytes(data_col_params.get_dtype()) *
                                   mluOpGetTensorElementNum(data_col_desc_)));
      }
      kernel_w_ = std::get<4>(GetParam());
      kernel_h_ = std::get<5>(GetParam());

      target_device_ = std::get<6>(GetParam());
      expected_status_ = std::get<7>(GetParam());

      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&workspace_, MLUOP_DTYPE_FLOAT * workspace_size_));
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in masked_im2col_forward_general.";
    }
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpMaskedIm2colForward(
        handle_, feature_desc_, feature_, mask_h_idx_desc_, mask_h_idx_,
        mask_w_idx_desc_, mask_w_idx_, kernel_h_, kernel_w_, pad_h_, pad_w_,
        workspace_, workspace_size_, data_col_desc_, data_col_);
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

    if (feature_desc_) {
      VLOG(4) << "Destroy feature_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(feature_desc_));
      feature_desc_ = nullptr;
    }

    if (feature_) {
      VLOG(4) << "Destroy feature_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(feature_));
      feature_ = nullptr;
    }

    if (mask_h_idx_desc_) {
      VLOG(4) << "Destroy mask_h_idx_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(mask_h_idx_desc_));
      mask_h_idx_desc_ = nullptr;
    }

    if (mask_h_idx_) {
      VLOG(4) << "Destroy mask_h_idx_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(mask_h_idx_));
      mask_h_idx_ = nullptr;
    }

    if (mask_w_idx_desc_) {
      VLOG(4) << "Destroy mask_w_idx_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(mask_w_idx_desc_));
      mask_w_idx_desc_ = nullptr;
    }

    if (mask_w_idx_) {
      VLOG(4) << "Destroy mask_w_idx_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(mask_w_idx_));
      mask_w_idx_ = nullptr;
    }

    if (data_col_desc_) {
      VLOG(4) << "Destroy data_col_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(data_col_desc_));
      data_col_desc_ = nullptr;
    }

    if (data_col_) {
      VLOG(4) << "Destroy data_col_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(data_col_));
      data_col_ = nullptr;
    }

    if (workspace_) {
      VLOG(4) << "Destroy workspace_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t feature_desc_ = nullptr;
  void *feature_ = nullptr;
  mluOpTensorDescriptor_t mask_h_idx_desc_ = nullptr;
  void *mask_h_idx_ = nullptr;
  mluOpTensorDescriptor_t mask_w_idx_desc_ = nullptr;
  void *mask_w_idx_ = nullptr;
  mluOpTensorDescriptor_t data_col_desc_ = nullptr;
  void *data_col_ = nullptr;
  int kernel_w_ = 2;
  int kernel_h_ = 2;
  int pad_w_ = 2;
  int pad_h_ = 1;
  size_t workspace_size_ = 10;
  void *workspace_ = nullptr;
  mluOpDevType_t target_device_;
  mluOpStatus_t expected_status_;
};

TEST_P(masked_im2col_forward_general, negative) { EXPECT_TRUE(compute()); }
INSTANTIATE_TEST_CASE_P(
    zero_element, masked_im2col_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({144, 0})}),
        testing::Values(2), testing::Values(2),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    feature_error_layout, masked_im2col_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({144, 88})}),
        testing::Values(2), testing::Values(2),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    feature_error_dtype, masked_im2col_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({1, 36, 37, 3})},
                        MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_HALF, 4,
                                         std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({144, 88})}),
        testing::Values(2), testing::Values(2),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    mask_h_error_dtype, masked_im2col_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({144, 88})}),
        testing::Values(2), testing::Values(2),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    mask_w_error_dtype, masked_im2col_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({144, 88})}),
        testing::Values(2), testing::Values(2),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    output_error_dtype, masked_im2col_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({144, 88})}),
        testing::Values(2), testing::Values(2),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    mask_w_error_dim, masked_im2col_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({88, 88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({144, 88})}),
        testing::Values(2), testing::Values(2),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    mask_h_error_dim, masked_im2col_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({88, 1})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({144, 88})}),
        testing::Values(2), testing::Values(2),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    output_error_dim, masked_im2col_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({88})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({144, 88, 1})}),
        testing::Values(2), testing::Values(2),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    feature_zero_element, masked_im2col_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 36, 37, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({144, 88})}),
        testing::Values(2), testing::Values(2),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    output_zero_element, masked_im2col_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 88})}),
        testing::Values(2), testing::Values(2),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    mask_error_shape, masked_im2col_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({66})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({144, 88})}),
        testing::Values(2), testing::Values(2),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    output_error_shape, masked_im2col_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({14, 88})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({144, 66})}),
        testing::Values(2), testing::Values(2),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    error_kernalw_shape, masked_im2col_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({144, 88})}),
        testing::Values(-1), testing::Values(-4),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    error_kernalh_shape, masked_im2col_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({144, 88})}),
        testing::Values(-2), testing::Values(-2),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));
}  // namespace mluopapitest
