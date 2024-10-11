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
                   MLUOpTensorParam, mluOpDevType_t, mluOpStatus_t>
    MaskedCol2imForward;
class masked_col2im_forward_general
    : public testing::TestWithParam<MaskedCol2imForward> {
 public:
  void SetUp() {
    try {
      MLUOP_CHECK(mluOpCreate(&handle_));

      MLUOpTensorParam col_params = std::get<0>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&col_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          col_desc_, col_params.get_layout(), col_params.get_dtype(),
          col_params.get_dim_nb(), col_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(col_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&col_, mluOpDataTypeBytes(col_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&col_, mluOpDataTypeBytes(col_params.get_dtype()) *
                                  mluOpGetTensorElementNum(col_desc_)));
      }

      MLUOpTensorParam mask_h_idx_params = std::get<1>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&mask_h_idx_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          mask_h_idx_desc_, mask_h_idx_params.get_layout(),
          mask_h_idx_params.get_dtype(), mask_h_idx_params.get_dim_nb(),
          mask_h_idx_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(mask_h_idx_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&mask_h_idx_,
                       mluOpDataTypeBytes(mask_h_idx_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            cnrtSuccess ==
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
            cnrtSuccess ==
            cnrtMalloc(&mask_w_idx_,
                       mluOpDataTypeBytes(mask_w_idx_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&mask_w_idx_,
                       mluOpDataTypeBytes(mask_w_idx_params.get_dtype()) *
                           mluOpGetTensorElementNum(mask_w_idx_desc_)));
      }

      MLUOpTensorParam im_params = std::get<3>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&im_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          im_desc_, im_params.get_layout(), im_params.get_dtype(),
          im_params.get_dim_nb(), im_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(im_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&im_, mluOpDataTypeBytes(im_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&im_, mluOpDataTypeBytes(im_params.get_dtype()) *
                                         mluOpGetTensorElementNum(im_desc_)));
      }

      target_device_ = std::get<4>(GetParam());
      expected_status_ = std::get<5>(GetParam());

      GTEST_CHECK(cnrtSuccess ==
                  cnrtMalloc(&workspace_, MLUOP_DTYPE_FLOAT * workspace_size_));
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in masked_col2im_forward_general.";
    }
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status =
        mluOpMaskedCol2imForward(handle_, col_desc_, col_, mask_h_idx_desc_,
                                 mask_h_idx_, mask_w_idx_desc_, mask_w_idx_,
                                 workspace_size_, workspace_, im_desc_, im_);
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

    if (col_desc_) {
      VLOG(4) << "Destroy col_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(col_desc_));
      col_desc_ = nullptr;
    }

    if (col_) {
      VLOG(4) << "Destroy col_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(col_));
      col_ = nullptr;
    }

    if (mask_h_idx_desc_) {
      VLOG(4) << "Destroy mask_h_idx_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(mask_h_idx_desc_));
      mask_h_idx_desc_ = nullptr;
    }

    if (mask_h_idx_) {
      VLOG(4) << "Destroy mask_h_idx_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(mask_h_idx_));
      mask_h_idx_ = nullptr;
    }

    if (mask_w_idx_desc_) {
      VLOG(4) << "Destroy mask_w_idx_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(mask_w_idx_desc_));
      mask_w_idx_desc_ = nullptr;
    }

    if (mask_w_idx_) {
      VLOG(4) << "Destroy mask_w_idx_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(mask_w_idx_));
      mask_w_idx_ = nullptr;
    }

    if (im_desc_) {
      VLOG(4) << "Destroy im_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(im_desc_));
      im_desc_ = nullptr;
    }

    if (im_) {
      VLOG(4) << "Destroy im_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(im_));
      im_ = nullptr;
    }

    if (workspace_) {
      VLOG(4) << "Destroy workspace_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(workspace_));
      workspace_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t col_desc_ = nullptr;
  void *col_ = nullptr;
  mluOpTensorDescriptor_t mask_h_idx_desc_ = nullptr;
  void *mask_h_idx_ = nullptr;
  mluOpTensorDescriptor_t mask_w_idx_desc_ = nullptr;
  void *mask_w_idx_ = nullptr;
  mluOpTensorDescriptor_t im_desc_ = nullptr;
  void *im_ = nullptr;
  size_t workspace_size_ = 10;
  void *workspace_ = nullptr;
  mluOpDevType_t target_device_;
  mluOpStatus_t expected_status_;
};

TEST_P(masked_col2im_forward_general, negative) { EXPECT_TRUE(compute()); }

INSTANTIATE_TEST_CASE_P(
    zero_element, masked_col2im_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({36, 0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    im_error_layout, masked_col2im_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({36, 88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    col_error_dtype, masked_col2im_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({36, 88})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({36, 88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    mask_h_error_dtype, masked_col2im_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({36, 88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    mask_w_error_dtype, masked_col2im_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({36, 88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    im_error_dtype, masked_col2im_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({36, 88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    mask_h_error_dim, masked_col2im_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({36, 88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({88, 88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    mask_w_error_dim, masked_col2im_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({36, 88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({88, 88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    col_error_dim, masked_col2im_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({36})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({36, 88, 88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    im_zero_element, masked_col2im_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({36, 88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 0, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    col_zero_element, masked_col2im_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 0, 37, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    mask_error_shape, masked_col2im_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({36, 88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({66})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    col_error_shape, masked_col2im_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({36, 66})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({16, 88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({88})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 36, 37, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));
}  // namespace mluopapitest
