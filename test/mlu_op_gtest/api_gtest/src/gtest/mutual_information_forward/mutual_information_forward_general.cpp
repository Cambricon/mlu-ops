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

typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, mluOpDevType_t,
                   mluOpStatus_t>
    MutualInformationForward;
class mutual_information_forward_general
    : public testing::TestWithParam<MutualInformationForward> {
 public:
  void SetUp() {
    try {
      MLUOP_CHECK(mluOpCreate(&handle_));

      MLUOpTensorParam px_params = std::get<0>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&px_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          px_desc_, px_params.get_layout(), px_params.get_dtype(),
          px_params.get_dim_nb(), px_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(px_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&px_, mluOpDataTypeBytes(px_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&px_, mluOpDataTypeBytes(px_params.get_dtype()) *
                                         mluOpGetTensorElementNum(px_desc_)));
      }

      MLUOpTensorParam py_params = std::get<1>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&py_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          py_desc_, py_params.get_layout(), py_params.get_dtype(),
          py_params.get_dim_nb(), py_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(py_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&py_, mluOpDataTypeBytes(py_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&py_, mluOpDataTypeBytes(py_params.get_dtype()) *
                                         mluOpGetTensorElementNum(py_desc_)));
      }

      MLUOpTensorParam opt_boundary_params = std::get<2>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&opt_boundary_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          opt_boundary_desc_, opt_boundary_params.get_layout(),
          opt_boundary_params.get_dtype(), opt_boundary_params.get_dim_nb(),
          opt_boundary_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(opt_boundary_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(
                &opt_boundary_,
                mluOpDataTypeBytes(opt_boundary_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&opt_boundary_,
                       mluOpDataTypeBytes(opt_boundary_params.get_dtype()) *
                           mluOpGetTensorElementNum(opt_boundary_desc_)));
      }

      MLUOpTensorParam p_params = std::get<3>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&p_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          p_desc_, p_params.get_layout(), p_params.get_dtype(),
          p_params.get_dim_nb(), p_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(p_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&p_, mluOpDataTypeBytes(p_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&p_, mluOpDataTypeBytes(p_params.get_dtype()) *
                                        mluOpGetTensorElementNum(p_desc_)));
      }

      MLUOpTensorParam ans_params = std::get<4>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&ans_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          ans_desc_, ans_params.get_layout(), ans_params.get_dtype(),
          ans_params.get_dim_nb(), ans_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(ans_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&ans_, mluOpDataTypeBytes(ans_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&ans_, mluOpDataTypeBytes(ans_params.get_dtype()) *
                                  mluOpGetTensorElementNum(ans_desc_)));
      }

      target_device_ = std::get<5>(GetParam());
      expected_status_ = std::get<6>(GetParam());

      GTEST_CHECK(cnrtSuccess ==
                  cnrtMalloc(&workspace_, MLUOP_DTYPE_FLOAT * workspace_size_));
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in mutual_information_forward_general.";
    }
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpMutualInformationForward(
        handle_, px_desc_, px_, py_desc_, py_, opt_boundary_desc_,
        opt_boundary_, p_desc_, p_, workspace_, workspace_size_, ans_desc_,
        ans_);
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

    if (px_desc_) {
      VLOG(4) << "Destroy px_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(px_desc_));
      px_desc_ = nullptr;
    }

    if (px_) {
      VLOG(4) << "Destroy px_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(px_));
      px_ = nullptr;
    }

    if (py_desc_) {
      VLOG(4) << "Destroy py_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(py_desc_));
      py_desc_ = nullptr;
    }

    if (py_) {
      VLOG(4) << "Destroy py_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(py_));
      py_ = nullptr;
    }

    if (opt_boundary_desc_) {
      VLOG(4) << "Destroy opt_boundary_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(opt_boundary_desc_));
      opt_boundary_desc_ = nullptr;
    }

    if (opt_boundary_) {
      VLOG(4) << "Destroy opt_boundary_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(opt_boundary_));
      opt_boundary_ = nullptr;
    }

    if (p_desc_) {
      VLOG(4) << "Destroy p_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(p_desc_));
      p_desc_ = nullptr;
    }

    if (p_) {
      VLOG(4) << "Destroy p_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(p_));
      p_ = nullptr;
    }

    if (ans_desc_) {
      VLOG(4) << "Destroy ans_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(ans_desc_));
      ans_desc_ = nullptr;
    }

    if (ans_) {
      VLOG(4) << "Destroy ans_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(ans_));
      ans_ = nullptr;
    }

    if (workspace_) {
      VLOG(4) << "Destroy workspace_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(workspace_));
      workspace_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t px_desc_ = nullptr;
  void *px_ = nullptr;
  mluOpTensorDescriptor_t py_desc_ = nullptr;
  void *py_ = nullptr;
  mluOpTensorDescriptor_t opt_boundary_desc_ = nullptr;
  void *opt_boundary_ = nullptr;
  mluOpTensorDescriptor_t p_desc_ = nullptr;
  void *p_ = nullptr;
  mluOpTensorDescriptor_t ans_desc_ = nullptr;
  void *ans_ = nullptr;
  size_t workspace_size_ = 10;
  void *workspace_ = nullptr;
  mluOpDevType_t target_device_;
  mluOpStatus_t expected_status_;
};

TEST_P(mutual_information_forward_general, negative) { EXPECT_TRUE(compute()); }

INSTANTIATE_TEST_CASE_P(
    zero_element_1, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({0, 15, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({0, 16, 104})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({0, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({0, 16, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({0})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_2, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 0, 1})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 1, 0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 1, 1})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    py_error_dim, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 15, 105, 1})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 15})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 104})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    px_error_dim, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 15, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 104, 1})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 16})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    opt_boundary_error_dim, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 15, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 104})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         1, std::vector<int>({4})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         3, std::vector<int>({4, 4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    p_error_dim, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 15, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 104})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 105, 1})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 16})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    ans_error_dim, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 15, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 104})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    py_error_shape1, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 15, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({3, 16, 104})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 15, 104})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    py_error_shape2, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 15, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 107})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    opt_boundary_error_shape, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 15, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 104})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({3, 4})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({4, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    p_error_shape, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 15, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 104})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({3, 16, 105})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 20, 105})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 10})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    ans_error_shape, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 15, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 104})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    px_error_dtype, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 15, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 104})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    py_error_dtype, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 15, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_DOUBLE,
                                         3, std::vector<int>({4, 16, 104})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    opt_boundary_error_dtype, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 15, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 104})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    p_error_dtype, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 15, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 104})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 16, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    ans_error_dtype, mutual_information_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 15, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 104})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 105})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

}  // namespace mluopapitest
