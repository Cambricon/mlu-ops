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

typedef std::tuple<mluOpComputationPreference_t, mluOpLossReduction_t,
                   MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, float, mluOpDevType_t, mluOpStatus_t>
    FocalLossSigmoidBackward;
class focal_loss_sigmoid_backward_general
    : public testing::TestWithParam<FocalLossSigmoidBackward> {
 public:
  void SetUp() {
    try {
      MLUOP_CHECK(mluOpCreate(&handle_));

      prefer_ = std::get<0>(GetParam());
      reduction_ = std::get<1>(GetParam());

      MLUOpTensorParam input_params = std::get<2>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          input_desc_, input_params.get_layout(), input_params.get_dtype(),
          input_params.get_dim_nb(), input_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(input_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&input_,
                       mluOpDataTypeBytes(input_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&input_, mluOpDataTypeBytes(input_params.get_dtype()) *
                                    mluOpGetTensorElementNum(input_desc_)));
      }

      MLUOpTensorParam target_params = std::get<3>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&target_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          target_desc_, target_params.get_layout(), target_params.get_dtype(),
          target_params.get_dim_nb(), target_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(target_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&target_,
                       mluOpDataTypeBytes(target_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&target_, mluOpDataTypeBytes(target_params.get_dtype()) *
                                     mluOpGetTensorElementNum(target_desc_)));
      }

      MLUOpTensorParam weight_params = std::get<4>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&weight_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          weight_desc_, weight_params.get_layout(), weight_params.get_dtype(),
          weight_params.get_dim_nb(), weight_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(weight_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&weight_,
                       mluOpDataTypeBytes(weight_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&weight_, mluOpDataTypeBytes(weight_params.get_dtype()) *
                                     mluOpGetTensorElementNum(weight_desc_)));
      }

      MLUOpTensorParam grad_input_params = std::get<5>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_input_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_input_desc_, grad_input_params.get_layout(),
          grad_input_params.get_dtype(), grad_input_params.get_dim_nb(),
          grad_input_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(grad_input_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&grad_input_,
                       mluOpDataTypeBytes(grad_input_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&grad_input_,
                       mluOpDataTypeBytes(grad_input_params.get_dtype()) *
                           mluOpGetTensorElementNum(grad_input_desc_)));
      }

      gamma_ = std::get<6>(GetParam());
      target_device_ = std::get<7>(GetParam());
      expected_status_ = std::get<8>(GetParam());
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in focal_loss_sigmoid_backward_general.";
    }
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpFocalLossSigmoidBackward(
        handle_, prefer_, reduction_, input_desc_, input_, target_desc_,
        target_, weight_desc_, weight_, alpha_, gamma_, grad_input_desc_,
        grad_input_);
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

    if (input_desc_) {
      VLOG(4) << "Destroy input_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_desc_));
      input_desc_ = nullptr;
    }

    if (input_) {
      VLOG(4) << "Destroy input_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_));
      input_ = nullptr;
    }

    if (target_desc_) {
      VLOG(4) << "Destroy target_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(target_desc_));
      target_desc_ = nullptr;
    }

    if (target_) {
      VLOG(4) << "Destroy target_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(target_));
      target_ = nullptr;
    }

    if (weight_desc_) {
      VLOG(4) << "Destroy weight_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(weight_desc_));
      weight_desc_ = nullptr;
    }

    if (weight_) {
      VLOG(4) << "Destroy weight_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(weight_));
      weight_ = nullptr;
    }

    if (grad_input_desc_) {
      VLOG(4) << "Destroy grad_input_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_input_desc_));
      grad_input_desc_ = nullptr;
    }

    if (grad_input_) {
      VLOG(4) << "Destroy grad_input_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_input_));
      grad_input_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t input_desc_ = nullptr;
  void *input_ = nullptr;
  mluOpTensorDescriptor_t target_desc_ = nullptr;
  void *target_ = nullptr;
  mluOpTensorDescriptor_t weight_desc_ = nullptr;
  void *weight_ = nullptr;
  mluOpTensorDescriptor_t grad_input_desc_ = nullptr;
  void *grad_input_ = nullptr;
  mluOpComputationPreference_t prefer_ = MLUOP_COMPUTATION_HIGH_PRECISION;
  mluOpLossReduction_t reduction_ = MLUOP_LOSS_REDUCTION_NONE;
  float alpha_ = 0.2;
  float gamma_ = 0.2;
  mluOpDevType_t target_device_;
  mluOpStatus_t expected_status_;
};

TEST_P(focal_loss_sigmoid_backward_general, negative) {
  EXPECT_TRUE(compute());
}

INSTANTIATE_TEST_CASE_P(
    zero_element_1, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 0})}),
        testing::Values(1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_2, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 100})}),
        testing::Values(1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    input_error_dims, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({100, 100, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    target_error_dims, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    target_error_shape, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({10})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    weight_error_dims, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1000})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    weight_error_shape, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({10})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    output_error_dims, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({100, 100, 100})}),
        testing::Values(1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    output_error_shape, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 1000})}),
        testing::Values(1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    input_error_dtype, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    target_error_dtype, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    weight_error_dtype, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    output_error_dtype, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    error_gamma, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(-1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    error_prefer, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_FAST),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    error_reduction, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_SUM),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    large_C, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 14849})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({14849})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 14849})}),
        testing::Values(1.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    large_gamma, focal_loss_sigmoid_backward_general,
    testing::Combine(
        testing::Values(MLUOP_COMPUTATION_HIGH_PRECISION),
        testing::Values(MLUOP_LOSS_REDUCTION_NONE),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({100})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({100, 100})}),
        testing::Values(15000.0), testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));
}  // namespace mluopapitest
