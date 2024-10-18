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

typedef std::tuple<int, int, int, int> MoeDispatchForwardParam;

typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, MoeDispatchForwardParam, MLUOpTensorParam,
                   mluOpDevType_t, mluOpStatus_t>
    MoeDispatchForward;

class moe_dispatch_forward_general
    : public testing::TestWithParam<MoeDispatchForward> {
 public:
  void SetUp() {
    device_ = std::get<6>(GetParam());
    expected_status_ = std::get<7>(GetParam());
    MLUOP_CHECK(mluOpCreate(&handle_));
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      return;
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&gates_desc_));
    MLUOpTensorParam gates_params = std::get<0>(GetParam());
    mluOpTensorLayout_t gates_layout = gates_params.get_layout();
    mluOpDataType_t gates_dtype = gates_params.get_dtype();
    int gates_dim = gates_params.get_dim_nb();
    std::vector<int> gates_shape = gates_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(gates_desc_, gates_layout, gates_dtype,
                                         gates_dim, gates_shape.data()));
    uint gates_ele_num = mluOpGetTensorElementNum(gates_desc_);
    if (gates_ele_num > 0) {
      if (mluOpGetTensorElementNum(gates_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&gates_, 12 * mluOpDataTypeBytes(gates_dtype)));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&gates_, gates_ele_num *
                                            mluOpDataTypeBytes(gates_dtype)));
      }
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
    uint indices_ele_num = mluOpGetTensorElementNum(indices_desc_);
    if (indices_ele_num > 0) {
      if (mluOpGetTensorElementNum(indices_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&indices_, 12 * mluOpDataTypeBytes(indices_dtype)));
      } else {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&indices_,
                       indices_ele_num * mluOpDataTypeBytes(indices_dtype)));
      }
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&locations_desc_));
    MLUOpTensorParam locations_params = std::get<2>(GetParam());
    mluOpTensorLayout_t locations_layout = locations_params.get_layout();
    mluOpDataType_t locations_dtype = locations_params.get_dtype();
    int locations_dim = locations_params.get_dim_nb();
    std::vector<int> locations_shape = locations_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(locations_desc_, locations_layout,
                                         locations_dtype, locations_dim,
                                         locations_shape.data()));
    uint locations_ele_num = mluOpGetTensorElementNum(locations_desc_);
    if (locations_ele_num > 0) {
      if (mluOpGetTensorElementNum(locations_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&locations_, 12 * mluOpDataTypeBytes(locations_dtype)));
      } else {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&locations_, locations_ele_num *
                                        mluOpDataTypeBytes(locations_dtype)));
      }
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
    MLUOpTensorParam input_params = std::get<3>(GetParam());
    mluOpTensorLayout_t input_layout = input_params.get_layout();
    mluOpDataType_t input_dtype = input_params.get_dtype();
    int input_dim = input_params.get_dim_nb();
    std::vector<int> input_shape = input_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(input_desc_, input_layout, input_dtype,
                                         input_dim, input_shape.data()));
    uint input_ele_num = mluOpGetTensorElementNum(input_desc_);
    if (input_ele_num > 0) {
      if (mluOpGetTensorElementNum(input_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&input_, 12 * mluOpDataTypeBytes(input_dtype)));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&input_, input_ele_num *
                                            mluOpDataTypeBytes(input_dtype)));
      }
    }

    MoeDispatchForwardParam MoeDispatchForward = std::get<4>(GetParam());
    std::tie(samples_, capacity_, hidden_, num_experts_) = MoeDispatchForward;

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&dispatch_desc_));
    MLUOpTensorParam dispatch_params = std::get<5>(GetParam());
    mluOpTensorLayout_t dispatch_layout = dispatch_params.get_layout();
    mluOpDataType_t dispatch_dtype = dispatch_params.get_dtype();
    int dispatch_dim = dispatch_params.get_dim_nb();
    std::vector<int> dispatch_shape = dispatch_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(dispatch_desc_, dispatch_layout,
                                         dispatch_dtype, dispatch_dim,
                                         dispatch_shape.data()));
    uint dispatch_ele_num = mluOpGetTensorElementNum(dispatch_desc_);
    if (dispatch_ele_num > 0) {
      if (mluOpGetTensorElementNum(dispatch_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&dispatch_, 12 * mluOpDataTypeBytes(dispatch_dtype)));
      } else {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&dispatch_,
                       dispatch_ele_num * mluOpDataTypeBytes(dispatch_dtype)));
      }
    }
  }

  bool compute() {
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpMoeDispatchForward(
        handle_, gates_desc_, gates_, indices_desc_, indices_, locations_desc_,
        locations_, input_desc_, input_, samples_, capacity_, hidden_,
        num_experts_, dispatch_desc_, dispatch_);
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
    if (gates_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(gates_desc_));
      gates_desc_ = NULL;
    }
    if (gates_) {
      GTEST_CHECK(cnrtSuccess == cnrtFree(gates_));
      gates_ = NULL;
    }
    if (indices_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indices_desc_));
      indices_desc_ = NULL;
    }
    if (indices_) {
      GTEST_CHECK(cnrtSuccess == cnrtFree(indices_));
      indices_ = NULL;
    }
    if (locations_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(locations_desc_));
      locations_desc_ = NULL;
    }
    if (locations_) {
      GTEST_CHECK(cnrtSuccess == cnrtFree(locations_));
      locations_ = NULL;
    }
    if (input_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_desc_));
      input_desc_ = NULL;
    }
    if (input_) {
      GTEST_CHECK(cnrtSuccess == cnrtFree(input_));
      input_ = NULL;
    }
    if (dispatch_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(dispatch_desc_));
      dispatch_desc_ = NULL;
    }
    if (dispatch_) {
      GTEST_CHECK(cnrtSuccess == cnrtFree(dispatch_));
      dispatch_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t gates_desc_ = NULL;
  void* gates_ = NULL;
  mluOpTensorDescriptor_t indices_desc_ = NULL;
  void* indices_ = NULL;
  mluOpTensorDescriptor_t locations_desc_ = NULL;
  void* locations_ = NULL;
  mluOpTensorDescriptor_t input_desc_ = NULL;
  void* input_ = NULL;
  int samples_;
  int capacity_;
  int hidden_;
  int num_experts_;
  mluOpTensorDescriptor_t dispatch_desc_ = NULL;
  void* dispatch_ = NULL;
  mluOpDevType_t device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(moe_dispatch_forward_general, api_test) { EXPECT_TRUE(compute()); }

INSTANTIATE_TEST_CASE_P(
    zero_element_samples_0, moe_dispatch_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 4}))),
        testing::Values(MoeDispatchForwardParam{0, 8, 4, 2}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({16, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_hidden_0, moe_dispatch_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({12, 0}))),
        testing::Values(MoeDispatchForwardParam{12, 8, 0, 2}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({16, 0}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_num_experts_0, moe_dispatch_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 4}))),
        testing::Values(MoeDispatchForwardParam{0, 8, 4, 0}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_capacity_0, moe_dispatch_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 4}))),
        testing::Values(MoeDispatchForwardParam{0, 0, 4, 2}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    bad_dtype_0, moe_dispatch_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({12, 4}))),
        testing::Values(MoeDispatchForwardParam{12, 8, 4, 2}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({16, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_dtype_1, moe_dispatch_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({12, 4}))),
        testing::Values(MoeDispatchForwardParam{12, 8, 4, 2}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({16, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_gates_dtype_shape, moe_dispatch_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         1, std::vector<int>({12})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({12})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({11})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({12, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({12, 4}))),
        testing::Values(MoeDispatchForwardParam{12, 8, 4, 2}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({16, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_indices_dtype_shape, moe_dispatch_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({12})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         1, std::vector<int>({12})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({11})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({12, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({12, 4}))),
        testing::Values(MoeDispatchForwardParam{12, 8, 4, 2}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({16, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_locations_dtype_shape, moe_dispatch_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({12})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         1, std::vector<int>({12})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({11})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({12, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({12, 4}))),
        testing::Values(MoeDispatchForwardParam{12, 8, 4, 2}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({16, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_input_dtype_shape, moe_dispatch_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({12, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({12, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({11, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({12, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({12})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({12, 4, 1}))),
        testing::Values(MoeDispatchForwardParam{12, 8, 4, 2}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({16, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_dispatch_dtype_shape, moe_dispatch_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({12, 4}))),
        testing::Values(MoeDispatchForwardParam{12, 8, 4, 2}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({16, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({16, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({15, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({16, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({16, 4, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({16}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_param_0, moe_dispatch_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({18}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({18}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({18}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({18, 4}))),
        testing::Values(MoeDispatchForwardParam{18, 8, 4, 2}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({16, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_param_1, moe_dispatch_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({12}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({12, 4}))),
        testing::Values(MoeDispatchForwardParam{-1, 8, 4, 2},
                        MoeDispatchForwardParam{12, -1, 4, 2},
                        MoeDispatchForwardParam{12, 8, -1, 2},
                        MoeDispatchForwardParam{12, 8, 4, -1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({16, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    large_tensor, moe_dispatch_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1073741825}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1073741825}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1073741825}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1073741825, 4}))),
        testing::Values(MoeDispatchForwardParam{1073741825, 536870913, 4, 2}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1073741826, 4}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

}  // namespace mluopapitest
