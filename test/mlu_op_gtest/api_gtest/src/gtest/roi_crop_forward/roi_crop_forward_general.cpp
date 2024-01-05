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
                   mluOpDevType_t, mluOpStatus_t>
    ROICropForwardParam;
class roi_crop_forward_general
    : public testing::TestWithParam<ROICropForwardParam> {
 public:
  void SetUp() {
    device_ = std::get<3>(GetParam());
    expected_status_ = std::get<4>(GetParam());
    MLUOP_CHECK(mluOpCreate(&handle_));
    device_ = std::get<3>(GetParam());
    expected_status_ = std::get<4>(GetParam());
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      return;
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
    MLUOpTensorParam input_params = std::get<0>(GetParam());
    mluOpTensorLayout_t i_layout = input_params.get_layout();
    mluOpDataType_t i_dtype = input_params.get_dtype();
    int i_dim = input_params.get_dim_nb();
    std::vector<int> i_dim_size = input_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(input_desc_, i_layout, i_dtype, i_dim,
                                         i_dim_size.data()));
    uint64_t i_ele_num = mluOpGetTensorElementNum(input_desc_);
    uint64_t i_bytes = mluOpDataTypeBytes(i_dtype) * i_ele_num;
    if (i_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&input_, i_bytes));
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&grid_desc_));
    MLUOpTensorParam grid_params = std::get<1>(GetParam());
    mluOpTensorLayout_t g_layout = grid_params.get_layout();
    mluOpDataType_t g_dtype = grid_params.get_dtype();
    int g_dim = grid_params.get_dim_nb();
    std::vector<int> g_dim_size = grid_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(grid_desc_, g_layout, g_dtype, g_dim,
                                         g_dim_size.data()));
    uint64_t g_ele_num = mluOpGetTensorElementNum(grid_desc_);
    uint64_t g_bytes = mluOpDataTypeBytes(g_dtype) * g_ele_num;
    if (g_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&grid_, g_bytes));
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));
    MLUOpTensorParam output_params = std::get<2>(GetParam());
    mluOpTensorLayout_t o_layout = output_params.get_layout();
    mluOpDataType_t o_dtype = output_params.get_dtype();
    int o_dim = output_params.get_dim_nb();
    std::vector<int> o_dim_size = output_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(output_desc_, o_layout, o_dtype, o_dim,
                                         o_dim_size.data()));
    uint64_t o_ele_num = mluOpGetTensorElementNum(output_desc_);
    uint64_t o_bytes = mluOpDataTypeBytes(o_dtype) * o_ele_num;
    if (o_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&output_, o_bytes));
    }
  }

  bool compute() {
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpRoiCropForward(
        handle_, input_desc_, input_, grid_desc_, grid_, output_desc_, output_);
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
    if (input_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_desc_));
      input_desc_ = NULL;
    }
    if (grid_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grid_desc_));
      grid_desc_ = NULL;
    }
    if (output_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
      output_desc_ = NULL;
    }
    if (input_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_));
      input_ = NULL;
    }
    if (grid_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grid_));
      grid_ = NULL;
    }
    if (output_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_));
      output_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t input_desc_ = NULL;
  mluOpTensorDescriptor_t grid_desc_ = NULL;
  mluOpTensorDescriptor_t output_desc_ = NULL;
  void* input_ = NULL;
  void* grid_ = NULL;
  void* output_ = NULL;
  mluOpDevType_t device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(roi_crop_forward_general, api_test) { EXPECT_TRUE(compute()); }

INSTANTIATE_TEST_CASE_P(
    zero_element_0, roi_crop_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 7, 7, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 3, 3, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 3, 3, 9}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_input_layout_dtype_0, roi_crop_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 7, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_HALF, 4,
                                         std::vector<int>({2, 7, 7, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 3, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 3, 9}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_dtype_0, roi_crop_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 7, 7, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 3, 3, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 3, 3, 9}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_grid_dtype_shape_0, roi_crop_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 7, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         4, std::vector<int>({2, 3, 3, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 3, 3, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 3, 3, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 4, 3, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 3, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 3, 3, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 3, 9}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_output_layout_dtype_shape_0, roi_crop_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 7, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 3, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 3, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_HALF, 4,
                                         std::vector<int>({2, 3, 3, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 3, 3, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 3, 7}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));
}  // namespace mluopapitest
