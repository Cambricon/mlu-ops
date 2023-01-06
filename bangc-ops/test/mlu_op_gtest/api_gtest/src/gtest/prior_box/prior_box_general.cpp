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
typedef std::tuple<int, int, int, int, float, float, float, bool, bool>
    PriorBoxDescParam;
typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   PriorBoxDescParam, mluOpDevType_t, mluOpStatus_t>
    PriorBoxParam;
class prior_box_general : public testing::TestWithParam<PriorBoxParam> {
 public:
  void SetUp() {
    device_ = std::get<7>(GetParam());
    expected_status_ = std::get<8>(GetParam());
    MLUOP_CHECK(mluOpCreate(&handle_));
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      return;
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&min_desc_));
    MLUOpTensorParam min_params = std::get<0>(GetParam());
    const mluOpTensorLayout_t min_layout = min_params.get_layout();
    const mluOpDataType_t min_dtype = min_params.get_dtype();
    const int min_dim = min_params.get_dim_nb();
    std::vector<int> min_dim_size = min_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(min_desc_, min_layout, min_dtype,
                                         min_dim, min_dim_size.data()));
    const uint64_t min_ele_num = mluOpGetTensorElementNum(min_desc_);
    if (min_ele_num > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&min_, 8))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&aspect_ratios_desc_));
    MLUOpTensorParam aspect_params = std::get<1>(GetParam());
    const mluOpTensorLayout_t aspect_layout = aspect_params.get_layout();
    const mluOpDataType_t aspect_dtype = aspect_params.get_dtype();
    const int aspect_dim = aspect_params.get_dim_nb();
    std::vector<int> aspect_dim_size = aspect_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(aspect_ratios_desc_, aspect_layout,
                                         aspect_dtype, aspect_dim,
                                         aspect_dim_size.data()));
    const uint64_t aspect_ele_num =
        mluOpGetTensorElementNum(aspect_ratios_desc_);
    if (aspect_ele_num > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&aspect_ratios_, 8));
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&variance_desc_));
    MLUOpTensorParam variance_params = std::get<2>(GetParam());
    const mluOpTensorLayout_t variance_layout = variance_params.get_layout();
    const mluOpDataType_t variance_dtype = variance_params.get_dtype();
    const int variance_dim = variance_params.get_dim_nb();
    std::vector<int> variance_dim_size = variance_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(variance_desc_, variance_layout,
                                         variance_dtype, variance_dim,
                                         variance_dim_size.data()));
    const uint64_t variance_ele_num = mluOpGetTensorElementNum(variance_desc_);
    if (variance_ele_num > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&variance_, 8));
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&max_desc_));
    MLUOpTensorParam max_params = std::get<3>(GetParam());
    const mluOpTensorLayout_t max_layout = max_params.get_layout();
    const mluOpDataType_t max_dtype = max_params.get_dtype();
    const int max_dim = max_params.get_dim_nb();
    std::vector<int> max_dim_size = max_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(max_desc_, max_layout, max_dtype,
                                         max_dim, max_dim_size.data()));
    const uint64_t max_ele_num = mluOpGetTensorElementNum(max_desc_);
    if (max_ele_num > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&max_, 8));
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));
    MLUOpTensorParam o_params = std::get<4>(GetParam());
    const mluOpTensorLayout_t o_layout = o_params.get_layout();
    const mluOpDataType_t o_dtype = o_params.get_dtype();
    const int o_dim = o_params.get_dim_nb();
    std::vector<int> o_dim_size = o_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(output_desc_, o_layout, o_dtype, o_dim,
                                         o_dim_size.data()));
    const uint64_t o_ele_num = mluOpGetTensorElementNum(output_desc_);
    if (o_ele_num > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&output_, 8));
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&var_desc_));
    MLUOpTensorParam var_params = std::get<5>(GetParam());
    const mluOpTensorLayout_t var_layout = var_params.get_layout();
    const mluOpDataType_t var_dtype = var_params.get_dtype();
    const int var_dim = var_params.get_dim_nb();
    std::vector<int> var_dim_size = var_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(var_desc_, var_layout, var_dtype,
                                         var_dim, var_dim_size.data()));
    const uint64_t var_ele_num = mluOpGetTensorElementNum(var_desc_);
    if (var_ele_num > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&var_, 8));
    }

    PriorBoxDescParam params = std::get<6>(GetParam());
    std::tie(height_, width_, im_height_, im_width_, step_w_, step_h_, offset_,
             is_clip_, min_max_aspect_order_) = params;
  }

  bool compute() {
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpPriorBox(
        handle_, min_desc_, min_, aspect_ratios_desc_, aspect_ratios_,
        variance_desc_, variance_, max_desc_, max_, height_, width_, im_height_,
        im_width_, step_h_, step_w_, offset_, is_clip_, min_max_aspect_order_,
        output_desc_, output_, var_desc_, var_);
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
    if (min_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(min_desc_));
      min_desc_ = NULL;
    }
    if (aspect_ratios_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(aspect_ratios_desc_));
      aspect_ratios_desc_ = NULL;
    }
    if (variance_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(variance_desc_));
      variance_desc_ = NULL;
    }
    if (max_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(max_desc_));
      max_desc_ = NULL;
    }
    if (output_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
      output_desc_ = NULL;
    }
    if (var_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(var_desc_));
      var_desc_ = NULL;
    }
    if (min_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(min_));
      min_ = NULL;
    }
    if (aspect_ratios_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(aspect_ratios_));
      aspect_ratios_ = NULL;
    }
    if (variance_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(variance_));
      variance_ = NULL;
    }
    if (max_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(max_));
      max_ = NULL;
    }
    if (output_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_));
      output_ = NULL;
    }
    if (var_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(var_));
      var_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t min_desc_ = NULL;
  mluOpTensorDescriptor_t aspect_ratios_desc_ = NULL;
  mluOpTensorDescriptor_t variance_desc_ = NULL;
  mluOpTensorDescriptor_t max_desc_ = NULL;
  mluOpTensorDescriptor_t output_desc_ = NULL;
  mluOpTensorDescriptor_t var_desc_ = NULL;
  void* min_ = NULL;
  void* aspect_ratios_ = NULL;
  void* variance_ = NULL;
  void* max_ = NULL;
  void* output_ = NULL;
  void* var_ = NULL;
  int height_ = 0;
  int width_ = 0;
  int im_height_ = 0;
  int im_width_ = 0;
  float step_w_ = 0.0;
  float step_h_ = 0.0;
  float offset_ = 0.0;
  bool is_clip_ = false;
  bool min_max_aspect_order_ = false;
  mluOpDevType_t device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(prior_box_general, api_test) {
  try {
    EXPECT_TRUE(compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in prior_box";
  }
}

INSTANTIATE_TEST_CASE_P(
    zero_element_0, prior_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(PriorBoxDescParam{3, 3, 9, 9, 3.0, 3.0, 0.5, true,
                                          true}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_1, prior_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(PriorBoxDescParam{3, 3, 9, 9, 3.0, 3.0, 0.5, true,
                                          true}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_2, prior_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 0, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 0, 2, 4}))),
        testing::Values(PriorBoxDescParam{0, 0, 9, 9, 3.0, 3.0, 0.5, true,
                                          true}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_3, prior_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(PriorBoxDescParam{3, 3, 0, 0, 3.0, 3.0, 0.5, true,
                                          true}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    bad_params_0, prior_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(
            PriorBoxDescParam{-1, 3, 9, 9, 3.0, 3.0, 0.5, true, true},
            PriorBoxDescParam{3, -1, 9, 9, 3.0, 3.0, 0.5, true, true},
            PriorBoxDescParam{3, 3, 9, 9, 0.0, 3.0, 0.5, true, true},
            PriorBoxDescParam{3, 3, 9, 9, 3.0, 0.0, 0.5, true, true}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_min_dtype_shape_0, prior_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         1, std::vector<int>({1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({1, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         1, std::vector<int>({2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(PriorBoxDescParam{3, 3, 9, 9, 3.0, 3.0, 0.5, true,
                                          true}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_aspect_dtype_shape_0, prior_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         1, std::vector<int>({1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(PriorBoxDescParam{3, 3, 9, 9, 3.0, 3.0, 0.5, true,
                                          true}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_variance_dtype_shape_0, prior_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         1, std::vector<int>({4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(PriorBoxDescParam{3, 3, 9, 9, 3.0, 3.0, 0.5, true,
                                          true}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_max_dtype_shape_0, prior_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         1, std::vector<int>({1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({1, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(PriorBoxDescParam{3, 3, 9, 9, 3.0, 3.0, 0.5, true,
                                          true}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_output_dtype_shape_0, prior_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         4, std::vector<int>({3, 3, 2, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 3, 2, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 4, 2, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 6})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 5, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(PriorBoxDescParam{3, 3, 9, 9, 3.0, 3.0, 0.5, true,
                                          true}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_num_box_0, prior_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 6, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 6, 4}))),
        testing::Values(PriorBoxDescParam{3, 3, 9, 9, 3.0, 3.0, 0.5, true,
                                          true}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_output_var_0, prior_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 6}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 6}))),
        testing::Values(PriorBoxDescParam{3, 3, 9, 9, 3.0, 3.0, 0.5, true,
                                          true}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_var_dtype_shape_0, prior_box_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         4, std::vector<int>({3, 3, 2, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 3, 2, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 4, 2, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 5, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 2, 6}))),
        testing::Values(PriorBoxDescParam{3, 3, 9, 9, 3.0, 3.0, 0.5, true,
                                          true}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));
}  // namespace mluopapitest
