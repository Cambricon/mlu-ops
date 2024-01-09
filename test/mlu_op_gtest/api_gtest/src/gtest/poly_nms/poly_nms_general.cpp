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
typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, mluOpDevType_t,
                   mluOpStatus_t>
    PolyNmsParam;
class poly_nms_general : public testing::TestWithParam<PolyNmsParam> {
 public:
  void SetUp() {
    MLUOP_CHECK(mluOpCreate(&handle_));
    device_ = std::get<2>(GetParam());
    expected_status_ = std::get<3>(GetParam());
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      return;
    }
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&boxes_desc_));
    MLUOpTensorParam boxes_params = std::get<0>(GetParam());
    mluOpTensorLayout_t b_layout = boxes_params.get_layout();
    mluOpDataType_t b_dtype = boxes_params.get_dtype();
    int b_dim = boxes_params.get_dim_nb();
    std::vector<int> b_dim_size = boxes_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(boxes_desc_, b_layout, b_dtype, b_dim,
                                         b_dim_size.data()));
    uint64_t b_ele_num = mluOpGetTensorElementNum(boxes_desc_);
    uint64_t b_bytes = mluOpDataTypeBytes(b_dtype) * b_ele_num;
    if (b_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&boxes_, b_bytes))
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));
    MLUOpTensorParam output_params = std::get<1>(GetParam());
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
    GTEST_CHECK(
        CNRT_RET_SUCCESS ==
        cnrtMalloc(&result_num_, mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
  }

  bool compute() {
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      destroy();
      return true;
    }
    mluOpStatus_t status =
        mluOpGetPolyNmsWorkspaceSize(handle_, boxes_desc_, &workspace_size_);
    if (MLUOP_STATUS_SUCCESS != status) {
      destroy();
      return status == expected_status_;
    }
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_));
    status =
        mluOpPolyNms(handle_, boxes_desc_, boxes_, iou_threshold_, workspace_,
                     workspace_size_, output_desc_, output_, result_num_);
    destroy();
    return status == expected_status_;
  }

 protected:
  void destroy() {
    if (handle_) {
      CNRT_CHECK(cnrtQueueSync(handle_->queue));
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = NULL;
    }
    if (boxes_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(boxes_desc_));
      boxes_desc_ = NULL;
    }
    if (boxes_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(boxes_));
      boxes_ = NULL;
    }
    if (workspace_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = NULL;
    }
    if (output_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
      output_desc_ = NULL;
    }
    if (output_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_));
      output_ = NULL;
    }
    if (result_num_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(result_num_));
      result_num_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t boxes_desc_ = NULL;
  void* boxes_ = NULL;
  float iou_threshold_ = 0.5;
  size_t workspace_size_ = 10;
  void* workspace_ = NULL;
  mluOpTensorDescriptor_t output_desc_ = NULL;
  void* output_ = NULL;
  void* result_num_ = NULL;
  mluOpDevType_t device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(poly_nms_general, api_test) {
  try {
    EXPECT_TRUE(compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in poly_nms";
  }
}

INSTANTIATE_TEST_CASE_P(
    zero_element_0, poly_nms_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    bad_input_dtype_shape, poly_nms_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({2, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 9, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({2}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_output_dtype_shape, poly_nms_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         1, std::vector<int>({2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    unsupported_large_shape, poly_nms_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({9771, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         1, std::vector<int>({9771}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));
}  // namespace mluopapitest
