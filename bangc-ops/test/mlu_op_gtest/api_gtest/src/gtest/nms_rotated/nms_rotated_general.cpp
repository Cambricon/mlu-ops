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
    MmsRotatedParam;
class nms_rotated_general : public testing::TestWithParam<MmsRotatedParam> {
 public:
  void SetUp() {
    MLUOP_CHECK(mluOpCreate(&handle_));
    device_ = std::get<3>(GetParam());
    expected_status_ = std::get<4>(GetParam());
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
      if (b_bytes < LARGE_TENSOR_NUM) {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&boxes_, b_bytes))
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&boxes_, 8))
      }
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&scores_desc_));
    MLUOpTensorParam scores_params = std::get<1>(GetParam());
    mluOpTensorLayout_t s_layout = scores_params.get_layout();
    mluOpDataType_t s_dtype = scores_params.get_dtype();
    int s_dim = socres_params.get_dim_nb();
    std::vector<int> s_dim_size = scores_params.get_dim_size();
    MLUOP_CHECK(mluOpSetTensorDescriptor(scores_desc_, s_layout, s_dtype, s_dim,
                                         s_dim_size.data()));
    uint64_t s_ele_num = mluOpGetTensorElementNum(scores_desc_);
    uint64_t s_bytes = mluOpDataTypeBytes(s_dtype) * s_ele_num;
    if (s_bytes > 0) {
      if (s_bytes < LARGE_TENSOR_NUM) {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&scores_, s_bytes))
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&scores_, 8))
      }
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
      if (o_bytes < LARGE_TENSOR_NUM) {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&output_, o_bytes))
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&output_, 8))
      }
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
        mluOpGetNmsRotatedWorkspaceSize(handle_, boxes_desc_, &workspace_size_);
    if (MLUOP_STATUS_SUCCESS != status) {
      destroy();
      return status == expected_status_;
    }
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_));
    status = mluOpNmsRotated(handle_, iou_threshold_, boxes_desc_, boxes_,
                             scores_desc_, scores_, workspace_, workspace_size_,
                             output_desc_, output_, result_num_);
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
    if (scores_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(scores_desc_));
      scores_desc_ = NULL;
    }
    if (scores_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(scores_));
      scores_ = NULL;
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
  size_t workspace_size_ = 10;
  mluOpTensorDescriptor_t boxes_desc_ = NULL;
  void* boxes_ = NULL;
  mluOpTensorDescriptor_t scores_desc_ = NULL;
  void* scores_ = NULL;
  float iou_threshold_ = 0.5;
  void* workspace_ = NULL;
  mluOpTensorDescriptor_t output_desc_ = NULL;
  void* output_ = NULL;
  void* result_num_ = NULL;
  mluOpDevType_t device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(nms_rotated_general, api_test) {
  try {
    EXPECT_TRUE(compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in nms_rotated";
  }
}

INSTANTIATE_TEST_CASE_P(
    zero_element_0, nms_rotated_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    bad_boxes_dtype_shape_0, nms_rotated_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({2, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 6})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 5, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({2}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_scores_dtype_shape, nms_rotated_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         1, std::vector<int>({2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({2}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_output_dtype_shape, nms_rotated_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT16,
                                         1, std::vector<int>({2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({2, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    unsupported_large_shape, nms_rotated_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({357913942, 6}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({357913942}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({357913942}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));
}  // namespace mluopapitest
