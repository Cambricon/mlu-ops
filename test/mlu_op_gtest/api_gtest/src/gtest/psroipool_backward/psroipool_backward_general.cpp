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
#include <string>
#include <tuple>
#include <vector>

#include "api_test_tools.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "gtest/gtest.h"
#include "mlu_op.h"
#include "core/context.h"

namespace mluopapitest {
typedef std::tuple<int, int, float, int> PsRoiPoolBackwardDescParam;
typedef std::tuple<PsRoiPoolBackwardDescParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   mluOpDevType_t, mluOpStatus_t>
    PsRoiPoolBackwardParam;
class psroipool_backward_general
    : public testing::TestWithParam<PsRoiPoolBackwardParam> {
 public:
  void SetUp() {
    MLUOP_CHECK(mluOpCreate(&handle_));
    device_ = std::get<5>(GetParam());
    expected_status_ = std::get<6>(GetParam());
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      return;
    }
    PsRoiPoolBackwardDescParam op_param = std::get<0>(GetParam());
    std::tie(pooled_height_, pooled_width_, spatial_scale_, output_dim_) =
        op_param;

    MLUOpTensorParam b_params = std::get<1>(GetParam());
    mluOpTensorLayout_t b_layout = b_params.get_layout();
    mluOpDataType_t b_dtype = b_params.get_dtype();
    int b_dim = b_params.get_dim_nb();
    std::vector<int> b_dim_size = b_params.get_dim_size();
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&bottom_grad_desc_));
    MLUOP_CHECK(mluOpSetTensorDescriptor(bottom_grad_desc_, b_layout, b_dtype,
                                         b_dim, b_dim_size.data()));
    uint64_t b_ele_num = mluOpGetTensorElementNum(bottom_grad_desc_);
    uint64_t b_bytes = mluOpDataTypeBytes(b_dtype) * b_ele_num;
    if (b_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&bottom_grad_, b_bytes))
    }

    MLUOpTensorParam r_params = std::get<2>(GetParam());
    mluOpTensorLayout_t r_layout = r_params.get_layout();
    mluOpDataType_t r_dtype = r_params.get_dtype();
    int r_dim = r_params.get_dim_nb();
    std::vector<int> r_dim_size = r_params.get_dim_size();
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&rois_desc_));
    MLUOP_CHECK(mluOpSetTensorDescriptor(rois_desc_, r_layout, r_dtype, r_dim,
                                         r_dim_size.data()));
    uint64_t r_ele_num = mluOpGetTensorElementNum(rois_desc_);
    uint64_t r_bytes = mluOpDataTypeBytes(r_dtype) * r_ele_num;
    if (r_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&rois_, r_bytes))
    }

    MLUOpTensorParam o_params = std::get<3>(GetParam());
    mluOpTensorLayout_t o_layout = o_params.get_layout();
    mluOpDataType_t o_dtype = o_params.get_dtype();
    int o_dim = o_params.get_dim_nb();
    std::vector<int> o_dim_size = o_params.get_dim_size();
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&top_grad_desc_));
    MLUOP_CHECK(mluOpSetTensorDescriptor(top_grad_desc_, o_layout, o_dtype,
                                         o_dim, o_dim_size.data()));
    uint64_t o_ele_num = mluOpGetTensorElementNum(top_grad_desc_);
    uint64_t o_bytes = mluOpDataTypeBytes(o_dtype) * o_ele_num;
    if (o_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&top_grad_, o_bytes))
    }

    MLUOpTensorParam m_params = std::get<4>(GetParam());
    mluOpTensorLayout_t m_layout = m_params.get_layout();
    mluOpDataType_t m_dtype = m_params.get_dtype();
    int m_dim = m_params.get_dim_nb();
    std::vector<int> m_dim_size = m_params.get_dim_size();
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&mapping_channel_desc_));
    MLUOP_CHECK(mluOpSetTensorDescriptor(mapping_channel_desc_, m_layout,
                                         m_dtype, m_dim, m_dim_size.data()));
    uint64_t m_ele_num = mluOpGetTensorElementNum(mapping_channel_desc_);
    uint64_t m_bytes = mluOpDataTypeBytes(m_dtype) * m_ele_num;
    if (m_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&mapping_channel_, m_bytes))
    }
  }

  bool compute() {
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpPsRoiPoolBackward(
        handle_, pooled_height_, pooled_width_, spatial_scale_, output_dim_,
        top_grad_desc_, top_grad_, rois_desc_, rois_, mapping_channel_desc_,
        mapping_channel_, bottom_grad_desc_, bottom_grad_);
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
    if (bottom_grad_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(bottom_grad_desc_));
      bottom_grad_desc_ = NULL;
    }
    if (bottom_grad_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(bottom_grad_));
      bottom_grad_ = NULL;
    }
    if (rois_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(rois_desc_));
      rois_desc_ = NULL;
    }
    if (rois_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(rois_));
      rois_ = NULL;
    }
    if (top_grad_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(top_grad_desc_));
      top_grad_desc_ = NULL;
    }
    if (top_grad_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(top_grad_));
      top_grad_ = NULL;
    }
    if (mapping_channel_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(mapping_channel_desc_));
      mapping_channel_desc_ = NULL;
    }
    if (mapping_channel_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(mapping_channel_));
      mapping_channel_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t bottom_grad_desc_ = NULL;
  mluOpTensorDescriptor_t rois_desc_ = NULL;
  mluOpTensorDescriptor_t top_grad_desc_ = NULL;
  mluOpTensorDescriptor_t mapping_channel_desc_ = NULL;
  void* bottom_grad_ = NULL;
  void* rois_ = NULL;
  void* top_grad_ = NULL;
  void* mapping_channel_ = NULL;
  int pooled_height_ = 3;
  int pooled_width_ = 3;
  float spatial_scale_ = 0.25;
  int output_dim_ = 1;
  mluOpDevType_t device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(psroipool_backward_general, api_test) {
  try {
    EXPECT_TRUE(compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_backward";
  }
}

INSTANTIATE_TEST_CASE_P(
    zero_element_0, psroipool_backward_general,
    testing::Combine(
        testing::Values(PsRoiPoolBackwardDescParam{3, 3, 1.0, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 4, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 3, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 3, 3, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_1, psroipool_backward_general,
    testing::Combine(
        testing::Values(PsRoiPoolBackwardDescParam{3, 3, 1.0, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 4, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({0, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 3, 3, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({0, 3, 3, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_2, psroipool_backward_general,
    testing::Combine(
        testing::Values(PsRoiPoolBackwardDescParam{0, 0, 1.0, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 4, 4, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 0, 0, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 0, 0, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    bad_bottom_grad_dtype_shape_layout_0, psroipool_backward_general,
    testing::Combine(
        testing::Values(PsRoiPoolBackwardDescParam{3, 3, 1.0, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_HALF, 4,
                                         std::vector<int>({1, 4, 4, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 4, 4, 8})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({1, 4, 4, 9, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 4, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 3, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 3, 3, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_rois_dtype_shape_0, psroipool_backward_general,
    testing::Combine(
        testing::Values(PsRoiPoolBackwardDescParam{3, 3, 1.0, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 4, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({2, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 5, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 3, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 3, 3, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_top_grad_dtype_shape_layout_0, psroipool_backward_general,
    testing::Combine(
        testing::Values(PsRoiPoolBackwardDescParam{3, 3, 1.0, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 4, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_HALF, 4,
                                         std::vector<int>({2, 3, 3, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 3, 3, 1, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 3, 3, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 3, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 3, 3, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_mapping_channel_dtype_shape_layout_0, psroipool_backward_general,
    testing::Combine(
        testing::Values(PsRoiPoolBackwardDescParam{3, 3, 1.0, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 4, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 3, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 3, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 3, 3, 1, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({1, 3, 3, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 2, 3, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 3, 2, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 3, 3, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 3, 3, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_params_0, psroipool_backward_general,
    testing::Combine(
        testing::Values(PsRoiPoolBackwardDescParam{3, 3, 1.0, 2},
                        PsRoiPoolBackwardDescParam{3, 3, 0, 1},
                        PsRoiPoolBackwardDescParam{3, 2, 1.0, 1},
                        PsRoiPoolBackwardDescParam{2, 3, 1.0, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 4, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 3, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 3, 3, 1}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    bad_params_1, psroipool_backward_general,
    testing::Combine(
        testing::Values(PsRoiPoolBackwardDescParam{3, 3, 1.0, 1}),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 4, 4, 9}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({2, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 3, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 3, 3, 2}))),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));
}  // namespace mluopapitest
