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
typedef std::tuple<int, int, float, int, int> PsRoiPoolForwardDescParam;
typedef std::tuple<PsRoiPoolForwardDescParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   mluOpDevType_t, mluOpStatus_t>
    PsRoiPoolForwardParam;
class psroipool_forward_general
    : public testing::TestWithParam<PsRoiPoolForwardParam> {
 public:
  void SetUp() {
    MLUOP_CHECK(mluOpCreate(&handle_));
    device_ = std::get<5>(GetParam());
    expected_status_ = std::get<6>(GetParam());
    if (!(device_ == MLUOP_UNKNOWN_DEVICE || device_ == handle_->arch)) {
      VLOG(4) << "Device does not match, skip testing.";
      return;
    }
    PsRoiPoolForwardDescParam op_param = std::get<0>(GetParam());
    std::tie(pooled_height_, pooled_width_, spatial_scale_, group_size_,
             output_dim_) = op_param;

    MLUOpTensorParam i_params = std::get<1>(GetParam());
    mluOpTensorLayout_t i_layout = i_params.get_layout();
    mluOpDataType_t i_dtype = i_params.get_dtype();
    int i_dim = i_params.get_dim_nb();
    std::vector<int> i_dim_size = i_params.get_dim_size();
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
    MLUOP_CHECK(mluOpSetTensorDescriptor(input_desc_, i_layout, i_dtype, i_dim,
                                         i_dim_size.data()));
    uint64_t i_ele_num = mluOpGetTensorElementNum(input_desc_);
    uint64_t i_bytes = mluOpDataTypeBytes(i_dtype) * i_ele_num;
    if (i_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&input_, i_bytes))
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
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));
    MLUOP_CHECK(mluOpSetTensorDescriptor(output_desc_, o_layout, o_dtype, o_dim,
                                         o_dim_size.data()));
    uint64_t o_ele_num = mluOpGetTensorElementNum(output_desc_);
    uint64_t o_bytes = mluOpDataTypeBytes(o_dtype) * o_ele_num;
    if (o_bytes > 0) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&output_, o_bytes))
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
    mluOpStatus_t status = mluOpPsRoiPoolForward(
        handle_, pooled_height_, pooled_width_, spatial_scale_, group_size_,
        output_dim_, input_desc_, input_, rois_desc_, rois_, output_desc_,
        output_, mapping_channel_desc_, mapping_channel_);
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
    if (input_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_desc_));
      input_desc_ = NULL;
    }
    if (input_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_));
      input_ = NULL;
    }
    if (rois_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(rois_desc_));
      rois_desc_ = NULL;
    }
    if (rois_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(rois_));
      rois_ = NULL;
    }
    if (output_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
      output_desc_ = NULL;
    }
    if (output_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_));
      output_ = NULL;
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
  mluOpTensorDescriptor_t input_desc_ = NULL;
  mluOpTensorDescriptor_t rois_desc_ = NULL;
  mluOpTensorDescriptor_t output_desc_ = NULL;
  mluOpTensorDescriptor_t mapping_channel_desc_ = NULL;
  void* input_ = NULL;
  void* rois_ = NULL;
  void* output_ = NULL;
  void* mapping_channel_ = NULL;
  int pooled_height_ = 3;
  int pooled_width_ = 3;
  float spatial_scale_ = 0.25;
  int group_size_ = 3;
  int output_dim_ = 1;
  mluOpDevType_t device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(psroipool_forward_general, api_test) {
  try {
    EXPECT_TRUE(compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_forward";
  }
}

INSTANTIATE_TEST_CASE_P(
    zero_element_0, psroipool_forward_general,
    testing::Combine(
        testing::Values(PsRoiPoolForwardDescParam{3, 3, 1.0, 3, 1}),
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
    zero_element_1, psroipool_forward_general,
    testing::Combine(
        testing::Values(PsRoiPoolForwardDescParam{3, 3, 1.0, 3, 1}),
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
    bad_input_dtype_shape_layout_0, psroipool_forward_general,
    testing::Combine(
        testing::Values(PsRoiPoolForwardDescParam{3, 3, 1.0, 3, 1}),
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
    bad_rois_dtype_shape_0, psroipool_forward_general,
    testing::Combine(
        testing::Values(PsRoiPoolForwardDescParam{3, 3, 1.0, 3, 1}),
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
    bad_output_dtype_shape_layout_0, psroipool_forward_general,
    testing::Combine(
        testing::Values(PsRoiPoolForwardDescParam{3, 3, 1.0, 3, 1}),
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
    bad_mapping_channel_dtype_shape_layout_0, psroipool_forward_general,
    testing::Combine(
        testing::Values(PsRoiPoolForwardDescParam{3, 3, 1.0, 3, 1}),
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
    bad_params_0, psroipool_forward_general,
    testing::Combine(
        testing::Values(PsRoiPoolForwardDescParam{3, 3, 1.0, 3, 2},
                        PsRoiPoolForwardDescParam{3, 3, 1.0, 2, 1},
                        PsRoiPoolForwardDescParam{3, 3, 0, 3, 1},
                        PsRoiPoolForwardDescParam{3, 2, 1.0, 3, 1},
                        PsRoiPoolForwardDescParam{2, 3, 1.0, 3, 1}),
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
}  // namespace mluopapitest
