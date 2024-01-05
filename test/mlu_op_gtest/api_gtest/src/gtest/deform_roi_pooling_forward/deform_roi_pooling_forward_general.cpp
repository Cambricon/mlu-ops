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
typedef std::tuple<int, int, float, int, float>
    DeformRoiPoolingForwardAdditionalParam;

typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, DeformRoiPoolingForwardAdditionalParam,
                   mluOpDevType_t, mluOpStatus_t>
    DeformRoiPoolingForwardParams;

class deform_roi_pooling_forward_general
    : public testing::TestWithParam<DeformRoiPoolingForwardParams> {
 public:
  void SetUp() {
    try {
      target_device_ = std::get<5>(GetParam());
      expected_status_ = std::get<6>(GetParam());

      MLUOP_CHECK(mluOpCreate(&handle_));

      MLUOpTensorParam inputDescParam = std::get<0>(GetParam());
      mluOpTensorLayout_t input_layout = inputDescParam.get_layout();
      mluOpDataType_t input_dtype = inputDescParam.get_dtype();
      int input_dim_nb = inputDescParam.get_dim_nb();
      std::vector<int> input_dims = inputDescParam.get_dim_size();
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(input_desc_, input_layout,
                                           input_dtype, input_dim_nb,
                                           input_dims.data()));
      uint64_t i_ele_num = mluOpGetTensorElementNum(input_desc_);
      uint64_t i_bytes = mluOpDataTypeBytes(input_dtype) * i_ele_num;
      if (i_bytes > 0) {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&input_, i_bytes));
      }

      MLUOpTensorParam roisDescParam = std::get<1>(GetParam());
      mluOpTensorLayout_t rois_layout = roisDescParam.get_layout();
      mluOpDataType_t rois_dtype = roisDescParam.get_dtype();
      int rois_dim_nb = roisDescParam.get_dim_nb();
      std::vector<int> rois_dims = roisDescParam.get_dim_size();
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&rois_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(rois_desc_, rois_layout, rois_dtype,
                                           rois_dim_nb, rois_dims.data()));
      uint64_t roi_ele_num = mluOpGetTensorElementNum(rois_desc_);
      uint64_t roi_bytes = mluOpDataTypeBytes(rois_dtype) * roi_ele_num;
      if (roi_bytes > 0) {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&rois_, roi_bytes));
      }

      MLUOpTensorParam offsetDescParam = std::get<2>(GetParam());
      mluOpTensorLayout_t offset_layout = offsetDescParam.get_layout();
      mluOpDataType_t offset_dtype = offsetDescParam.get_dtype();
      int offset_dim_nb = offsetDescParam.get_dim_nb();
      std::vector<int> offset_dims = offsetDescParam.get_dim_size();
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&offset_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(offset_desc_, offset_layout,
                                           offset_dtype, offset_dim_nb,
                                           offset_dims.data()));
      uint64_t offset_ele_num = mluOpGetTensorElementNum(offset_desc_);
      uint64_t offset_bytes = mluOpDataTypeBytes(offset_dtype) * offset_ele_num;
      if (offset_bytes > 0) {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&offset_, offset_bytes));
      }

      DeformRoiPoolingForwardAdditionalParam additoinal_param_ =
          std::get<4>(GetParam());
      std::tie(pooled_height_, pooled_width_, spatial_scale_, sampling_ratio_,
               gamma_) = additoinal_param_;

      MLUOpTensorParam outputDescParam = std::get<3>(GetParam());
      mluOpTensorLayout_t output_layout = outputDescParam.get_layout();
      mluOpDataType_t output_dtype = outputDescParam.get_dtype();
      int output_dim_nb = outputDescParam.get_dim_nb();
      std::vector<int> output_dims = outputDescParam.get_dim_size();
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(output_desc_, output_layout,
                                           output_dtype, output_dim_nb,
                                           output_dims.data()));
      uint64_t o_ele_num = mluOpGetTensorElementNum(output_desc_);
      uint64_t o_bytes = mluOpDataTypeBytes(output_dtype) * o_ele_num;
      if (o_bytes > 0) {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&output_, o_bytes));
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in deform_roi_pooling_forward";
    }
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpDeformRoiPoolForward(
        handle_, input_desc_, input_, rois_desc_, rois_, offset_desc_, offset_,
        pooled_height_, pooled_width_, spatial_scale_, sampling_ratio_, gamma_,
        output_desc_, output_);
    destroy();
    return expected_status_ == status;
  }

 protected:
  void destroy() {
    try {
      if (handle_) {
        CNRT_CHECK(cnrtQueueSync(handle_->queue));
        VLOG(4) << "Destroy handle";
        MLUOP_CHECK(mluOpDestroy(handle_));
      }
      if (input_desc_) {
        VLOG(4) << "Destroy input_desc";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_desc_));
      }
      if (input_) {
        VLOG(4) << "Destroy input";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_));
        input_ = nullptr;
      }
      if (rois_desc_) {
        VLOG(4) << "Destroy rois_desc";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(rois_desc_));
      }
      if (rois_) {
        VLOG(4) << "Destroy rois";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(rois_));
        rois_ = nullptr;
      }
      if (offset_desc_) {
        VLOG(4) << "Destroy offset_desc";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(offset_desc_));
      }
      if (offset_) {
        VLOG(4) << "Destroy offset";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(offset_));
        offset_ = nullptr;
      }
      if (output_desc_) {
        VLOG(4) << "Destroy output_desc";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
      }
      if (output_) {
        VLOG(4) << "Destroy output";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_));
        output_ = nullptr;
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << "in deform_roi_pooling_forward";
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t input_desc_ = nullptr;
  mluOpTensorDescriptor_t rois_desc_ = nullptr;
  mluOpTensorDescriptor_t offset_desc_ = nullptr;
  mluOpTensorDescriptor_t output_desc_ = nullptr;
  void *input_ = nullptr;
  void *rois_ = nullptr;
  void *offset_ = nullptr;
  int pooled_height_ = 1;
  int pooled_width_ = 1;
  float spatial_scale_ = 1.0;
  int sampling_ratio_ = 1;
  float gamma_ = 1.0;
  void *output_ = nullptr;
  mluOpDevType_t target_device_;
  mluOpStatus_t expected_status_;
};

TEST_P(deform_roi_pooling_forward_general, negative) { EXPECT_TRUE(compute()); }

INSTANTIATE_TEST_CASE_P(
    not_support_input_dtype, deform_roi_pooling_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({1, 5, 5, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 2, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 3, 1}))),
        testing::Values(DeformRoiPoolingForwardAdditionalParam(3, 3, (float)1.0,
                                                               1, (float)1.0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    not_support_rois_dtype, deform_roi_pooling_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 5, 5, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT8,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 2, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 3, 3}))),
        testing::Values(DeformRoiPoolingForwardAdditionalParam(3, 3, (float)1.0,
                                                               1, (float)1.0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    not_support_offset_dtype, deform_roi_pooling_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_HALF, 4,
                                         std::vector<int>({1, 5, 5, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         4, std::vector<int>({3, 2, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_HALF, 4,
                                         std::vector<int>({3, 3, 3, 3}))),
        testing::Values(DeformRoiPoolingForwardAdditionalParam(3, 3, (float)1.0,
                                                               1, (float)1.0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    not_support_output_dtype, deform_roi_pooling_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_HALF, 4,
                                         std::vector<int>({1, 5, 5, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         4, std::vector<int>({3, 2, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT16,
                                         4, std::vector<int>({3, 3, 3, 3}))),
        testing::Values(DeformRoiPoolingForwardAdditionalParam(3, 3, (float)1.0,
                                                               1, (float)1.0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    dtype_missmatch, deform_roi_pooling_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_HALF, 4,
                                         std::vector<int>({1, 5, 5, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         4, std::vector<int>({3, 2, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 3, 3}))),
        testing::Values(DeformRoiPoolingForwardAdditionalParam(3, 3, (float)1.0,
                                                               1, (float)1.0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    not_support_input_layout, deform_roi_pooling_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_HALF, 4,
                                         std::vector<int>({1, 5, 5, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         4, std::vector<int>({3, 2, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_HALF, 4,
                                         std::vector<int>({3, 3, 3, 3}))),
        testing::Values(DeformRoiPoolingForwardAdditionalParam(3, 3, (float)1.0,
                                                               1, (float)1.0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    not_support_output_layout, deform_roi_pooling_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 5, 5, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 2, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_HWCN, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 3, 3}))),
        testing::Values(DeformRoiPoolingForwardAdditionalParam(3, 3, (float)1.0,
                                                               1, (float)1.0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    wrong_rois_shape, deform_roi_pooling_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 5, 5, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 15}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 2, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 3, 1}))),
        testing::Values(DeformRoiPoolingForwardAdditionalParam(3, 3, (float)1.0,
                                                               1, (float)1.0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    wrong_output_shape, deform_roi_pooling_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 5, 5, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 2, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 3, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 3, 3, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 6, 3, 10})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 6, 10}))),
        testing::Values(DeformRoiPoolingForwardAdditionalParam(3, 3, (float)1.0,
                                                               1, (float)1.0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    pooled_shape_mismatch, deform_roi_pooling_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 5, 5, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 2, 6, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 2, 3, 7}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 3, 1}))),
        testing::Values(DeformRoiPoolingForwardAdditionalParam(3, 3, (float)1.0,
                                                               1, (float)1.0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    offset_shape_mismatch, deform_roi_pooling_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 5, 5, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 3, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({5, 2, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 3, 1}))),
        testing::Values(DeformRoiPoolingForwardAdditionalParam(3, 3, (float)1.0,
                                                               1, (float)1.0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element, deform_roi_pooling_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 0, 5, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 5, 0, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 2, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 3, 1}))),
        testing::Values(DeformRoiPoolingForwardAdditionalParam(3, 3, (float)1.0,
                                                               1, (float)1.0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    wrong_zero_element, deform_roi_pooling_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 5, 5, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 2, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 3, 1}))),
        testing::Values(DeformRoiPoolingForwardAdditionalParam(3, 3, (float)1.0,
                                                               1, (float)1.0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    wrong_zero_element2, deform_roi_pooling_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 5, 5, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({3, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 2, 3, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 3, 0}))),
        testing::Values(DeformRoiPoolingForwardAdditionalParam(3, 3, (float)1.0,
                                                               1, (float)1.0)),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));
}  // namespace mluopapitest
