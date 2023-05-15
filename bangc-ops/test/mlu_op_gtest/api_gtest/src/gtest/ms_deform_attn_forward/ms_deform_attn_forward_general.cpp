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

typedef std::tuple<mluOpDevType_t, mluOpStatus_t> MsDeformAttnForwardParam;
typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MsDeformAttnForwardParam>
    MsDeformAttnForward;
class ms_deform_attn_forward_general
    : public testing::TestWithParam<MsDeformAttnForward> {
 public:
  void SetUp() {
    try {
      MLUOP_CHECK(mluOpCreate(&handle_));

      MLUOpTensorParam data_value_params = std::get<0>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_value_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          data_value_desc_, data_value_params.get_layout(),
          data_value_params.get_dtype(), data_value_params.get_dim_nb(),
          data_value_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(data_value_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&data_value_,
                       mluOpDataTypeBytes(data_value_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&data_value_,
                       mluOpDataTypeBytes(data_value_params.get_dtype()) *
                           mluOpGetTensorElementNum(data_value_desc_)));
      }

      MLUOpTensorParam data_spatial_shapes_params = std::get<1>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_spatial_shapes_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          data_spatial_shapes_desc_, data_spatial_shapes_params.get_layout(),
          data_spatial_shapes_params.get_dtype(),
          data_spatial_shapes_params.get_dim_nb(),
          data_spatial_shapes_params.get_dim_size().data()));

      if (mluOpGetTensorElementNum(data_spatial_shapes_desc_) >=
          LARGE_TENSOR_NUM) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&data_spatial_shapes_,
                               mluOpDataTypeBytes(
                                   data_spatial_shapes_params.get_dtype()) *
                                   2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &data_spatial_shapes_,
                mluOpDataTypeBytes(data_spatial_shapes_params.get_dtype()) *
                    mluOpGetTensorElementNum(data_spatial_shapes_desc_)));
      }

      MLUOpTensorParam data_level_start_index_params = std::get<2>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_level_start_index_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          data_level_start_index_desc_,
          data_level_start_index_params.get_layout(),
          data_level_start_index_params.get_dtype(),
          data_level_start_index_params.get_dim_nb(),
          data_level_start_index_params.get_dim_size().data()));

      if (mluOpGetTensorElementNum(data_level_start_index_desc_) >=
          LARGE_TENSOR_NUM) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&data_level_start_index_,
                               mluOpDataTypeBytes(
                                   data_level_start_index_params.get_dtype()) *
                                   2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &data_level_start_index_,
                mluOpDataTypeBytes(data_level_start_index_params.get_dtype()) *
                    mluOpGetTensorElementNum(data_level_start_index_desc_)));
      }

      MLUOpTensorParam data_sampling_loc_params = std::get<3>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_sampling_loc_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          data_sampling_loc_desc_, data_sampling_loc_params.get_layout(),
          data_sampling_loc_params.get_dtype(),
          data_sampling_loc_params.get_dim_nb(),
          data_sampling_loc_params.get_dim_size().data()));

      if (mluOpGetTensorElementNum(data_sampling_loc_desc_) >=
          LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &data_sampling_loc_,
                mluOpDataTypeBytes(data_sampling_loc_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &data_sampling_loc_,
                mluOpDataTypeBytes(data_sampling_loc_params.get_dtype()) *
                    mluOpGetTensorElementNum(data_sampling_loc_desc_)));
      }

      MLUOpTensorParam data_attn_weight_params = std::get<4>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_attn_weight_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          data_attn_weight_desc_, data_attn_weight_params.get_layout(),
          data_attn_weight_params.get_dtype(),
          data_attn_weight_params.get_dim_nb(),
          data_attn_weight_params.get_dim_size().data()));

      if (mluOpGetTensorElementNum(data_attn_weight_desc_) >=
          LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &data_attn_weight_,
                mluOpDataTypeBytes(data_attn_weight_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&data_attn_weight_,
                       mluOpDataTypeBytes(data_attn_weight_params.get_dtype()) *
                           mluOpGetTensorElementNum(data_attn_weight_desc_)));
      }

      MLUOpTensorParam data_col_params = std::get<5>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_col_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          data_col_desc_, data_col_params.get_layout(),
          data_col_params.get_dtype(), data_col_params.get_dim_nb(),
          data_col_params.get_dim_size().data()));

      if (mluOpGetTensorElementNum(data_col_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&data_col_,
                       mluOpDataTypeBytes(data_col_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&data_col_,
                               mluOpDataTypeBytes(data_col_params.get_dtype()) *
                                   mluOpGetTensorElementNum(data_col_desc_)));
      }

      MsDeformAttnForwardParam msDeformAttnForwardParam =
          std::get<6>(GetParam());
      std::tie(target_device_, expected_status_) = msDeformAttnForwardParam;
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in ms_deform_attn_forward general.";
    }
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpMsDeformAttnForward(
        handle_, data_value_desc_, data_value_, data_spatial_shapes_desc_,
        data_spatial_shapes_, data_level_start_index_desc_,
        data_level_start_index_, data_sampling_loc_desc_, data_sampling_loc_,
        data_attn_weight_desc_, data_attn_weight_, im2col_step_, data_col_desc_,
        data_col_);
    destroy();
    return expected_status_ == status;
  }

  void destroy() {
    if (handle_) {
      CNRT_CHECK(cnrtQueueSync(handle_->queue));
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = nullptr;
    }

    if (data_value_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(data_value_desc_));
      data_value_desc_ = nullptr;
    }

    if (data_value_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(data_value_));
      data_value_ = nullptr;
    }

    if (data_spatial_shapes_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(data_spatial_shapes_desc_));
      data_spatial_shapes_desc_ = nullptr;
    }

    if (data_spatial_shapes_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(data_spatial_shapes_));
      data_spatial_shapes_ = nullptr;
    }

    if (data_level_start_index_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(data_level_start_index_desc_));
      data_level_start_index_desc_ = nullptr;
    }

    if (data_level_start_index_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(data_level_start_index_));
      data_level_start_index_ = nullptr;
    }

    if (data_sampling_loc_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(data_sampling_loc_desc_));
      data_sampling_loc_desc_ = nullptr;
    }

    if (data_sampling_loc_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(data_sampling_loc_));
      data_sampling_loc_ = nullptr;
    }

    if (data_attn_weight_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(data_attn_weight_desc_));
      data_attn_weight_desc_ = nullptr;
    }

    if (data_attn_weight_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(data_attn_weight_));
      data_attn_weight_ = nullptr;
    }

    if (data_col_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(data_col_desc_));
      data_col_desc_ = nullptr;
    }

    if (data_col_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(data_col_));
      data_col_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t data_value_desc_ = nullptr;
  void *data_value_ = nullptr;
  mluOpTensorDescriptor_t data_spatial_shapes_desc_ = nullptr;
  void *data_spatial_shapes_ = nullptr;
  mluOpTensorDescriptor_t data_level_start_index_desc_ = nullptr;
  void *data_level_start_index_ = nullptr;
  mluOpTensorDescriptor_t data_sampling_loc_desc_ = nullptr;
  void *data_sampling_loc_ = nullptr;
  mluOpTensorDescriptor_t data_attn_weight_desc_ = nullptr;
  void *data_attn_weight_ = nullptr;
  mluOpTensorDescriptor_t data_col_desc_ = nullptr;
  void *data_col_ = nullptr;
  int32_t im2col_step_ = 1;
  mluOpDevType_t target_device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(ms_deform_attn_forward_general, negative) { EXPECT_TRUE(compute()); }

INSTANTIATE_TEST_CASE_P(
    zero_element_1, ms_deform_attn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 3, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({0, 7, 4, 6, 8, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({0, 7, 4, 6, 8})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 7, 4, 5})}),
        testing::Values(MsDeformAttnForwardParam{MLUOP_UNKNOWN_DEVICE,
                                                 MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    zero_element_2, ms_deform_attn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 0, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({0, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 0, 8, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 0, 8})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5})}),
        testing::Values(MsDeformAttnForwardParam{MLUOP_UNKNOWN_DEVICE,
                                                 MLUOP_STATUS_SUCCESS})));

INSTANTIATE_TEST_CASE_P(
    zero_element_3, ms_deform_attn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 0, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 0, 6, 8, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 0, 6, 8})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 0, 5})}),
        testing::Values(MsDeformAttnForwardParam{MLUOP_UNKNOWN_DEVICE,
                                                 MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    zero_element_4, ms_deform_attn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 0})}),
        testing::Values(MsDeformAttnForwardParam{MLUOP_UNKNOWN_DEVICE,
                                                 MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    zero_element_5, ms_deform_attn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 0, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5})}),
        testing::Values(MsDeformAttnForwardParam{MLUOP_UNKNOWN_DEVICE,
                                                 MLUOP_STATUS_SUCCESS})));

INSTANTIATE_TEST_CASE_P(
    negative_input_data_value, ms_deform_attn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 3, 4, 5})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 4, 5})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 5, 5})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 6})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 3, 4})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 3, 4, 5, 6})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5})}),
        testing::Values(MsDeformAttnForwardParam{MLUOP_UNKNOWN_DEVICE,
                                                 MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    negative_input_data_spatial_shapes, ms_deform_attn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({6, 2})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({7, 2})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 3})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({6, 2, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5})}),
        testing::Values(MsDeformAttnForwardParam{MLUOP_UNKNOWN_DEVICE,
                                                 MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    negative_input_data_level_start_index, ms_deform_attn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({6})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({7})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5})}),
        testing::Values(MsDeformAttnForwardParam{MLUOP_UNKNOWN_DEVICE,
                                                 MLUOP_STATUS_BAD_PARAM})));
INSTANTIATE_TEST_CASE_P(
    negative_input_data_sampling_loc, ms_deform_attn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6})}),
        testing::Values(
            MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 6,
                             std::vector<int>({2, 7, 4, 6, 8, 2})},
            MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({3, 7, 4, 6, 8, 2})},
            MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({2, 8, 4, 6, 8, 2})},
            MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({2, 7, 5, 6, 8, 2})},
            MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({2, 7, 4, 7, 8, 2})},
            MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({2, 7, 4, 6, 9, 2})},
            MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({2, 7, 4, 6, 8, 3})},
            MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 7,
                             std::vector<int>({2, 7, 4, 6, 8, 2, 1})},
            MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 5,
                             std::vector<int>({2, 7, 4, 6, 8})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5})}),
        testing::Values(MsDeformAttnForwardParam{MLUOP_UNKNOWN_DEVICE,
                                                 MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    negative_input_data_attn_weight, ms_deform_attn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 7, 4, 6, 8})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({3, 7, 4, 6, 8})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 8, 4, 6, 8})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 5, 6, 8})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 7, 8})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 9})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 1})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 6})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5})}),
        testing::Values(MsDeformAttnForwardParam{MLUOP_UNKNOWN_DEVICE,
                                                 MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    negative_input_data_col, ms_deform_attn_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 7, 4, 5})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 7, 4, 5})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 8, 4, 5})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 5, 5})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 6})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 5, 1})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 7, 4})}),
        testing::Values(MsDeformAttnForwardParam{MLUOP_UNKNOWN_DEVICE,
                                                 MLUOP_STATUS_BAD_PARAM})));

}  // namespace mluopapitest
