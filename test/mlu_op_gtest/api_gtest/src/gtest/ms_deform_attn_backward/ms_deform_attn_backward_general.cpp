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

#define LARGE_TENSOR_NUM ((uint64_t)2147483648)

namespace mluopapitest {

typedef std::tuple<int32_t, mluOpDevType_t, mluOpStatus_t>
    MsDeformAttnBackwardParam;
typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MsDeformAttnBackwardParam>
    MsDeformAttnBackward;
class ms_deform_attn_backward_general
    : public testing::TestWithParam<MsDeformAttnBackward> {
 public:
  void SetUp() {
    try {
      MLUOP_CHECK(mluOpCreate(&handle_));

      MLUOpTensorParam value_params = std::get<0>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&value_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          value_desc_, value_params.get_layout(), value_params.get_dtype(),
          value_params.get_dim_nb(), value_params.get_dim_size().data()));

      if (mluOpGetTensorElementNum(value_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&value_,
                       mluOpDataTypeBytes(value_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&value_, mluOpDataTypeBytes(value_params.get_dtype()) *
                                    mluOpGetTensorElementNum(value_desc_)));
      }

      MLUOpTensorParam spatial_shapes_params = std::get<1>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&spatial_shapes_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          spatial_shapes_desc_, spatial_shapes_params.get_layout(),
          spatial_shapes_params.get_dtype(), spatial_shapes_params.get_dim_nb(),
          spatial_shapes_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(spatial_shapes_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &spatial_shapes_,
                mluOpDataTypeBytes(spatial_shapes_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&spatial_shapes_,
                       mluOpDataTypeBytes(spatial_shapes_params.get_dtype()) *
                           mluOpGetTensorElementNum(spatial_shapes_desc_)));
      }

      MLUOpTensorParam level_start_index_params = std::get<2>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&level_start_index_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          level_start_index_desc_, level_start_index_params.get_layout(),
          level_start_index_params.get_dtype(),
          level_start_index_params.get_dim_nb(),
          level_start_index_params.get_dim_size().data()));

      if (mluOpGetTensorElementNum(level_start_index_desc_) >=
          LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &level_start_index_,
                mluOpDataTypeBytes(level_start_index_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &level_start_index_,
                mluOpDataTypeBytes(level_start_index_params.get_dtype()) *
                    mluOpGetTensorElementNum(level_start_index_desc_)));
      }

      MLUOpTensorParam sampling_loc_params = std::get<3>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&sampling_loc_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          sampling_loc_desc_, sampling_loc_params.get_layout(),
          sampling_loc_params.get_dtype(), sampling_loc_params.get_dim_nb(),
          sampling_loc_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(sampling_loc_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &sampling_loc_,
                mluOpDataTypeBytes(sampling_loc_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&sampling_loc_,
                       mluOpDataTypeBytes(sampling_loc_params.get_dtype()) *
                           mluOpGetTensorElementNum(sampling_loc_desc_)));
      }

      MLUOpTensorParam attn_weight_params = std::get<4>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&attn_weight_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          attn_weight_desc_, attn_weight_params.get_layout(),
          attn_weight_params.get_dtype(), attn_weight_params.get_dim_nb(),
          attn_weight_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(attn_weight_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&attn_weight_,
                       mluOpDataTypeBytes(attn_weight_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&attn_weight_,
                       mluOpDataTypeBytes(attn_weight_params.get_dtype()) *
                           mluOpGetTensorElementNum(attn_weight_desc_)));
      }

      MLUOpTensorParam grad_output_params = std::get<5>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_output_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_output_desc_, grad_output_params.get_layout(),
          grad_output_params.get_dtype(), grad_output_params.get_dim_nb(),
          grad_output_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(grad_output_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&grad_output_,
                       mluOpDataTypeBytes(grad_output_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&grad_output_,
                       mluOpDataTypeBytes(grad_output_params.get_dtype()) *
                           mluOpGetTensorElementNum(grad_output_desc_)));
      }

      MLUOpTensorParam grad_value_params = std::get<6>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_value_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_value_desc_, grad_value_params.get_layout(),
          grad_value_params.get_dtype(), grad_value_params.get_dim_nb(),
          grad_value_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(grad_value_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&grad_value_,
                       mluOpDataTypeBytes(grad_value_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&grad_value_,
                       mluOpDataTypeBytes(grad_value_params.get_dtype()) *
                           mluOpGetTensorElementNum(grad_value_desc_)));
      }

      MLUOpTensorParam grad_sampling_loc_params = std::get<7>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_sampling_loc_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_sampling_loc_desc_, grad_sampling_loc_params.get_layout(),
          grad_sampling_loc_params.get_dtype(),
          grad_sampling_loc_params.get_dim_nb(),
          grad_sampling_loc_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(grad_sampling_loc_desc_) >=
          LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &grad_sampling_loc_,
                mluOpDataTypeBytes(grad_sampling_loc_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &grad_sampling_loc_,
                mluOpDataTypeBytes(grad_sampling_loc_params.get_dtype()) *
                    mluOpGetTensorElementNum(grad_sampling_loc_desc_)));
      }

      MLUOpTensorParam grad_attn_weight_params = std::get<8>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_attn_weight_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_attn_weight_desc_, grad_attn_weight_params.get_layout(),
          grad_attn_weight_params.get_dtype(),
          grad_attn_weight_params.get_dim_nb(),
          grad_attn_weight_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(grad_attn_weight_desc_) >=
          LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &grad_attn_weight_,
                mluOpDataTypeBytes(grad_attn_weight_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&grad_attn_weight_,
                       mluOpDataTypeBytes(grad_attn_weight_params.get_dtype()) *
                           mluOpGetTensorElementNum(grad_attn_weight_desc_)));
      }

      MsDeformAttnBackwardParam msDeformAttnBackwardParam =
          std::get<9>(GetParam());
      std::tie(im2col_step_, target_device_, expected_status_) =
          msDeformAttnBackwardParam;
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in ms_deform_attn_backward general.";
    }
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpMsDeformAttnBackward(
        handle_, value_desc_, value_, spatial_shapes_desc_, spatial_shapes_,
        level_start_index_desc_, level_start_index_, sampling_loc_desc_,
        sampling_loc_, attn_weight_desc_, attn_weight_, grad_output_desc_,
        grad_output_, im2col_step_, grad_value_desc_, grad_value_,
        grad_sampling_loc_desc_, grad_sampling_loc_, grad_attn_weight_desc_,
        grad_attn_weight_);
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

    if (value_desc_) {
      VLOG(4) << "Destroy value_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(value_desc_));
      value_desc_ = nullptr;
    }

    if (value_) {
      VLOG(4) << "Destroy value";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(value_));
      value_ = nullptr;
    }

    if (spatial_shapes_desc_) {
      VLOG(4) << "Destroy spatial_shapes_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(spatial_shapes_desc_));
      spatial_shapes_desc_ = nullptr;
    }

    if (spatial_shapes_) {
      VLOG(4) << "Destroy spatial_shapes";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(spatial_shapes_));
      spatial_shapes_ = nullptr;
    }

    if (level_start_index_desc_) {
      VLOG(4) << "Destroy level_start_index_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(level_start_index_desc_));
      level_start_index_desc_ = nullptr;
    }

    if (level_start_index_) {
      VLOG(4) << "Destroy level_start_index";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(level_start_index_));
      level_start_index_ = nullptr;
    }

    if (sampling_loc_desc_) {
      VLOG(4) << "Destroy sampling_loc_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(sampling_loc_desc_));
      sampling_loc_desc_ = nullptr;
    }

    if (sampling_loc_) {
      VLOG(4) << "Destroy sampling_loc";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(sampling_loc_));
      sampling_loc_ = nullptr;
    }

    if (attn_weight_desc_) {
      VLOG(4) << "Destroy attn_weight_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(attn_weight_desc_));
      attn_weight_desc_ = nullptr;
    }

    if (attn_weight_) {
      VLOG(4) << "Destroy attn_weight";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(attn_weight_));
      attn_weight_ = nullptr;
    }

    if (grad_output_desc_) {
      VLOG(4) << "Destroy grad_output_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_output_desc_));
      grad_output_desc_ = nullptr;
    }

    if (grad_output_) {
      VLOG(4) << "Destroy grad_output";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_output_));
      grad_output_ = nullptr;
    }

    if (grad_value_desc_) {
      VLOG(4) << "Destroy grad_value_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_value_desc_));
      grad_value_desc_ = nullptr;
    }

    if (grad_value_) {
      VLOG(4) << "Destroy grad_value";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_value_));
      grad_value_ = nullptr;
    }

    if (grad_sampling_loc_desc_) {
      VLOG(4) << "Destroy grad_sampling_loc_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_sampling_loc_desc_));
      grad_sampling_loc_desc_ = nullptr;
    }

    if (grad_sampling_loc_) {
      VLOG(4) << "Destroy grad_sampling_loc";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_sampling_loc_));
      grad_sampling_loc_ = nullptr;
    }

    if (grad_attn_weight_desc_) {
      VLOG(4) << "Destroy grad_attn_weight_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_attn_weight_desc_));
      grad_attn_weight_desc_ = nullptr;
    }

    if (grad_attn_weight_) {
      VLOG(4) << "Destroy grad_attn_weight";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_attn_weight_));
      grad_attn_weight_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t value_desc_ = nullptr;
  void *value_ = nullptr;
  mluOpTensorDescriptor_t spatial_shapes_desc_ = nullptr;
  void *spatial_shapes_ = nullptr;
  mluOpTensorDescriptor_t level_start_index_desc_ = nullptr;
  void *level_start_index_ = nullptr;
  mluOpTensorDescriptor_t sampling_loc_desc_ = nullptr;
  void *sampling_loc_ = nullptr;
  mluOpTensorDescriptor_t attn_weight_desc_ = nullptr;
  void *attn_weight_ = nullptr;
  mluOpTensorDescriptor_t grad_output_desc_ = nullptr;
  void *grad_output_ = nullptr;
  mluOpTensorDescriptor_t grad_value_desc_ = nullptr;
  void *grad_value_ = nullptr;
  mluOpTensorDescriptor_t grad_sampling_loc_desc_ = nullptr;
  void *grad_sampling_loc_ = nullptr;
  mluOpTensorDescriptor_t grad_attn_weight_desc_ = nullptr;
  void *grad_attn_weight_ = nullptr;
  int32_t im2col_step_ = 1;
  mluOpDevType_t target_device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(ms_deform_attn_backward_general, negative) { EXPECT_TRUE(compute()); }

INSTANTIATE_TEST_CASE_P(
    zero_element_num_keys, ms_deform_attn_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 0, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 0, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MsDeformAttnBackwardParam{1, MLUOP_UNKNOWN_DEVICE,
                                                  MLUOP_STATUS_SUCCESS})));

INSTANTIATE_TEST_CASE_P(
    zero_element_num_heads, ms_deform_attn_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 0, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 0, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 0, 6, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 0, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 0, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 0, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 0, 6, 8}))),
        testing::Values(MsDeformAttnBackwardParam{1, MLUOP_UNKNOWN_DEVICE,
                                                  MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    zero_element_channels, ms_deform_attn_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MsDeformAttnBackwardParam{1, MLUOP_UNKNOWN_DEVICE,
                                                  MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    zero_element_num_levels, ms_deform_attn_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({0, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 0, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 0, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 0, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 0, 8}))),
        testing::Values(MsDeformAttnBackwardParam{1, MLUOP_UNKNOWN_DEVICE,
                                                  MLUOP_STATUS_SUCCESS})));

INSTANTIATE_TEST_CASE_P(
    zero_element_num_points, ms_deform_attn_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 0, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 0}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 0, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 0}))),
        testing::Values(MsDeformAttnBackwardParam{1, MLUOP_UNKNOWN_DEVICE,
                                                  MLUOP_STATUS_SUCCESS})));

INSTANTIATE_TEST_CASE_P(
    negative_bad_im2col_step, ms_deform_attn_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({4, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({4, 7, 4, 6, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 7, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({4, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({4, 7, 4, 6, 8}))),
        testing::Values(MsDeformAttnBackwardParam{0, MLUOP_UNKNOWN_DEVICE,
                                                  MLUOP_STATUS_BAD_PARAM},
                        MsDeformAttnBackwardParam{3, MLUOP_UNKNOWN_DEVICE,
                                                  MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    negative_input_value, ms_deform_attn_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 3, 4, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 4, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 5, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 6})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 3, 4})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 3, 4, 5, 6}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MsDeformAttnBackwardParam{1, MLUOP_UNKNOWN_DEVICE,
                                                  MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    negative_input_spatial_shapes, ms_deform_attn_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({6, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({7, 2})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 3})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({6, 2, 3}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MsDeformAttnBackwardParam{1, MLUOP_UNKNOWN_DEVICE,
                                                  MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    negative_input_level_start_index, ms_deform_attn_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         1, std::vector<int>({6})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({7})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MsDeformAttnBackwardParam{1, MLUOP_UNKNOWN_DEVICE,
                                                  MLUOP_STATUS_BAD_PARAM})));
INSTANTIATE_TEST_CASE_P(
    negative_input_sampling_loc, ms_deform_attn_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6}))),
        testing::Values(
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 6,
                             std::vector<int>({2, 7, 4, 6, 8, 2})),
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({3, 7, 4, 6, 8, 2})),
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({2, 8, 4, 6, 8, 2})),
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({2, 7, 5, 6, 8, 2})),
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({2, 7, 4, 7, 8, 2})),
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({2, 7, 4, 6, 9, 2})),
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({2, 7, 4, 6, 8, 3})),
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 7,
                             std::vector<int>({2, 7, 4, 6, 8, 2, 1})),
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 5,
                             std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MsDeformAttnBackwardParam{1, MLUOP_UNKNOWN_DEVICE,
                                                  MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    negative_input_attn_weight, ms_deform_attn_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 7, 4, 6, 8})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({3, 7, 4, 6, 8})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 8, 4, 6, 8})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 5, 6, 8})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 7, 8})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 6}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MsDeformAttnBackwardParam{1, MLUOP_UNKNOWN_DEVICE,
                                                  MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    negative_input_grad_output, ms_deform_attn_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 7, 4, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 7, 4, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 8, 4, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 5, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 6})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 5, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 7, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MsDeformAttnBackwardParam{1, MLUOP_UNKNOWN_DEVICE,
                                                  MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    negative_output_grad_value, ms_deform_attn_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 3, 4, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 3, 4, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 4, 4, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 5, 5})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 6})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 3, 4, 5, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 3, 4}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MsDeformAttnBackwardParam{1, MLUOP_UNKNOWN_DEVICE,
                                                  MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    negative_output_grad_sampling_loc, ms_deform_attn_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 6,
                             std::vector<int>({2, 7, 4, 6, 8, 2})),
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({3, 7, 4, 6, 8, 2})),
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({2, 8, 4, 6, 8, 2})),
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({2, 7, 5, 6, 8, 2})),
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({2, 7, 4, 7, 8, 2})),
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({2, 7, 4, 6, 9, 2})),
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({2, 7, 4, 6, 8, 3})),
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 7,
                             std::vector<int>({2, 7, 4, 6, 8, 2, 1})),
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 5,
                             std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MsDeformAttnBackwardParam{1, MLUOP_UNKNOWN_DEVICE,
                                                  MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    negative_output_grad_attn_weight, ms_deform_attn_backward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({6, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({6}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 8}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 3, 4, 5}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({2, 7, 4, 6, 8})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({3, 7, 4, 6, 8})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 8, 4, 6, 8})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 5, 6, 8})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 7, 8})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({2, 7, 4, 6, 9})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({2, 7, 4, 6, 8, 1})),
                        MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 7, 4, 6}))),
        testing::Values(MsDeformAttnBackwardParam{1, MLUOP_UNKNOWN_DEVICE,
                                                  MLUOP_STATUS_BAD_PARAM})));

INSTANTIATE_TEST_CASE_P(
    negative_large_tensor, ms_deform_attn_backward_general,
    testing::Combine(
        testing::Values(
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 4,
                             std::vector<int>({100, 1000, 100, 215}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({1, 2}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({1}))),
        testing::Values(
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({100, 1, 100, 1, 1, 2}))),
        testing::Values(
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 5,
                             std::vector<int>({100, 1, 100, 1, 1}))),
        testing::Values(MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4,
                                         std::vector<int>({100, 1, 100, 215}))),
        testing::Values(
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 4,
                             std::vector<int>({100, 1000, 100, 215}))),
        testing::Values(
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
                             std::vector<int>({100, 1, 100, 1, 1, 2}))),
        testing::Values(
            MLUOpTensorParam(MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 5,
                             std::vector<int>({100, 1, 100, 1, 1}))),
        testing::Values(MsDeformAttnBackwardParam{
            1, MLUOP_UNKNOWN_DEVICE, MLUOP_STATUS_NOT_SUPPORTED})));

}  // namespace mluopapitest
