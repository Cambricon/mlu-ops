/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
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

typedef std::tuple<MLUOpTensorParam, MLUOpTensorParam, MLUOpTensorParam,
                   MLUOpTensorParam, mluOpDevType_t, mluOpStatus_t>
    DiffIouRotatedSortVerticesForward;
class diff_iou_rotated_sort_vertices_forward_general
    : public testing::TestWithParam<DiffIouRotatedSortVerticesForward> {
 public:
  void SetUp() {
    try {
      MLUOP_CHECK(mluOpCreate(&handle_));

      MLUOpTensorParam vertices_params = std::get<0>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&vertices_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          vertices_desc_, vertices_params.get_layout(), vertices_params.get_dtype(),
          vertices_params.get_dim_nb(), vertices_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(vertices_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&vertices_, mluOpDataTypeBytes(vertices_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&vertices_, mluOpDataTypeBytes(vertices_params.get_dtype()) *
                                         mluOpGetTensorElementNum(vertices_desc_)));
      }

      MLUOpTensorParam mask_params = std::get<1>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&mask_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          mask_desc_, mask_params.get_layout(), mask_params.get_dtype(),
          mask_params.get_dim_nb(), mask_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(mask_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&mask_, mluOpDataTypeBytes(mask_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&mask_, mluOpDataTypeBytes(mask_params.get_dtype()) *
                                         mluOpGetTensorElementNum(mask_desc_)));
      }

      MLUOpTensorParam num_valid_params = std::get<2>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&num_valid_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          num_valid_desc_, num_valid_params.get_layout(),
          num_valid_params.get_dtype(), num_valid_params.get_dim_nb(),
          num_valid_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(num_valid_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(
                &num_valid_,
                mluOpDataTypeBytes(num_valid_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&num_valid_,
                       mluOpDataTypeBytes(num_valid_params.get_dtype()) *
                           mluOpGetTensorElementNum(num_valid_desc_)));
      }

      MLUOpTensorParam idx_params = std::get<3>(GetParam());
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&idx_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          idx_desc_, idx_params.get_layout(), idx_params.get_dtype(),
          idx_params.get_dim_nb(), idx_params.get_dim_size().data()));
      if (mluOpGetTensorElementNum(idx_desc_) >= LARGE_TENSOR_NUM) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&idx_, mluOpDataTypeBytes(idx_params.get_dtype()) * 2));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&idx_, mluOpDataTypeBytes(idx_params.get_dtype()) *
                                        mluOpGetTensorElementNum(idx_desc_)));
      }

      target_device_ = std::get<4>(GetParam());
      expected_status_ = std::get<5>(GetParam());
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in diff_iou_rotated_sort_vertices_forward_general.";
    }
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpDiffIouRotatedSortVerticesForward(
        handle_, vertices_desc_, vertices_, mask_desc_, mask_, num_valid_desc_,
        num_valid_, idx_desc_, idx_);
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

    if (vertices_desc_) {
      VLOG(4) << "Destroy vertices_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(vertices_desc_));
      vertices_desc_ = nullptr;
    }

    if (vertices_) {
      VLOG(4) << "Destroy vertices_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(vertices_));
      vertices_ = nullptr;
    }

    if (mask_desc_) {
      VLOG(4) << "Destroy mask_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(mask_desc_));
      mask_desc_ = nullptr;
    }

    if (mask_) {
      VLOG(4) << "Destroy mask_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(mask_));
      mask_ = nullptr;
    }

    if (num_valid_desc_) {
      VLOG(4) << "Destroy num_valid_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(num_valid_desc_));
      num_valid_desc_ = nullptr;
    }

    if (num_valid_) {
      VLOG(4) << "Destroy num_valid_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(num_valid_));
      num_valid_ = nullptr;
    }

    if (idx_desc_) {
      VLOG(4) << "Destroy idx_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(idx_desc_));
      idx_desc_ = nullptr;
    }

    if (idx_) {
      VLOG(4) << "Destroy idx_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(idx_));
      idx_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t vertices_desc_ = nullptr;
  void *vertices_ = nullptr;
  mluOpTensorDescriptor_t mask_desc_ = nullptr;
  void *mask_ = nullptr;
  mluOpTensorDescriptor_t num_valid_desc_ = nullptr;
  void *num_valid_ = nullptr;
  mluOpTensorDescriptor_t idx_desc_ = nullptr;
  void *idx_ = nullptr;
  mluOpDevType_t target_device_;
  mluOpStatus_t expected_status_;
};

TEST_P(diff_iou_rotated_sort_vertices_forward_general, negative) {
  EXPECT_TRUE(compute());
}

INSTANTIATE_TEST_CASE_P(
    zero_element_1, diff_iou_rotated_sort_vertices_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 16, 24, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         3, std::vector<int>({0, 16, 24})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({0, 16})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({0, 16, 9})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_2, diff_iou_rotated_sort_vertices_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 0, 24, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         3, std::vector<int>({4, 0, 24})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({4, 0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 0, 9})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    vertices_error_dim, diff_iou_rotated_sort_vertices_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({4, 16, 24, 2, 1})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 24})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         3, std::vector<int>({4, 16, 24})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({4, 16})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 16, 9})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    mask_error_dim, diff_iou_rotated_sort_vertices_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 24, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         4, std::vector<int>({4, 16, 24, 1})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         2, std::vector<int>({4, 16})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({4, 16})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 16, 9})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    num_valid_error_dim, diff_iou_rotated_sort_vertices_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 24, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         3, std::vector<int>({4, 16, 24})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 16, 1})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         1, std::vector<int>({4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 16, 9})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    idx_error_dim, diff_iou_rotated_sort_vertices_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 24, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         3, std::vector<int>({4, 16, 24})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({4, 16})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({4, 16, 9, 1})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({4, 16})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    vertices_error_shape, diff_iou_rotated_sort_vertices_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({5, 16, 24, 2})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 15, 24, 2})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 20, 2})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 24, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         3, std::vector<int>({4, 16, 24})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({4, 16})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 16, 9})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    mask_error_shape, diff_iou_rotated_sort_vertices_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 24, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         3, std::vector<int>({3, 16, 24})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         3, std::vector<int>({4, 17, 24})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         3, std::vector<int>({4, 16, 28})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({4, 16})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 16, 9})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    num_valid_error_shape, diff_iou_rotated_sort_vertices_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 24, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         3, std::vector<int>({4, 16, 24})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({5, 16})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({4, 15})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 16, 9})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    idx_error_shape, diff_iou_rotated_sort_vertices_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 24, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         3, std::vector<int>({4, 16, 24})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({4, 16})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({5, 16, 9})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({4, 17, 9})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 16, 8})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    vertices_error_dtype, diff_iou_rotated_sort_vertices_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({4, 16, 24, 2})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         4, std::vector<int>({4, 16, 24, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         3, std::vector<int>({4, 16, 24})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({4, 16})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 16, 9})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    mask_error_dtype, diff_iou_rotated_sort_vertices_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 24, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 24})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 16, 24})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({4, 16})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 16, 9})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    num_valid_error_dtype, diff_iou_rotated_sort_vertices_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 24, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         3, std::vector<int>({4, 16, 24})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         2, std::vector<int>({4, 16})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         2, std::vector<int>({4, 16})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({4, 16, 9})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    idx_error_dtype, diff_iou_rotated_sort_vertices_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 24, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         3, std::vector<int>({4, 16, 24})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({4, 16})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 9})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64,
                                         3, std::vector<int>({4, 16, 9})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    large_tensor_notsupported, diff_iou_rotated_sort_vertices_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({65536, 65536, 24, 2})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_BOOL,
                                         3, std::vector<int>({5536, 65536, 24})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         2, std::vector<int>({5536, 65536})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({5536, 65536, 9})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

}  // namespace mluopapitest
