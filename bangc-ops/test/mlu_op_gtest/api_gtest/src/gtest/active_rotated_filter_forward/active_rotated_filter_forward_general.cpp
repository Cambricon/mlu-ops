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
                   mluOpDevType_t, mluOpStatus_t>
    ActiveRotatedFilterForwardParams;

class active_rotated_filter_forward_general
    : public testing::TestWithParam<ActiveRotatedFilterForwardParams> {
 public:
  void SetUp() {
    try {
      MLUOP_CHECK(mluOpCreate(&handle_));

      MLUOpTensorParam input_desc = std::get<0>(GetParam());
      mluOpTensorLayout_t input_layout = input_desc.get_layout();
      mluOpDataType_t input_dtype = input_desc.get_dtype();
      int input_doutput_nb = input_desc.get_dim_nb();
      std::vector<int> input_dims = input_desc.get_dim_size();
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(input_desc_, input_layout,
                                           input_dtype, input_doutput_nb,
                                           input_dims.data()));
      int input_elenum = mluOpGetTensorElementNum(input_desc_);
      if (input_elenum > 0) {
        VLOG(4) << "malloc input_";
        uint64_t i_bytes = input_elenum * mluOpDataTypeBytes(input_dtype);
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&input_, i_bytes))
      }

      MLUOpTensorParam indices_desc = std::get<1>(GetParam());
      mluOpTensorLayout_t indices_layout = indices_desc.get_layout();
      mluOpDataType_t indices_dtype = indices_desc.get_dtype();
      int indices_doutput_nb = indices_desc.get_dim_nb();
      std::vector<int> indices_dims = indices_desc.get_dim_size();
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&indices_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(indices_desc_, indices_layout,
                                           indices_dtype, indices_doutput_nb,
                                           indices_dims.data()));
      int indices_elenum = mluOpGetTensorElementNum(indices_desc_);
      if (indices_elenum > 0) {
        VLOG(4) << "malloc indices_";
        uint64_t id_bytes = indices_elenum * mluOpDataTypeBytes(indices_dtype);
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&indices_, id_bytes))
      }

      MLUOpTensorParam output_desc = std::get<2>(GetParam());
      mluOpTensorLayout_t output_layout = output_desc.get_layout();
      mluOpDataType_t output_dtype = output_desc.get_dtype();
      int output_doutput_nb = output_desc.get_dim_nb();
      std::vector<int> featuret_dims = output_desc.get_dim_size();
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(output_desc_, output_layout,
                                           output_dtype, output_doutput_nb,
                                           featuret_dims.data()));
      int output_elenum = mluOpGetTensorElementNum(output_desc_);
      if (output_elenum > 0) {
        VLOG(4) << "malloc output_";
        uint64_t o_bytes = output_elenum * mluOpDataTypeBytes(output_dtype);
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&output_, o_bytes))
      }

      target_device_ = std::get<3>(GetParam());
      expected_status_ = std::get<4>(GetParam());
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in active_rotated_filter_forward";
    }
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status;
    status = mluOpGetActiveRotatedFilterForwardWorkspaceSize(
        handle_, input_desc_, &workspace_size_);
    if (status != MLUOP_STATUS_SUCCESS) {
      destroy();
      return expected_status_ == status;
    }
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_))
    status = mluOpActiveRotatedFilterForward(
        handle_, input_desc_, input_, indices_desc_, indices_, workspace_,
        workspace_size_, output_desc_, output_);
    destroy();
    return expected_status_ == status;
  }

 protected:
  void destroy() {
    try {
      if (handle_) {
        CNRT_CHECK(cnrtQueueSync(handle_->queue));
        VLOG(4) << "Destroy handle_";
        MLUOP_CHECK(mluOpDestroy(handle_));
        handle_ = nullptr;
      }
      if (input_desc_) {
        VLOG(4) << "Destroy input_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_desc_));
        input_desc_ = nullptr;
      }
      if (input_) {
        VLOG(4) << "Destroy input_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_));
        input_ = nullptr;
      }
      if (indices_desc_) {
        VLOG(4) << "Destroy indices_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(indices_desc_));
        indices_desc_ = nullptr;
      }
      if (indices_) {
        VLOG(4) << "Destroy indices_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(indices_));
        indices_ = nullptr;
      }
      if (workspace_) {
        VLOG(4) << "Destroy workspace_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
        workspace_ = nullptr;
      }
      if (output_desc_) {
        VLOG(4) << "Destroy output_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
        output_desc_ = nullptr;
      }
      if (output_) {
        VLOG(4) << "Destroy output_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_));
        output_ = nullptr;
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in active_rotated_filter_forward";
    }
  }

  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t input_desc_ = nullptr;
  void *input_ = nullptr;
  mluOpTensorDescriptor_t indices_desc_ = nullptr;
  void *indices_ = nullptr;
  void *workspace_ = nullptr;
  size_t workspace_size_ = 64;
  mluOpTensorDescriptor_t output_desc_ = nullptr;
  void *output_ = nullptr;
  mluOpDevType_t target_device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(active_rotated_filter_forward_general, negative) {
  EXPECT_TRUE(compute());
}

// INSTANTIATE_TEST_CASE_P(
//     success,
//     active_rotated_filter_forward_general,
//     testing::Combine(testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY,
//     MLUOP_DTYPE_FLOAT, 5,
//                                                        std::vector<int>({1,
//                                                        2, 8, 3, 3})}),
//                      testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY,
//                      MLUOP_DTYPE_INT32, 4,
//                                                        std::vector<int>({8,
//                                                        3, 3, 4})}),
//                      testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY,
//                      MLUOP_DTYPE_FLOAT, 4,
//                                                        std::vector<int>({4,
//                                                        16, 3, 3})}),
//                      testing::Values(MLUOP_UNKNOWN_DEVICE),
//                      testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    zero_element_input_dim0, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({0, 2, 8, 3, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({8, 3, 3, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 16, 3, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_input_dim1, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({1, 0, 8, 3, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({8, 3, 3, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 0, 3, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_input_dim2, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({1, 2, 0, 3, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({0, 3, 3, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 0, 3, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    zero_element_input_dim3, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({1, 2, 8, 0, 0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({8, 0, 0, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 0, 0})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));
INSTANTIATE_TEST_CASE_P(
    zero_element_input_indices, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({1, 2, 8, 3, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({8, 3, 3, 0})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({0, 16, 3, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_input_dtype, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({1, 2, 8, 3, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({8, 3, 3, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 3, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_indices_dtype, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({1, 2, 8, 3, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({8, 3, 3, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 3, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_output_dtype, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({1, 2, 8, 3, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({8, 3, 3, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({4, 16, 3, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_input_indices_shape, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({1, 2, 8, 3, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({16, 3, 3, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 3, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_input_indices_shape_1, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({1, 2, 7, 3, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({7, 3, 3, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 3, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_input_indices_shape_2, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5,
                                         std::vector<int>({1, 2, 256, 3, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({256, 3, 3, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 3, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_w_h_1, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({1, 2, 8, 1, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({8, 3, 3, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 3, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_w_h_2, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({1, 2, 8, 1, 1})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({8, 3, 3, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 3, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_w_h_3, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({1, 2, 8, 1, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({8, 1, 3, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 1, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_w_h_4, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({1, 2, 8, 4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({8, 4, 4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 4, 4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_indices, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({1, 2, 8, 3, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({8, 3, 3, 1})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 16, 3, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_output, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({1, 2, 8, 3, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({8, 3, 3, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({5, 16, 3, 3})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 17, 3, 3})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 1, 3})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 3, 1})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 1, 1})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({4, 16, 3, 3, 1})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({4, 16, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_input_dim, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         6,
                                         std::vector<int>({1, 2, 8, 3, 3, 1})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 2, 8, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({8, 3, 3, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 3, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_indices_dim, active_rotated_filter_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         5, std::vector<int>({1, 2, 8, 3, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         5, std::vector<int>({8, 3, 3, 4, 1})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({8, 3, 3})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({4, 16, 3, 3})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

}  // namespace mluopapitest
