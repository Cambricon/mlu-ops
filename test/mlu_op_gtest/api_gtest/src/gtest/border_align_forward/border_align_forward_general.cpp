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
                   MLUOpTensorParam, mluOpDevType_t, mluOpStatus_t>
    BorderAlignForwardParams;

class border_align_forward_general
    : public testing::TestWithParam<BorderAlignForwardParams> {
 public:
  void SetUp() {
    try {
      mluOpCreate(&handle_);

      MLUOpTensorParam input_desc = std::get<0>(GetParam());
      mluOpTensorLayout_t input_layout = input_desc.get_layout();
      mluOpDataType_t input_dtype = input_desc.get_dtype();
      int input_dim_nb = input_desc.get_dim_nb();
      std::vector<int> input_dims = input_desc.get_dim_size();
      std::vector<int> input_stride = input_desc.get_dim_stride();
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
      if (input_stride.empty()) {
        MLUOP_CHECK(mluOpSetTensorDescriptor(input_desc_, input_layout,
                                            input_dtype, input_dim_nb,
                                            input_dims.data()));
      } else {
        MLUOP_CHECK(mluOpSetTensorDescriptorEx(input_desc_, input_layout,
                                            input_dtype, input_dim_nb,
                                            input_dims.data(),
                                            input_stride.data()));
      }
      int input_elenum = mluOpGetTensorElementNum(input_desc_);
      if (input_elenum > 0) {
        VLOG(4) << "malloc input_";
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&input_,
                               input_elenum * mluOpDataTypeBytes(input_dtype)));
      }

      MLUOpTensorParam boxes_desc = std::get<1>(GetParam());
      mluOpTensorLayout_t boxes_layout = boxes_desc.get_layout();
      mluOpDataType_t boxes_dtype = boxes_desc.get_dtype();
      int boxes_dim_nb = boxes_desc.get_dim_nb();
      std::vector<int> boxes_dims = boxes_desc.get_dim_size();
      std::vector<int> boxes_stride = boxes_desc.get_dim_stride();
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&boxes_desc_));
      if (boxes_stride.empty()) {
        MLUOP_CHECK(mluOpSetTensorDescriptor(boxes_desc_, boxes_layout,
                                            boxes_dtype, boxes_dim_nb,
                                            boxes_dims.data()));
      } else {
        MLUOP_CHECK(mluOpSetTensorDescriptorEx(boxes_desc_, boxes_layout,
                                            boxes_dtype, boxes_dim_nb,
                                            boxes_dims.data(),
                                            boxes_stride.data()));
      }
      int boxes_elenum = mluOpGetTensorElementNum(boxes_desc_);
      if (boxes_elenum > 0) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&boxes_,
                               boxes_elenum * mluOpDataTypeBytes(boxes_dtype)));
      }

      MLUOpTensorParam output_desc = std::get<2>(GetParam());
      mluOpTensorLayout_t output_layout = output_desc.get_layout();
      mluOpDataType_t output_dtype = output_desc.get_dtype();
      int output_dim_nb = output_desc.get_dim_nb();
      std::vector<int> output_dims = output_desc.get_dim_size();
      std::vector<int> output_stride = output_desc.get_dim_stride();
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));
      if (output_stride.empty()) {
        MLUOP_CHECK(mluOpSetTensorDescriptor(output_desc_, output_layout,
                                            output_dtype, output_dim_nb,
                                            output_dims.data()));
      } else {
        MLUOP_CHECK(mluOpSetTensorDescriptorEx(output_desc_, output_layout,
                                            output_dtype, output_dim_nb,
                                            output_dims.data(),
                                            output_stride.data()));
      }
      int output_elenum = mluOpGetTensorElementNum(output_desc_);
      if (output_elenum > 0) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&output_, output_elenum *
                                             mluOpDataTypeBytes(output_dtype)));
      }

      MLUOpTensorParam argmax_idx_desc = std::get<3>(GetParam());
      mluOpTensorLayout_t argmax_idx_layout = argmax_idx_desc.get_layout();
      mluOpDataType_t argmax_idx_dtype = argmax_idx_desc.get_dtype();
      int argmax_idx_dim_nb = argmax_idx_desc.get_dim_nb();
      std::vector<int> argmax_idx_dims = argmax_idx_desc.get_dim_size();
      std::vector<int> argmax_idx_stride = argmax_idx_desc.get_dim_stride();
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&argmax_idx_desc_));
      if (argmax_idx_stride.empty()) {
        MLUOP_CHECK(mluOpSetTensorDescriptor(argmax_idx_desc_, argmax_idx_layout,
                                            argmax_idx_dtype, argmax_idx_dim_nb,
                                            argmax_idx_dims.data()));
      } else {
        MLUOP_CHECK(mluOpSetTensorDescriptorEx(argmax_idx_desc_, argmax_idx_layout,
                                            argmax_idx_dtype, argmax_idx_dim_nb,
                                            argmax_idx_dims.data(),
                                            argmax_idx_stride.data()));
      }
      int argmax_idx_elenum = mluOpGetTensorElementNum(argmax_idx_desc_);
      if (argmax_idx_elenum > 0) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&argmax_idx_, argmax_idx_elenum *
                                         mluOpDataTypeBytes(argmax_idx_dtype)));
      }

      target_device_ = std::get<4>(GetParam());
      expected_status_ = std::get<5>(GetParam());
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in border_align_forward";
    }
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }
    mluOpStatus_t status = mluOpBorderAlignForward(
        handle_, input_desc_, input_, boxes_desc_, boxes_, pool_size_,
        output_desc_, output_, argmax_idx_desc_, argmax_idx_);
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
      }
      if (input_desc_) {
        VLOG(4) << "Destroy input_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_desc_));
      }
      if (input_) {
        VLOG(4) << "Destroy input_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_));
      }
      if (boxes_desc_) {
        VLOG(4) << "Destroy boxes_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(boxes_desc_));
      }
      if (boxes_) {
        VLOG(4) << "Destroy boxes_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(boxes_));
      }
      if (output_desc_) {
        VLOG(4) << "Destroy output_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
      }
      if (output_) {
        VLOG(4) << "Destroy output_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_));
      }
      if (argmax_idx_desc_) {
        VLOG(4) << "Destroy argmax_idx_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(argmax_idx_desc_));
      }
      if (argmax_idx_) {
        VLOG(4) << "Destroy argmax_idx_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(argmax_idx_));
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in border_align_forward";
    }
  }

  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t input_desc_ = nullptr;
  void *input_ = nullptr;
  mluOpTensorDescriptor_t boxes_desc_ = nullptr;
  void *boxes_ = nullptr;
  mluOpTensorDescriptor_t output_desc_ = nullptr;
  void *output_ = nullptr;
  int32_t pool_size_ = 9;
  mluOpTensorDescriptor_t argmax_idx_desc_ = nullptr;
  void *argmax_idx_ = nullptr;
  mluOpDevType_t target_device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(border_align_forward_general, negative) { EXPECT_TRUE(compute()); }

INSTANTIATE_TEST_CASE_P(
    error_output_shape, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 20})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 100, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 7})},
                        MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({3, 100, 4, 5})},
                        MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 30, 4, 5})},
                        MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 5, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    error_argmax_idx_shape, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 20})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 100, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({1, 100, 4, 5})},
                        MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 200, 4, 5})},
                        MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 100, 5, 5})},
                        MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 100, 4, 7})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    error_input_shape, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 15})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 100, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 100, 4, 4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    error_boxex_shape, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 20})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 100, 5})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({1, 100, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    error_input_layout, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 20})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 100, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    error_argmax_idx_layout, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 20})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 100, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    error_input_dtype, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 20})},
                        MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 10, 10, 20})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 100, 4})},
                        MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 100, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 5})},
                        MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    error_argmax_idx_dtype, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 20})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 100, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    error_dtype_combine, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 20})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_HALF,
                                         3, std::vector<int>({2, 100, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    error_input_dim, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({20, 10, 20})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 100, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    error_boxex_dim, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 20})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({1, 2, 10, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    error_argmax_idx_dim, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 20})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 100, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         3, std::vector<int>({2, 100, 4})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    error_output_dim, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 20})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 100, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 100, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    input_unsupport_stride, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 20}),
                                         std::vector<int>({1, 2, 20, 300})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 100, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    boxex_unsupport_stride, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 20})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 100, 4}),
                                         std::vector<int>({1, 2, 300})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    output_unsupport_stride, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 20})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 100, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 5}),
                                         std::vector<int>({1, 2, 200, 1000})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    argmax_idx_unsupport_stride, border_align_forward_general,
    testing::Combine(
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 10, 10, 20})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                         3, std::vector<int>({2, 100, 4})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
                                         4, std::vector<int>({2, 100, 4, 5})}),
        testing::Values(MLUOpTensorParam{MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                         4, std::vector<int>({2, 100, 4, 5}),
                                         std::vector<int>({1, 2, 200, 1000})}),
        testing::Values(MLUOP_UNKNOWN_DEVICE),
        testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));
}  // namespace mluopapitest
