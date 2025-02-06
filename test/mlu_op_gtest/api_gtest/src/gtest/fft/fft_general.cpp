/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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

#include "gtest/gtest.h"
#include "mlu_op.h"
#include "core/context.h"
#include "core/logging.h"
#include "api_test_tools.h"

namespace mluopapitest {
typedef std::tuple<MLUOpTensorParamInt64, MLUOpTensorParamInt64, int64_t,
                   int64_t, mluOpDevType_t, mluOpStatus_t>
    FFTParams;

class fft_general : public testing::TestWithParam<FFTParams> {
 public:
  void SetUp() {
    target_device_ = std::get<4>(GetParam());
    expected_status_ = std::get<5>(GetParam());
    MLUOP_CHECK(mluOpCreate(&handle_));
    MLUOP_CHECK(mluOpCreateFFTPlan(&fft_plan_));

    MLUOpTensorParamInt64 input_params = std::get<0>(GetParam());
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
    MLUOP_CHECK(mluOpSetTensorDescriptorEx_v2(
        input_desc_, input_params.get_layout(), input_params.get_dtype(),
        input_params.get_dim_nb(), input_params.get_dim_size().data(),
        input_params.get_dim_stride().data()));

    MLUOP_CHECK(mluOpSetTensorDescriptorOnchipDataType(
        input_desc_, input_params.get_onchip_dtype()));

    MLUOpTensorParamInt64 output_params = std::get<1>(GetParam());
    MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));

    MLUOP_CHECK(mluOpSetTensorDescriptorEx_v2(
        output_desc_, output_params.get_layout(), output_params.get_dtype(),
        output_params.get_dim_nb(), output_params.get_dim_size().data(),
        output_params.get_dim_stride().data()));
    n_[0] = std::get<3>(GetParam());
    rank_ = std::get<2>(GetParam());
  }

  bool compute() {
    if (!(target_device_ == MLUOP_UNKNOWN_DEVICE ||
          target_device_ == handle_->arch)) {
      destroy();
      return true;
    }

    mluOpStatus_t status;
    status = mluOpMakeFFTPlanMany(handle_, fft_plan_, input_desc_, output_desc_,
                                  rank_, n_, &reserveSpaceSizeInBytes_,
                                  &workSpaceSizeInBytes_);
    destroy();

    return expected_status_ == status;
  }

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
      if (output_desc_) {
        VLOG(4) << "Destroy output_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
        output_desc_ = nullptr;
      }
      if (fft_plan_) {
        VLOG(4) << "Destroy fft_plan_";
        MLUOP_CHECK(mluOpDestroyFFTPlan(fft_plan_));
        fft_plan_ = nullptr;
      }
      if (workspace_size_) {
        VLOG(4) << "Destroy workspace_size_";
        GTEST_CHECK(cnrtSuccess == cnrtFree(workspace_size_));
        workspace_size_ = nullptr;
      }
      if (reservespace_size_) {
        VLOG(4) << "Destroy reservespace_size_";
        GTEST_CHECK(cnrtSuccess == cnrtFree(reservespace_size_));
        reservespace_size_ = nullptr;
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in fft_general";
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpFFTPlan_t fft_plan_ = nullptr;
  mluOpTensorDescriptor_t input_desc_ = nullptr;
  mluOpTensorDescriptor_t output_desc_ = nullptr;
  int rank_ = 1;
  int n_[1] = {1};
  size_t *reservespace_size_ = nullptr;
  size_t *workspace_size_ = nullptr;
  size_t reserveSpaceSizeInBytes_ = 64;
  size_t workSpaceSizeInBytes_ = 64;
  mluOpDevType_t target_device_ = MLUOP_UNKNOWN_DEVICE;
  mluOpStatus_t expected_status_ = MLUOP_STATUS_BAD_PARAM;
};

TEST_P(fft_general, negative) { EXPECT_TRUE(compute()); }

INSTANTIATE_TEST_CASE_P(
    zero_element, fft_general,
    testing::Combine(testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({0, 1}),
                         std::vector<int64_t>({1, 1}), MLUOP_DTYPE_FLOAT}),
                     testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({1, 1}),
                         std::vector<int64_t>({1, 1})}),
                     testing::Values(1), testing::Values(1),
                     testing::Values(MLUOP_UNKNOWN_DEVICE),
                     testing::Values(MLUOP_STATUS_SUCCESS)));

INSTANTIATE_TEST_CASE_P(
    negative_2_n,  // half,complex_half，fft length can be broken down into 2^m
    fft_general,
    testing::Combine(testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_HALF, 2,
                         std::vector<int64_t>({1, 7}),
                         std::vector<int64_t>({1, 1}), MLUOP_DTYPE_HALF}),
                     testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_HALF, 2,
                         std::vector<int64_t>({1, 7}),
                         std::vector<int64_t>({1, 1})}),
                     testing::Values(1), testing::Values(7),
                     testing::Values(MLUOP_UNKNOWN_DEVICE),
                     testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    negative_2_m_l_370,  // float/complex_float，n>4096, fft length can be
                         // broken down into 2^m*l
    fft_general,
    testing::Combine(testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 2,
                         std::vector<int64_t>({1, 4097}),
                         std::vector<int64_t>({1, 1}), MLUOP_DTYPE_FLOAT}),
                     testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 2,
                         std::vector<int64_t>({1, 4097}),
                         std::vector<int64_t>({1, 1})}),
                     testing::Values(1), testing::Values(4097),
                     testing::Values(MLUOP_MLU370),
                     testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    negative_2_m_l,  // float/complex_float，n>4096, fft length can be broken
                     // down into 2^m*l
    fft_general,
    testing::Combine(testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 2,
                         std::vector<int64_t>({1, 4099}),
                         std::vector<int64_t>({1, 1}), MLUOP_DTYPE_FLOAT}),
                     testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 2,
                         std::vector<int64_t>({1, 4099}),
                         std::vector<int64_t>({1, 1})}),
                     testing::Values(1), testing::Values(4099),
                     testing::Values(MLUOP_UNKNOWN_DEVICE),
                     testing::Values(MLUOP_STATUS_NOT_SUPPORTED)));

INSTANTIATE_TEST_CASE_P(
    negative_rank_1, fft_general,
    testing::Combine(testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({1}), std::vector<int64_t>({1}),
                         MLUOP_DTYPE_FLOAT}),
                     testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({1}), std::vector<int64_t>({1})}),
                     testing::Values(4), testing::Values(1),
                     testing::Values(MLUOP_UNKNOWN_DEVICE),
                     testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_N_le_0, fft_general,
    testing::Combine(testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({1}), std::vector<int64_t>({1}),
                         MLUOP_DTYPE_FLOAT}),
                     testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({1}), std::vector<int64_t>({1})}),
                     testing::Values(1), testing::Values(0, -1),
                     testing::Values(MLUOP_UNKNOWN_DEVICE),
                     testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_batch, fft_general,
    testing::Combine(testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 2,
                         std::vector<int64_t>({1, 1}),
                         std::vector<int64_t>({1, 1}), MLUOP_DTYPE_FLOAT}),
                     testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 2,
                         std::vector<int64_t>({2, 1}),
                         std::vector<int64_t>({1, 1})}),
                     testing::Values(1), testing::Values(1),
                     testing::Values(MLUOP_UNKNOWN_DEVICE),
                     testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_input_stride, fft_general,
    testing::Combine(testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({1}), std::vector<int64_t>({-1}),
                         MLUOP_DTYPE_FLOAT}),
                     testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({1}), std::vector<int64_t>({1})}),
                     testing::Values(1), testing::Values(1),
                     testing::Values(MLUOP_UNKNOWN_DEVICE),
                     testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_output_stride, fft_general,
    testing::Combine(testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({1}), std::vector<int64_t>({1}),
                         MLUOP_DTYPE_FLOAT}),
                     testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({1}),
                         std::vector<int64_t>({-1})}),
                     testing::Values(1), testing::Values(1),
                     testing::Values(MLUOP_UNKNOWN_DEVICE),
                     testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_unsupported_dtype_combination, fft_general,
    testing::Combine(testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_HALF, 1,
                         std::vector<int64_t>({1}), std::vector<int64_t>({1}),
                         MLUOP_DTYPE_FLOAT}),
                     testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({1}), std::vector<int64_t>({1})}),
                     testing::Values(1), testing::Values(1),
                     testing::Values(MLUOP_UNKNOWN_DEVICE),
                     testing::Values(MLUOP_STATUS_BAD_PARAM)));

INSTANTIATE_TEST_CASE_P(
    negative_onchip_dtype, fft_general,
    testing::Combine(testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({1}), std::vector<int64_t>({1}),
                         MLUOP_DTYPE_HALF}),
                     testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({1}), std::vector<int64_t>({1})}),
                     testing::Values(1), testing::Values(1),
                     testing::Values(MLUOP_UNKNOWN_DEVICE),
                     testing::Values(MLUOP_STATUS_BAD_PARAM)));

// r2c,output!=n/2+1
INSTANTIATE_TEST_CASE_P(
    negative_r2c_length, fft_general,
    testing::Combine(testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT, 1,
                         std::vector<int64_t>({4}), std::vector<int64_t>({1}),
                         MLUOP_DTYPE_FLOAT}),
                     testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({1}), std::vector<int64_t>({1})}),
                     testing::Values(1), testing::Values(4),
                     testing::Values(MLUOP_UNKNOWN_DEVICE),
                     testing::Values(MLUOP_STATUS_BAD_PARAM)));

// c2c,output != n
INSTANTIATE_TEST_CASE_P(
    negative_c2c_length, fft_general,
    testing::Combine(testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({1}), std::vector<int64_t>({1}),
                         MLUOP_DTYPE_FLOAT}),
                     testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({0}), std::vector<int64_t>({1})}),
                     testing::Values(1), testing::Values(1),
                     testing::Values(MLUOP_UNKNOWN_DEVICE),
                     testing::Values(MLUOP_STATUS_BAD_PARAM)));

// c2r,output!=n
INSTANTIATE_TEST_CASE_P(
    negative_c2r_length, fft_general,
    testing::Combine(testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_COMPLEX_FLOAT, 1,
                         std::vector<int64_t>({4}), std::vector<int64_t>({1}),
                         MLUOP_DTYPE_FLOAT}),
                     testing::Values(MLUOpTensorParamInt64{
                         MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT, 1,
                         std::vector<int64_t>({3}), std::vector<int64_t>({1})}),
                     testing::Values(1), testing::Values(4),
                     testing::Values(MLUOP_UNKNOWN_DEVICE),
                     testing::Values(MLUOP_STATUS_BAD_PARAM)));
}  // namespace mluopapitest
