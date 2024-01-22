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
#include "fft.h"

namespace mluoptest {

void FftExecutor::paramCheck() {
  GTEST_CHECK(parser_->getInputNum() == 1, "fft input number is wrong.");
  GTEST_CHECK(parser_->getOutputNum() == 1, "fft input number is wrong.");
}

void FftExecutor::workspaceMalloc() {
  auto input_tensor = tensor_desc_[0].tensor;
  auto output_tensor = tensor_desc_[1].tensor;

  auto fft_param = parser_->getProtoNode()->fft_param();
  int rank = fft_param.rank();
  std::vector<int> n;
  for (int i = 0; i < rank; i++) {
    n.push_back(fft_param.n(i));
  }

  MLUOP_CHECK(mluOpCreateFFTPlan(&fft_plan_));
  MLUOP_CHECK(mluOpMakeFFTPlanMany(handle_, fft_plan_, input_tensor,
                                   output_tensor, rank, n.data(),
                                   &reservespace_size_, &workspace_size_));

  VLOG(4) << "reserve space size: " << reservespace_size_;
  VLOG(4) << "workspace size: " << workspace_size_;

  if (reservespace_size_ > 0) {
    GTEST_CHECK(reservespace_addr_ = mlu_runtime_.allocate(reservespace_size_));
    workspace_.push_back(reservespace_addr_);
  }
  //  interface_timer_.start();
  /* reserve space is the compiling time process before FFT execution */
  MLUOP_CHECK(mluOpSetFFTReserveArea(handle_, fft_plan_, reservespace_addr_));
  //  interface_timer_.stop();
  if (workspace_size_ > 0) {
    GTEST_CHECK(workspace_addr_ = mlu_runtime_.allocate(workspace_size_));
    workspace_.push_back(workspace_addr_);
  }
}

void FftExecutor::compute() {
  VLOG(4) << "FftExecutor compute ";
  auto input_dev = data_vector_[0].device_ptr;
  auto output_dev = data_vector_[1].device_ptr;

  auto fft_param = parser_->getProtoNode()->fft_param();
  int direction = fft_param.direction();
  float scale_factor = fft_param.scale_factor();

  VLOG(4) << "call mluOpFFT";

  interface_timer_.start();
  MLUOP_CHECK(mluOpExecFFT(handle_, fft_plan_, input_dev, scale_factor,
                           workspace_addr_, output_dev, direction));
  interface_timer_.stop();
}

void FftExecutor::workspaceFree() {
  MLUOP_CHECK(mluOpDestroyFFTPlan(fft_plan_));
  for (auto &addr : workspace_) {
    mlu_runtime_.deallocate(addr);
  }
  workspace_.clear();
}

void FftExecutor::cpuCompute() {
  // TODO(sunhui): use fftw? librosa? OTFFT? other thrid-party library.
}

int64_t FftExecutor::getTheoryOps() {
  auto input_tensor = tensor_desc_[0].tensor;
  auto fft_param = parser_->getProtoNode()->fft_param();
  int rank = fft_param.rank();
  int bc = 1;
  if (input_tensor->dim != rank) {
    bc = input_tensor->dims[0];
  }
  int n = fft_param.n(0);

  int64_t ops_each_batch;
  // Convert LT and CT computing power. The computing power of a single LT is
  // 4096 * 2, the computing power of a single CT is 128.
  int cp_ratio = 4096 * 2 / 128;
  if (n <= 4096) {
    // fft_plan->fft_strategy = CNFFT_FUNC_MATMUL. Mainly use LT.
    ops_each_batch = n * n * 2 / cp_ratio;
  } else {
    ops_each_batch = n * int(std::log(n)) * 2;
    // fft_plan->fft_strategy = CNFFT_FUNC_COOLEY_TUKEY or CNFFT_FUNC_STOCKHAM.
    // Half use LT and half use CT.
    ops_each_batch = ops_each_batch * (0.5 / cp_ratio + 0.5);
  }
  int64_t theory_ops = bc * ops_each_batch;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

int64_t FftExecutor::getTheoryIoSize() {
  // dtype check
  auto input_tensor = tensor_desc_[0].tensor;
  auto output_tensor = tensor_desc_[1].tensor;
  mluOpDataType_t input_dtype = input_tensor->dtype;
  mluOpDataType_t output_dtype = output_tensor->dtype;

  auto fft_param = parser_->getProtoNode()->fft_param();
  int rank = fft_param.rank();
  int bc = 1;
  if (input_tensor->dim != rank) {
    bc = input_tensor->dims[0];
  }
  int n = fft_param.n(0);

  int64_t theory_ios = 0;
  if (n <= 4096) {
    if (input_dtype == output_dtype) {
      theory_ios += bc * n * 4;  // matmul io
    } else {                     // r2c or c2r
      theory_ios += bc * n * 2;  // matmul io
    }
    theory_ios += n * n * 2;  // W io
  } else {
    if (input_dtype == output_dtype) {
      theory_ios += bc * n * 4;  // matmul io
      theory_ios += bc * n * 4;  // stockham or cooley_tukey io
    } else {                     // r2c or c2r
      theory_ios += bc * n * 2;  // matmul
      theory_ios += bc * n * 2;  // stockham or cooley_tukey io
    }

    // W io
    int n_temp = n;
    while (n_temp >= 128 && n_temp % 2 == 0) {
      n_temp = n_temp / 2;
    }
    theory_ios += n_temp * 2;
  }
  VLOG(4) << "getTheoryIoSize: " << theory_ios << " ops";
  return theory_ios;
}

std::set<Evaluator::Formula> FftExecutor::getCriterionsUse() const {
  return {Evaluator::DIFF1, Evaluator::DIFF2, Evaluator::DIFF3,
          Evaluator::DIFF4};
}

}  // namespace mluoptest
