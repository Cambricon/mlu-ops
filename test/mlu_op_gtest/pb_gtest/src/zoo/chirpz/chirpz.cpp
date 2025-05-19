/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell coM_PIes of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all coM_PIes or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "chirpz.h"
using namespace std;
namespace mluoptest {

void ChirpzExecutor::initData() {
  length_ = parser_->getProtoNode()->chirpz_param().length();
  n_ = parser_->getProtoNode()->chirpz_param().n();
  pad_n_ = parser_->getProtoNode()->chirpz_param().pad_n();
  chirpz_ = parser_->getProtoNode()->chirpz_param().chirpz();
  VLOG(4) << length_ << " " << n_ << " " << chirpz_;
}

void ChirpzExecutor::paramCheck() {
  GTEST_CHECK(parser_->outputs().size() == 1,
              "chirpz tensor output number is wrong.");
}

void ChirpzExecutor::compute() {
  VLOG(4) << "chirpzExecutor compute ";
  initData();

  auto tensor_y = tensor_desc_[1].tensor;
  auto dev_y = data_vector_[1].device_ptr;

  VLOG(4) << "call mluOpchirpz()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpChirpz(handle_, length_, n_, pad_n_, chirpz_, tensor_y, dev_y));
  interface_timer_.stop();
}

void ChirpzExecutor::cpuCompute() {
  VLOG(4) << "call cpuCompute()" << length_ << " M_PI " << M_PI << " n_ " << n_;
  for (int i = 0; i < pad_n_; i++) {
    cpu_fp32_output_[0][2 * i] = 0;
    cpu_fp32_output_[0][2 * i + 1] = 0;
    if(i < length_) {
      if(chirpz_) {
        // printf("%f ", M_PI * i * i / n_);
        // cpu_fp32_output_[0][2 * i] = M_PI * i * i / n_;
        // cpu_fp32_output_[0][2 * i + 1] = M_PI * i * i / n_;
        cpu_fp32_output_[0][2 * i] = cos(double(M_PI) * i * i / n_);
        cpu_fp32_output_[0][2 * i + 1] = -1 * sin(double(M_PI) * i * i / n_);
      } else {
          cpu_fp32_output_[0][2 * i] = cos(double(M_PI) * i * i / n_);
          cpu_fp32_output_[0][2 * i + 1] = 1 * sin(double(M_PI) * i * i / n_);
      }
    }
    if(!chirpz_ && i >= pad_n_ - n_ + 1) {
        // printf("%d ", pad_n_ - i);
        cpu_fp32_output_[0][2 * i] =
            cos(double(M_PI) * (pad_n_ - i) * (pad_n_ - i) / n_);
        cpu_fp32_output_[0][2 * i + 1] = sin(double(M_PI) * (pad_n_  - i) * (pad_n_ - i) / n_);
    }
  }

  // for (int i = 0; i < pad_n_; i++) {
  //   printf("%f %f ", cpu_fp32_output_[0][2*i], cpu_fp32_output_[0][2*i+1]);
  // }
}

int64_t ChirpzExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->output(0)->total_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
