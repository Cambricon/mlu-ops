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
#include <fftw3.h>
namespace mluoptest {

void FftExecutor::paramCheck() {
  GTEST_CHECK(parser_->getInputNum() == 1, "fft input number is wrong.");
  GTEST_CHECK(parser_->getOutputNum() == 1, "fft input number is wrong.");
}

void FftExecutor::workspaceMalloc() {
  // return;
  auto input_tensor = tensor_desc_[0].tensor;
  auto output_tensor = tensor_desc_[1].tensor;

  auto fft_param = parser_->getProtoNode()->fft_param();
  int rank = fft_param.rank();
  std::vector<int> n;
  for (int i = 0; i < rank; i++) {
    n.push_back(fft_param.n(i));
  }
  int direction = fft_param.direction();

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
  // return;
  VLOG(4) << "FftExecutor compute ";
  auto input_dev = data_vector_[0].device_ptr;
  auto output_dev = data_vector_[1].device_ptr;
  // auto input_tensor = tensor_desc_[0].tensor;
  // auto output_tensor = tensor_desc_[1].tensor;
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
  // auto batch = parser_->getInputDataCount(0);
  auto count = parser_->getInputDataCount(0);
  // printf("\n\n\nbatch: %ld, count: %ld\n\n\n", batch, count);
  // cpu_fp32_output_[0][i] = (cpu_fp32_input_[0][i]);

  for (int i = 0; i < count; ++i) {
    cpu_fp32_output_[0][i] = (cpu_fp32_input_[0][i]);
  }
  auto fft_param = parser_->getProtoNode()->fft_param();
  int direction = (fft_param.direction() == 0) ? FFTW_FORWARD : FFTW_BACKWARD;

#define TEST_C2C1D_FP32 0
#define TEST_C2C1D_STRIDE_FP32 0
#define TEST_C2C2D_FP32 0
#define TEST_C2C2D_STRIDE_FP32 0
#define TEST_R2C2D_STRIDE_FP32 0
#define TEST_R2C1D_STRIDE_FP32 0
#define TEST_C2R1D_STRIDE_FP32 1
#define TEST_C2R2D_STRIDE_FP32 0

#if TEST_C2C1D_FP32
  int n[1];
  n[0] = parser_->getProtoNode()->fft_param().n()[0];

  int batch = count / n[0];

  auto size = count / batch;
  fftwf_plan fft;

  fftwf_complex *fftw_out = ((fftwf_complex *)cpu_fp32_output_[0]);
  fftwf_complex *fftw_in = ((fftwf_complex *)cpu_fp32_input_[0]);

  // int rank = 1;
  // int rank = fft_param.rank();
  int howmany = batch;
  // fft = fftwf_plan_many_dft(rank, &size, howmany, fftw_in, inembed, istride,
  // idist,
  //                           fftw_out, onembed, ostride, odist,
  //                           FFTW_FORWARD, FFTW_MEASURE);

  // fft = fftwf_plan_dft_1d(size, fftw_in, fftw_out, FFTW_FORWARD,
  //                         FFTW_ESTIMATE);  // Setup fftw plan for fft

  // fftwf_execute(fft);

  for (int batch_id = 0; batch_id < batch; batch_id++) {
    fft = fftwf_plan_dft_1d(size, fftw_in + batch_id * size,
                            fftw_out + batch_id * size, direction,
                            FFTW_ESTIMATE);  // Setup fftw plan for fft

    fftwf_execute(fft);
  }
  fftwf_destroy_plan(fft);
#endif

#if TEST_C2C2D_FP32
  fftwf_plan fft;

  fftwf_complex *fftw_out = ((fftwf_complex *)cpu_fp32_output_[0]);
  fftwf_complex *fftw_in = ((fftwf_complex *)cpu_fp32_input_[0]);

  int n[2];
  n[0] = parser_->getProtoNode()->fft_param().n()[0];
  n[1] = parser_->getProtoNode()->fft_param().n()[1];
  int howmany = count / (n[0] * n[1]);
  int *inembed = n;
  int *onembed = n;
  // int istride = howmany;
  // int ostride = howmany;
  // int idist = 1;
  // int odist = 1;
  int istride = 1;
  int ostride = 1;
  int idist = (n[0] * n[1]);
  int odist = (n[0] * n[1]);
  //   input[b * idist + (y * inembed[1] + x) * istride]
  //   output[b * odist + (y * onembed[1] + x) * ostride]

  // for(int i = 0; i <6; i ++) {
  //     for(int j = 0; j <2; j ++) {

  //   printf("(%f, %f)  ", ((float *)fftw_in)[(i*2+j)*2],((float
  //   *)fftw_in)[(i*2+j)*2+1]);
  // }
  // printf("\n");
  // }

  fft = fftwf_plan_many_dft(2, n, howmany, fftw_in, inembed, istride, idist,
                            fftw_out, onembed, ostride, odist, direction,
                            FFTW_ESTIMATE);  // Setup fftw plan for fft
  printf("fftw:\n");
  printf("howmany: %d\n", howmany);
  printf("n[0]: %d\n", n[0]);
  printf("n[1]: %d\n", n[1]);

  fftwf_execute(fft);

  fftwf_destroy_plan(fft);

#endif

#if TEST_C2C2D_STRIDE_FP32

  fftwf_plan fft;

  fftwf_complex *fftw_out = ((fftwf_complex *)cpu_fp32_output_[0]);
  fftwf_complex *fftw_in = ((fftwf_complex *)cpu_fp32_input_[0]);

  int n[2];
  n[0] = parser_->getProtoNode()->fft_param().n()[0];
  n[1] = parser_->getProtoNode()->fft_param().n()[1];
  int howmany = count / (n[0] * n[1]);
  int *inembed = n;
  int *onembed = n;
  // int istride = howmany;
  // int ostride = howmany;
  // int idist = 1;
  // int odist = 1;
  int istride = 1;
  int ostride = 1;
  int idist = (n[0] * n[1]);
  int odist = (n[0] * n[1]);
  //   input[b * idist + (y * inembed[1] + x) * istride]
  //   output[b * odist + (y * onembed[1] + x) * ostride]

  // for(int i = 0; i <6; i ++) {
  //     for(int j = 0; j <2; j ++) {

  //   printf("(%f, %f)  ", ((float *)fftw_in)[(i*2+j)*2],((float
  //   *)fftw_in)[(i*2+j)*2+1]);
  // }
  // printf("\n");
  // }

  fft = fftwf_plan_many_dft(2, n, howmany, fftw_in, inembed, istride, idist,
                            fftw_out, onembed, ostride, odist, direction,
                            FFTW_ESTIMATE);  // Setup fftw plan for fft
  printf("fftw:\n");
  printf("howmany: %d\n", howmany);
  printf("n[0]: %d\n", n[0]);
  printf("n[1]: %d\n", n[1]);

  fftwf_execute(fft);

  // for(int i = 0; i <6; i ++) {
  //     for(int j = 0; j <2; j ++) {

  //   printf("(%f, %f)  ", ((float *)fftw_out)[(i*2+j)*2],((float
  //   *)fftw_out)[(i*2+j)*2+1]);
  // }
  // printf("\n");
  // }

  fftwf_destroy_plan(fft);

#endif

#if TEST_R2C2D_STRIDE_FP32

  fftwf_plan fft;

  fftwf_complex *fftw_out = ((fftwf_complex *)cpu_fp32_output_[0]);
  float *fftw_in = ((float *)cpu_fp32_input_[0]);

  int n[2];
  n[0] = parser_->getProtoNode()->fft_param().n()[0];
  n[1] = parser_->getProtoNode()->fft_param().n()[1];
  int howmany = count / (n[0] * n[1]);
  int inembed[2] = {n[0], n[1]};
  int onembed[2] = {n[0], n[1] / 2 + 1};
  // onembed[1] = n[1]/2 +1;
  // int istride = howmany;
  // int ostride = howmany;
  // int idist = 1;
  // int odist = 1;
  int istride = 1;
  int ostride = 1;
  int idist = (n[0] * n[1]);
  int odist = (n[0] * (n[1] / 2 + 1));
  //   input[b * idist + (y * inembed[1] + x) * istride]
  //   output[b * odist + (y * onembed[1] + x) * ostride]

  // for(int i = 0; i <6; i ++) {
  //     for(int j = 0; j <2; j ++) {

  //   printf("(%f, %f)  ", ((float *)fftw_in)[(i*2+j)*2],((float
  //   *)fftw_in)[(i*2+j)*2+1]);
  // }
  // printf("\n");
  // }

  fft = fftwf_plan_many_dft_r2c(2, n, howmany, fftw_in, inembed, istride, idist,
                                fftw_out, onembed, ostride, odist,
                                FFTW_ESTIMATE);  // Setup fftw plan for fft
  // printf("fftw:\n");
  // printf("howmany: %d\n", howmany);
  // printf("n[0]: %d\n", n[0]);
  // printf("n[1]: %d\n", n[1]);

  fftwf_execute(fft);
  // fftwf_execute_dft_r2c(fft, fftw_in, fftw_out);
  // for(int i = 0; i <6; i ++) {
  //     for(int j = 0; j <2; j ++) {

  //   printf("(%f, %f)  ", ((float *)fftw_out)[(i*2+j)*2],((float
  //   *)fftw_out)[(i*2+j)*2+1]);
  // }
  // printf("\n");
  // }

  //   for (int i = 0; i < n[0]; i++) {
  //   int ld = (n[1]/2+1)*howmany;
  //   for (int j = 0; j < ld; j++) {
  //     printf("[%d][%d]: (%f, %f)  ",i, j, ((float*)fftw_out)[(i * (ld) + j)
  //     *2], ((float*)fftw_out)[((i * (ld) + j) *2 + 1)]);
  //   }
  //   printf("\n");
  // }
  //   for (int i = 0; i < howmany; i++) {
  //   // int ld = (n[1]/2+1)*howmany;
  //   int ld = (n[1]/2+1)*n[0];
  //   for (int j = 0; j < ld; j++) {
  //     printf("[%d][%d]: (%f, %f)  ",i, j, ((float*)fftw_out)[(i * (ld) + j)
  //     *2], ((float*)fftw_out)[((i * (ld) + j) *2 + 1)]);
  //   }
  //   printf("\n");
  // }

  fftwf_destroy_plan(fft);

#endif

#if TEST_R2C2D_STRIDE_FP32

  fftwf_plan fft;

  fftwf_complex *fftw_out = ((fftwf_complex *)cpu_fp32_output_[0]);
  float *fftw_in = ((float *)cpu_fp32_input_[0]);

  int n[1];
  n[0] = parser_->getProtoNode()->fft_param().n()[0];
  int howmany = count / (n[0]);
  int inembed[1] = {n[0]};
  int onembed[1] = {n[0] / 2 + 1};
  // onembed[1] = n[1]/2 +1;
  // int istride = howmany;
  // int ostride = howmany;
  // int idist = 1;
  // int odist = 1;
  int istride = 1;
  int ostride = 1;
  int idist = (n[0]);
  int odist = ((n[0] / 2 + 1));

  fft = fftwf_plan_many_dft_r2c(1, n, howmany, fftw_in, inembed, istride, idist,
                                fftw_out, onembed, ostride, odist,
                                FFTW_ESTIMATE);  // Setup fftw plan for fft

  fftwf_execute(fft);

  fftwf_destroy_plan(fft);

#endif

#if TEST_C2R2D_STRIDE_FP32

  fftwf_plan fft;

  float *fftw_out = ((float *)cpu_fp32_output_[0]);
  fftwf_complex *fftw_in = ((fftwf_complex *)cpu_fp32_input_[0]);

  int n[2];
  n[0] = parser_->getProtoNode()->fft_param().n()[0];
  n[1] = parser_->getProtoNode()->fft_param().n()[1];
  int howmany = count / (n[0] * (n[1] / 2 + 1));
  // int howmany = 768;
  int inembed[2] = {n[0], n[1] / 2 + 1};
  int onembed[2] = {n[0], n[1]};
  // onembed[1] = n[1]/2 +1;
  // int istride = howmany;
  // int ostride = howmany;
  // int idist = 1;
  // int odist = 1;
  int istride = 1;
  int ostride = 1;
  int idist = (n[0] * (n[1] / 2 + 1));
  int odist = (n[0] * (n[1]));
  //   input[b * idist + (y * inembed[1] + x) * istride]
  //   output[b * odist + (y * onembed[1] + x) * ostride]

  // for(int i = 0; i <6; i ++) {
  //     for(int j = 0; j <2; j ++) {

  //   printf("(%f, %f)  ", ((float *)fftw_in)[(i*2+j)*2],((float
  //   *)fftw_in)[(i*2+j)*2+1]);
  // }
  // printf("\n");
  // }

  fft = fftwf_plan_many_dft_c2r(2, n, howmany, fftw_in, inembed, istride, idist,
                                fftw_out, onembed, ostride, odist,
                                FFTW_ESTIMATE);  // Setup fftw plan for fft
  printf("fftw:\n");
  printf("howmany: %d\n", howmany);
  printf("n[0]: %d\n", n[0]);
  printf("n[1]: %d\n", n[1]);

  fftwf_execute(fft);
  // fftwf_execute_dft_r2c(fft, fftw_in, fftw_out);
  // for(int i = 0; i <6; i ++) {
  //     for(int j = 0; j <2; j ++) {

  //   printf("(%f, %f)  ", ((float *)fftw_out)[(i*2+j)*2],((float
  //   *)fftw_out)[(i*2+j)*2+1]);
  // }
  // printf("\n");
  // }

  //   for (int i = 0; i < n[0]; i++) {
  //   int ld = (n[1]/2+1)*howmany;
  //   for (int j = 0; j < ld; j++) {
  //     printf("[%d][%d]: (%f, %f)  ",i, j, ((float*)fftw_out)[(i * (ld) + j)
  //     *2], ((float*)fftw_out)[((i * (ld) + j) *2 + 1)]);
  //   }
  //   printf("\n");
  // }
  //   for (int i = 0; i < howmany; i++) {
  //   // int ld = (n[1]/2+1)*howmany;
  //   int ld = (n[1]/2+1)*n[0];
  //   for (int j = 0; j < ld; j++) {
  //     printf("[%d][%d]: (%f, %f)  ",i, j, ((float*)fftw_out)[(i * (ld) + j)
  //     *2], ((float*)fftw_out)[((i * (ld) + j) *2 + 1)]);
  //   }
  //   printf("\n");
  // }

  fftwf_destroy_plan(fft);

#endif

#if TEST_C2R1D_STRIDE_FP32

  fftwf_plan fft;

  float *fftw_out = ((float *)cpu_fp32_output_[0]);
  fftwf_complex *fftw_in = ((fftwf_complex *)cpu_fp32_input_[0]);

  int n[1];
  n[0] = parser_->getProtoNode()->fft_param().n()[0];
  int howmany = count / (n[0] / 2 + 1);
  int inembed[1] = {n[0] / 2 + 1};
  int onembed[1] = {n[0]};

  int istride = 1;
  int ostride = 1;
  int idist = (n[0] / 2 + 1);
  int odist = n[0];

  fft = fftwf_plan_many_dft_c2r(1, n, howmany, fftw_in, inembed, istride, idist,
                                fftw_out, onembed, ostride, odist,
                                FFTW_ESTIMATE);  // Setup fftw plan for fft
  printf("fftw:\n");
  printf("howmany: %d\n", howmany);
  printf("n[0]: %d\n", n[0]);

  fftwf_execute(fft);

  fftwf_destroy_plan(fft);

#endif

#if TEST_C2C1D_STRIDE_FP32

  fftwf_plan fft;

  fftwf_complex *fftw_out = ((fftwf_complex *)cpu_fp32_output_[0]);
  fftwf_complex *fftw_in = ((fftwf_complex *)cpu_fp32_input_[0]);

  int n[1];
  n[0] = parser_->getProtoNode()->fft_param().n()[0];

  int howmany = count / (n[0]);
  int *inembed = n;
  int *onembed = n;
  // int istride = howmany;
  // int ostride = howmany;
  // int idist = 1;
  // int odist = 1;
  int istride = 1;
  int ostride = 1;
  int idist = (n[0]);
  int odist = (n[0]);
  //   input[b * idist + (y * inembed[1] + x) * istride]
  //   output[b * odist + (y * onembed[1] + x) * ostride]

  // for(int i = 0; i <6; i ++) {
  //     for(int j = 0; j <2; j ++) {

  //   printf("(%f, %f)  ", ((float *)fftw_in)[(i*2+j)*2],((float
  //   *)fftw_in)[(i*2+j)*2+1]);
  // }
  // printf("\n");
  // }

  fft = fftwf_plan_many_dft(1, n, howmany, fftw_in, inembed, istride, idist,
                            fftw_out, onembed, ostride, odist, direction,
                            FFTW_ESTIMATE);  // Setup fftw plan for fft
  // printf("fftw:\n");
  // printf("howmany: %d\n", howmany);
  // printf("n[0]: %d\n", n[0]);
  // printf("n[1]: %d\n", n[1]);

  fftwf_execute(fft);

  fftwf_destroy_plan(fft);

#endif
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
