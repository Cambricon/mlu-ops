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

#include "kernels/fft/c2c_fft/c2c_fft.h"
#include <algorithm>
#include <string>

#define DIRECTION 2  // FORWARD and BACKWARD

static mluOpStatus_t selectFFT1dStrategy(mluOpHandle_t handle,
                                         mluOpFFTPlan_t fft_plan) {
  const std::string make_plan_api = "[selectFFT1dStrategy]";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  /* there are plenty of algorithms for FFT, depending on the fft length.
   * Iterative FFT:
   *   Stockham FFT, Cooley-Tukey FFT, peaseFFT, Kron-Lambiotte FFT
   * Recursive FFT:
   *   Recursive Cooley-Tukey FFT, Four-step FFT, Six-step FFT, Multicore FFT,
   * SIMD short vector FFT. General FFT: chirp-Z Bluestein FFT.
   */
  // select Stockham FFT, Cooley-Tukey FFT or MATMUL strategy logic
  fft_plan->fft_strategy = CNFFT_FUNC_MATMUL;
  status = selectFFTStrategy(handle, fft_plan, make_plan_api);
  return status;
}

/*
 * Make the policy of FFT1d.
 */
mluOpStatus_t makeFFT1dPolicy(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan) {
  std::string api = "[mluOpMakeFFTPlanMany]";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  CHECK_RETURN(api, selectFFT1dStrategy(handle, fft_plan));

  mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
  mluOpDataType_t in_r_dtype = (in_c_dtype == MLUOP_DTYPE_COMPLEX_HALF)
                                   ? MLUOP_DTYPE_HALF
                                   : MLUOP_DTYPE_FLOAT;
  mluOpDataType_t in_e_dtype = fft_plan->execution_dtype;
  size_t in_c_dtype_size = mluOpDataTypeBytes(in_c_dtype);
  size_t in_r_dtype_size = mluOpDataTypeBytes(in_r_dtype);
  int batch = fft_plan->batch;
  int n = fft_plan->n[0];
  int FFT_L_LIMIT_MATMUL = FFT_L_LIMIT_MATMUL_300;
  if (handle->arch > MLUOP_MLU370) {
    FFT_L_LIMIT_MATMUL = FFT_L_LIMIT_MATMUL_500;
  }

  switch (fft_plan->fft_strategy) {
    case CNFFT_FUNC_MATMUL: {
      if (n > FFT_L_LIMIT_MATMUL) {
        LOG(ERROR)
            << "[mluOpMakeFFTPlanMany]: FFT1d CNFFT_FUNC_MATMUL length > "
            << FFT_L_LIMIT_MATMUL << " is not supported currently.";
        return MLUOP_STATUS_NOT_SUPPORTED;
      }

      // Matmul Input  : 2 * [batch, n]
      // Matmul Matrix : 2 * 2 * [n, n] (forward and backward)
      // Matmul Result : 4 * [batch, n]
      int dft_mat_times = COMPLEX;
      int dim0 = n;
      int dim1 = n;
      int dft_mat_num = DIRECTION * dft_mat_times * dim0 * dim1;

      // reservespace size allocation
      fft_plan->reservespace_size = 0;
      fft_plan->reservespace_size +=
          dft_mat_num * mluOpDataTypeBytes(in_r_dtype);

      /* CNFFT_FUNC_MATMUL :
         -------------------------
         |        input          |
         -------------------------
                    |
                    | input contiguous
                   \|/
         -------------------------
         |    input_contiguous   |
         -------------------------
                    |
                    | input pad
                   \|/
         -------------------------
         |      input_pad        |
         -------------------------
                    |
                    | input trans: batch * n * 2 --> 2 * batch * n
                   \|/
         -------------------------
         |      input_re         |
         |      input_im         |
         -------------------------
                    |
                    | matmul
                    | optensor(re_mul_re - im_mul_im, re_mul_im + im_mul_re)
                   \|/
         -------------------------
         |     matmul_re_mul_re   | (matmul_re)
         |     matmul_re_mul_im   | (matmul_im)
         |     matmul_im_mul_re   |
         |     matmul_im_mul_im   |
         -------------------------
                    |
                    | output trans: 2 * batch * n --> batch * n * 2
                   \|/
         -------------------------
         |   output_contiguous   |
         -------------------------
                    |
                    | output contiguous
                   \|/
         -------------------------
         |        output         |
         -------------------------
      */
      // worksapce size allocation
      fft_plan->matmul_addrs.internal_workspace_size = 0;
      fft_plan->workspace_size = 0;

      // input contiguous
      size_t input_size = in_c_dtype_size * fft_plan->inum;
      fft_plan->workspace_size +=
          fft_plan->is_input_contiguous ? 0 : input_size;

      // input pad
      bool need_pad = (fft_plan->inembed[0] != n);
      int padded_input_num = batch * n;
      size_t padded_input_size = in_c_dtype_size * padded_input_num;
      fft_plan->workspace_size += need_pad ? padded_input_size : 0;

      // input trans and workspace
      size_t transed_input_size = padded_input_size;
      fft_plan->workspace_size += transed_input_size;
      // input trans workspace: batch * n * 2 --> 2 * batch * n
      const int in_trans_dim_num = 2;
      int64_t in_trans_input_dims[in_trans_dim_num] = {padded_input_num,
                                                       COMPLEX};
      int in_trans_permute[in_trans_dim_num] = {1, 0};
      size_t in_trans_workspace_size = 0;
      status = fftGetTransposeWorkspaceSize(
          handle, in_trans_workspace_size, in_trans_dim_num,
          in_trans_input_dims, in_trans_permute, in_r_dtype, api);
      fft_plan->matmul_addrs.internal_workspace_size =
          std::max(fft_plan->matmul_addrs.internal_workspace_size,
                   in_trans_workspace_size);

      // matmul output
      const int matmul_times =
          4;  // real mul real, real mul imag, imag mul real, imag mul imag
      int per_matmul_output_num = batch * n;
      size_t per_matmul_output_size = in_r_dtype_size * per_matmul_output_num;
      size_t matmul_output_size = matmul_times * per_matmul_output_size;
      fft_plan->workspace_size += matmul_output_size;
      // matmul workspace
      size_t matmul_workspace_size = 0;
      status = fftGetMatMulWorkspaceSize(handle, matmul_workspace_size, batch,
                                         dim1, dim0, false, true, in_e_dtype,
                                         in_e_dtype, in_r_dtype, api);
      fft_plan->matmul_addrs.internal_workspace_size =
          std::max(fft_plan->matmul_addrs.internal_workspace_size,
                   matmul_workspace_size);
      // optensor workspace
      size_t optensor_workspace_size = 0;
      status =
          fftGetOptensorWorkspaceSize(handle, optensor_workspace_size,
                                      per_matmul_output_num, in_r_dtype, api);
      fft_plan->matmul_addrs.internal_workspace_size =
          std::max(fft_plan->matmul_addrs.internal_workspace_size,
                   optensor_workspace_size);

      // output trans workspace: 2 * batch * n --> batch * n * 2
      const int out_trans_dim_num = 2;
      int64_t out_trans_input_dims[out_trans_dim_num] = {COMPLEX,
                                                         per_matmul_output_num};
      int out_trans_permute[out_trans_dim_num] = {1, 0};
      size_t out_trans_workspace_size = 0;
      status = fftGetTransposeWorkspaceSize(
          handle, out_trans_workspace_size, out_trans_dim_num,
          out_trans_input_dims, out_trans_permute, in_r_dtype, api);
      fft_plan->matmul_addrs.internal_workspace_size =
          std::max(fft_plan->matmul_addrs.internal_workspace_size,
                   out_trans_workspace_size);

      // output contiguous
      size_t output_size =
          mluOpDataTypeBytes(fft_plan->output_dtype) * fft_plan->onum;
      fft_plan->workspace_size +=
          fft_plan->is_output_contiguous ? 0 : output_size;

      // internal_workspace
      fft_plan->workspace_size +=
          fft_plan->matmul_addrs.internal_workspace_size;
      VLOG(5) << "internal workspace size: "
              << fft_plan->matmul_addrs.internal_workspace_size;
      VLOG(5) << "total workspace size: " << fft_plan->workspace_size;
    }; break;
    case CNFFT_FUNC_COOLEY_TUKEY:
    case CNFFT_FUNC_STOCKHAM: {
      int L = fft_plan->L;
      int m = (1 << fft_plan->m);
      if (L > FFT_L_LIMIT) {
        LOG(ERROR) << "[mluOpMakeFFTPlanMany]: FFT1d CNFFT_FUNC_COOLEY_TUKEY "
                   << "n = L * 2^m and L > 4096 is not supported currently.";
        return MLUOP_STATUS_NOT_SUPPORTED;
      }

      // Matmul Input  : 2 * [batch, 2^m, L]
      // Matmul Matrix : 2 * 2 * [L, L] (forward and backward)
      // Matmul Result : 4 * [batch, 2^m, L]
      int dft_mat_times = COMPLEX;
      int dim0 = L;
      int dim1 = L;
      int dft_mat_num = DIRECTION * dft_mat_times * dim0 * dim1;

      // reservespace size allocation
      fft_plan->reservespace_size = 0;
      fft_plan->reservespace_size += dft_mat_num * in_r_dtype_size;

      /* CNFFT_FUNC_COOLEY_TUKEY :
         -------------------------
         |        input          |
         -------------------------
                    |
                    | input contiguous
                   \|/
         -------------------------
         |    input_contiguous   |
         -------------------------
                    |
                    | input pad
                   \|/
         -------------------------
         |      input_pad        |
         -------------------------
                    |
                    | input trans: batch * n * 2 --> 2 * batch * n
                   \|/
         -------------------------
         |      input_transed    |
         -------------------------
                    |
                    | input trans: 2 * batch * L * 2^m --> 2 * batch * 2^m * L
                   \|/
         -------------------------
         |      input_re         |
         |      input_im         |
         -------------------------
                    |
                    | matmul
                    | optensor(re_mul_re - im_mul_im, re_mul_im + im_mul_re)
                   \|/
         -------------------------
         |     matmul_re_mul_re   | (matmul_re)
         |     matmul_re_mul_im   | (matmul_im)
         |     matmul_im_mul_re   |
         |     matmul_im_mul_im   |
         -------------------------
                    |
                    | output merge
                   \|/
         -------------------------
         |   output_contiguous   |
         -------------------------
                    |
                    | output contiguous
                   \|/
         -------------------------
         |        output         |
         -------------------------
      */
      // worksapce size allocation
      fft_plan->matmul_addrs.internal_workspace_size = 0;
      fft_plan->workspace_size = 0;

      // input contiguous
      size_t input_size = in_c_dtype_size * fft_plan->inum;
      fft_plan->workspace_size +=
          fft_plan->is_input_contiguous ? 0 : input_size;

      // input pad
      bool need_pad = (fft_plan->inembed[0] != n);
      int padded_input_num = fft_plan->batch * n;
      size_t padded_input_size = in_c_dtype_size * padded_input_num;
      fft_plan->workspace_size += need_pad ? padded_input_size : 0;

      // input trans and workspace
      const int trans_times = 2;
      size_t transed_input_size = trans_times * padded_input_size;
      fft_plan->workspace_size += transed_input_size;
      // input trans workspace
      // 1st transpose: batch * n * 2 --> 2 * batch * n
      const int in_trans_1st_dim_num = 2;
      int64_t in_trans_1st_input_dims[in_trans_1st_dim_num] = {padded_input_num,
                                                               COMPLEX};
      int in_trans_1st_permute[in_trans_1st_dim_num] = {1, 0};
      size_t in_trans_1st_workspace_size = 0;
      status = fftGetTransposeWorkspaceSize(
          handle, in_trans_1st_workspace_size, in_trans_1st_dim_num,
          in_trans_1st_input_dims, in_trans_1st_permute, in_r_dtype, api);
      fft_plan->matmul_addrs.internal_workspace_size =
          std::max(fft_plan->matmul_addrs.internal_workspace_size,
                   in_trans_1st_workspace_size);
      // 2nd transpose: 2 * batch * L * 2^m --> 2 * batch * 2^m * L
      const int in_trans_2nd_dim_num = 3;
      int64_t in_trans_2nd_input_dims[in_trans_2nd_dim_num] = {COMPLEX * batch,
                                                               L, m};
      int in_trans_2nd_permute[in_trans_2nd_dim_num] = {0, 2, 1};
      size_t in_trans_2nd_workspace_size = 0;
      status = fftGetTransposeWorkspaceSize(
          handle, in_trans_2nd_workspace_size, in_trans_2nd_dim_num,
          in_trans_2nd_input_dims, in_trans_2nd_permute, in_r_dtype, api);
      fft_plan->matmul_addrs.internal_workspace_size =
          std::max(fft_plan->matmul_addrs.internal_workspace_size,
                   in_trans_2nd_workspace_size);

      // matmul output
      const int matmul_times =
          4;  // real mul real, real mul imag, imag mul real, imag mul imag
      int per_matmul_output_num = batch * n;
      size_t per_matmul_output_size = in_r_dtype_size * per_matmul_output_num;
      fft_plan->workspace_size += matmul_times * per_matmul_output_size;
      // matmul workspace
      size_t matmul_workspace_size = 0;
      if (fft_plan->fft_strategy == CNFFT_FUNC_COOLEY_TUKEY) {
        status = fftGetMatMulWorkspaceSize(
            handle, matmul_workspace_size, batch * m, L, L, false, true,
            in_e_dtype, in_e_dtype, in_r_dtype, api);
        fft_plan->matmul_addrs.internal_workspace_size =
            std::max(fft_plan->matmul_addrs.internal_workspace_size,
                     matmul_workspace_size);
      } else {
        status = fftGetBatchMatMulBcastWorkspaceSize(
            handle, 2 * L, L, m * 2, batch,
            fft_plan->matmul_addrs.dft_im_matrix_addr,
            fft_plan->matmul_addrs.input_pad_addr,
            fft_plan->matmul_addrs.matmul_re_mul_re_addr, false, false, 1.0,
            0.0, in_e_dtype, in_e_dtype, in_r_dtype,
            fft_plan->matmul_addrs.internal_workspace_addr,
            fft_plan->matmul_addrs.internal_workspace_size, api);
      }

      // optensor workspace
      size_t optensor_workspace_size = 0;
      status =
          fftGetOptensorWorkspaceSize(handle, optensor_workspace_size,
                                      per_matmul_output_num, in_r_dtype, api);
      fft_plan->matmul_addrs.internal_workspace_size =
          std::max(fft_plan->matmul_addrs.internal_workspace_size,
                   optensor_workspace_size);

      // output merge workspace
      size_t merge_workspace_size =
          COMPLEX * in_r_dtype_size * per_matmul_output_num;
      fft_plan->matmul_addrs.internal_workspace_size = std::max(
          fft_plan->matmul_addrs.internal_workspace_size, merge_workspace_size);

      // output contiguous
      size_t output_size =
          mluOpDataTypeBytes(fft_plan->output_dtype) * fft_plan->onum;
      fft_plan->workspace_size +=
          fft_plan->is_output_contiguous ? 0 : output_size;

      // internal_workspace
      fft_plan->workspace_size +=
          fft_plan->matmul_addrs.internal_workspace_size;
      VLOG(5) << "internal workspace size: "
              << fft_plan->matmul_addrs.internal_workspace_size;
      VLOG(5) << "total workspace size: " << fft_plan->workspace_size;
    }; break;
    default: {
      status = MLUOP_STATUS_NOT_SUPPORTED;
      return status;
    }
  }
  return status;
}

static void configureFFT1dMatmulReserveAddrs(mluOpHandle_t handle,
                                             mluOpFFTPlan_t fft_plan) {
  size_t dft_mat_size = 0;
  // forward real, forward imag, backward real, backward imag
  const int dft_mat_times = DIRECTION * COMPLEX;
  mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
  mluOpDataType_t in_r_dtype = (in_c_dtype == MLUOP_DTYPE_COMPLEX_HALF)
                                   ? MLUOP_DTYPE_HALF
                                   : MLUOP_DTYPE_FLOAT;
  size_t in_r_dtype_size = mluOpDataTypeBytes(in_r_dtype);
  int n = fft_plan->n[0];

  switch (fft_plan->fft_strategy) {
    case CNFFT_FUNC_MATMUL: {
      // Matmul Matrix : 2 * 2 * [n, n] (forward and backward)
      size_t per_dft_mat_size = n * n * in_r_dtype_size;
      dft_mat_size = dft_mat_times * per_dft_mat_size;
      fft_plan->matmul_addrs.dft_matrix_addr = fft_plan->reservespace_addr;
      fft_plan->matmul_addrs.dft_re_matrix_addr =
          fft_plan->matmul_addrs.dft_matrix_addr;
      fft_plan->matmul_addrs.dft_im_matrix_addr =
          (uint8_t *)fft_plan->matmul_addrs.dft_matrix_addr + per_dft_mat_size;
      fft_plan->matmul_addrs.ifft_dft_matrix_addr =
          (uint8_t *)fft_plan->matmul_addrs.dft_im_matrix_addr +
          per_dft_mat_size;
      fft_plan->matmul_addrs.ifft_dft_re_matrix_addr =
          fft_plan->matmul_addrs.ifft_dft_matrix_addr;
      fft_plan->matmul_addrs.ifft_dft_im_matrix_addr =
          (int8_t *)fft_plan->matmul_addrs.ifft_dft_matrix_addr +
          per_dft_mat_size;
    }; break;
    case CNFFT_FUNC_COOLEY_TUKEY:
    case CNFFT_FUNC_STOCKHAM: {
      // Matmul Matrix : 2 * 2 * [L, L] (forward and backward)
      int L = fft_plan->L;
      size_t per_dft_mat_size = L * L * in_r_dtype_size;
      dft_mat_size = dft_mat_times * per_dft_mat_size;
      fft_plan->matmul_addrs.dft_matrix_addr = fft_plan->reservespace_addr;
      fft_plan->matmul_addrs.dft_re_matrix_addr =
          fft_plan->matmul_addrs.dft_matrix_addr;
      fft_plan->matmul_addrs.dft_im_matrix_addr =
          (uint8_t *)fft_plan->matmul_addrs.dft_matrix_addr + per_dft_mat_size;
      fft_plan->matmul_addrs.ifft_dft_matrix_addr =
          (uint8_t *)fft_plan->matmul_addrs.dft_im_matrix_addr +
          per_dft_mat_size;
      fft_plan->matmul_addrs.ifft_dft_re_matrix_addr =
          fft_plan->matmul_addrs.ifft_dft_matrix_addr;
      fft_plan->matmul_addrs.ifft_dft_im_matrix_addr =
          (uint8_t *)fft_plan->matmul_addrs.ifft_dft_matrix_addr +
          per_dft_mat_size;
    }; break;
    default: {
      break;
    }
  }
}

mluOpStatus_t setFFT1dReserveArea(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                                  const std::string api) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  if (fft_plan->prime) {
    configureFFT1dMatmulReserveAddrs(handle, fft_plan);

    mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
    mluOpDataType_t in_r_dtype = (in_c_dtype == MLUOP_DTYPE_COMPLEX_HALF)
                                     ? MLUOP_DTYPE_HALF
                                     : MLUOP_DTYPE_FLOAT;
    mluOpDataType_t in_e_dtype = fft_plan->execution_dtype;
    const int n = fft_plan->n[0];
    const int dft_mat_times =
        DIRECTION *
        COMPLEX;  // forward real, forward imag, backward real, backward imag

    const unsigned int cluster_number =
        mluop::runtime::getClusterLimitCapability(handle);
    const unsigned int core_dim = handle->core_num_per_cluster;
    cnrtDim3_t k_dim = {core_dim, cluster_number, 1};
    cnrtFunctionType_t k_type = cnrtFuncTypeBlock;

    switch (fft_plan->fft_strategy) {
      case CNFFT_FUNC_MATMUL: {
        // Matmul Matrix : 2 * 2 * [n, n] (forward and backward)
        int dft_mat_num = dft_mat_times * n * n;
        kernelC2CFFTDFTMatrix(k_dim, k_type, handle->queue, fft_plan,
                              in_r_dtype, n);
      }; break;
      case CNFFT_FUNC_COOLEY_TUKEY:
      case CNFFT_FUNC_STOCKHAM: {
        // Matmul Matrix : 2 * 2 * [L, L] (forward and backward)
        int L = fft_plan->L;
        int dft_mat_num = dft_mat_times * L * L;
        kernelC2CFFTDFTMatrix(k_dim, k_type, handle->queue, fft_plan,
                              in_r_dtype, L);
      }; break;
      default: {
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }

  } else {
    mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
    size_t in_c_dtype_size = mluOpDataTypeBytes(in_c_dtype);

    int nfft = fft_plan->n[0];
    size_t factors_size = FFT_MAXFACTORS * sizeof(int);  // bytes
    size_t twiddles_size = in_c_dtype_size * nfft * 2;

    size_t reservespace_offset = 0;
    fft_plan->mlu_addrs.twiddles =
        (uint8_t *)fft_plan->reservespace_addr + reservespace_offset;
    reservespace_offset += twiddles_size;
    fft_plan->mlu_addrs.twiddles_end =
        (uint8_t *)fft_plan->mlu_addrs.twiddles +
        ((uint8_t *)fft_plan->twiddles_end - (uint8_t *)fft_plan->twiddles);

    fft_plan->mlu_addrs.dft_matrix =
        (int *)((uint8_t *)fft_plan->reservespace_addr + reservespace_offset);
    reservespace_offset += DFT_TABLE_SIZE;

    fft_plan->mlu_addrs.twiddles_inv =
        (uint8_t *)fft_plan->reservespace_addr + reservespace_offset;
    reservespace_offset += twiddles_size;
    fft_plan->mlu_addrs.twiddles_inv_end =
        (uint8_t *)fft_plan->mlu_addrs.twiddles_inv +
        ((uint8_t *)fft_plan->twiddles_inv_end -
         (uint8_t *)fft_plan->twiddles_inv);

    fft_plan->mlu_addrs.idft_matrix =
        (int *)((uint8_t *)fft_plan->reservespace_addr + reservespace_offset);
    reservespace_offset += DFT_TABLE_SIZE;

    fft_plan->mlu_addrs.factors =
        (int *)((uint8_t *)fft_plan->reservespace_addr + reservespace_offset);
    reservespace_offset += factors_size;

    CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.factors, fft_plan->factors,
                               FFT_MAXFACTORS * sizeof(int), handle->queue,
                               cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.twiddles, fft_plan->twiddles,
                               twiddles_size, handle->queue,
                               cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.dft_matrix,
                               fft_plan->dft_matrix, DFT_TABLE_SIZE,
                               handle->queue, cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.twiddles_inv,
                               fft_plan->twiddles_inv, twiddles_size,
                               handle->queue, cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.idft_matrix,
                               fft_plan->idft_matrix, DFT_TABLE_SIZE,
                               handle->queue, cnrtMemcpyHostToDev));
  }
  return status;
}

mluOpStatus_t setFFT2dReserveArea(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                                  const std::string api) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  const std::string make_plan_api = "[setFFT2dReserveArea]";

  size_t CPX_TYPE_SIZE = 0;

  switch (fft_plan->fft_type) {
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF: {
      CPX_TYPE_SIZE = 2 * 2;
    } break;
    case CNFFT_FLOAT2COMPLEX_FLOAT:
    case CNFFT_COMPLEX_FLOAT2FLOAT:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
      CPX_TYPE_SIZE = 4 * 2;
    }; break;
    default: {
      LOG(ERROR) << make_plan_api << ": invalid 2d fft type.";
      status = MLUOP_STATUS_NOT_SUPPORTED;
      return status;
    }
  }

  int n0_ori = fft_plan->n[0];
  int n1_ori = fft_plan->n[1];

  if (fft_plan->fft_strategy == CNFFT_FUNC_TWO_LEVEL_STOCKHAM) {
    size_t factors_size = FFT_MAXFACTORS * sizeof(int);  // bytes
    size_t twiddles_size = CPX_TYPE_SIZE * n1_ori;
    size_t twiddles_size_2d = CPX_TYPE_SIZE * n0_ori;

    size_t reservespace_offset = 0;

    fft_plan->mlu_addrs.twiddles =
        (uint8_t *)fft_plan->reservespace_addr + reservespace_offset;
    reservespace_offset += twiddles_size;
    fft_plan->mlu_addrs.twiddles_end =
        (uint8_t *)fft_plan->mlu_addrs.twiddles +
        ((uint8_t *)fft_plan->twiddles_end - (uint8_t *)fft_plan->twiddles);

    // dft matrix
    fft_plan->mlu_addrs.dft_matrix =
        (int *)((uint8_t *)fft_plan->reservespace_addr + reservespace_offset);
    reservespace_offset += DFT_TABLE_SIZE;

    // factors
    fft_plan->mlu_addrs.factors =
        (int *)((uint8_t *)fft_plan->reservespace_addr + reservespace_offset);
    reservespace_offset += factors_size;

    fft_plan->mlu_addrs.factors_2d =
        (int *)((uint8_t *)fft_plan->reservespace_addr + reservespace_offset);
    reservespace_offset += factors_size;

    CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.factors, fft_plan->factors,
                               factors_size, handle->queue,
                               cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.factors_2d,
                               fft_plan->factors_2d, factors_size,
                               handle->queue, cnrtMemcpyHostToDev));

    CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.twiddles, fft_plan->twiddles,
                               twiddles_size, handle->queue,
                               cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.dft_matrix,
                               fft_plan->dft_matrix, DFT_TABLE_SIZE,
                               handle->queue, cnrtMemcpyHostToDev));

    if (fft_plan->fft_type == CNFFT_HALF2COMPLEX_HALF ||
        fft_plan->fft_type == CNFFT_FLOAT2COMPLEX_FLOAT) {
      fft_plan->mlu_addrs.twiddles_2d =
          (uint8_t *)fft_plan->reservespace_addr + reservespace_offset;
      reservespace_offset += twiddles_size_2d;
      fft_plan->mlu_addrs.twiddles_2d_end =
          (uint8_t *)fft_plan->mlu_addrs.twiddles_2d +
          ((uint8_t *)fft_plan->twiddles_2d_end -
           (uint8_t *)fft_plan->twiddles_2d);
      fft_plan->mlu_addrs.dft_matrix_2d =
          (int *)((uint8_t *)fft_plan->reservespace_addr + reservespace_offset);
      reservespace_offset += DFT_TABLE_SIZE;

      CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.twiddles_2d,
                                 fft_plan->twiddles_2d, twiddles_size_2d,
                                 handle->queue, cnrtMemcpyHostToDev));
      CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.dft_matrix_2d,
                                 fft_plan->dft_matrix_2d, DFT_TABLE_SIZE,
                                 handle->queue, cnrtMemcpyHostToDev));
    } else if (fft_plan->fft_type == CNFFT_COMPLEX_HALF2HALF ||
               fft_plan->fft_type == CNFFT_COMPLEX_FLOAT2FLOAT) {
      fft_plan->mlu_addrs.twiddles_inv_2d =
          (uint8_t *)fft_plan->reservespace_addr + reservespace_offset;
      reservespace_offset += twiddles_size_2d;
      fft_plan->mlu_addrs.twiddles_inv_2d_end =
          (uint8_t *)fft_plan->mlu_addrs.twiddles_inv_2d +
          ((uint8_t *)fft_plan->twiddles_inv_2d_end -
           (uint8_t *)fft_plan->twiddles_inv_2d);

      fft_plan->mlu_addrs.idft_matrix_2d =
          (int *)((uint8_t *)fft_plan->reservespace_addr + reservespace_offset);
      reservespace_offset += DFT_TABLE_SIZE;

      CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.twiddles_inv_2d,
                                 fft_plan->twiddles_inv_2d, twiddles_size_2d,
                                 handle->queue, cnrtMemcpyHostToDev));

      CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.idft_matrix_2d,
                                 fft_plan->idft_matrix_2d, DFT_TABLE_SIZE,
                                 handle->queue, cnrtMemcpyHostToDev));
    } else {
      fft_plan->mlu_addrs.twiddles_2d =
          (uint8_t *)fft_plan->reservespace_addr + reservespace_offset;
      reservespace_offset += twiddles_size_2d;
      fft_plan->mlu_addrs.twiddles_2d_end =
          (uint8_t *)fft_plan->mlu_addrs.twiddles_2d +
          ((uint8_t *)fft_plan->twiddles_2d_end -
           (uint8_t *)fft_plan->twiddles_2d);
      fft_plan->mlu_addrs.dft_matrix_2d =
          (int *)((uint8_t *)fft_plan->reservespace_addr + reservespace_offset);
      reservespace_offset += DFT_TABLE_SIZE;
      fft_plan->mlu_addrs.twiddles_inv =
          (uint8_t *)fft_plan->reservespace_addr + reservespace_offset;
      reservespace_offset += twiddles_size;
      fft_plan->mlu_addrs.twiddles_inv_end =
          (uint8_t *)fft_plan->mlu_addrs.twiddles_inv +
          ((uint8_t *)fft_plan->twiddles_inv_end -
           (uint8_t *)fft_plan->twiddles_inv);
      fft_plan->mlu_addrs.twiddles_inv_2d =
          (uint8_t *)fft_plan->reservespace_addr + reservespace_offset;
      reservespace_offset += twiddles_size_2d;
      fft_plan->mlu_addrs.twiddles_inv_2d_end =
          (uint8_t *)fft_plan->mlu_addrs.twiddles_inv_2d +
          ((uint8_t *)fft_plan->twiddles_inv_2d_end -
           (uint8_t *)fft_plan->twiddles_inv_2d);

      fft_plan->mlu_addrs.idft_matrix_2d =
          (int *)((uint8_t *)fft_plan->reservespace_addr + reservespace_offset);
      reservespace_offset += DFT_TABLE_SIZE;
      fft_plan->mlu_addrs.idft_matrix =
          (int *)((uint8_t *)fft_plan->reservespace_addr + reservespace_offset);
      reservespace_offset += DFT_TABLE_SIZE;

      CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.twiddles_2d,
                                 fft_plan->twiddles_2d, twiddles_size_2d,
                                 handle->queue, cnrtMemcpyHostToDev));
      CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.dft_matrix_2d,
                                 fft_plan->dft_matrix_2d, DFT_TABLE_SIZE,
                                 handle->queue, cnrtMemcpyHostToDev));
      CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.twiddles_inv_2d,
                                 fft_plan->twiddles_inv_2d, twiddles_size_2d,
                                 handle->queue, cnrtMemcpyHostToDev));
      CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.twiddles_inv,
                                 fft_plan->twiddles_inv, twiddles_size,
                                 handle->queue, cnrtMemcpyHostToDev));
      CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.idft_matrix_2d,
                                 fft_plan->idft_matrix_2d, DFT_TABLE_SIZE,
                                 handle->queue, cnrtMemcpyHostToDev));
      CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.idft_matrix,
                                 fft_plan->idft_matrix, DFT_TABLE_SIZE,
                                 handle->queue, cnrtMemcpyHostToDev));
    }

  } else if (fft_plan->fft_strategy == CNFFT_FUNC_MANY_DIST1_2D) {
    switch (fft_plan->fft_type) {
      case CNFFT_HALF2COMPLEX_HALF:
      case CNFFT_FLOAT2COMPLEX_FLOAT: {
        // R2C
        size_t reservespace_offset = 0;
        fft_plan->mlu_addrs.dft_matrix =
            (uint8_t *)fft_plan->reservespace_addr + reservespace_offset;
        reservespace_offset += CPX_TYPE_SIZE * (n1_ori / 2 + 1) * n1_ori;

        fft_plan->mlu_addrs.dft_matrix_2d =
            (uint8_t *)fft_plan->reservespace_addr + reservespace_offset;
        reservespace_offset += CPX_TYPE_SIZE * n0_ori * n0_ori;

        CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.dft_matrix,
                                   fft_plan->dft_matrix,
                                   CPX_TYPE_SIZE * (n1_ori / 2 + 1) * n1_ori,
                                   handle->queue, cnrtMemcpyHostToDev));
        CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.dft_matrix_2d,
                                   fft_plan->dft_matrix_2d,
                                   CPX_TYPE_SIZE * n0_ori * n0_ori,
                                   handle->queue, cnrtMemcpyHostToDev));
      } break;
      case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
      case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
        // C2C
        size_t reservespace_offset = 0;
        fft_plan->mlu_addrs.dft_matrix =
            (uint8_t *)fft_plan->reservespace_addr + reservespace_offset;
        reservespace_offset += CPX_TYPE_SIZE * n1_ori * n1_ori;

        fft_plan->mlu_addrs.dft_matrix_2d =
            (uint8_t *)fft_plan->reservespace_addr + reservespace_offset;
        reservespace_offset += CPX_TYPE_SIZE * n0_ori * n0_ori;

        fft_plan->mlu_addrs.idft_matrix =
            (uint8_t *)fft_plan->reservespace_addr + reservespace_offset;
        reservespace_offset += CPX_TYPE_SIZE * n1_ori * n1_ori;

        fft_plan->mlu_addrs.idft_matrix_2d =
            (uint8_t *)fft_plan->reservespace_addr + reservespace_offset;
        reservespace_offset += CPX_TYPE_SIZE * n0_ori * n0_ori;

        CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.dft_matrix,
                                   fft_plan->dft_matrix,
                                   CPX_TYPE_SIZE * n1_ori * n1_ori,
                                   handle->queue, cnrtMemcpyHostToDev));
        CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.dft_matrix_2d,
                                   fft_plan->dft_matrix_2d,

                                   CPX_TYPE_SIZE * n0_ori * n0_ori,
                                   handle->queue, cnrtMemcpyHostToDev));

        CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.idft_matrix,
                                   fft_plan->idft_matrix,
                                   CPX_TYPE_SIZE * n1_ori * n1_ori,
                                   handle->queue, cnrtMemcpyHostToDev));
        CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.idft_matrix_2d,
                                   fft_plan->idft_matrix_2d,
                                   CPX_TYPE_SIZE * n0_ori * n0_ori,
                                   handle->queue, cnrtMemcpyHostToDev));
      }; break;
      case CNFFT_COMPLEX_HALF2HALF:
      case CNFFT_COMPLEX_FLOAT2FLOAT: {
        // C2R
        size_t reservespace_offset = 0;
        fft_plan->mlu_addrs.dft_matrix =
            (uint8_t *)fft_plan->reservespace_addr + reservespace_offset;
        reservespace_offset += CPX_TYPE_SIZE * (n1_ori / 2 + 1) * n1_ori;

        fft_plan->mlu_addrs.dft_matrix_2d =
            (uint8_t *)fft_plan->reservespace_addr + reservespace_offset;
        reservespace_offset += CPX_TYPE_SIZE * n0_ori * n0_ori;

        CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.dft_matrix,
                                   fft_plan->dft_matrix,
                                   CPX_TYPE_SIZE * (n1_ori / 2 + 1) * n1_ori,
                                   handle->queue, cnrtMemcpyHostToDev));
        CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.dft_matrix_2d,
                                   fft_plan->dft_matrix_2d,
                                   CPX_TYPE_SIZE * n0_ori * n0_ori,
                                   handle->queue, cnrtMemcpyHostToDev));
      }; break;
      default: {
        LOG(ERROR) << make_plan_api << ": invalid 2d fft type.";
        status = MLUOP_STATUS_NOT_SUPPORTED;
        return status;
      }
    }
  }
  return status;
}

static void configureFFT1dMatmulWorkspaceAddrs(mluOpHandle_t handle,
                                               mluOpFFTPlan_t fft_plan,
                                               void *input, void *workspace,
                                               void *output) {
  VLOG(5) << "Into configure FFT1d Matmul Workspace Addrs";
  size_t workspace_cur_offset = 0;
  size_t workspace_cur_offset_to_end = 0;
  size_t workspace_total_size = fft_plan->workspace_size;
  void *workspace_end = (uint8_t *)workspace + workspace_total_size;

  mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
  mluOpDataType_t in_r_dtype = (in_c_dtype == MLUOP_DTYPE_COMPLEX_HALF)
                                   ? MLUOP_DTYPE_HALF
                                   : MLUOP_DTYPE_FLOAT;
  mluOpDataType_t in_e_dtype = fft_plan->execution_dtype;
  size_t in_c_dtype_size = mluOpDataTypeBytes(in_c_dtype);
  size_t in_r_dtype_size = mluOpDataTypeBytes(in_r_dtype);
  int batch = fft_plan->batch;
  int n = fft_plan->n[0];

  // input contiguous
  size_t input_size = in_c_dtype_size * fft_plan->inum;
  if (!fft_plan->is_input_contiguous) {
    fft_plan->matmul_addrs.input_contiguous_addr =
        (uint8_t *)workspace + workspace_cur_offset;
    workspace_cur_offset += input_size;
  } else {
    fft_plan->matmul_addrs.input_contiguous_addr = input;
  }

  // input pad
  bool need_pad = (fft_plan->inembed[0] != n);
  int padded_input_num = batch * n;
  size_t padded_input_size = in_c_dtype_size * padded_input_num;
  if (need_pad) {
    fft_plan->matmul_addrs.input_pad_addr =
        (uint8_t *)workspace + workspace_cur_offset;
    workspace_cur_offset += padded_input_size;
  } else {
    fft_plan->matmul_addrs.input_pad_addr =
        fft_plan->matmul_addrs.input_contiguous_addr;
  }

  if (fft_plan->fft_strategy == CNFFT_FUNC_MATMUL) {
    // input trans: batch * n * 2 --> 2 * batch * n
    size_t transed_input_size = padded_input_size;
    fft_plan->matmul_addrs.input_re_addr =
        (uint8_t *)workspace + workspace_cur_offset;
    fft_plan->matmul_addrs.input_im_addr = (uint8_t *)workspace +
                                           workspace_cur_offset +
                                           transed_input_size / COMPLEX;
    workspace_cur_offset += transed_input_size;
  } else if (fft_plan->fft_strategy == CNFFT_FUNC_COOLEY_TUKEY) {
    // input 1st trans: batch * n * 2 --> 2 * batch * n
    size_t per_transed_input_size = padded_input_size;
    fft_plan->matmul_addrs.input_transed_addr =
        (uint8_t *)workspace + workspace_cur_offset;
    workspace_cur_offset += per_transed_input_size;
    // input 2nd trans: 2 * batch * L * 2^m --> 2 * batch * 2^m * L
    fft_plan->matmul_addrs.input_re_addr =
        (uint8_t *)workspace + workspace_cur_offset;
    fft_plan->matmul_addrs.input_im_addr = (uint8_t *)workspace +
                                           workspace_cur_offset +
                                           per_transed_input_size / COMPLEX;
    workspace_cur_offset += per_transed_input_size;
  }

  // internal workspace
  workspace_cur_offset_to_end += fft_plan->matmul_addrs.internal_workspace_size;
  fft_plan->matmul_addrs.internal_workspace_addr =
      (uint8_t *)workspace_end - workspace_cur_offset_to_end;

  // output contiguous
  size_t output_size =
      mluOpDataTypeBytes(fft_plan->output_dtype) * fft_plan->onum;
  if (!fft_plan->is_output_contiguous) {
    workspace_cur_offset_to_end += output_size;
    fft_plan->matmul_addrs.output_contiguous_addr =
        (uint8_t *)workspace_end - workspace_cur_offset_to_end;
  } else {
    fft_plan->matmul_addrs.output_contiguous_addr = output;
  }

  // matmul output
  int per_matmul_output_num = batch * n;
  size_t per_matmul_output_size = in_r_dtype_size * per_matmul_output_num;
  if (fft_plan->fft_strategy == CNFFT_FUNC_MATMUL) {
    workspace_cur_offset_to_end += per_matmul_output_size;
    fft_plan->matmul_addrs.matmul_im_mul_im_addr =
        (uint8_t *)workspace_end - workspace_cur_offset_to_end;
    workspace_cur_offset_to_end += per_matmul_output_size;
    fft_plan->matmul_addrs.matmul_im_mul_re_addr =
        (uint8_t *)workspace_end - workspace_cur_offset_to_end;
    workspace_cur_offset_to_end += per_matmul_output_size;
    fft_plan->matmul_addrs.matmul_re_mul_im_addr =
        (uint8_t *)workspace_end - workspace_cur_offset_to_end;
    workspace_cur_offset_to_end += per_matmul_output_size;
    fft_plan->matmul_addrs.matmul_re_mul_re_addr =
        (uint8_t *)workspace_end - workspace_cur_offset_to_end;
  } else if (fft_plan->fft_strategy == CNFFT_FUNC_COOLEY_TUKEY ||
             fft_plan->fft_strategy == CNFFT_FUNC_STOCKHAM) {
    workspace_cur_offset_to_end += per_matmul_output_size;
    fft_plan->matmul_addrs.matmul_im_mul_im_addr =
        (uint8_t *)workspace_end - workspace_cur_offset_to_end;
    workspace_cur_offset_to_end += per_matmul_output_size;
    fft_plan->matmul_addrs.matmul_im_mul_re_addr =
        (uint8_t *)workspace_end - workspace_cur_offset_to_end;
    workspace_cur_offset_to_end += per_matmul_output_size;
    fft_plan->matmul_addrs.matmul_re_mul_im_addr =
        (uint8_t *)workspace_end - workspace_cur_offset_to_end;
    workspace_cur_offset_to_end += per_matmul_output_size;
    fft_plan->matmul_addrs.matmul_re_mul_re_addr =
        (uint8_t *)workspace_end - workspace_cur_offset_to_end;
  }
}

// zrg

static void configureFFT1dWorkspaceAddrs(mluOpHandle_t handle,
                                         mluOpFFTPlan_t fft_plan, void *input,
                                         void *workspace, void *output) {
  VLOG(5) << "Into configure FFT1d Workspace Addrs";
  const std::string make_plan_api = "[configureFFT1dWorkspaceAddrs]";
  size_t workspace_size = 0;
  size_t reservespace_size = 0;

  // c2c
  mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
  mluOpDataType_t out_c_dtype = fft_plan->output_dtype;

  size_t in_c_dtype_size = mluOpDataTypeBytes(in_c_dtype);
  size_t out_c_dtype_size = mluOpDataTypeBytes(out_c_dtype);

  int batch = fft_plan->batch;
  int nfft = fft_plan->n[0];

  size_t buffer_size = batch * in_c_dtype_size * nfft;

  size_t offset = 0;
  fft_plan->mlu_addrs.buffer_buf = (uint8_t *)workspace + offset;
  offset += buffer_size * 2;

  if ((fft_plan->is_input_contiguous &&
       fft_plan->inembed[0] <= fft_plan->n[0]) ||
      fft_plan->is_batch_contiguous) {
    fft_plan->mlu_addrs.input = input;
  } else {
    fft_plan->mlu_addrs.input = (uint8_t *)workspace + offset;
    offset += buffer_size;
  }

  if (fft_plan->is_output_contiguous || fft_plan->is_batch_contiguous) {
    fft_plan->mlu_addrs.output = output;
  } else {
    fft_plan->mlu_addrs.output = (uint8_t *)workspace + offset;
    offset += buffer_size;
  }
  if (fft_plan->n[0] > fft_plan->inembed[0]) {
    fft_plan->mlu_addrs.input_pad_addr = (uint8_t *)workspace + offset;
  }
}

static void configureFFT2dWorkspaceAddrs(mluOpHandle_t handle,
                                         mluOpFFTPlan_t fft_plan, void *input,
                                         void *workspace, void *output) {
  const std::string make_plan_api = "[configureFFT2dWorkspaceAddrs]";

  mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
  mluOpDataType_t out_c_dtype = fft_plan->output_dtype;

  size_t in_c_dtype_size = mluOpDataTypeBytes(in_c_dtype);
  size_t out_c_dtype_size = mluOpDataTypeBytes(out_c_dtype);

  int batch = fft_plan->batch;
  int n0_ori = fft_plan->n[0];
  int n1_ori = fft_plan->n[1];

  size_t offset = 0;
  if (fft_plan->fft_strategy == CNFFT_FUNC_MANY_DIST1_2D) {
    // rr ri ir ii
    size_t buffer_size = batch * in_c_dtype_size * n0_ori * n1_ori * 2;
    fft_plan->mlu_addrs.input = input;
    fft_plan->mlu_addrs.output = output;
    fft_plan->mlu_addrs.buffer_in = (uint8_t *)workspace + offset;
    offset += buffer_size;
    fft_plan->mlu_addrs.buffer_out = (uint8_t *)workspace + offset;
    offset += buffer_size;
  }

  if (fft_plan->fft_strategy == CNFFT_FUNC_TWO_LEVEL_STOCKHAM) {
    fft_plan->mlu_addrs.buffer_buf = (uint8_t *)workspace + offset;
    offset += batch * in_c_dtype_size * n0_ori * n1_ori * 2;

    if ((fft_plan->is_input_contiguous &&
         fft_plan->inembed[0] <= fft_plan->n[0] &&
         fft_plan->inembed[1] <= fft_plan->n[1])) {
      fft_plan->mlu_addrs.input = input;
    } else {
      fft_plan->mlu_addrs.input = (uint8_t *)workspace + offset;
      offset += batch * in_c_dtype_size * n0_ori * n1_ori;
    }

    if (fft_plan->is_output_contiguous) {
      fft_plan->mlu_addrs.output = output;
    } else {
      fft_plan->mlu_addrs.output = (uint8_t *)workspace + offset;
      offset += batch * in_c_dtype_size * n0_ori * n1_ori;
    }
  }
  if (fft_plan->n[0] > fft_plan->inembed[0] ||
      fft_plan->n[1] > fft_plan->inembed[1]) {
    fft_plan->mlu_addrs.input_pad_addr =
        (uint8_t *)workspace + offset;  // batch * in_c_dtype_size * n0_ori *
                                        // n1_ori * 2; // buffer_size;
  }
}
// input    : in input
// output   : in input_contiguous_addr
static mluOpStatus_t makeFFT1dContiguousInput(mluOpHandle_t handle,
                                              mluOpFFTPlan_t fft_plan,
                                              const void *input) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into makeFFT1dContiguousInput";
  auto status = MLUOP_STATUS_SUCCESS;
  if (fft_plan->prime) {
    if (!fft_plan->is_input_contiguous) {
      VLOG(5) << "launch mluOpContiguous for fft1d input";
      mluOpTensorDescriptor_t input_desc;
      status = mluOpCreateTensorDescriptor(&input_desc);
      CHECK_RETURN(api, status);

      const int IN_DIM_NUM = 2;
      int64_t dims[IN_DIM_NUM] = {fft_plan->batch, fft_plan->inembed[0]};
      int64_t strides[IN_DIM_NUM] = {fft_plan->idist, fft_plan->istride};
      status = mluOpSetTensorDescriptorEx_v2(input_desc, MLUOP_LAYOUT_ARRAY,
                                             fft_plan->input_dtype, IN_DIM_NUM,
                                             dims, strides);
      CHECK_RETURN(api, status);

      status = mluOpContiguous(handle, input_desc, input,
                               fft_plan->matmul_addrs.input_contiguous_addr);
      CHECK_RETURN(api, status);

      status = mluOpDestroyTensorDescriptor(input_desc);
      CHECK_RETURN(api, status);
    }

  } else {
    if (!(fft_plan->is_input_contiguous &&
          fft_plan->inembed[0] <= fft_plan->n[0]) &&
        !fft_plan->is_batch_contiguous) {
      VLOG(5) << "launch mluOpContiguous for fft1d input";
      mluOpTensorDescriptor_t input_desc;
      status = mluOpCreateTensorDescriptor(&input_desc);
      CHECK_RETURN(api, status);

      const int IN_DIM_NUM = 2;
      int64_t dims[IN_DIM_NUM] = {
          fft_plan->batch, std::min(fft_plan->n[0], fft_plan->inembed[0])};
      int64_t strides[IN_DIM_NUM] = {fft_plan->idist, fft_plan->istride};
      status = mluOpSetTensorDescriptorEx_v2(input_desc, MLUOP_LAYOUT_ARRAY,
                                             fft_plan->input_dtype, IN_DIM_NUM,
                                             dims, strides);
      CHECK_RETURN(api, status);

      status =
          mluOpContiguous(handle, input_desc, input, fft_plan->mlu_addrs.input);
      CHECK_RETURN(api, status);

      status = mluOpDestroyTensorDescriptor(input_desc);
      CHECK_RETURN(api, status);
    }
  }
  return status;
}

// input    : in input
// output   : in input_contiguous_addr
static mluOpStatus_t makeFFT2dContiguousInput(mluOpHandle_t handle,
                                              mluOpFFTPlan_t fft_plan,
                                              const void *input,
                                              void *input_contiguous) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into makeFFT2dContiguousInput";
  auto status = MLUOP_STATUS_SUCCESS;
  if (!fft_plan->is_input_contiguous || fft_plan->inembed[0] > fft_plan->n[0] ||
      fft_plan->inembed[1] > fft_plan->n[1]) {
    VLOG(5) << "launch mluOpContiguous for fft2d input";
    mluOpTensorDescriptor_t input_desc;
    status = mluOpCreateTensorDescriptor(&input_desc);
    CHECK_RETURN(api, status);

    const int IN_DIM_NUM = 3;
    int64_t dims[IN_DIM_NUM] = {fft_plan->batch,
                                std::min(fft_plan->n[0], fft_plan->inembed[0]),
                                std::min(fft_plan->n[1], fft_plan->inembed[1])};

    int64_t strides[IN_DIM_NUM];  // IN_DIM_NUM
    for (int i = 0; i < IN_DIM_NUM; i++) {
      strides[i] = fft_plan->in_stride[i];
    }
    status = mluOpSetTensorDescriptorEx_v2(input_desc, MLUOP_LAYOUT_ARRAY,
                                           fft_plan->input_dtype, IN_DIM_NUM,
                                           dims, strides);
    CHECK_RETURN(api, status);

    status = mluOpContiguous(handle, input_desc, input, input_contiguous);
    CHECK_RETURN(api, status);

    status = mluOpDestroyTensorDescriptor(input_desc);
    CHECK_RETURN(api, status);
  }
  return status;
}

// input    : in input_contiguous_addr
// output   : in input_pad_addr
static mluOpStatus_t padFFT1dContiguousInput(mluOpHandle_t handle,
                                             mluOpFFTPlan_t fft_plan) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into padFFT1dContiguousInput";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
  mluOpDataType_t in_r_dtype = (in_c_dtype == MLUOP_DTYPE_COMPLEX_HALF)
                                   ? MLUOP_DTYPE_HALF
                                   : MLUOP_DTYPE_FLOAT;
  int batch = fft_plan->batch;
  int n = fft_plan->n[0];
  bool need_pad = (fft_plan->inembed[0] != n);
  if (need_pad) {
    VLOG(5) << "launch cnnlOpPad for input pad";
    mluOpTensorDescriptor_t input_desc, padded_input_desc;
    status = mluOpCreateTensorDescriptor(&input_desc);
    CHECK_RETURN(api, status);
    status = mluOpCreateTensorDescriptor(&padded_input_desc);
    CHECK_RETURN(api, status);

    const int IN_DIM_NUM = 2;
    int64_t dims[IN_DIM_NUM] = {batch, fft_plan->inembed[0] * COMPLEX};
    status = mluOpSetTensorDescriptor_v2(input_desc, MLUOP_LAYOUT_ARRAY,
                                         in_r_dtype, IN_DIM_NUM, dims);
    CHECK_RETURN(api, status);

    int64_t padded_dims[IN_DIM_NUM] = {batch, n * COMPLEX};
    status = mluOpSetTensorDescriptor_v2(padded_input_desc, MLUOP_LAYOUT_ARRAY,
                                         in_r_dtype, IN_DIM_NUM, padded_dims);
    CHECK_RETURN(api, status);

    const int pad_dim_num = 4;
    int paddings[pad_dim_num] = {0, 0, 0, (n - fft_plan->inembed[0]) * COMPLEX};
    uint64_t padding_value = 0x00000000;

    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                      cnnl_handle);  // convert to cnnl_handle

    // convert to cnnl_tensor_descriptor
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(padded_input_desc,
                                                 cnnl_padded_input_desc);
    CALL_CNNL(cnnlPad(cnnl_handle, cnnl_input_desc,
                      fft_plan->prime
                          ? fft_plan->matmul_addrs.input_contiguous_addr
                          : fft_plan->mlu_addrs.input,
                      paddings, &padding_value, cnnl_padded_input_desc,
                      fft_plan->prime ? fft_plan->matmul_addrs.input_pad_addr
                                      : fft_plan->mlu_addrs.input_pad_addr));

    status = mluOpDestroyTensorDescriptor(input_desc);
    CHECK_RETURN(api, status);
    status = mluOpDestroyTensorDescriptor(padded_input_desc);
    CHECK_RETURN(api, status);
    // destroy cnnl descriptor
    VLOG(5) << "c2cfft cnnlOpPad end";
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_padded_input_desc);

    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  return status;
}

// input    : in input_contiguous_addr
// output   : in input_pad_addr
static mluOpStatus_t padFFT2dContiguousInput(mluOpHandle_t handle,
                                             mluOpFFTPlan_t fft_plan) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into padFFT1dContiguousInput";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
  mluOpDataType_t in_r_dtype = (in_c_dtype == MLUOP_DTYPE_COMPLEX_HALF)
                                   ? MLUOP_DTYPE_HALF
                                   : MLUOP_DTYPE_FLOAT;
  int batch = fft_plan->batch;
  int n0 = fft_plan->n[0];
  int n1 = fft_plan->n[1];
  bool need_pad = (fft_plan->inembed[0] != n0 || fft_plan->inembed[1] != n1);
  if (need_pad) {
    VLOG(5) << "launch cnnlOpPad for input pad";
    mluOpTensorDescriptor_t input_desc, padded_input_desc;
    status = mluOpCreateTensorDescriptor(&input_desc);
    CHECK_RETURN(api, status);
    status = mluOpCreateTensorDescriptor(&padded_input_desc);
    CHECK_RETURN(api, status);

    const int IN_DIM_NUM = 3;
    int64_t dims[IN_DIM_NUM] = {batch, std::min(fft_plan->inembed[0], n0),
                                std::min(fft_plan->inembed[1], n1) * COMPLEX};
    status = mluOpSetTensorDescriptor_v2(input_desc, MLUOP_LAYOUT_ARRAY,
                                         in_r_dtype, IN_DIM_NUM, dims);
    CHECK_RETURN(api, status);

    int64_t padded_dims[IN_DIM_NUM] = {batch, n0, n1 * COMPLEX};
    status = mluOpSetTensorDescriptor_v2(padded_input_desc, MLUOP_LAYOUT_ARRAY,
                                         in_r_dtype, IN_DIM_NUM, padded_dims);
    CHECK_RETURN(api, status);

    const int pad_dim_num = 6;
    int paddings[pad_dim_num] = {
        0, 0,
        0, std::max((n0 - fft_plan->inembed[0]), 0),
        0, std::max((n1 - fft_plan->inembed[1]) * COMPLEX, 0)};
    uint64_t padding_value = 0x00000000;

    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                      cnnl_handle);  // convert to cnnl_handle

    // convert to cnnl_tensor_descriptor
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(padded_input_desc,
                                                 cnnl_padded_input_desc);
    CALL_CNNL(cnnlPad(cnnl_handle, cnnl_input_desc,
                      (fft_plan->n[0] > fft_plan->inembed[0] ||
                       fft_plan->n[1] > fft_plan->inembed[1])
                          ? fft_plan->mlu_addrs.input
                          : fft_plan->matmul_addrs.input_contiguous_addr,
                      paddings, &padding_value, cnnl_padded_input_desc,

                      (fft_plan->n[0] > fft_plan->inembed[0] ||
                       fft_plan->n[1] > fft_plan->inembed[1])
                          ? fft_plan->mlu_addrs.input_pad_addr
                          : fft_plan->matmul_addrs.input_pad_addr));

    status = mluOpDestroyTensorDescriptor(input_desc);
    CHECK_RETURN(api, status);
    status = mluOpDestroyTensorDescriptor(padded_input_desc);
    CHECK_RETURN(api, status);
    // destroy cnnl descriptor
    VLOG(5) << "c2cfft cnnlOpPad end";
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_padded_input_desc);

    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  return status;
}

/* CNFFT_FUNC_MATMUL:
         -------------------------
         |      input_pad        |
         -------------------------
                    |
                    | input trans: batch * n * 2 --> 2 * batch * n
                   \|/
         -------------------------
         |      input_re         |
         |      input_im         |
         -------------------------

   CNFFT_FUNC_COOLEY_TUKEY:
         -------------------------
         |      input_pad        |
         -------------------------
                    |
                    | input trans: batch * n * 2 --> 2 * batch * n
                   \|/
         -------------------------
         |      input_transed    |
         -------------------------
                    |
                    | input trans: 2 * batch * L * 2^m --> 2 * batch * 2^m * L
                   \|/
         -------------------------
         |      input_re         |
         |      input_im         |
         -------------------------
*/
static mluOpStatus_t transposeFFT1dPaddedInput(mluOpHandle_t handle,
                                               mluOpFFTPlan_t fft_plan) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into transposeFFT1dPaddedInput";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
  mluOpDataType_t in_r_dtype = (in_c_dtype == MLUOP_DTYPE_COMPLEX_HALF)
                                   ? MLUOP_DTYPE_HALF
                                   : MLUOP_DTYPE_FLOAT;
  int batch = fft_plan->batch;
  int n = fft_plan->n[0];

  if (fft_plan->fft_strategy == CNFFT_FUNC_MATMUL) {
    // transpose: batch * n * 2 --> 2 * batch * n
    VLOG(5) << "launch mluOpTranspose for input CNFFT_FUNC_MATMUL";
    int padded_input_num = batch * n;
    const int trans_dim_num = 2;
    int64_t trans_input_dims[trans_dim_num] = {padded_input_num, COMPLEX};
    int64_t trans_output_dims[trans_dim_num] = {COMPLEX, padded_input_num};
    int trans_permute[trans_dim_num] = {1, 0};

    status =
        fftTranspose(handle, trans_dim_num, trans_input_dims, trans_output_dims,
                     trans_permute, fft_plan->matmul_addrs.input_pad_addr,
                     fft_plan->matmul_addrs.input_re_addr, in_r_dtype,
                     fft_plan->matmul_addrs.internal_workspace_addr,
                     fft_plan->matmul_addrs.internal_workspace_size, api);
    CHECK_RETURN(api, status);
  } else if (fft_plan->fft_strategy == CNFFT_FUNC_COOLEY_TUKEY) {
    VLOG(5) << "launch mluOpTranspose for input CNFFT_FUNC_COOLEY_TUKEY";
    int L = fft_plan->L;
    int m = (1 << fft_plan->m);

    // 1st transpose: batch * n * 2 --> 2 * batch * n
    int padded_input_num = batch * n;
    const int trans_dim_num = 2;
    int64_t trans_input_dims[trans_dim_num] = {padded_input_num, COMPLEX};
    int64_t trans_output_dims[trans_dim_num] = {COMPLEX, padded_input_num};
    int trans_permute[trans_dim_num] = {1, 0};

    status =
        fftTranspose(handle, trans_dim_num, trans_input_dims, trans_output_dims,
                     trans_permute, fft_plan->matmul_addrs.input_pad_addr,
                     fft_plan->matmul_addrs.input_transed_addr, in_r_dtype,
                     fft_plan->matmul_addrs.internal_workspace_addr,
                     fft_plan->matmul_addrs.internal_workspace_size, api);

    // 2nd transpose: 2 * batch * L * 2^m --> 2 * batch * 2^m * L
    const int trans_2nd_dim_num = 3;
    int64_t trans_2nd_input_dims[trans_2nd_dim_num] = {COMPLEX * batch, L, m};
    int64_t trans_2nd_output_dims[trans_2nd_dim_num] = {COMPLEX * batch, m, L};
    int trans_2nd_permute[trans_2nd_dim_num] = {0, 2, 1};

    status = fftTranspose(handle, trans_2nd_dim_num, trans_2nd_input_dims,
                          trans_2nd_output_dims, trans_2nd_permute,
                          fft_plan->matmul_addrs.input_transed_addr,
                          fft_plan->matmul_addrs.input_re_addr, in_r_dtype,
                          fft_plan->matmul_addrs.internal_workspace_addr,
                          fft_plan->matmul_addrs.internal_workspace_size, api);
  }
  return status;
}

/* CNFFT_FUNC_MATMUL and CNFFT_FUNC_COOLEY_TUKEY:
         -------------------------
         |      input_re         |
         |      input_im         |
         -------------------------
                    |
                    | matmul
                    | optensor(re_mul_re - im_mul_im, re_mul_im + im_mul_re)
                   \|/
         -------------------------
         |     matmul_re_mul_re   | (matmul_re)
         |     matmul_re_mul_im   | (matmul_im)
         |     matmul_im_mul_re   |
         |     matmul_im_mul_im   |
         -------------------------
*/
static mluOpStatus_t computeFFT1dMatmulResult(mluOpHandle_t handle,
                                              mluOpFFTPlan_t fft_plan,
                                              const float scale_factor,
                                              const int direction) {
  std::string api = "[mluOpExecFFT]";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
  mluOpDataType_t in_r_dtype = (in_c_dtype == MLUOP_DTYPE_COMPLEX_HALF)
                                   ? MLUOP_DTYPE_HALF
                                   : MLUOP_DTYPE_FLOAT;
  mluOpDataType_t in_e_dtype = fft_plan->execution_dtype;
  int batch = fft_plan->batch;
  int n = fft_plan->n[0];

  void *dft_re_matrix_addr =
      (direction == 0) ? fft_plan->matmul_addrs.dft_re_matrix_addr
                       : fft_plan->matmul_addrs.ifft_dft_re_matrix_addr;
  void *dft_im_matrix_addr =
      (direction == 0) ? fft_plan->matmul_addrs.dft_im_matrix_addr
                       : fft_plan->matmul_addrs.ifft_dft_im_matrix_addr;

  if (fft_plan->fft_strategy == CNFFT_FUNC_MATMUL) {
    VLOG(5) << "into computeFFT1dMatmulResult CNFFT_FUNC_MATMUL";
    // input real matmul dft real
    status = fftMatMul(
        handle, batch, n, n, fft_plan->matmul_addrs.input_re_addr,
        dft_re_matrix_addr, fft_plan->matmul_addrs.matmul_re_mul_re_addr, false,
        true, scale_factor, 0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    CHECK_RETURN(api, status);

    // input imag matmul dft imag
    status = fftMatMul(
        handle, batch, n, n, fft_plan->matmul_addrs.input_im_addr,
        dft_im_matrix_addr, fft_plan->matmul_addrs.matmul_im_mul_im_addr, false,
        true, scale_factor, 0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    CHECK_RETURN(api, status);

    // input real matmul dft imag
    status = fftMatMul(
        handle, batch, n, n, fft_plan->matmul_addrs.input_re_addr,
        dft_im_matrix_addr, fft_plan->matmul_addrs.matmul_re_mul_im_addr, false,
        true, scale_factor, 0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    CHECK_RETURN(api, status);

    // input imag matmul dft real
    status = fftMatMul(
        handle, batch, n, n, fft_plan->matmul_addrs.input_im_addr,
        dft_re_matrix_addr, fft_plan->matmul_addrs.matmul_im_mul_re_addr, false,
        true, scale_factor, 0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    CHECK_RETURN(api, status);

    // real mul real sub imag mul imag
    int per_matmul_output_num = batch * n;
    status = fftOptensor(handle, per_matmul_output_num,
                         fft_plan->matmul_addrs.matmul_re_mul_re_addr,
                         fft_plan->matmul_addrs.matmul_im_mul_im_addr,
                         fft_plan->matmul_addrs.matmul_re_mul_re_addr, 1.0, 1.0,
                         0.0, in_r_dtype, CNNL_OP_TENSOR_SUB,
                         fft_plan->matmul_addrs.internal_workspace_addr,
                         fft_plan->matmul_addrs.internal_workspace_size, api);
    CHECK_RETURN(api, status);

    // real mul imag add imag mul real
    status = fftOptensor(handle, per_matmul_output_num,
                         fft_plan->matmul_addrs.matmul_re_mul_im_addr,
                         fft_plan->matmul_addrs.matmul_im_mul_re_addr,
                         fft_plan->matmul_addrs.matmul_re_mul_im_addr, 1.0, 1.0,
                         0.0, in_r_dtype, CNNL_OP_TENSOR_ADD,
                         fft_plan->matmul_addrs.internal_workspace_addr,
                         fft_plan->matmul_addrs.internal_workspace_size, api);
    CHECK_RETURN(api, status);
  } else if (fft_plan->fft_strategy == CNFFT_FUNC_COOLEY_TUKEY) {
    int L = fft_plan->L;
    int m = (1 << fft_plan->m);

    // input real matmul dft real
    status = fftMatMul(
        handle, batch * m, L, L, fft_plan->matmul_addrs.input_re_addr,
        dft_re_matrix_addr, fft_plan->matmul_addrs.matmul_re_mul_re_addr, false,
        true, scale_factor, 0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    CHECK_RETURN(api, status);

    // input imag matmul dft imag
    status = fftMatMul(
        handle, batch * m, L, L, fft_plan->matmul_addrs.input_im_addr,
        dft_im_matrix_addr, fft_plan->matmul_addrs.matmul_im_mul_im_addr, false,
        true, scale_factor, 0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    CHECK_RETURN(api, status);

    // input real matmul dft imag
    status = fftMatMul(
        handle, batch * m, L, L, fft_plan->matmul_addrs.input_re_addr,
        dft_im_matrix_addr, fft_plan->matmul_addrs.matmul_re_mul_im_addr, false,
        true, scale_factor, 0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    CHECK_RETURN(api, status);

    // input imag matmul dft real
    status = fftMatMul(
        handle, batch * m, L, L, fft_plan->matmul_addrs.input_im_addr,
        dft_re_matrix_addr, fft_plan->matmul_addrs.matmul_im_mul_re_addr, false,
        true, scale_factor, 0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    CHECK_RETURN(api, status);
  } else if (fft_plan->fft_strategy == CNFFT_FUNC_STOCKHAM) {
    int L = fft_plan->L;
    int m = (1 << fft_plan->m);

    // origin: in_trans[batch, 2^m, L] * W_real[L, L] -> IN_real[batch, 2^m, L]
    //         in_trans[batch, 2^m, L] * W_imag[L, L] -> IN_imag[batch, 2^m, L]
    // update: W[c*L, L] * in[batch, L, 2^m] -> out[batch, c*L, 2^m]
    status = fftBatchMatMulBcast(
        handle, 2 * L, L, m * 2, batch, dft_re_matrix_addr,
        fft_plan->matmul_addrs.input_pad_addr,
        fft_plan->matmul_addrs.matmul_re_mul_re_addr, false, false,
        scale_factor, 0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
  }

  return status;
}

static mluOpStatus_t policyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                                cnrtFunctionType_t *k_type) {
  *k_type = cnrtFuncTypeUnion1;
  k_dim->x = handle->core_num_per_cluster;
  k_dim->y = mluop::runtime::getClusterLimitCapability(handle);
  k_dim->z = 1;
  return MLUOP_STATUS_SUCCESS;
}

// only for CNFFT_FUNC_COOLEY_TUKEY and CNFFT_FUNC_STOCKHAM
// input    : matmul real result in matmul_re_mul_re_addr
//            matmul imag result in matmul_re_mul_im_addr
// workspace: internal_workspace_addr
// output   : output real result in output_contiguous_addr
mluOpStatus_t mergeFFT1dOutput(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                               const float scale_factor, int direction) {
  std::string api = "[mluOpExecFFT]";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  if (fft_plan->fft_strategy == CNFFT_FUNC_COOLEY_TUKEY) {
    VLOG(5) << "launch merge fft1d output";
    // TODO(niyuming) luanch merge kernel
    int core_num = handle->core_num_per_cluster;
    cnrtFunctionType_t k_type = cnrtFuncTypeUnion1;
    int task_type = mluop::runtime::getJobLimitCapability(handle);
    int task_num = 1;

    switch (task_type) {
      default:
        task_num = core_num;
        break;
      case (int)cnrtFuncTypeUnion2:
        task_num = core_num * 2;
        break;
      case (int)cnrtFuncTypeUnion4:
        task_num = core_num * 4;
        break;
      case (int)cnrtFuncTypeUnion8:
        task_num = core_num * 8;
        break;
      case (int)cnrtFuncTypeUnion16:
        task_num = core_num * 16;
        break;
    }

    unsigned int dimx = task_num;
    cnrtDim3_t k_dim = {dimx, 1, 1};
    k_type = (cnrtFunctionType_t)dimx;
    kernelFFTCooleyTukey(k_dim, k_type, handle->queue, fft_plan, direction,
                         FFT_IFFT);
  } else if (fft_plan->fft_strategy == CNFFT_FUNC_STOCKHAM) {
    VLOG(5) << "launch mrege rfft1d output";
    cnrtDim3_t k_dim;
    cnrtFunctionType_t k_type;
    policyFunc(handle, &k_dim, &k_type);
    kernelFFTStockham(k_dim, k_type, handle->queue, fft_plan, direction,
                      scale_factor, FFT_IFFT);
  }
  return status;
}

// only for CNFFT_FUNC_MATMUL
// input    : matmul real result in matmul_re_mul_re_addr
//            matmul imag result in matmul_re_mul_im_addr
// output   : output complex result in output_contiguous_addr
static mluOpStatus_t transposeFFT1dOutput(mluOpHandle_t handle,
                                          mluOpFFTPlan_t fft_plan) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into transposeFFT1dOutput";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  if (fft_plan->fft_strategy == CNFFT_FUNC_MATMUL) {
    VLOG(5) << "launch mluOpTranspose";
    mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
    mluOpDataType_t in_r_dtype = (in_c_dtype == MLUOP_DTYPE_COMPLEX_HALF)
                                     ? MLUOP_DTYPE_HALF
                                     : MLUOP_DTYPE_FLOAT;
    int batch = fft_plan->batch;
    int n = fft_plan->n[0];

    int output_num = batch * n;
    const int trans_dim_num = 2;
    int64_t trans_input_dims[trans_dim_num] = {COMPLEX, output_num};
    int64_t trans_output_dims[trans_dim_num] = {output_num, COMPLEX};
    int trans_permute[trans_dim_num] = {1, 0};

    status = fftTranspose(
        handle, trans_dim_num, trans_input_dims, trans_output_dims,
        trans_permute, fft_plan->matmul_addrs.matmul_re_mul_re_addr,
        fft_plan->matmul_addrs.output_contiguous_addr, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    CHECK_RETURN(api, status);
  }
  return status;
}

static mluOpStatus_t makeFFT1dContiguousOutput(mluOpHandle_t handle,
                                               mluOpFFTPlan_t fft_plan,
                                               void *output) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into makeFFT1dContiguousOutput";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  if (!fft_plan->is_output_contiguous &&
      (fft_plan->prime || !fft_plan->is_batch_contiguous)) {
    VLOG(5) << "launch copy with stride";
    mluOpDataType_t out_c_dtype = fft_plan->output_dtype;
    // create tensor desc
    mluOpTensorDescriptor_t copy_src_desc, copy_dst_desc;
    status = mluOpCreateTensorDescriptor(&copy_src_desc);
    CHECK_RETURN(api, status);
    status = mluOpCreateTensorDescriptor(&copy_dst_desc);
    CHECK_RETURN(api, status);

    // set up tensor desc
    const int OUT_DIM_NUM = 2;
    int64_t dims[OUT_DIM_NUM] = {fft_plan->batch, (fft_plan->prime)
                                                      ? fft_plan->onembed[0]
                                                      : fft_plan->n[0]};
    int64_t strides[OUT_DIM_NUM] = {fft_plan->odist, fft_plan->ostride};
    status = mluOpSetTensorDescriptor_v2(copy_src_desc, MLUOP_LAYOUT_ARRAY,
                                         out_c_dtype, OUT_DIM_NUM, dims);
    CHECK_RETURN(api, status);
    status =
        mluOpSetTensorDescriptorEx_v2(copy_dst_desc, MLUOP_LAYOUT_ARRAY,
                                      out_c_dtype, OUT_DIM_NUM, dims, strides);
    CHECK_RETURN(api, status);

    void *copy_src_addr = (fft_plan->prime)
                              ? fft_plan->matmul_addrs.output_contiguous_addr
                              : fft_plan->mlu_addrs.output;
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                      cnnl_handle);  // convert to cnnl_handle
    // convert to cnnl_tensor_descriptor
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(copy_src_desc,
                                                 cnnl_copy_src_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(copy_dst_desc,
                                                 cnnl_copy_dst_desc);
    CALL_CNNL(cnnlCopy_v2(cnnl_handle, cnnl_copy_src_desc, copy_src_addr,
                          cnnl_copy_dst_desc, output, NULL, 0));
    status = mluOpDestroyTensorDescriptor(copy_src_desc);
    CHECK_RETURN(api, status);
    status = mluOpDestroyTensorDescriptor(copy_dst_desc);
    CHECK_RETURN(api, status);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_copy_src_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_copy_dst_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  return status;
}

static mluOpStatus_t makeFFT2dContiguousOutput(mluOpHandle_t handle,
                                               mluOpFFTPlan_t fft_plan,
                                               void *output,
                                               void *output_contiguous) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into makeFFT2dContiguousOutput";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  if (!fft_plan->is_output_contiguous) {
    VLOG(5) << "launch copy with stride";
    mluOpDataType_t out_c_dtype = fft_plan->output_dtype;
    // create tensor desc
    mluOpTensorDescriptor_t copy_src_desc, copy_dst_desc;
    status = mluOpCreateTensorDescriptor(&copy_src_desc);
    CHECK_RETURN(api, status);
    status = mluOpCreateTensorDescriptor(&copy_dst_desc);
    CHECK_RETURN(api, status);

    // set up tensor desc
    const int OUT_DIM_NUM = 3;
    int64_t dims[OUT_DIM_NUM] = {fft_plan->batch, fft_plan->n[0],
                                 fft_plan->n[1]};
    int64_t strides[OUT_DIM_NUM];  // OUT_DIM_NUM
    for (int i = 0; i < OUT_DIM_NUM; i++) {
      strides[i] = fft_plan->out_stride[i];
    }
    status = mluOpSetTensorDescriptor_v2(copy_src_desc, MLUOP_LAYOUT_ARRAY,
                                         out_c_dtype, OUT_DIM_NUM, dims);
    CHECK_RETURN(api, status);
    status =
        mluOpSetTensorDescriptorEx_v2(copy_dst_desc, MLUOP_LAYOUT_ARRAY,
                                      out_c_dtype, OUT_DIM_NUM, dims, strides);
    CHECK_RETURN(api, status);

    // void *copy_src_addr = fft_plan->matmul_addrs.output_contiguous_addr;
    void *copy_src_addr = output_contiguous;
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                      cnnl_handle);  // convert to cnnl_handle
    // convert to cnnl_tensor_descriptor
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(copy_src_desc,
                                                 cnnl_copy_src_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(copy_dst_desc,
                                                 cnnl_copy_dst_desc);
    CALL_CNNL(cnnlCopy_v2(cnnl_handle, cnnl_copy_src_desc, copy_src_addr,
                          cnnl_copy_dst_desc, output, NULL, 0));
    status = mluOpDestroyTensorDescriptor(copy_src_desc);
    CHECK_RETURN(api, status);
    status = mluOpDestroyTensorDescriptor(copy_dst_desc);
    CHECK_RETURN(api, status);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_copy_src_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_copy_dst_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  return status;
}

mluOpStatus_t execFFTc2c1d(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                           const float scale_factor, const int direction) {
  std::string api = "[execFFTc2c1d]";

  VLOG(5) << "launch c2c fft1d";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, &k_dim, &k_type);
  if (!fft_plan->is_batch_contiguous) {
    kernelFFT1dButterflyRow(k_dim, k_type, handle->queue, fft_plan, direction,
                            FFT_IFFT);
  } else {
    kernelFFT1dButterflyColumn(k_dim, k_type, handle->queue, fft_plan,
                               direction, FFT_IFFT);
  }

  return status;
}

mluOpStatus_t execFFTc2c2d(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                           const float scale_factor, const int direction) {
  std::string api = "[execFFTc2c2d]";

  VLOG(5) << "launch c2c fft2d";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  if (fft_plan->fft_strategy == CNFFT_FUNC_TWO_LEVEL_STOCKHAM) {
    cnrtDim3_t k_dim;
    cnrtFunctionType_t k_type;
    policyFunc(handle, &k_dim, &k_type);
    uint64_t idist = 0, odist = 0;  // bytes
    mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
    size_t in_c_dtype_size = mluOpDataTypeBytes(in_c_dtype);
    idist = in_c_dtype_size * fft_plan->n[0] * fft_plan->n[1];
    odist = in_c_dtype_size * fft_plan->n[0] * fft_plan->n[1];

    for (int batch_id = 0; batch_id < fft_plan->batch; batch_id++) {
      if (direction == FFT_FORWARD) {
        status = kernelFFT2dButterflyRow(k_dim, k_type, handle->queue, fft_plan,
                                         direction, FFT_IFFT);
        CHECK_RETURN(api, status);

        status = kernelFFT2dButterflyColumn(k_dim, k_type, handle->queue,
                                            fft_plan, direction, FFT_IFFT);
        CHECK_RETURN(api, status);
      } else {
        status = kernelFFT2dButterflyColumn(k_dim, k_type, handle->queue,
                                            fft_plan, direction, FFT_IFFT);

        CHECK_RETURN(api, status);

        status = kernelFFT2dButterflyRow(k_dim, k_type, handle->queue, fft_plan,
                                         direction, FFT_IFFT);
        CHECK_RETURN(api, status);
      }

      fft_plan->mlu_addrs.input =
          (void *)((uint64_t)(fft_plan->mlu_addrs.input) + idist);
      fft_plan->mlu_addrs.output =
          (void *)((uint64_t)(fft_plan->mlu_addrs.output) + odist);
    }
    fft_plan->mlu_addrs.input = (void *)((uint64_t)(fft_plan->mlu_addrs.input) -
                                         fft_plan->batch * idist);
    fft_plan->mlu_addrs.output =
        (void *)((uint64_t)(fft_plan->mlu_addrs.output) -
                 fft_plan->batch * odist);
  }

  if (fft_plan->fft_strategy == CNFFT_FUNC_MANY_DIST1_2D) {
    if (direction == FFT_FORWARD) {
      status = computeFFT2dMatMulRow(handle, fft_plan, scale_factor, direction);
      CHECK_RETURN(api, status);

      status =
          computeFFT2dMatMulColumn(handle, fft_plan, scale_factor, direction);
      CHECK_RETURN(api, status);
    } else {
      status =
          computeFFT2dMatMulColumn(handle, fft_plan, scale_factor, direction);
      CHECK_RETURN(api, status);

      status = computeFFT2dMatMulRow(handle, fft_plan, scale_factor, direction);
      CHECK_RETURN(api, status);
    }
  }
  return status;
}

mluOpStatus_t execFFT1d(mluOpHandle_t handle, const mluOpFFTPlan_t fft_plan,
                        const void *input, const float scale_factor,
                        void *workspace, void *output, const int direction) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  std::string api = "[mluOpExecFFT]";

  if (fft_plan->prime) {
    configureFFT1dMatmulWorkspaceAddrs(handle, fft_plan, (void *)input,
                                       workspace, output);

    status = makeFFT1dContiguousInput(handle, fft_plan, input);
    CHECK_RETURN(api, status);

    status = padFFT1dContiguousInput(handle, fft_plan);
    CHECK_RETURN(api, status);

    status = transposeFFT1dPaddedInput(handle, fft_plan);
    CHECK_RETURN(api, status);

    status =
        computeFFT1dMatmulResult(handle, fft_plan, scale_factor, direction);
    CHECK_RETURN(api, status);

    status = mergeFFT1dOutput(handle, fft_plan, scale_factor, direction);
    CHECK_RETURN(api, status);

    status = transposeFFT1dOutput(handle, fft_plan);
    CHECK_RETURN(api, status);

    status = makeFFT1dContiguousOutput(handle, fft_plan, output);
    CHECK_RETURN(api, status);
  } else {
    configureFFT1dWorkspaceAddrs(handle, fft_plan, (void *)input, workspace,
                                 output);

    status = makeFFT1dContiguousInput(handle, fft_plan, input);
    CHECK_RETURN(api, status);

    if (fft_plan->n[0] > fft_plan->inembed[0]) {
      status = padFFT1dContiguousInput(handle, fft_plan);
      CHECK_RETURN(api, status);
      fft_plan->mlu_addrs.input = fft_plan->mlu_addrs.input_pad_addr;
    }

    status = execFFTc2c1d(handle, fft_plan, scale_factor, direction);

    CHECK_RETURN(api, status);

    if (scale_factor != 1.0) {
      const float alpha[2] = {scale_factor, 0.0};
      const float beta[2] = {0.0, 0.0};
      mluOpTensorDescriptor_t c_desc = nullptr;
      status = mluOpCreateTensorDescriptor(&c_desc);
      CHECK_RETURN(api, status);
      const int OUT_DIM_NUM = 2;
      int64_t dims[OUT_DIM_NUM] = {fft_plan->batch, fft_plan->n[0]};
      status = mluOpSetTensorDescriptor_v2(c_desc, MLUOP_LAYOUT_ARRAY,
                                           fft_plan->output_dtype, OUT_DIM_NUM,
                                           dims);
      CHECK_RETURN(api, status);
      status = mluOpSetTensorDescriptorOnchipDataType(
          c_desc, fft_plan->execution_dtype);
      CHECK_RETURN(api, status);

      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                        cnnl_handle);  // convert to cnnl_handle

      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_output_desc);

      CALL_CNNL(cnnlTransform_v2(cnnl_handle, CNNL_POINTER_MODE_HOST, &alpha,
                                 cnnl_output_desc, fft_plan->mlu_addrs.output,
                                 &beta, cnnl_output_desc,
                                 fft_plan->mlu_addrs.output));
      status = mluOpDestroyTensorDescriptor(c_desc);
      CHECK_RETURN(api, status);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }
    CHECK_RETURN(api, status);
    status = makeFFT1dContiguousOutput(handle, fft_plan, output);
    CHECK_RETURN(api, status);
  }

  return status;
}

mluOpStatus_t execFFT2d(mluOpHandle_t handle, const mluOpFFTPlan_t fft_plan,
                        const void *input, const float scale_factor,
                        void *workspace, void *output, const int direction) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  std::string api = "[mluOpExecFFT]";

  // direction: 0(forward) 1(backward)

  configureFFT2dWorkspaceAddrs(handle, fft_plan, (void *)input, workspace,
                               output);
  if (fft_plan->fft_strategy != CNFFT_FUNC_MANY_DIST1_2D) {
    status = makeFFT2dContiguousInput(handle, fft_plan, input,
                                      fft_plan->mlu_addrs.input);
    CHECK_RETURN(api, status);
  }

  if (fft_plan->n[0] > fft_plan->inembed[0] ||
      fft_plan->n[1] > fft_plan->inembed[1]) {
    status = padFFT2dContiguousInput(handle, fft_plan);
    CHECK_RETURN(api, status);
    fft_plan->mlu_addrs.input = fft_plan->mlu_addrs.input_pad_addr;
  }

  if (fft_plan->n[0] == 1 || fft_plan->n[1] == 1) {
    cnrtDim3_t k_dim;
    cnrtFunctionType_t k_type;
    policyFunc(handle, &k_dim, &k_type);
    uint64_t idist = 0, odist = 0;  // bytes
    mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
    size_t in_c_dtype_size = mluOpDataTypeBytes(in_c_dtype);
    idist = in_c_dtype_size * fft_plan->n[0] * fft_plan->n[1];
    odist = in_c_dtype_size * fft_plan->n[0] * fft_plan->n[1];

    if ((fft_plan->n[0] == 1 && fft_plan->n[1] == 1) ||
        (fft_plan->n[1] != 1 && direction == FFT_BACKWARD) ||
        (fft_plan->n[0] != 1 && direction == FFT_FORWARD)) {
      mluOpTensorDescriptor_t c_desc = nullptr;
      status = mluOpCreateTensorDescriptor(&c_desc);
      CHECK_RETURN(api, status);
      const int OUT_DIM_NUM = 3;
      int64_t dims[OUT_DIM_NUM] = {fft_plan->batch, fft_plan->n[0],
                                   fft_plan->n[1]};
      status = mluOpSetTensorDescriptor_v2(c_desc, MLUOP_LAYOUT_ARRAY,
                                           fft_plan->output_dtype, OUT_DIM_NUM,
                                           dims);
      CHECK_RETURN(api, status);
      status = mluOpSetTensorDescriptorOnchipDataType(
          c_desc, fft_plan->execution_dtype);
      CHECK_RETURN(api, status);

      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                        cnnl_handle);  // convert to cnnl_handle

      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_output_desc);
      CALL_CNNL(cnnlCopy_v2(cnnl_handle, cnnl_output_desc,
                            fft_plan->mlu_addrs.input, cnnl_output_desc,
                            fft_plan->mlu_addrs.output, NULL, 0));
      status = mluOpDestroyTensorDescriptor(c_desc);
      CHECK_RETURN(api, status);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }

    if (fft_plan->n[0] == 1 && fft_plan->n[1] != 1) {
      for (int batch_id = 0; batch_id < fft_plan->batch; batch_id++) {
        if (direction == FFT_FORWARD) {
          status = kernelFFT2dButterflyRow(k_dim, k_type, handle->queue,
                                           fft_plan, direction, FFT_IFFT);
          CHECK_RETURN(api, status);
        } else {
          status = kernelFFT2dButterflyRow(k_dim, k_type, handle->queue,
                                           fft_plan, direction, FFT_IFFT);
          CHECK_RETURN(api, status);
        }

        fft_plan->mlu_addrs.input =
            (void *)((uint64_t)(fft_plan->mlu_addrs.input) + idist);
        fft_plan->mlu_addrs.output =
            (void *)((uint64_t)(fft_plan->mlu_addrs.output) + odist);
      }
      fft_plan->mlu_addrs.input =
          (void *)((uint64_t)(fft_plan->mlu_addrs.input) -
                   fft_plan->batch * idist);
      fft_plan->mlu_addrs.output =
          (void *)((uint64_t)(fft_plan->mlu_addrs.output) -
                   fft_plan->batch * odist);

    } else if (fft_plan->n[0] != 1 && fft_plan->n[1] == 1) {
      for (int batch_id = 0; batch_id < fft_plan->batch; batch_id++) {
        if (direction == FFT_FORWARD) {
          status = kernelFFT2dButterflyColumn(k_dim, k_type, handle->queue,
                                              fft_plan, direction, FFT_IFFT);
          CHECK_RETURN(api, status);
        } else {
          status = kernelFFT2dButterflyColumn(k_dim, k_type, handle->queue,
                                              fft_plan, direction, FFT_IFFT);
          CHECK_RETURN(api, status);
        }

        fft_plan->mlu_addrs.input =
            (void *)((uint64_t)(fft_plan->mlu_addrs.input) + idist);
        fft_plan->mlu_addrs.output =
            (void *)((uint64_t)(fft_plan->mlu_addrs.output) + odist);
      }
      fft_plan->mlu_addrs.input =
          (void *)((uint64_t)(fft_plan->mlu_addrs.input) -
                   fft_plan->batch * idist);
      fft_plan->mlu_addrs.output =
          (void *)((uint64_t)(fft_plan->mlu_addrs.output) -
                   fft_plan->batch * odist);
    }
  } else {
    status = execFFTc2c2d(handle, fft_plan, scale_factor, direction);
  }

  CHECK_RETURN(api, status);

  if (scale_factor != 1.0) {
    const float alpha[2] = {scale_factor, 0.0};
    const float beta[2] = {0.0, 0.0};
    mluOpTensorDescriptor_t c_desc = nullptr;
    status = mluOpCreateTensorDescriptor(&c_desc);
    CHECK_RETURN(api, status);
    const int OUT_DIM_NUM = 3;
    int64_t dims[OUT_DIM_NUM] = {fft_plan->batch, fft_plan->n[0],
                                 fft_plan->n[1]};
    status = mluOpSetTensorDescriptor_v2(
        c_desc, MLUOP_LAYOUT_ARRAY, fft_plan->output_dtype, OUT_DIM_NUM, dims);
    CHECK_RETURN(api, status);
    status = mluOpSetTensorDescriptorOnchipDataType(c_desc,
                                                    fft_plan->execution_dtype);
    CHECK_RETURN(api, status);

    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                      cnnl_handle);  // convert to cnnl_handle

    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_output_desc);

    CALL_CNNL(cnnlTransform_v2(cnnl_handle, CNNL_POINTER_MODE_HOST, &alpha,
                               cnnl_output_desc, fft_plan->mlu_addrs.output,
                               &beta, cnnl_output_desc,
                               fft_plan->mlu_addrs.output));
    status = mluOpDestroyTensorDescriptor(c_desc);
    CHECK_RETURN(api, status);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  CHECK_RETURN(api, status);

  if (!fft_plan->is_output_contiguous &&
      fft_plan->fft_strategy != CNFFT_FUNC_MANY_DIST1_2D) {
    status = makeFFT2dContiguousOutput(handle, fft_plan, output,
                                       fft_plan->mlu_addrs.output);
    CHECK_RETURN(api, status);
  }

  return status;
}

mluOpStatus_t computeFFT2dMatMulColumn(mluOpHandle_t handle,
                                       mluOpFFTPlan_t fft_plan,
                                       const float scale_factor,
                                       const int direction) {
  std::string api = "[computeFFT2dMatMulColumn]";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  mluOpDataType_t in_e_dtype = fft_plan->execution_dtype;
  int batch = fft_plan->batch;
  int n0 = fft_plan->n[0];
  int n1 = fft_plan->n[1];

  void *dft_matrix_addr = (direction == FFT_FORWARD)
                              ? fft_plan->mlu_addrs.dft_matrix_2d
                              : fft_plan->mlu_addrs.idft_matrix_2d;
  void *in_addr = (direction == FFT_FORWARD) ? fft_plan->mlu_addrs.buffer_in
                                             : fft_plan->mlu_addrs.input;
  void *out_addr = fft_plan->mlu_addrs.buffer_out;
  // W[n0 * c][n0] * In[n0][n1 * batch]
  const int m = n0 * 2, k = n0, n = n1 * batch * 2;

  // create descriptor
  mluOpTensorDescriptor_t a_desc = nullptr;
  mluOpTensorDescriptor_t b_desc = nullptr;
  mluOpTensorDescriptor_t c_desc = nullptr;
  status = mluOpCreateTensorDescriptor(&a_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&b_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&c_desc);
  CHECK_RETURN(api, status);

  // set descriptor
  int64_t a_dims[2] = {m, k};
  int64_t b_dims[2] = {k, n};
  int64_t c_dims[2] = {m, n};

  status = mluOpSetTensorDescriptor_v2(a_desc, MLUOP_LAYOUT_ARRAY, in_e_dtype,
                                       2, a_dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptorOnchipDataType(a_desc, in_e_dtype);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(b_desc, MLUOP_LAYOUT_ARRAY, in_e_dtype,
                                       2, b_dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptorOnchipDataType(b_desc, in_e_dtype);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(c_desc, MLUOP_LAYOUT_ARRAY, in_e_dtype,
                                       2, c_dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptorOnchipDataType(c_desc, in_e_dtype);
  CHECK_RETURN(api, status);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                    cnnl_handle);  // convert to cnnl_handle

  cnnlMatMulDescriptor_t matmul_desc;
  cnnlMatMulAlgo_t matmul_algo;
  cnnlMatMulHeuristicResult_t heuristic_result;
  size_t matmul_ws_size = 0, workspace_size = 0;

  CALL_CNNL(cnnlCreateMatMulDescriptor(&matmul_desc));
  CALL_CNNL(cnnlCreateMatMulAlgo(&matmul_algo));
  CALL_CNNL(cnnlCreateMatMulHeuristicResult(&heuristic_result));
  int32_t requested_algo_count = 1, return_algo_count = 0;

  // convert to cnnl_tensor_descriptor
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_d_desc);
  c_desc->setOnchipDtype(in_e_dtype);
  CALL_CNNL(cnnlGetMatMulAlgoHeuristic(cnnl_handle, matmul_desc, cnnl_a_desc,
                                       cnnl_b_desc, cnnl_c_desc, cnnl_d_desc,
                                       nullptr, requested_algo_count,
                                       &heuristic_result, &return_algo_count));
  CALL_CNNL(cnnlGetMatMulHeuristicResult(heuristic_result, matmul_algo,
                                         &workspace_size));
  float *workspace = nullptr;
  if (workspace_size > 0) {
    CNRT_CHECK(cnrtMalloc((void **)&workspace, workspace_size));
  }
  float alpha = 1.0;
  float beta = 0.0;
  CALL_CNNL(cnnlMatMul_v2(cnnl_handle, matmul_desc, matmul_algo, &alpha,
                          cnnl_a_desc, dft_matrix_addr, cnnl_b_desc, in_addr,
                          &beta, cnnl_c_desc, out_addr, workspace,
                          workspace_size, cnnl_d_desc, out_addr));

  status = mluOpDestroyTensorDescriptor(a_desc);
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(b_desc);
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(c_desc);
  CHECK_RETURN(api, status);
  // destroy cnnl descriptor
  CALL_CNNL(cnnlDestroyMatMulDescriptor(matmul_desc));
  CALL_CNNL(cnnlDestroyMatMulAlgo(matmul_algo));
  CALL_CNNL(cnnlDestroyMatMulHeuristicResult(heuristic_result));
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);

  DESTROY_CNNL_HANDLE(cnnl_handle);
  cnrtQueueSync(handle->queue);

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, &k_dim, &k_type);
  if (direction == FFT_FORWARD) {
    kernelFFTConjMerge(k_dim, k_type, handle->queue, fft_plan->mlu_addrs.output,
                       fft_plan->mlu_addrs.buffer_out, n0 * n1 * batch,
                       in_e_dtype);
  } else {
    kernelFFTConjMerge(
        k_dim, k_type, handle->queue, fft_plan->mlu_addrs.buffer_in,
        fft_plan->mlu_addrs.buffer_out, n0 * n1 * batch, in_e_dtype);
  }

  cnrtQueueSync(handle->queue);

  return status;
}

// a0 + b0 i a1 + b1 i a2 + b2 i a3 + b3 i a4 + b4 i
// a0 + 0 i a1 + b1 i a2 + b2 i a2 - b2 i a1 - b1 i
//  R2C DFT:
//  [r][r][2] []
// [180][180][768]
// [180][768]

// DFT[n1*2][n1] * In[n0][n1][batch*2]
// out[n0][n1*2][batch*2] -> merge out[n0][n1][batch*2]

mluOpStatus_t computeFFT2dMatMulRow(mluOpHandle_t handle,
                                    mluOpFFTPlan_t fft_plan,
                                    const float scale_factor,
                                    const int direction) {
  std::string api = "[computeFFT2dMatMulRow]";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  mluOpDataType_t in_e_dtype = fft_plan->execution_dtype;
  int batch = fft_plan->batch;
  int n0 = fft_plan->n[0];
  int n1 = fft_plan->n[1];

  void *dft_matrix_addr = (direction == FFT_FORWARD)
                              ? fft_plan->mlu_addrs.dft_matrix
                              : fft_plan->mlu_addrs.idft_matrix;
  void *in_addr = (direction == FFT_FORWARD) ? fft_plan->mlu_addrs.input
                                             : fft_plan->mlu_addrs.buffer_in;
  void *out_addr = fft_plan->mlu_addrs.buffer_out;
  // out[n0][n1*2][batch*2] = W[n1 * 2][n1] * In[n0][n1][batch *2]
  const int m = n1 * 2, k = n1, n = batch * 2;

  // create descriptor
  mluOpTensorDescriptor_t a_desc = nullptr;
  mluOpTensorDescriptor_t b_desc = nullptr;
  mluOpTensorDescriptor_t c_desc = nullptr;
  status = mluOpCreateTensorDescriptor(&a_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&b_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&c_desc);
  CHECK_RETURN(api, status);

  // set descriptor
  int64_t a_dims[2] = {m, k};
  int64_t b_dims[3] = {n0, k, n};
  int64_t c_dims[3] = {n0, m, n};

  status = mluOpSetTensorDescriptor_v2(a_desc, MLUOP_LAYOUT_ARRAY, in_e_dtype,
                                       2, a_dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptorOnchipDataType(a_desc, in_e_dtype);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(b_desc, MLUOP_LAYOUT_ARRAY, in_e_dtype,
                                       3, b_dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptorOnchipDataType(b_desc, in_e_dtype);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(c_desc, MLUOP_LAYOUT_ARRAY, in_e_dtype,
                                       3, c_dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptorOnchipDataType(c_desc, in_e_dtype);
  CHECK_RETURN(api, status);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                    cnnl_handle);  // convert to cnnl_handle

  // convert to cnnl_tensor_descriptor
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);

  // c_desc->setOnchipDtype(MLUOP_DTYPE_FLOAT);
  c_desc->setOnchipDtype(in_e_dtype);
  float alpha = 1.0;
  float beta = 0.0;

  cnnlMatMulAlgo_t algo;
  CALL_CNNL(cnnlCreateMatMulAlgo(&algo));
  cnnlMatMulDescriptor_t bmm_bcast_desc;
  CALL_CNNL(cnnlCreateMatMulDescriptor(&bmm_bcast_desc));

  cnnlMatMulHeuristicResult_t heuristic_result;
  CALL_CNNL(cnnlCreateMatMulHeuristicResult(&heuristic_result));

  int requested_algo_count = 1, return_algo_count = 0;
  float *workspace;
  size_t workspace_size;
  cnnlGetBatchMatMulExAlgoHeuristic(
      cnnl_handle, bmm_bcast_desc, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc, NULL,
      requested_algo_count, &heuristic_result, &return_algo_count);

  cnnlGetBatchMatMulExHeuristicResult(heuristic_result, algo, &workspace_size);

  if (workspace_size > 0) {
    CNRT_CHECK(cnrtMalloc((void **)&workspace, workspace_size));
  } else {
    CNRT_CHECK(cnrtMalloc((void **)&workspace, m * n * sizeof(float)));
  }

  CALL_CNNL(cnnlBatchMatMulEx(cnnl_handle, bmm_bcast_desc, algo, &alpha,
                              cnnl_a_desc, dft_matrix_addr, cnnl_b_desc,
                              in_addr, &beta, cnnl_c_desc, out_addr,
                              (void *)workspace, workspace_size));

  status = mluOpDestroyTensorDescriptor(a_desc);
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(b_desc);
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(c_desc);
  CHECK_RETURN(api, status);
  // destroy cnnl descriptor
  CALL_CNNL(cnnlDestroyMatMulDescriptor(bmm_bcast_desc));
  CALL_CNNL(cnnlDestroyMatMulAlgo(algo));
  CALL_CNNL(cnnlDestroyMatMulHeuristicResult(heuristic_result));
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);

  DESTROY_CNNL_HANDLE(cnnl_handle);
  cnrtQueueSync(handle->queue);

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, &k_dim, &k_type);

  if (direction == FFT_FORWARD) {
    kernelFFTBatchConjMerge(
        k_dim, k_type, handle->queue, fft_plan->mlu_addrs.buffer_in,
        fft_plan->mlu_addrs.buffer_out, n1 * batch, n0, in_e_dtype);

  } else {
    kernelFFTBatchConjMerge(
        k_dim, k_type, handle->queue, fft_plan->mlu_addrs.output,
        fft_plan->mlu_addrs.buffer_out, n1 * batch, n0, in_e_dtype);
  }

  cnrtQueueSync(handle->queue);

  return status;
}
