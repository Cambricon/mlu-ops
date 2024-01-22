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
  INTERNAL_CHECK(api,
                 selectFFT1dStrategy(handle, fft_plan) == MLUOP_STATUS_SUCCESS);

  mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
  mluOpDataType_t in_r_dtype = (in_c_dtype == MLUOP_DTYPE_COMPLEX_HALF)
                                   ? MLUOP_DTYPE_HALF
                                   : MLUOP_DTYPE_FLOAT;
  mluOpDataType_t in_e_dtype = fft_plan->execution_dtype;
  size_t in_c_dtype_size = mluOpDataTypeBytes(in_c_dtype);
  size_t in_r_dtype_size = mluOpDataTypeBytes(in_r_dtype);
  int batch = fft_plan->batch;
  int n = fft_plan->n[0];

  switch (fft_plan->fft_strategy) {
    case CNFFT_FUNC_MATMUL: {
      if (n > FFT_L_LIMIT) {
        LOG(ERROR) << "[mluOpMakeFFTPlanMany]: FFT1d CNFFT_FUNC_MATMUL "
                   << "length > 4096 is not supported currently.";
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
      if (fftIsIntDtype(in_e_dtype)) {
        fft_plan->reservespace_size += sizeof(int32_t) + sizeof(float);
        size_t required_size = 0;
        status = fftGetQuantizeParamWorkspaceSize(
            handle, required_size, dft_mat_num, in_r_dtype, in_e_dtype, api);
        fft_plan->reservespace_size += required_size;
      }

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
      int in_trans_input_dims[in_trans_dim_num] = {padded_input_num, COMPLEX};
      int in_trans_permute[in_trans_dim_num] = {1, 0};
      size_t in_trans_workspace_size = 0;
      status = fftGetTransposeWorkspaceSize(
          handle, in_trans_workspace_size, in_trans_dim_num,
          in_trans_input_dims, in_trans_permute, in_r_dtype, api);
      fft_plan->matmul_addrs.internal_workspace_size =
          std::max(fft_plan->matmul_addrs.internal_workspace_size,
                   in_trans_workspace_size);

      // input quantize param and workspace
      if (fftIsIntDtype(in_e_dtype)) {
        fft_plan->workspace_size += sizeof(int32_t) + sizeof(float);
        size_t input_quant_workspace_size = 0;
        status = fftGetQuantizeParamWorkspaceSize(
            handle, input_quant_workspace_size, COMPLEX * padded_input_num,
            in_r_dtype, in_e_dtype, api);
        fft_plan->matmul_addrs.internal_workspace_size =
            std::max(fft_plan->matmul_addrs.internal_workspace_size,
                     input_quant_workspace_size);
      }

      // matmul output
      const int matmul_times =
          4;  // real mul real, real mul imag, imag mul real, imag mul imag
      int per_matmul_output_num = batch * n;
      size_t per_matmul_output_size = in_r_dtype_size * per_matmul_output_num;
      size_t matmul_output_size = matmul_times * per_matmul_output_size;
      fft_plan->workspace_size += matmul_output_size;
      // matmul workspace
      size_t matmul_workspace_size = 0;
      status = fftGetQuantizeMatMulWorkspaceSize(
          handle, matmul_workspace_size, batch, dim1, dim0, false, true,
          in_e_dtype, in_e_dtype, in_r_dtype, api);
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
      int out_trans_input_dims[out_trans_dim_num] = {COMPLEX,
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
      if (fftIsIntDtype(in_e_dtype)) {
        fft_plan->reservespace_size += sizeof(int32_t) + sizeof(float);
        size_t required_size = 0;
        status = fftGetQuantizeParamWorkspaceSize(
            handle, required_size, dft_mat_num, in_r_dtype, in_e_dtype, api);
        fft_plan->reservespace_size += required_size;
      }

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
      int in_trans_1st_input_dims[in_trans_1st_dim_num] = {padded_input_num,
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
      int in_trans_2nd_input_dims[in_trans_2nd_dim_num] = {COMPLEX * batch, L,
                                                           m};
      int in_trans_2nd_permute[in_trans_2nd_dim_num] = {0, 2, 1};
      size_t in_trans_2nd_workspace_size = 0;
      status = fftGetTransposeWorkspaceSize(
          handle, in_trans_2nd_workspace_size, in_trans_2nd_dim_num,
          in_trans_2nd_input_dims, in_trans_2nd_permute, in_r_dtype, api);
      fft_plan->matmul_addrs.internal_workspace_size =
          std::max(fft_plan->matmul_addrs.internal_workspace_size,
                   in_trans_2nd_workspace_size);

      // input quantize param and workspace
      if (fftIsIntDtype(in_e_dtype)) {
        fft_plan->workspace_size += sizeof(int32_t) + sizeof(float);
        size_t input_quant_workspace_size = 0;
        fftGetQuantizeParamWorkspaceSize(handle, input_quant_workspace_size,
                                         COMPLEX * padded_input_num, in_r_dtype,
                                         in_e_dtype, api);
        fft_plan->matmul_addrs.internal_workspace_size =
            std::max(fft_plan->matmul_addrs.internal_workspace_size,
                     input_quant_workspace_size);
      }

      // matmul output
      const int matmul_times =
          4;  // real mul real, real mul imag, imag mul real, imag mul imag
      int per_matmul_output_num = batch * n;
      size_t per_matmul_output_size = in_r_dtype_size * per_matmul_output_num;
      fft_plan->workspace_size += matmul_times * per_matmul_output_size;
      // matmul workspace
      size_t matmul_workspace_size = 0;
      status = fftGetQuantizeMatMulWorkspaceSize(
          handle, matmul_workspace_size, batch * m, L, L, false, true,
          in_e_dtype, in_e_dtype, in_r_dtype, api);
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
  if (fftIsIntDtype(fft_plan->execution_dtype)) {
    fft_plan->matmul_addrs.dft_pos_addr =
        (uint8_t *)fft_plan->reservespace_addr + dft_mat_size;
    fft_plan->matmul_addrs.dft_scale_addr =
        (uint8_t *)fft_plan->matmul_addrs.dft_pos_addr + sizeof(int32_t);
    fft_plan->matmul_addrs.dft_quantize_workspace_addr =
        (uint8_t *)fft_plan->matmul_addrs.dft_scale_addr + sizeof(float);
    fft_plan->matmul_addrs.dft_quantize_workspace_size =
        fft_plan->reservespace_size - dft_mat_size - sizeof(int32_t) -
        sizeof(float);
  } else {
    fft_plan->matmul_addrs.dft_pos_addr = nullptr;
    fft_plan->matmul_addrs.dft_scale_addr = nullptr;
    fft_plan->matmul_addrs.dft_quantize_workspace_addr = nullptr;
    fft_plan->matmul_addrs.dft_quantize_workspace_size = 0;
  }
}

mluOpStatus_t setFFT1dReserveArea(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                                  const std::string api) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
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
  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_BLOCK;

  switch (fft_plan->fft_strategy) {
    case CNFFT_FUNC_MATMUL: {
      // Matmul Matrix : 2 * 2 * [n, n] (forward and backward)
      int dft_mat_num = dft_mat_times * n * n;
      kernelC2CFFTDFTMatrix(k_dim, k_type, handle->queue, fft_plan, in_r_dtype,
                            n);
      status = fftQuantizePositionScale(
          handle, dft_mat_num, in_r_dtype, in_e_dtype,
          fft_plan->matmul_addrs.dft_matrix_addr,
          fft_plan->matmul_addrs.dft_pos_addr,
          fft_plan->matmul_addrs.dft_scale_addr,
          fft_plan->matmul_addrs.dft_quantize_workspace_addr,
          fft_plan->matmul_addrs.dft_quantize_workspace_size, api);
      INTERNAL_CHECK("[mluOpSetFFTReserveArea]",
                     status == MLUOP_STATUS_SUCCESS);
    }; break;
    case CNFFT_FUNC_COOLEY_TUKEY:
    case CNFFT_FUNC_STOCKHAM: {
      // Matmul Matrix : 2 * 2 * [L, L] (forward and backward)
      int L = fft_plan->L;
      int dft_mat_num = dft_mat_times * L * L;
      kernelC2CFFTDFTMatrix(k_dim, k_type, handle->queue, fft_plan, in_r_dtype,
                            L);
      status = fftQuantizePositionScale(
          handle, dft_mat_num, in_r_dtype, in_e_dtype,
          fft_plan->matmul_addrs.dft_matrix_addr,
          fft_plan->matmul_addrs.dft_pos_addr,
          fft_plan->matmul_addrs.dft_scale_addr,
          fft_plan->matmul_addrs.dft_quantize_workspace_addr,
          fft_plan->matmul_addrs.dft_quantize_workspace_size, api);
      INTERNAL_CHECK("[mluOpSetFFTReserveArea]",
                     status == MLUOP_STATUS_SUCCESS);
    }; break;
    default: {
      status = MLUOP_STATUS_NOT_SUPPORTED;
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

  // input quantize
  if (fftIsIntDtype(in_e_dtype)) {
    fft_plan->matmul_addrs.input_pos_addr =
        (uint8_t *)workspace + workspace_cur_offset;
    workspace_cur_offset += sizeof(int32_t);
    fft_plan->matmul_addrs.input_scale_addr =
        (uint8_t *)workspace + workspace_cur_offset;
    workspace_cur_offset += sizeof(float);
  } else {
    fft_plan->matmul_addrs.input_pos_addr = nullptr;
    fft_plan->matmul_addrs.input_scale_addr = nullptr;
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

// input    : in input
// output   : in input_contiguous_addr
static mluOpStatus_t makeFFT1dContiguousInput(mluOpHandle_t handle,
                                              mluOpFFTPlan_t fft_plan,
                                              const void *input) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into makeFFT1dContiguousInput";
  auto status = MLUOP_STATUS_SUCCESS;
  if (!fft_plan->is_input_contiguous) {
    VLOG(5) << "launch mluOpContiguous for fft1d input";
    mluOpTensorDescriptor_t input_desc;
    status = mluOpCreateTensorDescriptor(&input_desc);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    const int in_dim_num = 2;
    int64_t dims[in_dim_num] = {fft_plan->batch, fft_plan->inembed[0]};
    int64_t strides[in_dim_num] = {fft_plan->idist, fft_plan->istride};
    status = mluOpSetTensorDescriptorEx_v2(input_desc, MLUOP_LAYOUT_ARRAY,
                                           fft_plan->input_dtype, in_dim_num,
                                           dims, strides);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    status = mluOpContiguous(handle, input_desc, input,
                             fft_plan->matmul_addrs.input_contiguous_addr);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    status = mluOpDestroyTensorDescriptor(input_desc);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
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
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
    status = mluOpCreateTensorDescriptor(&padded_input_desc);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    const int in_dim_num = 2;
    int64_t dims[in_dim_num] = {batch, fft_plan->inembed[0] * COMPLEX};
    status = mluOpSetTensorDescriptor_v2(input_desc, MLUOP_LAYOUT_ARRAY,
                                         in_r_dtype, in_dim_num, dims);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    int64_t padded_dims[in_dim_num] = {batch, n * COMPLEX};
    status = mluOpSetTensorDescriptor_v2(padded_input_desc, MLUOP_LAYOUT_ARRAY,
                                         in_r_dtype, in_dim_num, padded_dims);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

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
                      fft_plan->matmul_addrs.input_contiguous_addr, paddings,
                      &padding_value, cnnl_padded_input_desc,
                      fft_plan->matmul_addrs.input_pad_addr));

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
    int trans_input_dims[trans_dim_num] = {padded_input_num, COMPLEX};
    int trans_output_dims[trans_dim_num] = {COMPLEX, padded_input_num};
    int trans_permute[trans_dim_num] = {1, 0};

    status =
        fftTranspose(handle, trans_dim_num, trans_input_dims, trans_output_dims,
                     trans_permute, fft_plan->matmul_addrs.input_pad_addr,
                     fft_plan->matmul_addrs.input_re_addr, in_r_dtype,
                     fft_plan->matmul_addrs.internal_workspace_addr,
                     fft_plan->matmul_addrs.internal_workspace_size, api);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  } else if (fft_plan->fft_strategy == CNFFT_FUNC_COOLEY_TUKEY) {
    VLOG(5) << "launch mluOpTranspose for input CNFFT_FUNC_COOLEY_TUKEY";
    int L = fft_plan->L;
    int m = (1 << fft_plan->m);

    // 1st transpose: batch * n * 2 --> 2 * batch * n
    int padded_input_num = batch * n;
    const int trans_dim_num = 2;
    int trans_input_dims[trans_dim_num] = {padded_input_num, COMPLEX};
    int trans_output_dims[trans_dim_num] = {COMPLEX, padded_input_num};
    int trans_permute[trans_dim_num] = {1, 0};

    status =
        fftTranspose(handle, trans_dim_num, trans_input_dims, trans_output_dims,
                     trans_permute, fft_plan->matmul_addrs.input_pad_addr,
                     fft_plan->matmul_addrs.input_transed_addr, in_r_dtype,
                     fft_plan->matmul_addrs.internal_workspace_addr,
                     fft_plan->matmul_addrs.internal_workspace_size, api);

    // 2nd transpose: 2 * batch * L * 2^m --> 2 * batch * 2^m * L
    const int trans_2nd_dim_num = 3;
    int trans_2nd_input_dims[trans_2nd_dim_num] = {COMPLEX * batch, L, m};
    int trans_2nd_output_dims[trans_2nd_dim_num] = {COMPLEX * batch, m, L};
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

// input    : in input_pad_addr
// output   : in input_pos_addr and input_scale_addr
static mluOpStatus_t quantizeFFT1dPaddedInput(mluOpHandle_t handle,
                                              mluOpFFTPlan_t fft_plan) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into quantizeFFT1dPaddedInput";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
  mluOpDataType_t in_r_dtype = (in_c_dtype == MLUOP_DTYPE_COMPLEX_HALF)
                                   ? MLUOP_DTYPE_HALF
                                   : MLUOP_DTYPE_FLOAT;
  mluOpDataType_t in_e_dtype = fft_plan->execution_dtype;
  int padded_input_num = fft_plan->batch * fft_plan->n[0];

  status = fftQuantizePositionScale(
      handle, COMPLEX * padded_input_num, in_r_dtype, in_e_dtype,
      fft_plan->matmul_addrs.input_re_addr,
      fft_plan->matmul_addrs.input_pos_addr,
      fft_plan->matmul_addrs.input_scale_addr,
      fft_plan->matmul_addrs.internal_workspace_addr,
      fft_plan->matmul_addrs.internal_workspace_size, api);

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
                                              int direction) {
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
    status = fftQuantMatMul(
        handle, batch, n, n, fft_plan->matmul_addrs.input_re_addr,
        fft_plan->matmul_addrs.input_pos_addr,
        fft_plan->matmul_addrs.input_scale_addr, dft_re_matrix_addr,
        fft_plan->matmul_addrs.dft_pos_addr,
        fft_plan->matmul_addrs.dft_scale_addr,
        fft_plan->matmul_addrs.matmul_re_mul_re_addr, false, true, scale_factor,
        0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    // input imag matmul dft imag
    status = fftQuantMatMul(
        handle, batch, n, n, fft_plan->matmul_addrs.input_im_addr,
        fft_plan->matmul_addrs.input_pos_addr,
        fft_plan->matmul_addrs.input_scale_addr, dft_im_matrix_addr,
        fft_plan->matmul_addrs.dft_pos_addr,
        fft_plan->matmul_addrs.dft_scale_addr,
        fft_plan->matmul_addrs.matmul_im_mul_im_addr, false, true, scale_factor,
        0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    // input real matmul dft imag
    status = fftQuantMatMul(
        handle, batch, n, n, fft_plan->matmul_addrs.input_re_addr,
        fft_plan->matmul_addrs.input_pos_addr,
        fft_plan->matmul_addrs.input_scale_addr, dft_im_matrix_addr,
        fft_plan->matmul_addrs.dft_pos_addr,
        fft_plan->matmul_addrs.dft_scale_addr,
        fft_plan->matmul_addrs.matmul_re_mul_im_addr, false, true, scale_factor,
        0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    // input imag matmul dft real
    status = fftQuantMatMul(
        handle, batch, n, n, fft_plan->matmul_addrs.input_im_addr,
        fft_plan->matmul_addrs.input_pos_addr,
        fft_plan->matmul_addrs.input_scale_addr, dft_re_matrix_addr,
        fft_plan->matmul_addrs.dft_pos_addr,
        fft_plan->matmul_addrs.dft_scale_addr,
        fft_plan->matmul_addrs.matmul_im_mul_re_addr, false, true, scale_factor,
        0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    // real mul real sub imag mul imag
    int per_matmul_output_num = batch * n;
    status = fftOptensor(handle, per_matmul_output_num,
                         fft_plan->matmul_addrs.matmul_re_mul_re_addr,
                         fft_plan->matmul_addrs.matmul_im_mul_im_addr,
                         fft_plan->matmul_addrs.matmul_re_mul_re_addr, 1.0, 1.0,
                         0.0, in_r_dtype, CNNL_OP_TENSOR_SUB,
                         fft_plan->matmul_addrs.internal_workspace_addr,
                         fft_plan->matmul_addrs.internal_workspace_size, api);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    // real mul imag add imag mul real
    status = fftOptensor(handle, per_matmul_output_num,
                         fft_plan->matmul_addrs.matmul_re_mul_im_addr,
                         fft_plan->matmul_addrs.matmul_im_mul_re_addr,
                         fft_plan->matmul_addrs.matmul_re_mul_im_addr, 1.0, 1.0,
                         0.0, in_r_dtype, CNNL_OP_TENSOR_ADD,
                         fft_plan->matmul_addrs.internal_workspace_addr,
                         fft_plan->matmul_addrs.internal_workspace_size, api);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  } else if (fft_plan->fft_strategy == CNFFT_FUNC_COOLEY_TUKEY) {
    int L = fft_plan->L;
    int m = (1 << fft_plan->m);

    // input real matmul dft real
    status = fftQuantMatMul(
        handle, batch * m, L, L, fft_plan->matmul_addrs.input_re_addr,
        fft_plan->matmul_addrs.input_pos_addr,
        fft_plan->matmul_addrs.input_scale_addr, dft_re_matrix_addr,
        fft_plan->matmul_addrs.dft_pos_addr,
        fft_plan->matmul_addrs.dft_scale_addr,
        fft_plan->matmul_addrs.matmul_re_mul_re_addr, false, true, scale_factor,
        0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    // input imag matmul dft imag
    status = fftQuantMatMul(
        handle, batch * m, L, L, fft_plan->matmul_addrs.input_im_addr,
        fft_plan->matmul_addrs.input_pos_addr,
        fft_plan->matmul_addrs.input_scale_addr, dft_im_matrix_addr,
        fft_plan->matmul_addrs.dft_pos_addr,
        fft_plan->matmul_addrs.dft_scale_addr,
        fft_plan->matmul_addrs.matmul_im_mul_im_addr, false, true, scale_factor,
        0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    // input real matmul dft imag
    status = fftQuantMatMul(
        handle, batch * m, L, L, fft_plan->matmul_addrs.input_re_addr,
        fft_plan->matmul_addrs.input_pos_addr,
        fft_plan->matmul_addrs.input_scale_addr, dft_im_matrix_addr,
        fft_plan->matmul_addrs.dft_pos_addr,
        fft_plan->matmul_addrs.dft_scale_addr,
        fft_plan->matmul_addrs.matmul_re_mul_im_addr, false, true, scale_factor,
        0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    // input imag matmul dft real
    status = fftQuantMatMul(
        handle, batch * m, L, L, fft_plan->matmul_addrs.input_im_addr,
        fft_plan->matmul_addrs.input_pos_addr,
        fft_plan->matmul_addrs.input_scale_addr, dft_re_matrix_addr,
        fft_plan->matmul_addrs.dft_pos_addr,
        fft_plan->matmul_addrs.dft_scale_addr,
        fft_plan->matmul_addrs.matmul_im_mul_re_addr, false, true, scale_factor,
        0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  } else if (fft_plan->fft_strategy == CNFFT_FUNC_STOCKHAM) {
    int L = fft_plan->L;
    int m = (1 << fft_plan->m);

    // origin: in_trans[batch, 2^m, L] * W_real[L, L] -> IN_real[batch, 2^m, L]
    //         in_trans[batch, 2^m, L] * W_imag[L, L] -> IN_imag[batch, 2^m, L]
    // update: W[c*L, L] * in[batch, L, 2^m] -> out[batch, c*L, 2^m]
    status = fftBatchMatMulBcast(
        handle, 2 * L, L, m * 2, batch, dft_re_matrix_addr,
        fft_plan->matmul_addrs.dft_pos_addr,
        fft_plan->matmul_addrs.dft_scale_addr,
        fft_plan->matmul_addrs.input_pad_addr,
        fft_plan->matmul_addrs.input_pos_addr,
        fft_plan->matmul_addrs.input_scale_addr,
        fft_plan->matmul_addrs.matmul_re_mul_re_addr, false, false,
        scale_factor, 0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
  }

  return status;
}

static mluOpStatus_t policyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                                cnrtFunctionType_t *k_type) {
  *k_type = CNRT_FUNC_TYPE_UNION1;
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
    cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
    int task_type = mluop::runtime::getJobLimitCapability(handle);
    int task_num = 1;

    switch (task_type) {
      default:
        task_num = core_num;
        break;
      case (int)CNRT_FUNC_TYPE_UNION2:
        task_num = core_num * 2;
        break;
      case (int)CNRT_FUNC_TYPE_UNION4:
        task_num = core_num * 4;
        break;
      case (int)CNRT_FUNC_TYPE_UNION8:
        task_num = core_num * 8;
        break;
      case (int)CNRT_FUNC_TYPE_UNION16:
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
    int trans_input_dims[trans_dim_num] = {COMPLEX, output_num};
    int trans_output_dims[trans_dim_num] = {output_num, COMPLEX};
    int trans_permute[trans_dim_num] = {1, 0};

    status = fftTranspose(
        handle, trans_dim_num, trans_input_dims, trans_output_dims,
        trans_permute, fft_plan->matmul_addrs.matmul_re_mul_re_addr,
        fft_plan->matmul_addrs.output_contiguous_addr, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  }
  return status;
}

static mluOpStatus_t makeFFT1dContiguousOutput(mluOpHandle_t handle,
                                               mluOpFFTPlan_t fft_plan,
                                               void *output) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into makeFFT1dContiguousOutput";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  if (!fft_plan->is_output_contiguous) {
    VLOG(5) << "launch copy with stride";
    mluOpDataType_t out_c_dtype = fft_plan->output_dtype;
    // create tensor desc
    mluOpTensorDescriptor_t copy_src_desc, copy_dst_desc;
    status = mluOpCreateTensorDescriptor(&copy_src_desc);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
    status = mluOpCreateTensorDescriptor(&copy_dst_desc);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    // set up tensor desc
    const int out_dim_num = 2;
    int64_t dims[out_dim_num] = {fft_plan->batch, fft_plan->onembed[0]};
    int64_t strides[out_dim_num] = {fft_plan->odist, fft_plan->ostride};
    status = mluOpSetTensorDescriptor_v2(copy_src_desc, MLUOP_LAYOUT_ARRAY,
                                         out_c_dtype, out_dim_num, dims);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
    status =
        mluOpSetTensorDescriptorEx_v2(copy_dst_desc, MLUOP_LAYOUT_ARRAY,
                                      out_c_dtype, out_dim_num, dims, strides);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    void *copy_src_addr = fft_plan->matmul_addrs.output_contiguous_addr;
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                      cnnl_handle);  // convert to cnnl_handle
    // convert to cnnl_tensor_descriptor
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(copy_src_desc,
                                                 cnnl_copy_src_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(copy_dst_desc,
                                                 cnnl_copy_dst_desc);

    CALL_CNNL(cnnlCopy(cnnl_handle, cnnl_copy_src_desc, copy_src_addr,
                       cnnl_copy_dst_desc, output));

    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_copy_src_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_copy_dst_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  return status;
}

mluOpStatus_t execFFT1d(mluOpHandle_t handle, const mluOpFFTPlan_t fft_plan,
                        const void *input, const float scale_factor,
                        void *workspace, void *output, int direction) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  std::string api = "[mluOpExecFFT]";
  configureFFT1dMatmulWorkspaceAddrs(handle, fft_plan, (void *)input, workspace,
                                     output);

  status = makeFFT1dContiguousInput(handle, fft_plan, input);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

  status = padFFT1dContiguousInput(handle, fft_plan);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

  status = transposeFFT1dPaddedInput(handle, fft_plan);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

  status = quantizeFFT1dPaddedInput(handle, fft_plan);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

  status = computeFFT1dMatmulResult(handle, fft_plan, scale_factor, direction);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

  status = mergeFFT1dOutput(handle, fft_plan, scale_factor, direction);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

  status = transposeFFT1dOutput(handle, fft_plan);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

  status = makeFFT1dContiguousOutput(handle, fft_plan, output);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

  return status;
}
