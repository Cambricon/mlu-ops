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

#include "kernels/fft/rfft/rfft.h"
#include <algorithm>
#include <string>

static mluOpStatus_t selectRFFT1dStrategy(mluOpHandle_t handle,
                                          mluOpFFTPlan_t fft_plan) {
  const std::string make_plan_api = "[selectRFFT1dStrategy]";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  /* there are plenty of algorithms for FFT, depending on the fft length.
   * Iterative FFT:
   *   Stockham FFT, Cooley-Tukey FFT, peaseFFT, Kron-Lambiotte FFT
   * Recursive FFT:
   *   Recursive Cooley-Tukey FFT, Four-step FFT, Six-step FFT, Multicore FFT,
   * SIMD short vector FFT. General FFT: chirp-Z Bluestein FFT.
   */
  // select Four-Step FFT or MATMUL strategy logic
  fft_plan->fft_strategy = CNFFT_FUNC_MATMUL;
  status = selectFFTStrategy(handle, fft_plan, make_plan_api);
  return status;
}

/*
 * Make the policy of RFFT1d.
 */
mluOpStatus_t makeRFFT1dPolicy(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan) {
  std::string api = "[mluOpMakeFFTPlanMany]";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  INTERNAL_CHECK(
      api, selectRFFT1dStrategy(handle, fft_plan) == MLUOP_STATUS_SUCCESS);

  mluOpDataType_t in_r_dtype = fft_plan->input_dtype;
  mluOpDataType_t in_e_dtype = fft_plan->execution_dtype;
  size_t in_r_dtype_size = mluOpDataTypeBytes(in_r_dtype);
  int batch = fft_plan->batch;
  int n = fft_plan->n[0];

  switch (fft_plan->fft_strategy) {
    case CNFFT_FUNC_MATMUL: {
      if (n > FFT_L_LIMIT) {
        LOG(ERROR) << "[mluOpMakeFFTPlanMany]: RFFT1d CNFFT_FUNC_MATMUL "
                   << "length > 4096 is not supported currently.";
        return MLUOP_STATUS_NOT_SUPPORTED;
      }

      // Matmul Input  : [batch, n]
      // Matmul Matrix : [(n / 2 + 1), 2, n]
      // Matmul Result : [batch, (n / 2 + 1), 2]
      int dim0 = FFT_HALF(n);
      int dim1 = COMPLEX;  // complex
      int dim2 = n;
      int dft_mat_num = dim0 * dim1 * dim2;

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
                    | matmul
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
      size_t input_size = in_r_dtype_size * fft_plan->inum;
      fft_plan->workspace_size +=
          fft_plan->is_input_contiguous ? 0 : input_size;

      // input pad
      bool need_pad = (fft_plan->inembed[0] != n);
      int padded_input_num = batch * n;
      size_t padded_input_size = in_r_dtype_size * padded_input_num;
      fft_plan->workspace_size += need_pad ? padded_input_size : 0;

      // input quantize param and workspace
      if (fftIsIntDtype(in_e_dtype)) {
        fft_plan->workspace_size += sizeof(int32_t) + sizeof(float);
        size_t input_quant_workspace_size = 0;
        status = fftGetQuantizeParamWorkspaceSize(
            handle, input_quant_workspace_size, padded_input_num, in_r_dtype,
            in_e_dtype, api);
        fft_plan->matmul_addrs.internal_workspace_size =
            std::max(fft_plan->matmul_addrs.internal_workspace_size,
                     input_quant_workspace_size);
      }

      // matmul workspace
      size_t matmul_workspace_size = 0;
      status = fftGetQuantizeMatMulWorkspaceSize(
          handle, matmul_workspace_size, batch, dim2, dim0 * dim1, false, true,
          in_e_dtype, in_e_dtype, in_r_dtype, api);
      fft_plan->matmul_addrs.internal_workspace_size =
          std::max(fft_plan->matmul_addrs.internal_workspace_size,
                   matmul_workspace_size);

      // output contiguous
      int padded_output_num = batch * FFT_HALF(n);
      size_t padded_output_size =
          mluOpDataTypeBytes(fft_plan->output_dtype) * padded_output_num;
      fft_plan->workspace_size +=
          fft_plan->is_output_contiguous ? 0 : padded_output_size;

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
        LOG(ERROR) << "[mluOpMakeFFTPlanMany]: RFFT1d CNFFT_FUNC_COOLEY_TUKEY "
                   << "n = L * 2^m and L > 4096 is not supported currently.";
        return MLUOP_STATUS_NOT_SUPPORTED;
      }

      // Matmul Input  : [batch, 2^m, L]
      // Matmul Matrix : 2 * [L, L]
      // Matmul Result : 2 * [batch, 2^m, L]
      int dft_mat_times = COMPLEX;
      int dim0 = L;
      int dim1 = L;
      int dft_mat_num = dft_mat_times * dim0 * dim1;

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
                    | input trans: batch * L * 2^m --> batch * 2^m * L
                   \|/
         -------------------------
         |    input_transed      |
         -------------------------
                    |
                    | matmul
                   \|/
         -------------------------
         |     matmul_re_mul_re   |
         |     matmul_re_mul_im   |
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
      size_t input_size = in_r_dtype_size * fft_plan->inum;
      fft_plan->workspace_size +=
          fft_plan->is_input_contiguous ? 0 : input_size;

      // input pad
      bool need_pad = (fft_plan->inembed[0] != n);
      int padded_input_num = batch * n;
      size_t padded_input_size = in_r_dtype_size * padded_input_num;
      fft_plan->workspace_size += need_pad ? padded_input_size : 0;

      // input trans
      size_t transed_input_size = padded_input_size;
      fft_plan->workspace_size += transed_input_size;
      // input trans workspace: batch * L * 2^m --> batch * 2^m * L
      const int trans_dim_num = 3;
      int64_t trans_input_dims[trans_dim_num] = {batch, L, m};
      int trans_permute[trans_dim_num] = {0, 2, 1};
      size_t trans_workspace_size = 0;
      status = fftGetTransposeWorkspaceSize(handle, trans_workspace_size,
                                            trans_dim_num, trans_input_dims,
                                            trans_permute, in_r_dtype, api);
      fft_plan->matmul_addrs.internal_workspace_size = std::max(
          fft_plan->matmul_addrs.internal_workspace_size, trans_workspace_size);

      // input quantize param and workspace
      if (fftIsIntDtype(in_e_dtype)) {
        fft_plan->workspace_size += sizeof(int32_t) + sizeof(float);
        size_t input_quant_workspace_size = 0;
        status = fftGetQuantizeParamWorkspaceSize(
            handle, input_quant_workspace_size, padded_input_num, in_r_dtype,
            in_e_dtype, api);
        fft_plan->matmul_addrs.internal_workspace_size =
            std::max(fft_plan->matmul_addrs.internal_workspace_size,
                     input_quant_workspace_size);
      }

      // matmul output
      int matmul_times = COMPLEX;  // real and imag
      int per_matmul_output_num = batch * n;
      size_t matmul_output_size =
          matmul_times * in_r_dtype_size * per_matmul_output_num;
      fft_plan->workspace_size += matmul_output_size;
      // matmul workspace
      size_t matmul_workspace_size = 0;
      if (fft_plan->fft_strategy == CNFFT_FUNC_COOLEY_TUKEY) {
        status = fftGetQuantizeMatMulWorkspaceSize(
            handle, matmul_workspace_size, batch * m, L, L, false, true,
            in_e_dtype, in_e_dtype, in_r_dtype, api);
        fft_plan->matmul_addrs.internal_workspace_size =
            std::max(fft_plan->matmul_addrs.internal_workspace_size,
                     matmul_workspace_size);
      } else {
        status = fftGetBatchMatMulBcastWorkspaceSize(
            handle,
            L <= fft_plan->L_sub ? (2 * L)
                                 : (2 * (PAD_UP(L / 2, fft_plan->L_sub) + 1)),
            L, m, batch, fft_plan->matmul_addrs.dft_re_matrix_addr,
            fft_plan->matmul_addrs.dft_pos_addr,
            fft_plan->matmul_addrs.dft_scale_addr,
            fft_plan->matmul_addrs.input_pad_addr,
            fft_plan->matmul_addrs.input_pos_addr,
            fft_plan->matmul_addrs.input_scale_addr,
            fft_plan->matmul_addrs.matmul_re_mul_re_addr, false, false, 1.0,
            0.0, in_e_dtype, in_e_dtype, in_r_dtype,
            fft_plan->matmul_addrs.internal_workspace_addr,
            fft_plan->matmul_addrs.internal_workspace_size, api);
      }

      // output merge workspace
      size_t merge_workspace_size = matmul_output_size;
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

static void configureRFFT1dMatmulReserveAddrs(mluOpHandle_t handle,
                                              mluOpFFTPlan_t fft_plan) {
  size_t dft_mat_size = 0;
  mluOpDataType_t in_r_dtype = fft_plan->input_dtype;
  mluOpDataType_t in_e_dtype = fft_plan->execution_dtype;
  size_t in_r_dtype_size = mluOpDataTypeBytes(in_r_dtype);
  int n = fft_plan->n[0];

  switch (fft_plan->fft_strategy) {
    case CNFFT_FUNC_MATMUL: {
      // Matmul Matrix : [(n / 2 + 1), 2, n]
      int dim0 = FFT_HALF(n);
      int dim1 = COMPLEX;
      int dim2 = n;
      dft_mat_size = dim0 * dim1 * dim2 * in_r_dtype_size;
      fft_plan->matmul_addrs.dft_matrix_addr = fft_plan->reservespace_addr;
    }; break;
    case CNFFT_FUNC_COOLEY_TUKEY:
    case CNFFT_FUNC_STOCKHAM: {
      // Matmul Matrix : 2 * [L, L]
      int L = fft_plan->L;
      int dft_mat_times = COMPLEX;
      size_t per_dft_mat_size = L * L * in_r_dtype_size;
      dft_mat_size = dft_mat_times * per_dft_mat_size;
      fft_plan->matmul_addrs.dft_matrix_addr = fft_plan->reservespace_addr;
      fft_plan->matmul_addrs.dft_re_matrix_addr = fft_plan->reservespace_addr;
      fft_plan->matmul_addrs.dft_im_matrix_addr =
          (uint8_t *)fft_plan->reservespace_addr + per_dft_mat_size;
    }; break;
    default: {
      break;
    }
  }
  if (fftIsIntDtype(in_e_dtype)) {
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

static void configureRFFT1dWorkspaceAddrs(mluOpHandle_t handle,
                                          mluOpFFTPlan_t fft_plan, void *input,
                                          void *workspace, void *output) {
  VLOG(5) << "Into configure FFT1d Workspace Addrs";
  const std::string make_plan_api = "[configureFFT1dWorkspaceAddrs]";
  size_t workspace_size = 0;
  size_t reservespace_size = 0;

  mluOpDataType_t out_c_dtype = fft_plan->output_dtype;

  size_t out_c_dtype_size = mluOpDataTypeBytes(out_c_dtype);

  int batch = fft_plan->batch;
  int nfft = fft_plan->n[0];

  size_t buffer_size = batch * out_c_dtype_size * nfft;

  size_t offset = 0;
  fft_plan->mlu_addrs.buffer_buf = (uint8_t *)workspace + offset;
  offset += buffer_size * 2;

  if (fft_plan->is_input_contiguous && fft_plan->inembed[0] <= fft_plan->n[0]) {
    fft_plan->mlu_addrs.input = input;
  } else {
    fft_plan->mlu_addrs.input = (uint8_t *)workspace + offset;
    offset += buffer_size;
  }

  if (fft_plan->is_output_contiguous) {
    fft_plan->mlu_addrs.output = output;
  } else {
    fft_plan->mlu_addrs.output = (uint8_t *)workspace + offset;
    offset += buffer_size;
  }
  if (fft_plan->n[0] > fft_plan->inembed[0]) {
    fft_plan->mlu_addrs.input_pad_addr = (uint8_t *)workspace + offset;
  }
}

static void configureRFFT2dWorkspaceAddrs(mluOpHandle_t handle,
                                          mluOpFFTPlan_t fft_plan, void *input,
                                          void *workspace, void *output) {
  const std::string make_plan_api = "[configureFFT2dWorkspaceAddrs]";

  mluOpDataType_t out_c_dtype = fft_plan->output_dtype;

  size_t out_c_dtype_size = mluOpDataTypeBytes(out_c_dtype);

  int batch = fft_plan->batch;
  int _n0 = fft_plan->n[0];
  int _n1 = fft_plan->n[1];

  size_t offset = 0;
  if (fft_plan->fft_strategy == CNFFT_FUNC_MANY_DIST1_2D) {
    // rr ri ir ii
    size_t buffer_size = batch * out_c_dtype_size * _n0 * _n1 * 2;
    fft_plan->mlu_addrs.input = input;
    fft_plan->mlu_addrs.output = output;
    fft_plan->mlu_addrs.buffer_in = (uint8_t *)workspace + offset;
    offset += buffer_size;
    fft_plan->mlu_addrs.buffer_out = (uint8_t *)workspace + offset;
    offset += buffer_size;
  }

  if (fft_plan->fft_strategy == CNFFT_FUNC_TWO_LEVEL_STOCKHAM) {
    fft_plan->mlu_addrs.buffer_buf = (uint8_t *)workspace + offset;
    offset += batch * out_c_dtype_size * _n0 * _n1 * 2;

    if (fft_plan->is_input_contiguous) {
      fft_plan->mlu_addrs.input = input;
    } else {
      fft_plan->mlu_addrs.input = (uint8_t *)workspace + offset;
      offset += batch * out_c_dtype_size * _n0 * _n1;
    }

    if (fft_plan->is_output_contiguous) {
      fft_plan->mlu_addrs.output = output;
    } else {
      fft_plan->mlu_addrs.output = (uint8_t *)workspace + offset;
      offset += batch * out_c_dtype_size * _n0 * _n1;
    }
  }

  if (fft_plan->n[0] > fft_plan->inembed[0] ||
      fft_plan->n[1] > fft_plan->inembed[1]) {
    fft_plan->mlu_addrs.input_pad_addr = (uint8_t *)workspace + offset;
  }
}

mluOpStatus_t setRFFT1dReserveArea(mluOpHandle_t handle,
                                   mluOpFFTPlan_t fft_plan,
                                   const std::string api) {
  VLOG(5) << "setRFFT1dReserveArea";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  if (fft_plan->prime) {
    configureRFFT1dMatmulReserveAddrs(handle, fft_plan);

    mluOpDataType_t in_r_dtype = fft_plan->input_dtype;
    mluOpDataType_t in_e_dtype = fft_plan->execution_dtype;
    int n = fft_plan->n[0];

    const unsigned int cluster_number =
        mluop::runtime::getClusterLimitCapability(handle);
    const unsigned int core_dim = handle->core_num_per_cluster;
    cnrtDim3_t k_dim = {core_dim, cluster_number, 1};
    cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_BLOCK;

    switch (fft_plan->fft_strategy) {
      case CNFFT_FUNC_MATMUL: {
        // Matmul Matrix : [(n / 2 + 1), 2, n]
        int dim0 = FFT_HALF(n);
        int dim1 = COMPLEX;
        int dim2 = n;
        int dft_mat_num = dim0 * dim1 * dim2;
        kernelGenerateRFFTHalfDFTMatrix(k_dim, k_type, handle->queue, fft_plan,
                                        in_r_dtype, n);
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
      case CNFFT_FUNC_COOLEY_TUKEY: {
        // Matmul Matrix : 2 * [L, L]
        int L = fft_plan->L;
        int dft_mat_times = COMPLEX;
        int dft_mat_num = dft_mat_times * L * L;
        kernelGenerateRFFTFullDFTMatrix(k_dim, k_type, handle->queue, fft_plan,
                                        in_r_dtype, L, L);
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
      case CNFFT_FUNC_STOCKHAM: {
        // Matmul Matrix : 2 * [L, L]
        int L = fft_plan->L;
        int row =
            L <= fft_plan->L_sub ? L : (PAD_UP(L / 2, fft_plan->L_sub) + 1);
        int dft_mat_times = COMPLEX;
        int dft_mat_num = dft_mat_times * L * L;
        VLOG(5) << "CNFFT_FUNC_STOCKHAM generateRFFTFullDFTMatrix";
        kernelGenerateRFFTFullDFTMatrix(k_dim, k_type, handle->queue, fft_plan,
                                        in_r_dtype, row, L);
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

  } else {
    mluOpDataType_t out_c_dtype = fft_plan->output_dtype;
    size_t out_c_dtype_size = mluOpDataTypeBytes(out_c_dtype);

    int nfft = fft_plan->n[0];
    size_t twiddles_size = out_c_dtype_size * nfft * 2;

    size_t factors_size = FFT_MAXFACTORS * sizeof(int);  // bytes
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

    fft_plan->mlu_addrs.factors =
        (int *)((uint8_t *)fft_plan->reservespace_addr + reservespace_offset);
    reservespace_offset += factors_size;

    // CNRT_CHECK(cnrtMemcpy(fft_plan->mlu_addrs.factors, fft_plan->factors,
    //                      FFT_MAXFACTORS * sizeof(int), cnrtMemcpyHostToDev));
    // CNRT_CHECK(cnrtMemcpy(fft_plan->mlu_addrs.twiddles, fft_plan->twiddles,
    //                      twiddles_size, cnrtMemcpyHostToDev));
    // CNRT_CHECK(cnrtMemcpy(fft_plan->mlu_addrs.dft_matrix,
    // fft_plan->dft_matrix,
    //                      DFT_TABLE_SIZE, cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.factors, fft_plan->factors,
                               FFT_MAXFACTORS * sizeof(int), handle->queue,
                               cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.twiddles, fft_plan->twiddles,
                               twiddles_size, handle->queue,
                               cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpyAsync(fft_plan->mlu_addrs.dft_matrix,
                               fft_plan->dft_matrix, DFT_TABLE_SIZE,
                               handle->queue, cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtFreeHost(fft_plan->factors));
    CNRT_CHECK(cnrtFreeHost(fft_plan->twiddles));
    CNRT_CHECK(cnrtFreeHost(fft_plan->dft_matrix));
  }
  return status;
}

static void configureRFFT1dMatmulWorkspaceAddrs(mluOpHandle_t handle,
                                                mluOpFFTPlan_t fft_plan,
                                                void *input, void *workspace,
                                                void *output) {
  VLOG(5) << "Into configure RFFT1d Matmul Workspace Addrs";
  size_t workspace_cur_offset = 0;
  size_t workspace_cur_offset_to_end = 0;
  size_t workspace_total_size = fft_plan->workspace_size;
  void *workspace_end = (uint8_t *)workspace + workspace_total_size;

  mluOpDataType_t in_r_dtype = fft_plan->input_dtype;
  mluOpDataType_t in_e_dtype = fft_plan->execution_dtype;
  size_t in_r_dtype_size = mluOpDataTypeBytes(in_r_dtype);
  int batch = fft_plan->batch;
  int n = fft_plan->n[0];

  // input contiguous
  size_t input_size = in_r_dtype_size * fft_plan->inum;
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
  size_t padded_input_size = in_r_dtype_size * padded_input_num;
  if (need_pad) {
    fft_plan->matmul_addrs.input_pad_addr =
        (uint8_t *)workspace + workspace_cur_offset;
    workspace_cur_offset += padded_input_size;
  } else {
    fft_plan->matmul_addrs.input_pad_addr =
        fft_plan->matmul_addrs.input_contiguous_addr;
  }

  // input trans
  if (fft_plan->fft_strategy == CNFFT_FUNC_COOLEY_TUKEY ||
      fft_plan->fft_strategy == CNFFT_FUNC_STOCKHAM) {
    fft_plan->matmul_addrs.input_transed_addr =
        (uint8_t *)workspace + workspace_cur_offset;
    workspace_cur_offset += padded_input_size;
  } else {
    fft_plan->matmul_addrs.input_transed_addr =
        fft_plan->matmul_addrs.input_pad_addr;
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
  if (fft_plan->fft_strategy == CNFFT_FUNC_COOLEY_TUKEY ||
      fft_plan->fft_strategy == CNFFT_FUNC_STOCKHAM) {
    int per_matmul_output_num = batch * n;
    size_t per_matmul_output_size = in_r_dtype_size * per_matmul_output_num;
    workspace_cur_offset_to_end += per_matmul_output_size;
    fft_plan->matmul_addrs.matmul_re_mul_im_addr =
        (uint8_t *)workspace_end - workspace_cur_offset_to_end;
    workspace_cur_offset_to_end += per_matmul_output_size;
    fft_plan->matmul_addrs.matmul_re_mul_re_addr =
        (uint8_t *)workspace_end - workspace_cur_offset_to_end;
  } else {
    fft_plan->matmul_addrs.matmul_re_mul_im_addr = nullptr;
    fft_plan->matmul_addrs.matmul_re_mul_re_addr = nullptr;
  }
}

// input    : in input
// output   : in input_contiguous_addr
static mluOpStatus_t makeRFFT1dContiguousInput(mluOpHandle_t handle,
                                               mluOpFFTPlan_t fft_plan,
                                               const void *input) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into makeRFFT1dContiguousInput";
  auto status = MLUOP_STATUS_SUCCESS;
  if (!fft_plan->is_input_contiguous ||
      (!fft_plan->prime && fft_plan->inembed[0] > fft_plan->n[0])) {
    VLOG(5) << "launch mluOpContiguous";
    mluOpTensorDescriptor_t input_desc;
    status = mluOpCreateTensorDescriptor(&input_desc);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    const int in_dim_num = 2;
    if (fft_plan->prime) {
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

    } else {
      int64_t dims[in_dim_num] = {
          fft_plan->batch, std::min(fft_plan->inembed[0], fft_plan->n[0])};
      int64_t strides[in_dim_num] = {fft_plan->idist, fft_plan->istride};
      status = mluOpSetTensorDescriptorEx_v2(input_desc, MLUOP_LAYOUT_ARRAY,
                                             fft_plan->input_dtype, in_dim_num,
                                             dims, strides);
      INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

      status =
          mluOpContiguous(handle, input_desc, input, fft_plan->mlu_addrs.input);
      INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

      status = mluOpDestroyTensorDescriptor(input_desc);
      INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
    }
  }
  return status;
}

// input    : in input_contiguous_addr
// output   : in input_pad_addr
static mluOpStatus_t padRFFT1dContiguousInput(mluOpHandle_t handle,
                                              mluOpFFTPlan_t fft_plan) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into padRFFT1dContiguousInput";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  mluOpDataType_t in_r_dtype = fft_plan->input_dtype;
  int batch = fft_plan->batch;
  int n = fft_plan->n[0];
  bool need_pad = (fft_plan->inembed[0] != n);
  if (need_pad) {
    mluOpTensorDescriptor_t input_desc, padded_input_desc;
    status = mluOpCreateTensorDescriptor(&input_desc);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
    status = mluOpCreateTensorDescriptor(&padded_input_desc);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    const int in_dim_num = 2;
    int64_t dims[in_dim_num] = {batch, fft_plan->inembed[0]};
    status = mluOpSetTensorDescriptor_v2(input_desc, MLUOP_LAYOUT_ARRAY,
                                         in_r_dtype, in_dim_num, dims);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    int64_t padded_dims[in_dim_num] = {batch, n};
    status = mluOpSetTensorDescriptor_v2(padded_input_desc, MLUOP_LAYOUT_ARRAY,
                                         in_r_dtype, in_dim_num, padded_dims);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    const int pad_dim_num = 4;
    int paddings[pad_dim_num] = {0, 0, 0, n - fft_plan->inembed[0]};
    uint64_t padding_value = 0x00000000;
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                      cnnl_handle);  // convert to cnnl_handle

    // convert to cnnl_tensor_descriptor
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(padded_input_desc,
                                                 cnnl_padded_input_desc);
    CALL_CNNL(cnnlPad(cnnl_handle, cnnl_input_desc,
                      (fft_plan->prime)
                          ? fft_plan->matmul_addrs.input_contiguous_addr
                          : fft_plan->mlu_addrs.input,
                      paddings, &padding_value, cnnl_padded_input_desc,
                      (fft_plan->prime) ? fft_plan->matmul_addrs.input_pad_addr
                                        : fft_plan->mlu_addrs.input_pad_addr));

    // destroy cnnl descriptor
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_padded_input_desc);

    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  return status;
}

static mluOpStatus_t padRFFT2dContiguousInput(mluOpHandle_t handle,
                                              mluOpFFTPlan_t fft_plan) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into padRFFT1dContiguousInput";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  mluOpDataType_t in_r_dtype = fft_plan->input_dtype;
  int batch = fft_plan->batch;
  int n0 = fft_plan->n[0];
  int n1 = fft_plan->n[1];
  bool need_pad = fft_plan->inembed[0] != n0 || fft_plan->inembed[1] != n1;
  if (need_pad) {
    mluOpTensorDescriptor_t input_desc, padded_input_desc;
    status = mluOpCreateTensorDescriptor(&input_desc);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
    status = mluOpCreateTensorDescriptor(&padded_input_desc);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    const int in_dim_num = 3;
    int64_t dims[in_dim_num] = {batch, std::min(n0, fft_plan->inembed[0]),
                                std::min(n1, fft_plan->inembed[1])};
    status = mluOpSetTensorDescriptor_v2(input_desc, MLUOP_LAYOUT_ARRAY,
                                         in_r_dtype, in_dim_num, dims);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    int64_t padded_dims[in_dim_num] = {batch, n0, n1};
    status = mluOpSetTensorDescriptor_v2(padded_input_desc, MLUOP_LAYOUT_ARRAY,
                                         in_r_dtype, in_dim_num, padded_dims);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    const int pad_dim_num = 6;
    int paddings[pad_dim_num] = {0, 0,
                                 0, std::max((n0 - fft_plan->inembed[0]), 0),
                                 0, std::max((n1 - fft_plan->inembed[1]), 0)};
    uint64_t padding_value = 0x00000000;
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                      cnnl_handle);  // convert to cnnl_handle

    // convert to cnnl_tensor_descriptor
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(padded_input_desc,
                                                 cnnl_padded_input_desc);
    CALL_CNNL(cnnlPad(cnnl_handle, cnnl_input_desc,
                      (fft_plan->prime)
                          ? fft_plan->matmul_addrs.input_contiguous_addr
                          : fft_plan->mlu_addrs.input,
                      paddings, &padding_value, cnnl_padded_input_desc,
                      (fft_plan->prime) ? fft_plan->matmul_addrs.input_pad_addr
                                        : fft_plan->mlu_addrs.input_pad_addr));

    // destroy cnnl descriptor
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_padded_input_desc);

    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  return status;
}

// only for CNFFT_FUNC_COOLEY_TUKEY
// batch * L * 2^m --> batch * 2^m * L
// input    : in input_pad_addr
// output   : in input_transed_addr
static mluOpStatus_t transposeRFFT1dPaddedInput(mluOpHandle_t handle,
                                                mluOpFFTPlan_t fft_plan) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into transposeRFFT1dPaddedInput";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  if (fft_plan->fft_strategy == CNFFT_FUNC_COOLEY_TUKEY) {
    VLOG(5) << "launch mluOpTranspose";

    mluOpDataType_t in_r_dtype = fft_plan->input_dtype;
    int batch = fft_plan->batch;
    int L = fft_plan->L;
    int m = (1 << fft_plan->m);

    const int trans_dim_num = 3;
    int64_t trans_input_dims[trans_dim_num] = {batch, L, m};
    int64_t trans_output_dims[trans_dim_num] = {batch, m, L};
    int trans_permute[trans_dim_num] = {0, 2, 1};

    status =
        fftTranspose(handle, trans_dim_num, trans_input_dims, trans_output_dims,
                     trans_permute, fft_plan->matmul_addrs.input_pad_addr,
                     fft_plan->matmul_addrs.input_transed_addr, in_r_dtype,
                     fft_plan->matmul_addrs.internal_workspace_addr,
                     fft_plan->matmul_addrs.internal_workspace_size, api);
  }
  return status;
}

// input    : in input_pad_addr
// output   : in input_pos_addr and input_scale_addr
static mluOpStatus_t quantizeRFFT1dPaddedInput(mluOpHandle_t handle,
                                               mluOpFFTPlan_t fft_plan) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into quantizeRFFT1dPaddedInput";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  mluOpDataType_t in_r_dtype = fft_plan->input_dtype;
  mluOpDataType_t in_e_dtype = fft_plan->execution_dtype;
  int padded_num = fft_plan->batch * fft_plan->n[0];

  status = fftQuantizePositionScale(
      handle, padded_num, in_r_dtype, in_e_dtype,
      fft_plan->matmul_addrs.input_pad_addr,
      fft_plan->matmul_addrs.input_pos_addr,
      fft_plan->matmul_addrs.input_scale_addr,
      fft_plan->matmul_addrs.internal_workspace_addr,
      fft_plan->matmul_addrs.internal_workspace_size, api);

  return status;
}

// CNFFT_FUNC_MATMUL
// input    : in input_pad_addr
// output   : in output_contiguous_addr
// CNFFT_FUNC_COOLEY_TUKEY
// input    : in input_transed_addr
// output   : input real matmul dft real result in matmul_re_mul_re_addr
//            input real matmul dft imag result in matmul_re_mul_im_addr
static mluOpStatus_t computeRFFT1dMatmulResult(mluOpHandle_t handle,
                                               mluOpFFTPlan_t fft_plan,
                                               const float scale_factor) {
  std::string api = "[mluOpExecFFT]";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  mluOpDataType_t in_r_dtype = fft_plan->input_dtype;
  mluOpDataType_t in_e_dtype = fft_plan->execution_dtype;
  int batch = fft_plan->batch;
  int n = fft_plan->n[0];

  if (fft_plan->fft_strategy == CNFFT_FUNC_MATMUL) {
    VLOG(5) << "into CNFFT_FUNC_MATMUL";
    status = fftQuantMatMul(
        handle, batch, n, FFT_HALF(n) * COMPLEX,
        fft_plan->matmul_addrs.input_pad_addr,
        fft_plan->matmul_addrs.input_pos_addr,
        fft_plan->matmul_addrs.input_scale_addr,
        fft_plan->matmul_addrs.dft_matrix_addr,
        fft_plan->matmul_addrs.dft_pos_addr,
        fft_plan->matmul_addrs.dft_scale_addr,
        fft_plan->matmul_addrs.output_contiguous_addr, false, true,
        scale_factor, 0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  } else if (fft_plan->fft_strategy == CNFFT_FUNC_COOLEY_TUKEY) {
    VLOG(5) << "into CNFFT_FUNC_COOLEY_TUKEY";
    int L = fft_plan->L;
    int m = (1 << fft_plan->m);
    // input real matmul dft real
    status = fftQuantMatMul(
        handle, batch * m, L, L, fft_plan->matmul_addrs.input_transed_addr,
        fft_plan->matmul_addrs.input_pos_addr,
        fft_plan->matmul_addrs.input_scale_addr,
        fft_plan->matmul_addrs.dft_re_matrix_addr,
        fft_plan->matmul_addrs.dft_pos_addr,
        fft_plan->matmul_addrs.dft_scale_addr,
        fft_plan->matmul_addrs.matmul_re_mul_re_addr, false, true, scale_factor,
        0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    // input re matmul dft imag
    status = fftQuantMatMul(
        handle, batch * m, L, L, fft_plan->matmul_addrs.input_transed_addr,
        fft_plan->matmul_addrs.input_pos_addr,
        fft_plan->matmul_addrs.input_scale_addr,
        fft_plan->matmul_addrs.dft_im_matrix_addr,
        fft_plan->matmul_addrs.dft_pos_addr,
        fft_plan->matmul_addrs.dft_scale_addr,
        fft_plan->matmul_addrs.matmul_re_mul_im_addr, false, true, scale_factor,
        0.0, in_e_dtype, in_e_dtype, in_r_dtype,
        fft_plan->matmul_addrs.internal_workspace_addr,
        fft_plan->matmul_addrs.internal_workspace_size, api);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  } else if (fft_plan->fft_strategy == CNFFT_FUNC_STOCKHAM) {
    VLOG(5) << "into CNFFT_FUNC_STOCKHAM";
    int L = fft_plan->L;
    int m = (1 << fft_plan->m);

    // origin: in_trans[batch, 2^m, L] * W_real[L, L] -> IN_real[batch, 2^m, L]
    //         in_trans[batch, 2^m, L] * W_imag[L, L] -> IN_imag[batch, 2^m, L]
    // update: W[c*L, L] * in[batch, L, 2^m] -> out[batch, c*L, 2^m]
    status = fftBatchMatMulBcast(
        handle,
        L <= fft_plan->L_sub ? (2 * L)
                             : (2 * (PAD_UP(L / 2, fft_plan->L_sub) + 1)),
        L, m, batch, fft_plan->matmul_addrs.dft_re_matrix_addr,
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
// input    : input real matmul dft real result in matmul_re_mul_re_addr
//            input real matmul dft imag result in matmul_re_mul_im_addr
// output   : output complex result in output_contiguous_addr
static mluOpStatus_t mergeRFFT1dOutput(mluOpHandle_t handle,
                                       mluOpFFTPlan_t fft_plan,
                                       const float scale_factor) {
  std::string api = "[mluOpExecFFT]";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  if (fft_plan->fft_strategy == CNFFT_FUNC_COOLEY_TUKEY) {
    VLOG(5) << "launch merge rfft1d output";
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
    kernelFFTCooleyTukey(k_dim, k_type, handle->queue, fft_plan, -1, RFFT);
    // direction, -1 means invalid(only FFT_IFFT use)
  } else if (fft_plan->fft_strategy == CNFFT_FUNC_STOCKHAM) {
    VLOG(5) << "launch mrege four-step rfft1d output";
    cnrtDim3_t k_dim;
    cnrtFunctionType_t k_type;
    policyFunc(handle, &k_dim, &k_type);
    kernelFFTStockham(k_dim, k_type, handle->queue, fft_plan, -1, scale_factor,
                      RFFT);
    // direction, -1 means invalid(only FFT_IFFT use).
  }
  return status;
}

// input    : in output_contiguous_addr
// output   : in output
static mluOpStatus_t makeRFFT1dContiguousOutput(mluOpHandle_t handle,
                                                mluOpFFTPlan_t fft_plan,
                                                void *output) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into makeRFFT1dContiguousOutput";
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
    int64_t dims[out_dim_num] = {
        fft_plan->batch,
        (fft_plan->prime) ? fft_plan->onembed[0] : (fft_plan->n[0] / 2 + 1)};
    int64_t strides[out_dim_num] = {fft_plan->odist, fft_plan->ostride};
    status = mluOpSetTensorDescriptor_v2(copy_src_desc, MLUOP_LAYOUT_ARRAY,
                                         out_c_dtype, out_dim_num, dims);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
    status =
        mluOpSetTensorDescriptorEx_v2(copy_dst_desc, MLUOP_LAYOUT_ARRAY,
                                      out_c_dtype, out_dim_num, dims, strides);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    // copy
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

    size_t workspace_size = 0;
    CALL_CNNL(cnnlGetCopyWorkspaceSize(cnnl_handle, cnnl_copy_src_desc,
                                       cnnl_copy_dst_desc, &workspace_size));

    void *workspace = nullptr;
    if (workspace_size > 0) {
      CNRT_CHECK(cnrtMalloc((void **)&workspace, workspace_size));
    }
    CALL_CNNL(cnnlCopy_v2(cnnl_handle, cnnl_copy_src_desc, copy_src_addr,
                          cnnl_copy_dst_desc, output, workspace,
                          workspace_size));

    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_copy_src_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_copy_dst_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  return status;
}

// input    : in input
// output   : in input_contiguous_addr
static mluOpStatus_t makeRFFT2dContiguousInput(mluOpHandle_t handle,
                                               mluOpFFTPlan_t fft_plan,
                                               const void *input) {
  std::string api = "[mluOpExecFFT]";
  VLOG(5) << "into makeIRFFT2dContiguousInput";
  auto status = MLUOP_STATUS_SUCCESS;
  if ((!fft_plan->is_input_contiguous ||
       (fft_plan->inembed[0] > fft_plan->n[0] ||
        fft_plan->inembed[1] > fft_plan->n[1]) &&
           !fft_plan->prime) &&
      fft_plan->fft_strategy != CNFFT_FUNC_MANY_DIST1_2D) {
    VLOG(5) << "launch mluOpContiguous for rfft2d input";
    mluOpTensorDescriptor_t input_desc;
    status = mluOpCreateTensorDescriptor(&input_desc);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    const int in_dim_num = 3;
    int64_t dims[in_dim_num] = {
        fft_plan->batch,
        fft_plan->n[0] > fft_plan->inembed[0] ? fft_plan->inembed[0]
                                              : fft_plan->n[0],
        fft_plan->n[1] > fft_plan->inembed[1] ? fft_plan->inembed[1]
                                              : fft_plan->n[1]};
    int64_t strides[in_dim_num] = {fft_plan->idist,
                                   (fft_plan->istride * fft_plan->inembed[1]),
                                   fft_plan->istride};
    status = mluOpSetTensorDescriptorEx_v2(input_desc, MLUOP_LAYOUT_ARRAY,
                                           fft_plan->input_dtype, in_dim_num,
                                           dims, strides);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    status =
        mluOpContiguous(handle, input_desc, input, fft_plan->mlu_addrs.input);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    status = mluOpDestroyTensorDescriptor(input_desc);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  }
  return status;
}

static mluOpStatus_t makeRFFT2dContiguousOutput(mluOpHandle_t handle,
                                                mluOpFFTPlan_t fft_plan,
                                                void *output) {
  std::string api = "[mluOpExecFFT]";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  if (!fft_plan->is_output_contiguous &&
      fft_plan->fft_strategy != CNFFT_FUNC_MANY_DIST1_2D) {
    VLOG(5) << "launch copy with stride";
    mluOpDataType_t out_c_dtype = fft_plan->output_dtype;
    // create tensor desc
    mluOpTensorDescriptor_t copy_src_desc, copy_dst_desc;
    status = mluOpCreateTensorDescriptor(&copy_src_desc);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
    status = mluOpCreateTensorDescriptor(&copy_dst_desc);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    // set up tensor desc
    const int out_dim_num = 3;
    int64_t dims[out_dim_num] = {fft_plan->batch, fft_plan->n[0],
                                 fft_plan->n[1] / 2 + 1};
    int64_t strides[out_dim_num] = {fft_plan->odist,
                                    fft_plan->ostride * fft_plan->onembed[1],
                                    fft_plan->ostride};
    status = mluOpSetTensorDescriptor_v2(copy_src_desc, MLUOP_LAYOUT_ARRAY,
                                         out_c_dtype, out_dim_num, dims);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
    status =
        mluOpSetTensorDescriptorEx_v2(copy_dst_desc, MLUOP_LAYOUT_ARRAY,
                                      out_c_dtype, out_dim_num, dims, strides);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    void *copy_src_addr = fft_plan->mlu_addrs.output;
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                      cnnl_handle);  // convert to cnnl_handle
    // convert to cnnl_tensor_descriptor
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(copy_src_desc,
                                                 cnnl_copy_src_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(copy_dst_desc,
                                                 cnnl_copy_dst_desc);

    size_t workspace_size = 0;
    CALL_CNNL(cnnlGetCopyWorkspaceSize(cnnl_handle, cnnl_copy_src_desc,
                                       cnnl_copy_dst_desc, &workspace_size));

    void *workspace = nullptr;
    if (workspace_size > 0) {
      CNRT_CHECK(cnrtMalloc((void **)&workspace, workspace_size));
    }
    CALL_CNNL(cnnlCopy_v2(cnnl_handle, cnnl_copy_src_desc, copy_src_addr,
                          cnnl_copy_dst_desc, output, workspace,
                          workspace_size));

    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_copy_src_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_copy_dst_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  return status;
}

mluOpStatus_t execRFFT1d(mluOpHandle_t handle, const mluOpFFTPlan_t fft_plan,
                         const void *input, const float scale_factor,
                         void *workspace, void *output) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  std::string api = "[mluOpExecFFT]";
  if (fft_plan->prime) {
    configureRFFT1dMatmulWorkspaceAddrs(handle, fft_plan, (void *)input,
                                        workspace, output);

    status = makeRFFT1dContiguousInput(handle, fft_plan, input);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    status = padRFFT1dContiguousInput(handle, fft_plan);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    status = transposeRFFT1dPaddedInput(handle, fft_plan);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    status = quantizeRFFT1dPaddedInput(handle, fft_plan);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    status = computeRFFT1dMatmulResult(handle, fft_plan, scale_factor);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    status = mergeRFFT1dOutput(handle, fft_plan, scale_factor);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    status = makeRFFT1dContiguousOutput(handle, fft_plan, output);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  } else {
    configureRFFT1dWorkspaceAddrs(handle, fft_plan, (void *)input, workspace,
                                  output);

    status = makeRFFT1dContiguousInput(handle, fft_plan, input);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    if (fft_plan->n[0] > fft_plan->inembed[0]) {
      status = padRFFT1dContiguousInput(handle, fft_plan);
      INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
      fft_plan->mlu_addrs.input = fft_plan->mlu_addrs.input_pad_addr;
    }

    cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
    cnrtDim3_t k_dim;
    k_dim.x = handle->core_num_per_cluster;
    k_dim.y = mluop::runtime::getClusterLimitCapability(handle);
    k_dim.z = 1;

    status = execFFTr2c1d(handle, fft_plan, scale_factor);

    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    if (scale_factor != 1.0) {
      const float alpha[2] = {scale_factor, 0.0};
      const float beta[2] = {0.0, 0.0};
      mluOpTensorDescriptor_t c_desc = nullptr;
      status = mluOpCreateTensorDescriptor(&c_desc);
      const int out_dim_num = 2;
      int64_t dims[out_dim_num] = {fft_plan->batch, (fft_plan->n[0] / 2 + 1)};
      status = mluOpSetTensorDescriptor_v2(c_desc, MLUOP_LAYOUT_ARRAY,
                                           fft_plan->output_dtype, 2, dims);
      status = mluOpSetTensorDescriptorOnchipDataType(
          c_desc, fft_plan->execution_dtype);

      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                        cnnl_handle);  // convert to cnnl_handle

      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_output_desc);

      CALL_CNNL(cnnlTransform_v2(cnnl_handle, CNNL_POINTER_MODE_HOST, &alpha,
                                 cnnl_output_desc, fft_plan->mlu_addrs.output,
                                 &beta, cnnl_output_desc,
                                 fft_plan->mlu_addrs.output));
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }

    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
    status = makeRFFT1dContiguousOutput(handle, fft_plan, output);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  }

  return status;
}

mluOpStatus_t execFFTr2c1d(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                           const float scale_factor) {
  std::string api = "[execFFTr2c1d]";

  VLOG(5) << "launch r2c fft1d";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, &k_dim, &k_type);

  kernelFFT1dButterflyR2C(k_dim, k_type, handle->queue, fft_plan, RFFT);

  return status;
}

// in: [2][n0][2][n1][batch]
mluOpStatus_t computeFFT2dMatMulColumnR2C(mluOpHandle_t handle,
                                          mluOpFFTPlan_t fft_plan,
                                          const float scale_factor) {
  std::string api = "[computeFFT2dMatMulColumnR2C]";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  mluOpDataType_t in_e_dtype = fft_plan->execution_dtype;
  int batch = fft_plan->batch;
  int n0 = fft_plan->n[0];
  int n1 = fft_plan->n[1];

  void *dft_matrix_addr = fft_plan->mlu_addrs.dft_matrix_2d;
  void *in_addr = fft_plan->mlu_addrs.buffer_in;
  void *out_addr = fft_plan->mlu_addrs.buffer_out;

  // out[n0 * 2][(n1/2+1)*2][batch] = W[n0 * 2][n0] * In[n0][(n1/2+1)*2][batch]
  const int m = n0 * 2, k = n0, n = (n1 / 2 + 1) * 2 * batch;

  // create descriptor
  mluOpTensorDescriptor_t a_desc = nullptr;
  mluOpTensorDescriptor_t b_desc = nullptr;
  mluOpTensorDescriptor_t c_desc = nullptr;
  status = mluOpCreateTensorDescriptor(&a_desc);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  status = mluOpCreateTensorDescriptor(&b_desc);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  status = mluOpCreateTensorDescriptor(&c_desc);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

  // set descriptor
  int64_t a_dims[2] = {m, k};
  int64_t b_dims[2] = {k, n};
  int64_t c_dims[2] = {m, n};
  size_t matmul_ws_size = 0, workspace_size = 0;

  status = mluOpSetTensorDescriptor_v2(a_desc, MLUOP_LAYOUT_ARRAY, in_e_dtype,
                                       2, a_dims);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  status = mluOpSetTensorDescriptorOnchipDataType(a_desc, in_e_dtype);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  status = mluOpSetTensorDescriptor_v2(b_desc, MLUOP_LAYOUT_ARRAY, in_e_dtype,
                                       2, b_dims);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  status = mluOpSetTensorDescriptorOnchipDataType(b_desc, in_e_dtype);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  status = mluOpSetTensorDescriptor_v2(c_desc, MLUOP_LAYOUT_ARRAY, in_e_dtype,
                                       2, c_dims);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  status = mluOpSetTensorDescriptorOnchipDataType(c_desc, in_e_dtype);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                    cnnl_handle);  // convert to cnnl_handle
  cnnlMatMulDescriptor_t matmul_desc;
  cnnlMatMulAlgo_t matmul_algo;
  cnnlMatMulHeuristicResult_t heuristic_result;

  CALL_CNNL(cnnlMatMulDescCreate(&matmul_desc));
  CALL_CNNL(cnnlMatMulAlgoCreate(&matmul_algo));
  CALL_CNNL(cnnlCreateMatMulHeuristicResult(&heuristic_result));
  int32_t requested_algo_count = 1, return_algo_count = 0;

  // convert to cnnl_tensor_descriptor
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_d_desc);
  c_desc->onchip_dtype = in_e_dtype;
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

  // destroy cnnl descriptor
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);

  DESTROY_CNNL_HANDLE(cnnl_handle);
  // cnrtQueueSync(handle->queue);

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, &k_dim, &k_type);
  kernelFFTBatchConjMergeR2C(
      k_dim, k_type, handle->queue, fft_plan->mlu_addrs.output,
      fft_plan->mlu_addrs.buffer_out, (n1 / 2 + 1) * batch, n0, in_e_dtype);
  // cnrtQueueSync(handle->queue);

  return status;
}

mluOpStatus_t computeFFT2dMatMulRowR2C(mluOpHandle_t handle,
                                       mluOpFFTPlan_t fft_plan,
                                       const float scale_factor) {
  std::string api = "[computeFFT2dMatMulRowR2C]";
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  mluOpDataType_t in_e_dtype = fft_plan->execution_dtype;
  int batch = fft_plan->batch;
  int n0 = fft_plan->n[0];
  int n1 = fft_plan->n[1];

  void *dft_matrix_addr = fft_plan->mlu_addrs.dft_matrix;
  void *in_addr = fft_plan->mlu_addrs.input;
  void *out_addr = fft_plan->mlu_addrs.buffer_in;
  // void *out_addr = fft_plan->mlu_addrs.output;

  // out[n0][(n1/2+1)*2][batch] = W[(n1/2+1) * 2][n1] * In[n0][n1][batch]
  const int m = (n1 / 2 + 1) * 2, k = n1, n = batch;

  // create descriptor
  mluOpTensorDescriptor_t a_desc = nullptr;
  mluOpTensorDescriptor_t b_desc = nullptr;
  mluOpTensorDescriptor_t c_desc = nullptr;
  status = mluOpCreateTensorDescriptor(&a_desc);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  status = mluOpCreateTensorDescriptor(&b_desc);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  status = mluOpCreateTensorDescriptor(&c_desc);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

  // set descriptor
  int64_t a_dims[2] = {m, k};
  int64_t b_dims[3] = {n0, k, n};
  int64_t c_dims[3] = {n0, m, n};

  status = mluOpSetTensorDescriptor_v2(a_desc, MLUOP_LAYOUT_ARRAY, in_e_dtype,
                                       2, a_dims);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  status = mluOpSetTensorDescriptorOnchipDataType(a_desc, in_e_dtype);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  status = mluOpSetTensorDescriptor_v2(b_desc, MLUOP_LAYOUT_ARRAY, in_e_dtype,
                                       3, b_dims);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  status = mluOpSetTensorDescriptorOnchipDataType(b_desc, in_e_dtype);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  status = mluOpSetTensorDescriptor_v2(c_desc, MLUOP_LAYOUT_ARRAY, in_e_dtype,
                                       3, c_dims);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  status = mluOpSetTensorDescriptorOnchipDataType(c_desc, in_e_dtype);
  INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                    cnnl_handle);  // convert to cnnl_handle

  // convert to cnnl_tensor_descriptor
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);

  // c_desc->onchip_dtype = MLUOP_DTYPE_FLOAT;
  c_desc->onchip_dtype = in_e_dtype;
  float alpha = 1.0;
  float beta = 0.0;

  cnnlMatMulAlgo_t algo;
  CALL_CNNL(cnnlMatMulAlgoCreate(&algo));
  cnnlMatMulDescriptor_t bmm_bcast_desc;
  CALL_CNNL(cnnlMatMulDescCreate(&bmm_bcast_desc));

  cnnlMatMulHeuristicResult_t heuristic_result;
  CALL_CNNL(cnnlCreateMatMulHeuristicResult(&heuristic_result));

  int requested_algo_count = 1, return_algo_count = 0;
  float *workspace;
  size_t workspace_size;
  cnnlGetBatchMatMulAlgoHeuristic(
      cnnl_handle, bmm_bcast_desc, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc, NULL,
      requested_algo_count, &heuristic_result, &return_algo_count);

  cnnlGetBatchMatMulHeuristicResult(heuristic_result, algo, &workspace_size);

  if (workspace_size > 0) {
    CNRT_CHECK(cnrtMalloc((void **)&workspace, workspace_size));
  } else {
    CNRT_CHECK(cnrtMalloc((void **)&workspace, m * n * sizeof(float)));
  }
  CALL_CNNL(cnnlBatchMatMulBCast_v2(cnnl_handle, bmm_bcast_desc, algo, &alpha,
                                    cnnl_a_desc, dft_matrix_addr, cnnl_b_desc,
                                    in_addr, &beta, cnnl_c_desc, out_addr,
                                    (void *)workspace, workspace_size));
  // destroy cnnl descriptor
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);

  DESTROY_CNNL_HANDLE(cnnl_handle);
  // cnrtQueueSync(handle->queue);

  return status;
}

mluOpStatus_t execRFFT2d(mluOpHandle_t handle, const mluOpFFTPlan_t fft_plan,
                         const void *input, const float scale_factor,
                         void *workspace, void *output) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  std::string api = "[mluOpExecFFT]";

  configureRFFT2dWorkspaceAddrs(handle, fft_plan, (void *)input, workspace,
                                output);

  if (fft_plan->fft_strategy == CNFFT_FUNC_TWO_LEVEL_STOCKHAM) {
    cnrtDim3_t k_dim;
    cnrtFunctionType_t k_type;
    policyFunc(handle, &k_dim, &k_type);
    uint64_t idist = 0, odist = 0;  // bytes
    mluOpDataType_t in_r_dtype = fft_plan->input_dtype;
    mluOpDataType_t in_c_dtype = fft_plan->output_dtype;
    size_t in_r_dtype_size = mluOpDataTypeBytes(in_r_dtype);
    size_t in_c_dtype_size = mluOpDataTypeBytes(in_c_dtype);
    // outplace
    idist = in_r_dtype_size * fft_plan->n[0] * fft_plan->n[1];
    odist = in_c_dtype_size * fft_plan->n[0] * (fft_plan->n[1] / 2 + 1);
    status = makeRFFT2dContiguousInput(handle, fft_plan, input);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    if (fft_plan->n[0] > fft_plan->inembed[0] ||
        fft_plan->n[1] > fft_plan->inembed[1]) {
      status = padRFFT2dContiguousInput(handle, fft_plan);
      INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

      fft_plan->mlu_addrs.input = fft_plan->mlu_addrs.input_pad_addr;
    }

    for (int batch_id = 0; batch_id < fft_plan->batch; batch_id++) {
      status = kernelRFFT2dButterflyRow(k_dim, k_type, handle->queue, fft_plan,
                                        RFFT);

      INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

      status = kernelRFFT2dButterflyColumn(k_dim, k_type, handle->queue,
                                           fft_plan, FFT_IFFT);
      INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

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

    if (scale_factor != 1.0) {
      const float alpha[2] = {scale_factor, 0.0};
      const float beta[2] = {0.0, 0.0};
      mluOpTensorDescriptor_t c_desc = nullptr;
      status = mluOpCreateTensorDescriptor(&c_desc);
      const int out_dim_num = 3;
      int64_t dims[out_dim_num] = {fft_plan->batch, fft_plan->n[0],
                                   fft_plan->n[1] / 2 + 1};
      status = mluOpSetTensorDescriptor_v2(c_desc, MLUOP_LAYOUT_ARRAY,
                                           fft_plan->output_dtype, 3, dims);
      status = mluOpSetTensorDescriptorOnchipDataType(
          c_desc, fft_plan->execution_dtype);

      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                        cnnl_handle);  // convert to cnnl_handle

      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_output_desc);

      CALL_CNNL(cnnlTransform_v2(cnnl_handle, CNNL_POINTER_MODE_HOST, &alpha,
                                 cnnl_output_desc, fft_plan->mlu_addrs.output,
                                 &beta, cnnl_output_desc,
                                 fft_plan->mlu_addrs.output));
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

    status = makeRFFT2dContiguousOutput(handle, fft_plan, output);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);

  } else if (fft_plan->fft_strategy == CNFFT_FUNC_MANY_DIST1_2D) {
    status = computeFFT2dMatMulRowR2C(handle, fft_plan, scale_factor);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
    status = computeFFT2dMatMulColumnR2C(handle, fft_plan, scale_factor);
    INTERNAL_CHECK(api, status == MLUOP_STATUS_SUCCESS);
  }
  return status;
}
