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

#include "kernels/fft/bluestein_fft/bluestein_fft.h"
#include "kernels/fft/c2c_fft/c2c_fft.h"
// #include "kernels/fft/fft_optm_device/fft_bluestein.h"
#include <algorithm>
#include <string>

#define DIRECTION 2  // FORWARD and BACKWARD

static mluOpStatus_t policyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                                cnrtFunctionType_t *k_type) {
  *k_type = cnrtFuncTypeUnion1;
  k_dim->x = handle->core_num_per_cluster;
  k_dim->y = mluop::runtime::getClusterLimitCapability(handle);
  k_dim->z = 1;
  return MLUOP_STATUS_SUCCESS;
}


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

VLOG(5) << "batch " << batch << "nfft " << nfft;

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

static void configureBluestein(mluOpHandle_t handle,
                                         mluOpFFTPlan_t fft_plan, void *input,
                                         void *workspace, void *output) {
  VLOG(5) << "Into configure BluesteinFFT Workspace Addrs";
  const std::string make_plan_api = "[configureBluesteinFFTdWorkspaceAddrs]";
  size_t workspace_size = 0;
  size_t reservespace_size = 0;

  // c2c
  mluOpDataType_t in_c_dtype = fft_plan->bluestein_plan->input_dtype;
  mluOpDataType_t out_c_dtype = fft_plan->bluestein_plan->output_dtype;

  size_t in_c_dtype_size = mluOpDataTypeBytes(in_c_dtype);
  size_t out_c_dtype_size = mluOpDataTypeBytes(out_c_dtype);

  int batch = fft_plan->bluestein_plan->batch;
  int nfft = fft_plan->bluestein_plan->n[0];

  if(fft_plan->rank == 2) {
      nfft = nfft * fft_plan->bluestein_plan->n[1];
  }

  // size_t offset = 0;

  size_t buffer_size = batch * in_c_dtype_size * nfft;

  size_t offset = 0;
  fft_plan->bluestein_plan->mlu_addrs.buffer_buf = (uint8_t *)workspace + offset;
  offset += buffer_size * 2;


  VLOG(5) << "fft_plan->workspace_size: " << fft_plan->workspace_size;
  VLOG(5) << "fft_plan->workspace_size: " << fft_plan->bluestein_plan->workspace_size;
  VLOG(5) << "buffer_size: " << buffer_size;
  VLOG(5) << "in_c_dtype_size: " << in_c_dtype_size;

  fft_plan->bluestein_input = (uint8_t *)workspace + fft_plan->bluestein_plan->workspace_size;
  VLOG(5) << "fft_plan->bluestein_input offset " << offset;
  offset += buffer_size*2;
  fft_plan->bluestein_output = (uint8_t *)workspace + fft_plan->bluestein_plan->workspace_size + offset;

  VLOG(5) << " fft_plan->bluestein_inputoffset " << offset;

  offset += buffer_size*2;
  fft_plan->bluestein_chirpz = (uint8_t *)workspace + fft_plan->bluestein_plan->workspace_size + offset;
  VLOG(5) << " fft_plan->bluestein_inputoffset " << offset;
}

// static void configureBlueFFT1dWorkspaceAddrs(mluOpHandle_t handle,
//   mluOpFFTPlan_t fft_plan, void *input,
//   void *workspace, void *output) {
// VLOG(5) << "Into configure BluesteinFFT Workspace Addrs";
// const std::string make_plan_api = "[configureBluesteinFFTdWorkspaceAddrs]";
// size_t workspace_size = 0;
// size_t reservespace_size = 0;

// // c2c
// mluOpDataType_t in_c_dtype = fft_plan->bluestein_plan->input_dtype;
// mluOpDataType_t out_c_dtype = fft_plan->bluestein_plan->output_dtype;

// size_t in_c_dtype_size = mluOpDataTypeBytes(in_c_dtype);
// size_t out_c_dtype_size = mluOpDataTypeBytes(out_c_dtype);

// // int batch = fft_plan->bluestein_plan->batch;
// // int nfft = fft_plan->bluestein_plan->n[0];

// if(fft_plan->rank == 2) {
// nfft = nfft * fft_plan->bluestein_plan->n[1];
// }

// // size_t offset = 0;

// // size_t buffer_size = batch * in_c_dtype_size * nfft;

// // VLOG(5) << "fft_plan->workspace_size: " << fft_plan->workspace_size;
// // VLOG(5) << "fft_plan->workspace_size: " << fft_plan->bluestein_plan->workspace_size;
// // VLOG(5) << "buffer_size: " << buffer_size;
// // VLOG(5) << "in_c_dtype_size: " << in_c_dtype_size;

// // fft_plan->bluestein_input = (uint8_t *)workspace + fft_plan->bluestein_plan->workspace_size;
// // VLOG(5) << "fft_plan->bluestein_input offset " << offset;
// // offset += buffer_size;
// // fft_plan->bluestein_output = (uint8_t *)workspace + fft_plan->bluestein_plan->workspace_size + offset;

// // VLOG(5) << " fft_plan->bluestein_inputoffset " << offset;

// // offset += buffer_size;
// // fft_plan->bluestein_chirpz = (uint8_t *)workspace + fft_plan->bluestein_plan->workspace_size + offset;
// // VLOG(5) << " fft_plan->bluestein_inputoffset " << offset;
// }

// static void configureFFT2dWorkspaceAddrs(mluOpHandle_t handle,
//   mluOpFFTPlan_t fft_plan, void *input,
//   void *workspace, void *output) {
// const std::string make_plan_api = "[configureFFT2dWorkspaceAddrs]";

// mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
// mluOpDataType_t out_c_dtype = fft_plan->output_dtype;

// size_t in_c_dtype_size = mluOpDataTypeBytes(in_c_dtype);
// size_t out_c_dtype_size = mluOpDataTypeBytes(out_c_dtype);

// int batch = fft_plan->batch;
// int n0_ori = fft_plan->n[0];
// int n1_ori = fft_plan->n[1];

// size_t offset = 0;
// if (fft_plan->fft_strategy == CNFFT_FUNC_MANY_DIST1_2D) {
// // rr ri ir ii
// size_t buffer_size = batch * in_c_dtype_size * n0_ori * n1_ori * 2;
// fft_plan->mlu_addrs.input = input;
// fft_plan->mlu_addrs.output = output;
// fft_plan->mlu_addrs.buffer_in = (uint8_t *)workspace + offset;
// offset += buffer_size;
// fft_plan->mlu_addrs.buffer_out = (uint8_t *)workspace + offset;
// offset += buffer_size;
// }

// if (fft_plan->fft_strategy == CNFFT_FUNC_TWO_LEVEL_STOCKHAM) {
// fft_plan->mlu_addrs.buffer_buf = (uint8_t *)workspace + offset;
// offset += batch * in_c_dtype_size * n0_ori * n1_ori * 2;

// if ((fft_plan->is_input_contiguous &&
// fft_plan->inembed[0] <= fft_plan->n[0] &&
// fft_plan->inembed[1] <= fft_plan->n[1])) {
// fft_plan->mlu_addrs.input = input;
// } else {
// fft_plan->mlu_addrs.input = (uint8_t *)workspace + offset;
// offset += batch * in_c_dtype_size * n0_ori * n1_ori;
// }

// if (fft_plan->is_output_contiguous) {
// fft_plan->mlu_addrs.output = output;
// } else {
// fft_plan->mlu_addrs.output = (uint8_t *)workspace + offset;
// offset += batch * in_c_dtype_size * n0_ori * n1_ori;
// }
// }
// if (fft_plan->n[0] > fft_plan->inembed[0] ||
// fft_plan->n[1] > fft_plan->inembed[1]) {
// fft_plan->mlu_addrs.input_pad_addr =
// (uint8_t *)workspace + offset;  // batch * in_c_dtype_size * n0_ori *
//  // n1_ori * 2; // buffer_size;
// }
// }

// static void configureBluesteinFFT2dWorkspaceAddrs(mluOpHandle_t handle,
//   mluOpFFTPlan_t fft_plan, void *input,
//   void *workspace, void *output) {
// const std::string make_plan_api = "[configureFFT2dWorkspaceAddrs]";

// mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
// mluOpDataType_t out_c_dtype = fft_plan->output_dtype;

// size_t in_c_dtype_size = mluOpDataTypeBytes(in_c_dtype);
// size_t out_c_dtype_size = mluOpDataTypeBytes(out_c_dtype);

// int batch = fft_plan->batch;
// int n0_ori = fft_plan->n[0];
// int n1_ori = fft_plan->n[1];

// size_t offset = 0;

// fft_plan->mlu_addrs.buffer_buf = (uint8_t *)workspace + offset;
// offset += batch * in_c_dtype_size * n0_ori * n1_ori * 2;

// // if ((fft_plan->is_input_contiguous &&
// // fft_plan->inembed[0] <= fft_plan->n[0] &&
// // fft_plan->inembed[1] <= fft_plan->n[1])) {
// // fft_plan->mlu_addrs.input = input;
// // } else {
// // fft_plan->mlu_addrs.input = (uint8_t *)workspace + offset;
// // offset += batch * in_c_dtype_size * n0_ori * n1_ori;
// // }

// // if (fft_plan->is_output_contiguous) {
// // fft_plan->mlu_addrs.output = output;
// // } else {
// // fft_plan->mlu_addrs.output = (uint8_t *)workspace + offset;
// // offset += batch * in_c_dtype_size * n0_ori * n1_ori;
// // }
// // }
// // if (fft_plan->n[0] > fft_plan->inembed[0] ||
// // fft_plan->n[1] > fft_plan->inembed[1]) {
// // fft_plan->mlu_addrs.input_pad_addr =
// // (uint8_t *)workspace + offset;  // batch * in_c_dtype_size * n0_ori *
//  // n1_ori * 2; // buffer_size;

// fft_plan->bluestein_input = (uint8_t *)workspace + fft_plan->bluestein_plan->workspace_size;
// VLOG(5) << "fft_plan->bluestein_input offset " << offset;
// offset += buffer_size;
// fft_plan->bluestein_output = (uint8_t *)workspace + fft_plan->bluestein_plan->workspace_size + offset;

// VLOG(5) << " fft_plan->bluestein_inputoffset " << offset;

// offset += buffer_size;
// fft_plan->bluestein_chirpz = (uint8_t *)workspace + fft_plan->bluestein_plan->workspace_size + offset;
// VLOG(5) << " fft_plan->bluestein_inputoffset " << offset;
// }

// mluOpStatus_t exec_bluestein_FFTc2c1d(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
//                            const float scale_factor, const int direction) {
//   std::string api = "[execFFTc2c1d]";

//   VLOG(5) << "launch c2c fft1d";
//   mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

//   cnrtDim3_t k_dim;
//   cnrtFunctionType_t k_type;
//   policyFunc(handle, &k_dim, &k_type);
//   kernelGenerateChripZ(k_dim, k_type, handle->queue, fft_plan, -1);
//   kernelFFT1dButterflyRow(k_dim, k_type, handle->queue, fft_plan, 0);
//   kernelChirpZDotFFT(k_dim, k_type, handle->queue, fft_plan);
//   kernelGenrate(k_dim, k_type, handle->queue, fft_plan, -1)
//   kernelFFT1dButterflyRow(k_dim, k_type, handle->queue, fft_plan, 1);

//   return status;
// }

mluOpStatus_t execGenerateChirpz(mluOpHandle_t handle, void *output, int length, int n, int pad_num, bool chirpz_flag, int direction) {
    cnrtDim3_t k_dim;
    cnrtFunctionType_t k_type;
    k_type = cnrtFuncTypeBlock;
    k_dim.x = 1;
     //handle->core_num_per_cluster;
    k_dim.y = 1;
    //mluop::runtime::getClusterLimitCapability(handle);
    k_dim.z = 1;
    VLOG(5) << "launch c2c execGenerateChirpz";

    mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

    VLOG(5) << "Launch Kernel execGenerateChirpz <<Union"
          << k_type / CORE_DIM << ", " << k_dim.x << ", " << k_dim.y << ", "
          << k_dim.z << ">>>";
    // KernelGenerateChripZ(k_dim, k_type, handle->queue, output, start, length, n, PAD_N, chirpz_flag, direction);
    KernelChirpz(k_dim, k_type, handle->queue, length, n, pad_num, direction,
      chirpz_flag, output);

    return status;
}

static void policyMatrixDotVectorFunc(const mluOpHandle_t &handle,
  cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type, int col_num, int row_num, bool row_major, bool *large_col) {
    int num_deal = 0;
    if(row_major) {
    num_deal = handle->nram_size  / (8 * sizeof(float));
    VLOG(5) << "nram_size: " << handle->nram_size;
    if (col_num > num_deal) {
        *large_col = true;
      } else {
        *large_col = false;
      }
    } else {
      num_deal = handle->nram_size  / (6 * sizeof(float));
      if (col_num <= num_deal) {
        *large_col = false;
      } else {
        *large_col = true;
      }
    }
    VLOG(5) << "if large col: " << *large_col << " col_num: " << col_num
    << "num_deal: " << num_deal;
    // *k_type = cnrtFuncTypeUnion1;
    // k_dim->x = handle->core_num_per_cluster;
    // k_dim->y = mluop::runtime::getClusterLimitCapability(handle);
    // k_dim->z = 1;
    *k_type = cnrtFuncTypeBlock;
    k_dim->x = 1;
    k_dim->y = 1;
    k_dim->z = 1;
}

mluOpStatus_t execComplexMatrixDotVector(mluOpHandle_t handle, const void *vector_input, 
                        const void *matrix_input, void *output, int batch, int row_num, int col_num, int pad_num,
                        bool row_major, bool real_input, int type, int output_type) {

    mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

    // VLOG(5) << "Launch Kernel execGenerateChirpz <<Union"
    //       << k_type / CORE_DIM << ", " << k_dim.x << ", " << k_dim.y << ", "
    //       << k_dim.z << ">>>";
    cnrtDim3_t k_dim;
    cnrtFunctionType_t k_type;
    bool large_col = false;
    policyMatrixDotVectorFunc(handle, &k_dim, &k_type, col_num, row_num, row_major, &large_col);

    // KernelComplexMatrixDotVector(k_dim, k_type, handle->queue, input_matrix,
    //                              input_vector, output, batch, h, w, PAD_N,
    //                              chirpz_flag, real_input);
    KernelComplexMatrixDotVector(k_dim, k_type, handle->queue, vector_input, matrix_input, 
                                  output, batch, row_num, col_num, pad_num, row_major,
                                  real_input, large_col,type, output_type);

    return MLUOP_STATUS_SUCCESS;
}


mluOpStatus_t bluesteinFFT1dRow(mluOpHandle_t handle,
                                     const mluOpFFTPlan_t fft_plan,
                                     const void *input,
                                     const float scale_factor, void *workspace,
                                     void *output, const int direction) {
        mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
        std::string api = "[bluesteinFFT1dRow]";
                                    
        VLOG(5) << "direction: " << direction
                << "1-direction: " << 1 - direction;
        CHECK_RETURN(api, status);
        // input should be contigues
        // k 和 n 需要算出来
        // // chirpz 信号生成
        // step1
        int type = 1;
        if(direction == 0) {
            type = 1;
        } else {
            type = -1;
        }
        bool real_input = false;
        int final_output_type = 0;
        if (fft_plan->input_dtype == MLUOP_DTYPE_FLOAT) {
            real_input = true;
        }

        if (fft_plan->input_dtype == MLUOP_DTYPE_COMPLEX_FLOAT && fft_plan->output_dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
          final_output_type = 1;
        }

        // if(real_input && fft_plan->output_dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
        //     final_output_type = 2;
        // }

        VLOG(5) << "final_output_type: " << final_output_type;


        status = execGenerateChirpz(handle, fft_plan->bluestein_chirpz,
                                    fft_plan->n[0], fft_plan->n[0],
                                    fft_plan->PAD_N0, true,
                                    type); // chirpz 信号生成

        // // // VLOG(5) << "execComplexMatrixDotVector";
        // // step2
        bool row_major = true;
        status = execComplexMatrixDotVector(handle, fft_plan->bluestein_chirpz,
                                            input, fft_plan->bluestein_input,
                                            fft_plan->batch, 1, fft_plan->n[0],
                                            fft_plan->PAD_N0, row_major, real_input, type, 0); //
        
        // // //step3
        status = execGenerateChirpz(handle, fft_plan->bluestein_chirpz,
          fft_plan->n[0], fft_plan->n[0],
          fft_plan->PAD_N0, false,
          type); // 辅助信号生成
        // // VLOG(5) << "scale_factor " << scale_factor;
        cnrtDim3_t k_dim;
        cnrtFunctionType_t k_type;
        

        // configure
        configureFFT1dWorkspaceAddrs(handle, fft_plan->bluestein_plan,
                                     fft_plan->bluestein_chirpz, workspace,
                                     fft_plan->bluestein_chirpz);

        policyFunc(handle, &k_dim, &k_type);
        kernelFFT1dButterflyRow(k_dim, k_type, handle->queue, fft_plan->bluestein_plan,
                                0, FFT_IFFT);
        // step4
        // status = execFFT1d(handle, fft_plan->bluestein_plan,
        //   fft_plan->bluestein_chirpz, scale_factor, workspace,
        //   fft_plan->bluestein_chirpz, FFT_FORWARD);

        // 向量乘矩阵

        // configure
        // //step5
        // configure
        configureFFT1dWorkspaceAddrs(handle, fft_plan->bluestein_plan,
          fft_plan->bluestein_input, workspace,
          fft_plan->bluestein_input);

        policyFunc(handle, &k_dim, &k_type);
        kernelFFT1dButterflyRow(k_dim, k_type, handle->queue, fft_plan->bluestein_plan,
            0, FFT_IFFT);
            
        /*
        status = execFFT1d(handle, fft_plan->bluestein_plan,
          fft_plan->bluestein_input, scale_factor, workspace,
                    fft_plan->bluestein_input,FFT_FORWARD);
        */
        
        // // // step6
        status = execComplexMatrixDotVector(handle, fft_plan->bluestein_chirpz,
          fft_plan->bluestein_input, fft_plan->bluestein_output,
          fft_plan->batch, 1, fft_plan->PAD_N0,
          fft_plan->PAD_N0, row_major, false, type, 0); //



        // configure

        // step7
        // float scale_factor_2 = ;
        // VLOG(5) << "last fft" << scale_factor_2;
        configureFFT1dWorkspaceAddrs(handle, fft_plan->bluestein_plan,
          fft_plan->bluestein_output, workspace,
          fft_plan->bluestein_input);
 
        policyFunc(handle, &k_dim, &k_type);
        kernelFFT1dButterflyRow(k_dim, k_type, handle->queue, fft_plan->bluestein_plan,
            1, FFT_IFFT);

        /*
        status =
            execFFT1d(handle, fft_plan->bluestein_plan,
                      fft_plan->bluestein_output, 1/ (float)fft_plan->PAD_N0,
                      workspace, fft_plan->bluestein_input, FFT_BACKWARD);
        */
        
        // /step8
        status = execGenerateChirpz(handle, fft_plan->bluestein_chirpz,
          fft_plan->n[0], fft_plan->n[0],
          fft_plan->PAD_N0, true,
          type); // chirpz 信号生成


        // // step9
        VLOG(5) << "final_output_type " << final_output_type;
        status = execComplexMatrixDotVector(
            handle, fft_plan->bluestein_chirpz, fft_plan->bluestein_input,
            output, fft_plan->batch, 1, fft_plan->n[0], fft_plan->PAD_N0,
            row_major, false, type, final_output_type); //
        return status;
}


mluOpStatus_t bluesteinFFT1dColumn(mluOpHandle_t handle,
  const mluOpFFTPlan_t fft_plan,
  const void *input,
  const float scale_factor, void *workspace,
  void *output, const int direction) {
mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
std::string api = "[bluesteinFFT1dRow]";
 
VLOG(5) << "direction: " << direction
<< "1-direction: " << 1 - direction;
CHECK_RETURN(api, status);
// input should be contigues
// k 和 n 需要算出来
// // chirpz 信号生成
// step1
int type = 1;
// if(direction == 0) {
// type = -1;
// } else {
// type = 1;
// }
bool real_input = false;
// int final_output_type = 0;
if (fft_plan->input_dtype == MLUOP_DTYPE_FLOAT) {
real_input = true;
}

// if (fft_plan->input_dtype == MLUOP_DTYPE_COMPLEX_FLOAT && fft_plan->output_dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
// final_output_type = 1;
// }

// if(real_input && fft_plan->output_dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
// final_output_type = 2;
// }

// VLOG(5) << "final_output_type: " << final_output_type;

VLOG(5) << "n[0] pad_n0" << fft_plan->n[0] << " " << fft_plan->PAD_N0;



status = execGenerateChirpz(handle, fft_plan->bluestein_chirpz,
 fft_plan->n[0], fft_plan->n[0],
 fft_plan->PAD_N0, true,
 type); // chirpz 信号生成

// // // VLOG(5) << "execComplexMatrixDotVector";
// step2
bool row_major = false;


status = execComplexMatrixDotVector(handle, fft_plan->bluestein_chirpz,
         input, fft_plan->bluestein_input,
         fft_plan->batch, fft_plan->n[0], fft_plan->n[1],
         fft_plan->PAD_N0, row_major, real_input, type, 0); //


//step 3
status = execGenerateChirpz(handle, fft_plan->bluestein_chirpz,
        fft_plan->n[0], fft_plan->n[0],
        fft_plan->PAD_N0, false,
        type); // 辅助信号生成
// cnrtDim3_t k_dim;
// cnrtFunctionType_t k_type;
         
 


cnrtDim3_t k_dim;
cnrtFunctionType_t k_type;
policyFunc(handle, &k_dim, &k_type);
//          // configure
// // configureFFT1dWorkspaceAddrs(handle, fft_plan->bluestein_plan,
// //                              fft_plan->bluestein_chirpz, workspace,
// //                              fft_plan->bluestein_chirpz);
// fft_plan->mlu_addrs.buffer_buf = (int8_t *)workspace;


// step4
fft_plan->bluestein_plan->mlu_addrs.input = fft_plan->bluestein_chirpz;
fft_plan->bluestein_plan->mlu_addrs.output = fft_plan->bluestein_chirpz;

kernelFFT2dButterflyColumn(k_dim, k_type, handle->queue, fft_plan->bluestein_plan,
  FFT_FORWARD, FFT_IFFT);

// step5
fft_plan->bluestein_plan->mlu_addrs.input = fft_plan->bluestein_input;
fft_plan->bluestein_plan->mlu_addrs.output = fft_plan->bluestein_input;

// // policyFunc(handle, &k_dim, &k_type);
         


kernelFFT2dButterflyColumn(k_dim, k_type, handle->queue, fft_plan->bluestein_plan,
  FFT_FORWARD, FFT_IFFT);

// // // step6

status = execComplexMatrixDotVector(handle, fft_plan->bluestein_chirpz,
  fft_plan->bluestein_input, fft_plan->bluestein_output,
  fft_plan->batch, fft_plan->bluestein_plan->n[0], fft_plan->bluestein_plan->n[1],
  fft_plan->PAD_N0, row_major, real_input, type, 0); //

// status = execComplexMatrixDotVector(handle, fft_plan->bluestein_chirpz,
// fft_plan->bluestein_input, fft_plan->bluestein_output,
// fft_plan->batch, 1, fft_plan->PAD_N0,
// fft_plan->PAD_N0, row_major, false, type, 0); //


// step5
fft_plan->bluestein_plan->mlu_addrs.input = fft_plan->bluestein_output;
fft_plan->bluestein_plan->mlu_addrs.output = fft_plan->bluestein_output;

kernelFFT2dButterflyColumn(k_dim, k_type, handle->queue, fft_plan->bluestein_plan,
  FFT_BACKWARD, FFT_IFFT);

status = execGenerateChirpz(handle, fft_plan->bluestein_chirpz,
 fft_plan->n[0], fft_plan->n[0],
 fft_plan->PAD_N0, true,
 type); // chirpz 信号生成


status = execComplexMatrixDotVector(handle, fft_plan->bluestein_chirpz,
  fft_plan->bluestein_output, output,
  fft_plan->batch, fft_plan->n[0], fft_plan->n[1],
  fft_plan->PAD_N0, row_major, real_input, type, 0); //

// // step7
// VLOG(5) << "last fft" << scale_factor;
// status = execFFT1d(handle, fft_plan->bluestein_plan,
// fft_plan->bluestein_output, scale_factor, workspace,
// fft_plan->bluestein_input, FFT_FORWARD);

// /step8
// status = execGenerateChirpz(handle, fft_plan->bluestein_chirpz,
// fft_plan->n[0], fft_plan->n[0],
// fft_plan->PAD_N0, true,
// type); // chirpz 信号生成


// // // step9
// status = execComplexMatrixDotVector(handle, fft_plan->bluestein_chirpz,
// fft_plan->bluestein_input, output,
// fft_plan->batch, 1, fft_plan->n[0],
// fft_plan->PAD_N0, row_major, false, type, final_output_type); //

return status;
}


mluOpStatus_t execBluesteinFFT1d(mluOpHandle_t handle,
  const mluOpFFTPlan_t fft_plan,
  const void *input,
  const float scale_factor, void *workspace,
  void *output, const int direction) {
mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
std::string api = "[mluOpExecFFT]";
configureBluestein(handle, fft_plan, (void *)input,
         workspace, output);

status = bluesteinFFT1dRow(handle, fft_plan, input, scale_factor, workspace,
                           output, direction);
return status;
}

mluOpStatus_t execBluesteinFFT2d(mluOpHandle_t handle,
  const mluOpFFTPlan_t fft_plan,
  const void *input,
  const float scale_factor, void *workspace,
  void *output, const int direction) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  std::string api = "[mluOpExecFFT]";
  // configureBluesteinFFT2dWorkspaceAddrs(handle, fft_plan, (void *)input,
  //         workspace, output);
  configureBluestein(handle, fft_plan, (void *)input, workspace, output);

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
      if(fft_plan->bluestein_column && fft_plan->bluestein_row) {
      status = bluesteinFFT1dRow(handle, fft_plan, input, scale_factor, workspace,
                                output, direction);
      status = bluesteinFFT1dColumn(handle, fft_plan, input, scale_factor,
                                  workspace, output, direction);
      } else if (fft_plan->bluestein_column && !fft_plan->bluestein_row) {

        // if(fft_plan->n[0] !=1) {
        //   status = kernelFFT2dButterflyRow(k_dim, k_type, handle->queue, fft_plan,
        //                                    direction, FFT_IFFT);
        //   CHECK_RETURN(api, status);
        // }
        VLOG(5) << "bluesteinFFT1dColumn ONLY";
        status = bluesteinFFT1dColumn(handle, fft_plan, input, scale_factor,
                                      workspace, output, direction);

      } else {
        status = bluesteinFFT1dRow(handle, fft_plan, input, scale_factor, workspace,
          output, direction);
        // if(fft_plan->n[1] != 1) {
        //   status = kernelFFT2dButterflyColumn(k_dim, k_type, handle->queue,
        //       fft_plan, direction, FFT_IFFT);
        // }
      }
    }
  
    //   status = kernelFFT2dButterflyRow(k_dim, k_type, handle->queue, fft_plan,
    //                                    direction, FFT_IFFT);
    //   CHECK_RETURN(api, status);
        
    //   status = kernelFFT2dButterflyColumn(k_dim, k_type, handle->queue,
    //                                       fft_plan, direction, FFT_IFFT);
    //   CHECK_RETURN(api, status);
    // } else {
    //   status = kernelFFT2dButterflyColumn(k_dim, k_type, handle->queue,
    //                                       fft_plan, direction, FFT_IFFT);
        
    //   CHECK_RETURN(api, status);
        
    //   status = kernelFFT2dButterflyRow(k_dim, k_type, handle->queue, fft_plan,
    //                                              direction, FFT_IFFT);
    //   CHECK_RETURN(api, status);
    //   }
        
      fft_plan->mlu_addrs.input =
                  (void *)((uint64_t)(fft_plan->mlu_addrs.input) + idist);
      fft_plan->mlu_addrs.output =
                  (void *)((uint64_t)(fft_plan->mlu_addrs.output) + odist);
  }
  fft_plan->mlu_addrs.input = (void *)((uint64_t)(fft_plan->mlu_addrs.input) -
                                       fft_plan->batch * idist);
  fft_plan->mlu_addrs.output =(void *)((uint64_t)(fft_plan->mlu_addrs.output) -
                                       fft_plan->batch * odist);



  // if(direction == 0) {
  //   status = bluesteinFFT1dColumn(handle, fft_plan, input, scale_factor,
  //                                 workspace, output, direction);
  //   status = bluesteinFFT1dRow(handle, fft_plan, input, scale_factor, workspace,
  //                             output, direction);
  // } else {
  //   status = bluesteinFFT1dRow(handle, fft_plan, input, scale_factor, workspace,
  //     output, direction);
  //   status = bluesteinFFT1dColumn(handle, fft_plan, input, scale_factor,
  //       workspace, output, direction);
  // }
  return status;
}
