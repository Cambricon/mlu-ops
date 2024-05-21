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
#include <string>
#include "kernels/fft/fft.h"
#include "kernels/fft/rfft/rfft.h"
#include "kernels/fft/irfft/irfft.h"
#include "kernels/fft/c2c_fft/c2c_fft.h"

// May be use a common function is a better choice?
static inline bool supportFloatConv(mluOpHandle_t handle) {
  switch (handle->arch) {
    case MLUOP_MLU370:
      return true;
    default:
      return true;
  }
}

// Calculate whether the optimization strategy can be
// entered(CNFFT_FUNC_STOCKHAM and CNFFT_FUNC_COOLEY_TUKEY). If it can enter,
// select the optimal strategy and calculate corresponding parameters.
mluOpStatus_t selectFFTStrategy(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                                const std::string make_plan_api) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  fft_plan->fft_strategy = CNFFT_FUNC_MATMUL;
  // The basic conditions for entering the optimization.
  if (fft_plan->n[0] > 4096) {
    bool find_stockham = 0;
    // CNFFT_FUNC_STOCKHAM optimizaion currently has more retrictions as
    // follows:
    if (handle->arch >= 300 &&
        (fft_plan->execution_dtype == MLUOP_DTYPE_HALF ||
         fft_plan->execution_dtype == MLUOP_DTYPE_FLOAT)) {
      find_stockham = true;
    }
    // strategy_status: 0 means select MLUOP_FUNC_STOCKHAM, 1 means selelct
    // COOLEY_TUKEY,
    //                  -1 means still select CNFFT_FUNC_MATMUL.
    int strategy_status =
        findFFTOptLimit(handle, fft_plan->n[0], fft_plan->m, fft_plan->L,
                        fft_plan->s, fft_plan->L_sub, find_stockham);
    if (strategy_status == 1) {
      fft_plan->fft_strategy = CNFFT_FUNC_COOLEY_TUKEY;
    } else if (strategy_status == 0) {
      fft_plan->fft_strategy = CNFFT_FUNC_STOCKHAM;
    }
  }
  return status;
}

mluOpStatus_t MLUOP_WIN_API mluOpCreateFFTPlan(mluOpFFTPlan_t *fft_plan) {
  mluOpFFTStruct *ts = new (std::nothrow) mluOpFFTStruct();
  if (ts == nullptr) {
    LOG(ERROR) << "[mluOpCreateFFTPlan]: alloc failed";
    return MLUOP_STATUS_ALLOC_FAILED;
  }
  ts->factors = new int[FFT_MAXFACTORS];
  *fft_plan = ts;
  return MLUOP_STATUS_SUCCESS;
}

template <typename DT>
mluOpStatus_t MLUOP_WIN_API fftGenerateTwiddlesLine(
    void *_twiddles, const int butterfly_num, const int section_num,
    const int radix, const int nfft, const int dir) {
  int j, k;
  DT phase;
  DT *twiddles = (DT *)_twiddles;
  const int sign = (dir == FFT_FORWARD) ? -1 : 1;
  for (j = 0; j < butterfly_num; j++) {
    // phase = 1 when k = 0
    for (k = 1; k < radix; k++) {
      phase = sign * 2 * (DT)FFT_PI * section_num * k * j / nfft;
      twiddles[(butterfly_num * (k - 1) + j)] = (DT)cos(phase);  // r
      twiddles[(butterfly_num * (k - 1) + j) + butterfly_num * (radix - 1)] =
          (DT)sin(phase);  // i
      // twiddles[(butterfly_num * (k - 1) + j) * 2] = (DT)cos(phase);     // r
      // twiddles[(butterfly_num * (k - 1) + j) * 2 + 1] = (DT)sin(phase); // i
    }  // radix
  }    // butterfly_num
  return MLUOP_STATUS_SUCCESS;
}

/**
 * @details
 * @brief   The control interfaces of the generation of FFT's twiddles.
 * @param[in]   generator       twiddle generation function pointer.
 * @param[out]  *twiddles       stores the twiddles information generated in
 * this function, first stage's is ignored.
 * @param[in]   *factors         the way plan factoring the length of inpue
 * sequence.
 * @param[in]   nfft            the length of FFT.
 * @param[in]   dir             the flag for FFT/IFFT.
 * @return none
 */

template <typename DT>
mluOpStatus_t MLUOP_WIN_API fftGenerateTwiddles(void *&_twiddles, int *factors,
                                                const int _nfft,
                                                const int dir) {
  // twiddles = _twiddles;
  DT *twiddles = new DT[_nfft * 2 * 2];  // complex *2(large+small)
  _twiddles = twiddles;
  int stage_count = factors[0];
  int cur_large_radix, cur_small_radix, section_num, butterfly_num,
      loop_stage;  // current radix
  int tw_offset = 0;
  int small_stage_count, small_loop_stage, small_factors_offset;

  // for other stage, ignore first stage
  for (loop_stage = 2; loop_stage <= stage_count; loop_stage++) {
    cur_large_radix = factors[5 * loop_stage];
    section_num = factors[5 * loop_stage + 1];
    butterfly_num = factors[5 * loop_stage + 2];
    fftGenerateTwiddlesLine<DT>(twiddles, butterfly_num, section_num,
                                cur_large_radix, _nfft, dir);
    twiddles += butterfly_num * (cur_large_radix - 1) * 2;
    tw_offset += butterfly_num * (cur_large_radix - 1);
  }  // stage_count

  // do not ignore first stage
  for (loop_stage = 1; loop_stage <= stage_count; loop_stage++) {
    cur_large_radix = factors[5 * loop_stage];
    small_factors_offset = factors[5 * loop_stage + 4];
    small_stage_count = factors[small_factors_offset];
    factors[small_factors_offset + 2] = tw_offset;
    // cur_radix = factors[4 * loop_stage];
    // section_num = factors[4 * loop_stage + 1];
    // butterfly_num = factors[4 * loop_stage + 2];
    // butterfly_num = factors[4 * loop_stage + 2];
    // generator(twiddles, butterfly_num, section_num, cur_radix, _nfft, dir);
    // twiddles += butterfly_num * (cur_radix - 1);

    for (small_loop_stage = 2; small_loop_stage <= small_stage_count;
         small_loop_stage++) {
      cur_small_radix = factors[small_factors_offset + 4 * small_loop_stage];
      section_num = factors[small_factors_offset + 4 * small_loop_stage + 1];
      butterfly_num = factors[small_factors_offset + 4 * small_loop_stage + 2];
      // butterfly_num = factors[small_factors_offset + 4 * small_loop_stage +
      // 2];
      fftGenerateTwiddlesLine<DT>(twiddles, butterfly_num, section_num,
                                  cur_small_radix, cur_large_radix, dir);
      twiddles += butterfly_num * (cur_small_radix - 1) * 2;
      tw_offset +=
          butterfly_num * (cur_small_radix - 1);  // complex element offset
    }                                             // small_stage_count
  }                                               // stage_count

  return MLUOP_STATUS_SUCCESS;
}

template <typename DT>
mluOpStatus_t MLUOP_WIN_API fftGenerateDftMatrixKernel(DT *dft_matrix,
                                                       const int radix,
                                                       const int dir) {
  int j, k;
  DT phase;

  const int sign = (dir == FFT_FORWARD) ? -1 : 1;
  for (j = 0; j < radix; j++) {
    // phase = 1 when k = 0
    for (k = 0; k < radix; k++) {
      phase = sign * 2 * (DT)FFT_PI * k * j / radix;
      dft_matrix[radix * k + j] = (DT)cos(phase);                  // r
      dft_matrix[radix * k + j + radix * radix] = (DT)sin(phase);  // i
      // twiddles[(butterfly_num * (k - 1) + j) * 2] = (DT)cos(phase);     // r
      // twiddles[(butterfly_num * (k - 1) + j) * 2 + 1] = (DT)sin(phase); // i
    }  // radix
  }    // butterfly_num
  return MLUOP_STATUS_SUCCESS;
}

template <typename DT>
mluOpStatus_t MLUOP_WIN_API fftGenerateDftMatrix(void *&_dft_matrix,
                                                 int *factors, const int _nfft,
                                                 const int dir) {
  // allocate space for dft_matrix_table and dft_matrix
  DT *dft_matrix = new DT[DFT_TABLE_SIZE];  // complex *2(large+small)
  dft_table_entry *dft_matrix_table = (dft_table_entry *)dft_matrix;
  _dft_matrix = dft_matrix;

  dft_table_entry *dft_matrix_table_end = dft_matrix_table + MAX_DFT_MATRIX_NR;
  dft_matrix = (DT *)dft_matrix_table_end;

  int cur_table_entry = 0;

  // radix == -1 means the end of table
  // init table
  for (int i = 0; i < MAX_DFT_MATRIX_NR; i++) {
    dft_matrix_table[i] = {-1, -1};
  }
  // dft_matrix_table =
  int stage_count = factors[0];
  int cur_large_radix, cur_small_radix, section_num, butterfly_num,
      loop_stage;  // current radix
  int tw_offset = 0;
  int small_stage_count, small_loop_stage, small_factors_offset;

  // initialize offset as the end of table
  // transform  dft_table_entry to complex DT
  int cur_offset =
      (MAX_DFT_MATRIX_NR + 1) * sizeof(dft_table_entry) / (sizeof(DT) * 2);

  for (loop_stage = 1; loop_stage <= stage_count; loop_stage++) {
    cur_large_radix = factors[5 * loop_stage];
    small_factors_offset = factors[5 * loop_stage + 4];
    small_stage_count = factors[small_factors_offset];

    for (small_loop_stage = 2; small_loop_stage <= small_stage_count;
         small_loop_stage++) {
      cur_small_radix = factors[small_factors_offset + 4 * small_loop_stage];
      section_num = factors[small_factors_offset + 4 * small_loop_stage + 1];
      butterfly_num = factors[small_factors_offset + 4 * small_loop_stage + 2];

      for (int entry = 0;; entry++) {
        if (dft_matrix_table[entry].radix == -1) {
          DT *dft_matrix_real = dft_matrix;

          fftGenerateDftMatrixKernel<DT>(dft_matrix, cur_small_radix, dir);
          cur_table_entry++;
          dft_matrix_table[cur_table_entry] = {cur_small_radix, cur_offset};
          cur_offset += cur_small_radix * cur_small_radix;
          if (cur_table_entry == MAX_DFT_MATRIX_NR) {
            LOG(ERROR) << "[fftGenerateDftMatrix]: too much dft matrices";
          }

          break;
        }

        if (dft_matrix_table[entry].radix == cur_small_radix) {
          break;
        }
      }
    }  // small_stage_count
  }    // stage_count

  return MLUOP_STATUS_SUCCESS;
}

#define MaxLargeRadix 2048

// int fftTwoStepFactor(const int _n, int *facbuf)
// {
//   int n = _n;
//   if ((facbuf == NULL) || (n <= 0))
//   {
//     printf("ERROR, facbuf is NULL or n smaller and equal to 0.  __FILE__: %s,
//     __LINE__: %d. \n", __FILE__, __LINE__); exit(1);
//   }

//   int r, in_stride, section_num, stage_num = 0, out_stride = 1;
//   int radix_basic[] = {3, 4, 5, 6, 7, 8, 9, 11, 13, 16};

//   int large_radix = 1;

//   while (n > 1)
//   {

//     if ((n % 1024) == 0)
//     {
//       r = 1024;
//       // set small factors
//     }
//     else if ((n % 512) == 0)
//     {
//       r = 512;
//       // set small factors
//     }
//     else
//     {
//       // prime
//       // r = n;
//       large_radix = 1;
//       while (n > 1 || large_radix < MaxLargeRadix)
//       {
//         if ((n % 1024) == 0)
//         {
//           r = 1024;
//           // set small factors
//         }
//         else if ((n % 512) == 0)
//         {
//           r = 512;
//           // set small factors
//         }
//         else
//         {
//           r =n;
//         }
//       }
//     }

//     // printf("radix%d = %d\n", stage_num+1, r);
//     n /= r;
//     in_stride = _n / r;
//     section_num = n;
//     stage_num++;

//     facbuf[4 * stage_num] = r;
//     facbuf[4 * stage_num + 1] = section_num;
//     facbuf[4 * stage_num + 2] = out_stride;
//     facbuf[4 * stage_num + 3] = in_stride;

//     out_stride *= r;
//   }

//   facbuf[0] = stage_num;
//   facbuf[1] = _n;
//   facbuf[2] = 0;

//   if (stage_num > 21)
//   {
//     // Since nfft is openfft_int32_t, stage_num can never be greater than 21,
//     because 3^21 > 2^32 printf("ERROR, unsupported length for int32 type.
//     __FILE__: %s, __LINE__: %d. \n", __FILE__, __LINE__); exit(1);
//   }

//   return 0;
// }

#define MaxLargeRadix 2048

// mluOpStatus_t MLUOP_WIN_API mluOpMakeFFTPlanC2C1D (
//     mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
//     mluOpTensorDescriptor_t input_desc, mluOpTensorDescriptor_t output_desc,
//     const int rank, const int *n)

// data struct
// factors[0]: stage_count
// factors[1]: nfft
// factors[2]: null
// factors[3]: null
// factors[4]: null
// factors[5]: null

// i-th large radix info:
// factors[5*(i+1)+0]: radix
// factors[5*(i+1)+1]: section_num
// factors[5*(i+1)+2]: butterfly_num
// factors[5*(i+1)+3]: in_stride
// factors[5*(i+1)+4]: small_factors_offset

// factors[small_factors_offset+0]: small_stage_count
// factors[small_factors_offset+1]: large radix
// factors[small_factors_offset+2]: tw_offset

// i-th large radix, j-th small radix info:
// factors[small_factors_offset+4*(j+1)+0]: radix
// factors[small_factors_offset+4*(j+1)+1]: section_num
// factors[small_factors_offset+4*(j+1)+2]: butterfly_num
// factors[small_factors_offset+4*(j+1)+3]: in_stride

mluOpStatus_t MLUOP_WIN_API fftFactor(const int _n, int *facbuf,
                                      int &small_factors_offset) {
  int n = _n;
  // if ((facbuf == NULL) || (n <= 0))
  // {
  //   printf("ERROR, facbuf is NULL or n smaller and equal to 0.  __FILE__: %s,
  //   __LINE__: %d. \n", __FILE__, __LINE__); exit(1);
  // }

  int r, in_stride, section_num, stage_num = 0, out_stride = 1;
  int radix_basic[] = {3, 4, 5, 6, 7, 8, 9, 11, 13, 16};

  int large_radix = 1;
  facbuf += small_factors_offset;

  while (n > 1) {
    if ((n % 1024) == 0) {
      r = 1024;
      // set small factors
    } else if ((n % 512) == 0) {
      r = 512;
      // set small factors
    } else if ((n % 9) == 0) {
      r = 9;
    } else if ((n % 3) == 0) {
      r = 3;
    } else {
      // prime
      r = n;

      // large_radix = 1;
      // while (n > 1 || large_radix < MaxLargeRadix)
      // {
      //   if ((n % 1024) == 0)
      //   {
      //     r = 1024;
      //     // set small factors
      //   }
      //   else if ((n % 512) == 0)
      //   {
      //     r = 512;
      //     // set small factors
      //   }
      //   else
      //   {
      //     r =n;
      //   }
      // }
    }

    n /= r;
    in_stride = _n / r;
    section_num = n;
    stage_num++;

    facbuf[4 * stage_num + 0] = r;
    facbuf[4 * stage_num + 1] = section_num;
    facbuf[4 * stage_num + 2] = out_stride;
    facbuf[4 * stage_num + 3] = in_stride;

    out_stride *= r;
  }

  facbuf[0] = stage_num;
  facbuf[1] = _n;
  facbuf[2] = 0;  // tw_offset
  facbuf[3] = 0;

  if (stage_num > 21) {
    // Since nfft is openfft_int32_t, stage_num can never be greater than 21,
    // because 3^21 > 2^32 printf("ERROR, unsupported length for int32 type.
    // __FILE__: %s, __LINE__: %d. \n", __FILE__, __LINE__); exit(1);
  }

  small_factors_offset += (stage_num + 1) * 4;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API fftTwoStepFactor(const int _n, int *facbuf) {
  int n = _n;
  // if ((facbuf == NULL) || (n <= 0))
  // {
  //   printf("ERROR, facbuf is NULL or n smaller and equal to 0.  __FILE__: %s,
  //   __LINE__: %d. \n", __FILE__, __LINE__); exit(1);
  // }

  int r, in_stride, section_num, stage_num = 0, out_stride = 1;
  int radix_basic[] = {3, 4, 5, 6, 7, 8, 9, 11, 13, 16};

  int large_radix = 1;
  int small_factors_offset = 22 * 5;

  while (n > 1) {
    if ((n % 1024) == 0) {
      r = 1024;
      // set small factors
    } else if ((n % 512) == 0) {
      r = 512;
      // set small factors
    } else if ((n % 2187) == 0) {
      r = 2187;
    } else if ((n % 729) == 0) {
      r = 729;
    } else if ((n % 243) == 0) {
      r = 243;
    } else if ((n % 81) == 0) {
      r = 81;
    } else if ((n % 27) == 0) {
      r = 27;
    } else if ((n % 9) == 0) {
      r = 9;
    } else if ((n % 3) == 0) {
      r = 3;
      // fftFactor(r, facbuf, small_factors_offset);
      // small_stage_count = 2;
      // std::cout<< "r1=3, r2=3"<< std::endl;
      // set small factors

      // factors[small_factors_offset+5*j+0]: radix
      // factors[small_factors_offset+5*j+1]: section_num
      // factors[small_factors_offset+5*j+2]: butterfly_num
      // factors[small_factors_offset+5*j+3]: in_stride
      // factors[small_factors_offset+5*j+4]: tw_offset
    } else {
      // prime
      r = n;

      // large_radix = 1;
      // while (n > 1 || large_radix < MaxLargeRadix)
      // {
      //   if ((n % 1024) == 0)
      //   {
      //     r = 1024;
      //     // set small factors
      //   }
      //   else if ((n % 512) == 0)
      //   {
      //     r = 512;
      //     // set small factors
      //   }
      //   else
      //   {
      //     r =n;
      //   }
      // }
    }

    // printf("radix%d = %d\n", stage_num+1, r);
    n /= r;
    in_stride = _n / r;
    section_num = n;
    stage_num++;

    facbuf[5 * stage_num + 0] = r;
    facbuf[5 * stage_num + 1] = section_num;
    facbuf[5 * stage_num + 2] = out_stride;
    facbuf[5 * stage_num + 3] = in_stride;
    facbuf[5 * stage_num + 4] = small_factors_offset;

    fftFactor(r, facbuf, small_factors_offset);
    // facbuf[6*stage_num+4] = small_stage_count;
    // facbuf[6*stage_num+5] = small_factors_offset;

    // facbuf[4 * stage_num] = r;
    // facbuf[4 * stage_num + 1] = section_num;
    // facbuf[4 * stage_num + 2] = out_stride;
    // facbuf[4 * stage_num + 3] = in_stride;

    out_stride *= r;
  }

  facbuf[0] = stage_num;
  facbuf[1] = _n;
  facbuf[2] = 0;
  facbuf[3] = 0;
  facbuf[4] = 0;

  VLOG(5) << "stage_num: " << stage_num << " _n: " << _n;

  if (stage_num > 21) {
    // Since nfft is openfft_int32_t, stage_num can never be greater than 21,
    // because 3^21 > 2^32 printf("ERROR, unsupported length for int32 type.
    // __FILE__: %s, __LINE__: %d. \n", __FILE__, __LINE__); exit(1);
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpAllocateC2C1D(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                   mluOpTensorDescriptor_t input_desc,
                   mluOpTensorDescriptor_t output_desc, const int nfft) {
  const std::string make_plan_api = "[mluOpAllocateC2C1D]";
  size_t workspace_size = 0;
  size_t reservespace_size = 0;

  size_t CPX_TYPE_SIZE = 0;

  switch (fft_plan->fft_type) {
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF: {
      CPX_TYPE_SIZE = 2 * 2;
    } break;
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
      CPX_TYPE_SIZE = 4 * 2;
    }; break;
    default: {
      LOG(ERROR) << make_plan_api << ": invalid c2c 1d fft type.";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  int batch = fft_plan->batch;

  size_t buffer_size = batch * sizeof(CPX_TYPE_SIZE) * nfft;

  workspace_size = buffer_size * 3;

  // reservespace_size = batch * sizeof(mluOpFFTPlan_t) + sizeof(int) *
  // (FFT_MAXFACTORS) /* factors */
  //                              + sizeof(CPX_TYPE_SIZE) * nfft * 2 /* twiddles
  //                              */
  //                             );

  size_t twiddles_size = sizeof(CPX_TYPE_SIZE) * nfft * 2;
  reservespace_size = sizeof(int) * (FFT_MAXFACTORS)    /* factors */
                      + twiddles_size + DFT_TABLE_SIZE; /* twiddles */

  fft_plan->workspace_size = workspace_size;
  fft_plan->reservespace_size = reservespace_size;

  // std::cout << "workspace_size: " << workspace_size << "bytes" << std::endl;
  // std::cout << "reservespace_size: " << reservespace_size << "bytes" <<
  // std::endl; CNAME(openfft_generate_twiddles)(st->twiddles, st->factors,
  // nfft, st->dir);

  return MLUOP_STATUS_SUCCESS;
}

/**
 * @degroup C2C_PLAN Floating Complex-to-Complex FFT plan
 */

mluOpStatus_t MLUOP_WIN_API mluOpMakeFFTPlanC2C1D(
    mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
    mluOpTensorDescriptor_t input_desc, mluOpTensorDescriptor_t output_desc,
    const int rank, const int *n, const int direction) {
  // reservespace_addr_ = mlu_runtime_.allocate(reservespace_size_)
  // st = CNAME(openfft_allocate_c2c_plan_1d)(nfft, fin, fout, dir);

  // std::cout<< "mluOpAllocateC2C1D"<<std::endl;
  mluOpAllocateC2C1D(handle, fft_plan, input_desc, output_desc, n[0]);
  // std::cout<< "mluOpAllocateC2C1D"<<std::endl;
  fftTwoStepFactor(n[0], fft_plan->factors);
  // result = openfft_factor(nfft, st->factors);
  // if (result == OPENFFT_ERR)
  // {
  //     openfft_aligned_free(st);
  //     return NULL;
  // }

  switch (fft_plan->fft_type) {
    case CNFFT_FLOAT2COMPLEX_FLOAT:
    case CNFFT_COMPLEX_FLOAT2FLOAT:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT:
      fftGenerateTwiddles<float>(fft_plan->twiddles, fft_plan->factors, n[0],
                                 direction);
      fftGenerateDftMatrix<float>(fft_plan->dft_matrix, fft_plan->factors, n[0],
                                  direction);
      break;
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
      // fftGenerateTwiddles<half>(fft_plan->twiddles,
      //                           fft_plan->factors,
      //                           n[0],
      //                           direction);

      // TODO(zrg): need to copy twiddles to device, and convert to half.
      fftGenerateTwiddles<float>(fft_plan->twiddles, fft_plan->factors, n[0],
                                 direction);
      fftGenerateDftMatrix<float>(fft_plan->dft_matrix, fft_plan->factors, n[0],
                                  direction);
      break;
    default:
      break;
  }

  // if(fft_plan->twiddles == NULL){
  //   std::cout<<" \n\n\n fft_plan->twiddles == NULL \n\n\n"<<std::endl;
  // }
  // CNAME(openfft_generate_twiddles)(st->twiddles, st->factors, nfft, st->dir);

  return MLUOP_STATUS_SUCCESS;
}

/**
 * This function
 * 1. receives parameters from user;
 * 2. checks the validity of the parameters;
 * 3. picks up the supported circumstances;
 * 4. make decisions which strategy should use.
 */
mluOpStatus_t MLUOP_WIN_API mluOpMakeFFTPlanMany(
    mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
    mluOpTensorDescriptor_t input_desc, mluOpTensorDescriptor_t output_desc,
    const int rank, const int *n, size_t *reservespace_size,
    size_t *workspace_size, const int direction) {
  // bad param check
  const std::string make_plan_api = "[mluOpMakeFFTPlanMany]";
  // plan NULL check
  PARAM_CHECK_NE(make_plan_api, handle, NULL);
  if (fft_plan == NULL) {
    LOG(ERROR) << make_plan_api + ": plan is not allocated.";
    return MLUOP_STATUS_NOT_INITIALIZED;
  }
  PARAM_CHECK_NE(make_plan_api, input_desc, NULL);
  PARAM_CHECK_NE(make_plan_api, output_desc, NULL);
  PARAM_CHECK_NE(make_plan_api, n, NULL);
  PARAM_CHECK_NE(make_plan_api, reservespace_size, NULL);
  PARAM_CHECK_NE(make_plan_api, workspace_size, NULL);

  // plan rank can only be 1, 2, 3
  if (rank < 1 || rank > FFT_DIM_MAX) {
    LOG(ERROR) << make_plan_api + ": invalid rank, should be 1, 2 or 3. Now is "
               << rank << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  for (auto i = 0; i < rank; i++) {
    if (n[i] <= 0) {
      LOG(ERROR)
          << make_plan_api +
                 ": fourier transform length should be greater than 0. Now n["
          << i << "] is " << n[i] << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  fft_plan->rank = rank;
  for (auto i = 0; i < rank; i++) {
    fft_plan->n[i] = n[i];
  }

  // dimension check
  fft_plan->idim = input_desc->dim;
  fft_plan->odim = output_desc->dim;
  fft_plan->inum = mluOpGetTensorElementNum(input_desc);
  fft_plan->onum = mluOpGetTensorElementNum(output_desc);
  PARAM_CHECK_GT(make_plan_api, input_desc->dim, 0);
  PARAM_CHECK_EQ(make_plan_api, fft_plan->idim, fft_plan->odim,
                 ": input and output dimension mismatch.");

  if (!(fft_plan->idim == rank || fft_plan->idim == rank + 1)) {
    LOG(ERROR) << make_plan_api << ": invalid input dimension, should be "
               << rank << " or " << rank + 1 << ". Now is " << fft_plan->idim
               << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // batch check
  if (fft_plan->idim == rank) {
    fft_plan->batch = 1;
  } else {  // idim == rank + 1
    fft_plan->batch = input_desc->dims[0];
    PARAM_CHECK_EQ(make_plan_api, fft_plan->batch, output_desc->dims[0],
                   ": batch size mismatch.");
  }

  // The FFT Struct is designed after cufftXtMakePlanMany.
  // An element of coordinates [z, y, x] in signal number b in the batch will
  // be associated with the following addresses in the memory
  // 1-D:
  //   input[b * idist + x * istride]
  //   output[b * odist + x * ostride]
  // 2-D:
  //   input[b * idist + (x * inembed[1] + y) * istride]
  //   output[b * odist + (x * onembed[1] + y) * istride]
  // 3-D:
  //   input[b * idist + ((x * inembed[1] + y) * inembed[2] + z) * istride]
  //   output[b * odist + ((x * onembed[1] + y) * onembed[2] + z) * ostride]
  // Thus, cufft and fftw advanced data layout is a subset of mluOp advanced
  // data layout with tensor dim strides. 2-D and 3-D should pay attention.
  // stride check, if an in-place fft is adopted check, `istride` should be
  // equal to `ostride`.
  fft_plan->istride = input_desc->strides[fft_plan->idim - 1];
  fft_plan->ostride = output_desc->strides[fft_plan->odim - 1];

  PARAM_CHECK_GE(make_plan_api, fft_plan->istride, 0,
                 ": input stride should be greater than or equal to 0.");
  PARAM_CHECK_GE(make_plan_api, fft_plan->ostride, 0,
                 ": output stride should be greater than or equal to 0.");

  for (auto i = 0; i < fft_plan->rank; i++) {
    fft_plan->inembed[i] = input_desc->dims[fft_plan->idim - rank + i];
    fft_plan->onembed[i] = output_desc->dims[fft_plan->odim - rank + i];
  }
  if (fft_plan->idim == rank + 1) {
    fft_plan->idist = input_desc->strides[0];
    fft_plan->odist = output_desc->strides[0];
  } else {  // batch == 1
    fft_plan->idist = mluOpGetTensorElementNum(input_desc) / fft_plan->batch;
    fft_plan->odist = mluOpGetTensorElementNum(output_desc) / fft_plan->batch;
  }
  fft_plan->is_input_contiguous = !mluop::ifNeedTensorStrideProcess(input_desc);
  fft_plan->is_output_contiguous =
      !mluop::ifNeedTensorStrideProcess(output_desc);

  // dtype check
  mluOpDataType_t input_dtype = input_desc->dtype;
  mluOpDataType_t output_dtype = output_desc->dtype;
  const mluOpDataType_t f_c_dtype = MLUOP_DTYPE_COMPLEX_FLOAT;
  const mluOpDataType_t f_r_dtype = MLUOP_DTYPE_FLOAT;
  const mluOpDataType_t hf_c_dtype = MLUOP_DTYPE_COMPLEX_HALF;
  const mluOpDataType_t hf_r_dtype = MLUOP_DTYPE_HALF;
  if (input_dtype == hf_r_dtype && output_dtype == hf_c_dtype) {
    fft_plan->fft_type = CNFFT_HALF2COMPLEX_HALF;
  } else if (input_dtype == hf_c_dtype && output_dtype == hf_c_dtype) {
    fft_plan->fft_type = CNFFT_COMPLEX_HALF2COMPLEX_HALF;
  } else if (input_dtype == hf_c_dtype && output_dtype == hf_r_dtype) {
    fft_plan->fft_type = CNFFT_COMPLEX_HALF2HALF;
  } else if (input_dtype == f_r_dtype && output_dtype == f_c_dtype) {
    fft_plan->fft_type = CNFFT_FLOAT2COMPLEX_FLOAT;
  } else if (input_dtype == f_c_dtype && output_dtype == f_c_dtype) {
    fft_plan->fft_type = CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT;
  } else if (input_dtype == f_c_dtype && output_dtype == f_r_dtype) {
    fft_plan->fft_type = CNFFT_COMPLEX_FLOAT2FLOAT;
  } else {
    LOG(ERROR) << make_plan_api
               << ": invalid data type combination. Now input data type is "
               << mluOpGetNameOfDataType(input_dtype)
               << ", and output data type is "
               << mluOpGetNameOfDataType(output_dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  fft_plan->input_dtype = input_desc->dtype;
  fft_plan->output_dtype = output_desc->dtype;
  fft_plan->execution_dtype = input_desc->onchip_dtype;

  VLOG(5) << "input data type: "
          << mluOpGetNameOfDataType(fft_plan->input_dtype);
  VLOG(5) << "output data type: "
          << mluOpGetNameOfDataType(fft_plan->output_dtype);
  VLOG(5) << "execution data type: "
          << mluOpGetNameOfDataType(fft_plan->execution_dtype);

  // fft length check
  for (auto i = 0; i < fft_plan->rank - 1; i++) {  // except the last dim
    PARAM_CHECK_EQ(
        make_plan_api, n[i], fft_plan->inembed[i],
        ": the signal lengths of fft and input mismatch in dimention ", i, ".");
    PARAM_CHECK_EQ(
        make_plan_api, n[i], fft_plan->onembed[i],
        ": the signal lengths of fft and output mismatch in dimension ", i,
        ".");
  }
  switch (fft_plan->fft_type) {
    // r2c
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_FLOAT2COMPLEX_FLOAT: {
      PARAM_CHECK_EQ(make_plan_api, fft_plan->n[rank - 1] / 2 + 1,
                     fft_plan->onembed[rank - 1],
                     ": the signal lengths of fft and output last dimention "
                     "mismatch in R2C.");
    }; break;
    // c2c
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
      PARAM_CHECK_EQ(make_plan_api, fft_plan->n[rank - 1],
                     fft_plan->onembed[rank - 1],
                     ": the signal lengths of fft and output last dimention "
                     "mismatch in C2C.");
    }; break;
    // c2r
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_FLOAT2FLOAT: {
      PARAM_CHECK_EQ(make_plan_api, fft_plan->n[rank - 1],
                     fft_plan->onembed[rank - 1],
                     ": the signal lengths of fft and output last dimention "
                     "mismatch in C2R.");
    }; break;
    default: {
      LOG(ERROR) << make_plan_api << ": invalid fft type.";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  mluOpDataType_t execution_dtype = fft_plan->execution_dtype;
  switch (fft_plan->fft_type) {
    // half
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_HALF2HALF: {
      if (supportFloatConv(handle)) {
        if (!(execution_dtype == hf_r_dtype ||
              execution_dtype == MLUOP_DTYPE_INT16)) {
          LOG(ERROR) << make_plan_api << ": invalid execution dtype "
                     << mluOpGetNameOfDataType(fft_plan->execution_dtype)
                     << ".";
          return MLUOP_STATUS_BAD_PARAM;
        }
      } else {
        if (!(execution_dtype == MLUOP_DTYPE_INT16)) {
          LOG(ERROR) << make_plan_api << ": invalid execution dtype "
                     << mluOpGetNameOfDataType(fft_plan->execution_dtype)
                     << ".";
          return MLUOP_STATUS_BAD_PARAM;
        }
      }
    }; break;
    // float
    case CNFFT_FLOAT2COMPLEX_FLOAT:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT:
    case CNFFT_COMPLEX_FLOAT2FLOAT: {
      if (supportFloatConv(handle)) {
        if (!(execution_dtype == f_r_dtype ||
              execution_dtype == MLUOP_DTYPE_INT31)) {
          LOG(ERROR) << make_plan_api << ": invalid execution dtype "
                     << mluOpGetNameOfDataType(fft_plan->execution_dtype)
                     << ".";
          return MLUOP_STATUS_BAD_PARAM;
        }
      } else {
        if (!(execution_dtype == MLUOP_DTYPE_INT31)) {
          LOG(ERROR) << make_plan_api << ": invalid execution dtype "
                     << mluOpGetNameOfDataType(fft_plan->execution_dtype)
                     << ".";
          return MLUOP_STATUS_BAD_PARAM;
        }
      }
    }; break;
    default: {
      LOG(ERROR) << make_plan_api << ": invalid execution dtype.";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  // unsupported param
  if (fft_plan->rank != 1) {
    LOG(ERROR)
        << make_plan_api
        << ": 2-dimensional and 3-dimensional FFT are not supported currently.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  if (fft_plan->fft_type == CNFFT_HALF2COMPLEX_HALF ||
      fft_plan->fft_type == CNFFT_COMPLEX_HALF2HALF ||
      fft_plan->fft_type == CNFFT_COMPLEX_HALF2COMPLEX_HALF) {
    if ((n[0] & (n[0] - 1)) != 0) {
      LOG(ERROR) << make_plan_api
                 << ": the signal lengths of half-precision FFT are"
                 << " restriced to power of two only, but now is " << n[0]
                 << ".";
      return MLUOP_STATUS_NOT_SUPPORTED;
    }
  }

  // create input and output descriptor for gen_case
  // because mluOpExecFFT don't have input and output descriptor
  mluOpTensorDescriptor_t fft_input_desc, fft_output_desc;
  INTERNAL_CHECK(make_plan_api, mluOpCreateTensorDescriptor(&fft_input_desc) ==
                                    MLUOP_STATUS_SUCCESS);
  INTERNAL_CHECK(make_plan_api, mluOpCreateTensorDescriptor(&fft_output_desc) ==
                                    MLUOP_STATUS_SUCCESS);
  INTERNAL_CHECK(make_plan_api,
                 mluOpSetTensorDescriptorEx_v2(
                     fft_input_desc, input_desc->layout, input_desc->dtype,
                     input_desc->dim, input_desc->dims,
                     input_desc->strides) == MLUOP_STATUS_SUCCESS);
  INTERNAL_CHECK(make_plan_api, mluOpSetTensorDescriptorOnchipDataType(
                                    fft_input_desc, input_desc->onchip_dtype) ==
                                    MLUOP_STATUS_SUCCESS);
  INTERNAL_CHECK(make_plan_api,
                 mluOpSetTensorDescriptorEx_v2(
                     fft_output_desc, output_desc->layout, output_desc->dtype,
                     output_desc->dim, output_desc->dims,
                     output_desc->strides) == MLUOP_STATUS_SUCCESS);
  fft_plan->input_desc = fft_input_desc;
  fft_plan->output_desc = fft_output_desc;

  /*
   * decision part
   */
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  switch (fft_plan->fft_type) {
    // r2c
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_FLOAT2COMPLEX_FLOAT: {
      if (rank == 1) {
        VLOG(5) << "into make RFFT1d Policy";
        status = makeRFFT1dPolicy(handle, fft_plan);
      }
    }; break;
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_FLOAT2FLOAT: {
      if (rank == 1) {
        VLOG(5) << "into make IRFFT1d Policy";
        status = makeIRFFT1dPolicy(handle, fft_plan);
      }
    }; break;
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
      if (rank == 1) {
        VLOG(5) << "into make FFT1d Policy";
        // status = makeFFT1dPolicy(handle, fft_plan);
        // C2C 1D
        status = mluOpMakeFFTPlanC2C1D(handle, fft_plan, input_desc,
                                       output_desc, rank, n, direction);
      }
    }; break;
  }
  if (status != MLUOP_STATUS_SUCCESS) {
    return status;
  }

  *reservespace_size = fft_plan->reservespace_size;
  *workspace_size = fft_plan->workspace_size;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpDestroyFFTPlan(mluOpFFTPlan_t fft_plan) {
  const std::string destroy_api = "[mluOpDestroyFFTPlan]";
  PARAM_CHECK_NE("[mluOpDestroyFFTPlan]", fft_plan, NULL);
  if (fft_plan->input_desc != NULL) {
    INTERNAL_CHECK(destroy_api,
                   mluOpDestroyTensorDescriptor(fft_plan->input_desc) ==
                       MLUOP_STATUS_SUCCESS);
  }
  if (fft_plan->output_desc != NULL) {
    INTERNAL_CHECK(destroy_api,
                   mluOpDestroyTensorDescriptor(fft_plan->output_desc) ==
                       MLUOP_STATUS_SUCCESS);
  }
  if (fft_plan->factors != NULL) {
    delete fft_plan->factors;
  }

  if (fft_plan->twiddles != NULL) {
    delete (char *)fft_plan->twiddles;
  }

  delete fft_plan;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetFFTReserveArea(mluOpHandle_t handle,
                                                   mluOpFFTPlan_t fft_plan,
                                                   void *reservespace) {
  const std::string api = "[mluOpSetReserveArea]";
  PARAM_CHECK_NE(api, handle, NULL);
  PARAM_CHECK_NE(api, fft_plan, NULL);
  PARAM_CHECK_NE(api, reservespace, NULL);
  fft_plan->reservespace_addr = reservespace;
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  switch (fft_plan->fft_type) {
    // r2c
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_FLOAT2COMPLEX_FLOAT: {
      if (fft_plan->rank == 1) {
        status = setRFFT1dReserveArea(handle, fft_plan, api);
      } else {
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
    // c2c
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
      if (fft_plan->rank == 1) {
        // status = setFFT1dReserveArea(handle, fft_plan, api);
        status = setFFT1dReserveArea_v2(handle, fft_plan, api);
      } else {
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
    // c2r
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_FLOAT2FLOAT: {
      if (fft_plan->rank == 1) {
        status = setIRFFT1dReserveArea(handle, fft_plan, api);
      } else {
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
  }
  return status;
}

mluOpStatus_t MLUOP_WIN_API mluOpExecFFT(
    mluOpHandle_t handle, const mluOpFFTPlan_t fft_plan, const void *input,
    const float scale_factor, void *workspace, void *output, int direction) {
  const std::string exec_api = "[mluOpExecFFT]";
  PARAM_CHECK_NE(exec_api, handle, NULL);
  PARAM_CHECK_NE(exec_api, fft_plan, NULL);
  VLOG(5) << "input contiguous ? " << fft_plan->is_input_contiguous;
  VLOG(5) << "output contiguous ? " << fft_plan->is_output_contiguous;

  if (fft_plan->batch == 0) {
    VLOG(5) << "[mluOpExecFFT] Skip zero element tensor";
    return MLUOP_STATUS_SUCCESS;
  }
  // generate mluOpFFTExec prototxt start!
  {
    TENSOR_NUM_CHECK("[mluOpFft]",
                     mluOpGetTensorElementNum(fft_plan->input_desc),
                     LARGE_TENSOR_NUM, "");
    TENSOR_NUM_CHECK("[mluOpFft]",
                     mluOpGetTensorElementNum(fft_plan->output_desc),
                     LARGE_TENSOR_NUM, "");
  }

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("fft");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", input, fft_plan->input_desc, 1, 0);
    GEN_CASE_DATA(false, "output", output, fft_plan->output_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "fft", "rank", fft_plan->rank);
    GEN_CASE_OP_PARAM_ARRAY(1, "fft", "n", fft_plan->n, fft_plan->rank);
    GEN_CASE_OP_PARAM_SINGLE(1, "fft", "direction", direction);
    GEN_CASE_OP_PARAM_SINGLE(2, "fft", "scale_factor", scale_factor);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  if (fft_plan->workspace_size > 0) {
    PARAM_CHECK_NE(exec_api, workspace, NULL);
  }
  if (fft_plan->inum > 0) {
    PARAM_CHECK_NE(exec_api, input, NULL);
  }
  PARAM_CHECK_NE(exec_api, output, NULL);
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  bool is_in_place = (input == output);
  VLOG(5) << exec_api << ": in place ? " << is_in_place;
  switch (fft_plan->fft_type) {
    // r2c
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_FLOAT2COMPLEX_FLOAT: {
      if ((fft_plan->idist < (fft_plan->odist * 2)) && is_in_place) {
        LOG(ERROR)
            << exec_api
            << ": output overwritten may occur during an in-place "
               "real-to-complex fft "
               "execution, input needs to be slightly padding. Please refer to "
               "mluOpExecFFT document for detail.";
        status = MLUOP_STATUS_BAD_PARAM;
      }
      if (fft_plan->rank == 1) {
        status = execRFFT1d(handle, fft_plan, input, scale_factor, workspace,
                            output);
      } else if (fft_plan->rank == 2) {
        // TODO(who)
        status = MLUOP_STATUS_NOT_SUPPORTED;
      } else if (fft_plan->rank == 3) {
        // TODO(who)
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
    // c2c
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
      if ((fft_plan->idist < fft_plan->odist) && is_in_place) {
        LOG(ERROR)
            << exec_api
            << ": output overwritten may occur during an in-place "
               "complex-to-complex fft "
               "execution, input needs to be slightly padding. Please refer to "
               "mluOpExecFFT document for detail.";
        status = MLUOP_STATUS_BAD_PARAM;
      }
      if (fft_plan->rank == 1) {
        status = execFFT1d(handle, fft_plan, input, scale_factor, workspace,
                           output, direction);
      } else if (fft_plan->rank == 2) {
        // TODO(who)
        status = MLUOP_STATUS_NOT_SUPPORTED;
      } else if (fft_plan->rank == 3) {
        // TODO(who)
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
    // c2r
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_FLOAT2FLOAT: {
      if (((fft_plan->idist * 2) < fft_plan->odist) && is_in_place) {
        LOG(ERROR)
            << exec_api
            << ": output overwritten may occur during an in-place "
               "complex-to-real fft "
               "execution, input needs to be slightly padding. Please refer to "
               "mluOpExecFFT document for detail.";
        status = MLUOP_STATUS_BAD_PARAM;
      }
      if (fft_plan->rank == 1) {
        status = execIRFFT1d(handle, fft_plan, input, scale_factor, workspace,
                             output);
      } else if (fft_plan->rank == 2) {
        // TODO(who)
        status = MLUOP_STATUS_NOT_SUPPORTED;
      } else if (fft_plan->rank == 3) {
        // TODO(who)
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
  }

  GEN_CASE_END();
  return status;
}
