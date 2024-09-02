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
#pragma once
#include "kernels/fft/fft_optm_device/fft_generic_butterfly.h"

// Compute the large butterfly for the first stage from real to complex (R2C)
template <typename DT>
__mlu_func__ void computeLargeButterflyFirststageR2C(
    DT *output, DT *input, const int large_radix, const int large_in_stride,
    const int section_num, const DT *small_twiddles,
    const int small_twiddles_size, const DT *dft_matrix, void *nram_buf,
    const int *small_factors, const int nfft, const int last_stage,
    const int load_once_twiddles) {
  // constant
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;

  const int K_num = 64 / sizeof(DT);
  int align_K = 0;
  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;
  int tw_offset;
  _small_stage_count = small_factors[0];
  tw_offset = small_factors[1];

  const int max_para_ldst_num = small_factors[3];

  // assign nram space
  int nram_buf_offset = 0;
  DT *nram_in_r = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  DT *nram_in_i = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  DT *nram_out_r = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  DT *nram_out_i = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  // parallel load/store space
  DT *nram_para_load_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *nram_para_load_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *nram_para_store_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *nram_para_store_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *_nram_tw = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * 2;  // complex       //todo
  if (small_twiddles_size) {
    if (load_once_twiddles) {
      __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, SRAM2NRAM);
    } else {
      __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, GDRAM2NRAM);
    }
  }

  // load dftmtx sample
  int ld_dft_radix = -1;
  const int max_radix = 64;
  DT *nram_dftmtx = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += max_radix * max_radix * 2;  // complex

  DT *nram_scratch = (DT *)nram_buf + nram_buf_offset;

  int repeat_num = (section_num + max_para_ldst_num - 1) / max_para_ldst_num;

  for (int repeat_id = 0; repeat_id < repeat_num + 2; ++repeat_id) {
    // pipeline: load-stage
    if (repeat_id < repeat_num) {
      int i = max_para_ldst_num * repeat_id;
      DT *nram_para_load =
          (repeat_id % 2 == 0) ? nram_para_load_ping : nram_para_load_pong;

      int para_load_num = (max_para_ldst_num > (section_num - i))
                              ? (section_num - i)
                              : max_para_ldst_num;
      if (section_num == 1) {
        __memcpy_async(nram_para_load, input, sizeof(DT) * large_radix,
                       GDRAM2NRAM);  // Real FFT
      } else {
        // gather load
        __memcpy_async(nram_para_load, input + i, sizeof(DT) * para_load_num,
                       GDRAM2NRAM, sizeof(DT) * para_load_num,
                       large_in_stride * sizeof(DT), large_radix - 1);
      }  // Real FFT
    }

    // pipeline: store-stage
    if (repeat_id >= 2) {
      int i = max_para_ldst_num * (repeat_id - 2);

      int para_store_num = (max_para_ldst_num > (section_num - i))
                               ? (section_num - i)
                               : max_para_ldst_num;

      DT *nram_para_store =
          (repeat_id % 2 == 0) ? nram_para_store_ping : nram_para_store_pong;

      if (last_stage) {
        if (section_num == 1) {
          __memcpy_async(output, nram_para_store,
                         sizeof(DT) * ((large_radix >> 1) + 1) * 2, NRAM2GDRAM);
        } else {
          // scatter-store
          __memcpy_async(output + i * (large_radix + 2), nram_para_store,
                         sizeof(DT) * para_store_num * (large_radix + 2),
                         NRAM2GDRAM);
        }
      } else {
        // real
        __memcpy_async(output + i * ((large_radix >> 1) + 1), nram_para_store,
                       para_store_num * ((large_radix >> 1) + 1) * sizeof(DT),
                       NRAM2GDRAM);
        // imag
        __memcpy_async(
            output + i * ((large_radix >> 1) + 1) +
                ((nfft / large_radix) * ((large_radix >> 1) + 1)),
            nram_para_store + max_para_ldst_num * ((large_radix >> 1) + 1),
            para_store_num * ((large_radix >> 1) + 1) * sizeof(DT), NRAM2GDRAM);
      }
    }

    // pipeline: compute-stage
    if (repeat_id >= 1 && repeat_id < repeat_num + 1) {
      int i = max_para_ldst_num * (repeat_id - 1);

      DT *nram_para_load =
          (repeat_id % 2 != 0) ? nram_para_load_ping : nram_para_load_pong;
      DT *nram_para_store =
          (repeat_id % 2 != 0) ? nram_para_store_ping : nram_para_store_pong;

      int para_ldst_num = (max_para_ldst_num > (section_num - i))
                              ? (section_num - i)
                              : max_para_ldst_num;

      {
        // load real & imag
        radix = small_factors[4];
        small_section_num = small_factors[5];
        small_in_stride = small_factors[7];
        small_stage_count = _small_stage_count;

        // first stage
        if (ld_dft_radix != radix) {
          ld_dft_radix = radix;
          for (int entry = 0;; entry++) {
            if (dft_table[entry].radix == ld_dft_radix) {
              align_K = K_num * ((radix + K_num - 1) / K_num);

              __memcpy_async(
                  nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                  sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
              __sync_move();
              break;
            }

            if (dft_table[entry].radix == -1) {
              break;
            }
          }
        }

        computeGenericButterflyFirststageMatR2C(
            nram_out_r, nram_out_i, nram_para_load, nram_scratch, nram_dftmtx,
            small_section_num * para_ldst_num,
            small_section_num * para_ldst_num, 1, radix);
        // [radix, small_section_num, para_ldst_num] ->
        // [small_section_num, para_ldst_num, radix] -> [para_ldst_num,
        // small_section_num, radix]

        small_stage_count--;

        if (small_stage_count == 0) {
          // nram to gdram

          if (last_stage) {
            if (nram_out_r == nram_para_store_pong) {
              FFT_SWAP_PTR(nram_para_load_pong, nram_para_store_pong)
            }
            __bang_transpose(nram_para_store, nram_out_r, 2,
                             max_para_ldst_num * large_radix);
          } else {
            __memcpy_async(
                nram_para_store, nram_out_r,
                para_ldst_num * ((large_radix >> 1) + 1) * sizeof(DT),
                NRAM2NRAM);
            __memcpy_async(
                nram_para_store + max_para_ldst_num * ((large_radix >> 1) + 1),
                nram_out_i,
                para_ldst_num * ((large_radix >> 1) + 1) * sizeof(DT),
                NRAM2NRAM);
          }

        } else {
          // [small_section_num, para_ldst_num, radix] -> [para_ldst_num,
          // small_section_num, radix]
          FFT_SWAP_PTR(nram_out_r, nram_in_r);
          FFT_SWAP_PTR(nram_out_i, nram_in_i);

          TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                                 small_section_num, para_ldst_num,
                                 radix / 2 + 1, DT)

          value_mul = 8;
          DT *nram_tw = _nram_tw;

          for (; small_stage_count > 1; small_stage_count--) {
            FFT_SWAP_PTR(nram_out_r, nram_in_r);
            FFT_SWAP_PTR(nram_out_i, nram_in_i);

            // // update parameter
            radix = small_factors[value_mul++];
            small_section_num = small_factors[value_mul++];
            small_butterfly_num = small_factors[value_mul++];
            small_in_stride = small_factors[value_mul++];
            if (ld_dft_radix != radix) {
              ld_dft_radix = radix;
              for (int entry = 0;; entry++) {
                if (dft_table[entry].radix == ld_dft_radix) {
                  align_K = K_num * ((radix + K_num - 1) / K_num);

                  __memcpy_async(
                      nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                      sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                  __sync_move();
                  break;
                }

                if (dft_table[entry].radix == -1) {
                  break;
                }
              }
            }

            computeGenericButterflyOtherstagesMatR2C(
                nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                nram_dftmtx, nram_tw, small_section_num, small_butterfly_num,
                para_ldst_num, small_in_stride, radix);

            nram_tw += ((small_butterfly_num + 2) / 2) * (radix - 1) * 2;
          }  // for (stage_count)
          {
            FFT_SWAP_PTR(nram_out_r, nram_in_r);
            FFT_SWAP_PTR(nram_out_i, nram_in_i);

            // update parameter
            radix = small_factors[value_mul++];
            small_section_num = small_factors[value_mul++];
            small_butterfly_num = small_factors[value_mul++];
            small_in_stride = small_factors[value_mul];

            if (ld_dft_radix != radix) {
              ld_dft_radix = radix;
              for (int entry = 0;; entry++) {
                if (dft_table[entry].radix == ld_dft_radix) {
                  align_K = K_num * ((radix + K_num - 1) / K_num);

                  __memcpy_async(
                      nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                      sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                  __sync_move();
                  break;
                }

                if (dft_table[entry].radix == -1) {
                  break;
                }
              }
            }

            computeGenericButterflyLaststageMatR2C(
                nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                nram_dftmtx, nram_tw, small_section_num, small_butterfly_num,
                para_ldst_num, small_in_stride, radix);

            if (last_stage) {
              //  [2, para_ldst_num, large_radix] -> [para_ldst_num,
              //  large_radix, 2]
              __bang_transpose(nram_para_store, nram_out_r, 2,
                               max_para_ldst_num * large_radix);
            } else {
              //  [2, para_ldst_num, large_radix] -> [2, para_ldst_num,
              //  large_radix]
              __memcpy_async(
                  nram_para_store, nram_out_r,
                  para_ldst_num * ((large_radix >> 1) + 1) * sizeof(DT),
                  NRAM2NRAM);
              __memcpy_async(
                  nram_para_store +
                      max_para_ldst_num * ((large_radix >> 1) + 1),
                  nram_out_i,
                  para_ldst_num * ((large_radix >> 1) + 1) * sizeof(DT),
                  NRAM2NRAM);
            }
          }
        }
      }
    }

    __sync();
  }
}

// Compute the large butterfly for other stages from real to complex (R2C)
template <typename DT>
__mlu_func__ void computeLargeButterflyOtherstagesR2C(
    DT *output, DT *input, const int large_radix, const DT *cur_large_twiddles,
    const DT *small_twiddles, const int small_twiddles_size,
    const DT *dft_matrix, const int large_section_num,
    const int large_butterfly_num, const int large_in_stride, void *nram_buf,
    const int *small_factors, const int nfft, const int last_stage,
    const int load_once_twiddles) {
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;
  const int K_num = 64 / sizeof(DT);
  int align_K = 0;

  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;

  const int large_out_stride = (large_butterfly_num + 2) / 2;
  int _nfft = ((nfft / large_section_num) / 2 + 1) * large_section_num;
  int tw_offset;

  _small_stage_count = small_factors[0];
  tw_offset = small_factors[1];

  const int max_para_ldst_num = small_factors[3];

  int nram_buf_offset = 0;
  DT *nram_in_r = (DT *)nram_buf + nram_buf_offset;
  DT *nram_out_tmp = (DT *)nram_in_r;
  nram_buf_offset += large_radix * max_para_ldst_num;

  DT *nram_in_i = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  DT *nram_out_r = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  DT *nram_out_i = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  // parallel load/store space
  FFT_CPX_T<DT> nram_para_load_in_ping = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_ldst_num};
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  FFT_CPX_T<DT> nram_para_load_in_pong = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_ldst_num};
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  FFT_CPX_T<DT> nram_para_load_tw_ping = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_ldst_num};
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  FFT_CPX_T<DT> nram_para_load_tw_pong = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_ldst_num};
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  FFT_CPX_T<DT> nram_para_store_ping = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_ldst_num};
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  FFT_CPX_T<DT> nram_para_store_pong = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_ldst_num};
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *_nram_tw = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * 2;  // complex
  if (small_twiddles_size) {
    if (load_once_twiddles) {
      __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, SRAM2NRAM);
    } else {
      __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, GDRAM2NRAM);
    }
  }

  int ld_dft_radix = -1;
  const int max_radix = 64;
  DT *nram_dftmtx = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += max_radix * max_radix * 2;  // complex

  DT *nram_scratch = (DT *)nram_buf + nram_buf_offset;
  // overlap nram_in

  // temp overlap with "nram_scratch"
  DT *CPX_MUL_RR = nram_scratch;
  DT *CPX_MUL_RI = &CPX_MUL_RR[large_radix * max_para_ldst_num];
  DT *CPX_MUL_IR = &CPX_MUL_RI[large_radix * max_para_ldst_num];
  DT *CPX_MUL_II = &CPX_MUL_IR[large_radix * max_para_ldst_num];

  nram_buf_offset += large_radix * max_para_ldst_num * 4;  // complex
  FFT_CPX_T<DT> nram_transpose_temp;
  // temp out-space before transpose

  int Fin_stride = 0, Fout_stride = 0;
  int sec_count;
  int repeat_num = ((large_butterfly_num + 2) / 2 + max_para_ldst_num - 1) /
                   max_para_ldst_num;

  for (sec_count = 0; sec_count < large_section_num; ++sec_count) {
    for (int repeat_id = 0; repeat_id < repeat_num + 2; ++repeat_id) {
      // pipeline: load-stage
      if (repeat_id < repeat_num) {
        int i = max_para_ldst_num * repeat_id;
        FFT_CPX_T<DT> nram_para_load_in = (repeat_id % 2 == 0)
                                              ? nram_para_load_in_ping
                                              : nram_para_load_in_pong;

        FFT_CPX_T<DT> nram_para_load_tw = (repeat_id % 2 == 0)
                                              ? nram_para_load_tw_ping
                                              : nram_para_load_tw_pong;

        int para_load_num =
            (max_para_ldst_num > ((large_butterfly_num + 2) / 2 - i))
                ? ((large_butterfly_num + 2) / 2 - i)
                : max_para_ldst_num;
        __memcpy_async(nram_para_load_in.r, input + Fin_stride + i,
                       sizeof(DT) * para_load_num, GDRAM2NRAM,
                       sizeof(DT) * para_load_num, large_in_stride * sizeof(DT),
                       large_radix - 1);
        __memcpy_async(nram_para_load_in.i,
                       input + large_in_stride * large_radix + Fin_stride + i,
                       sizeof(DT) * para_load_num, GDRAM2NRAM,
                       sizeof(DT) * para_load_num, large_in_stride * sizeof(DT),
                       large_radix - 1);

        if (load_once_twiddles) {
          __memcpy_async(nram_para_load_tw.r, cur_large_twiddles + i,
                         sizeof(DT) * para_load_num, SRAM2NRAM,
                         sizeof(DT) * para_load_num,
                         large_out_stride * sizeof(DT), large_radix - 2);
          __memcpy_async(
              nram_para_load_tw.i,
              cur_large_twiddles +
                  ((large_butterfly_num + 2) / 2) * (large_radix - 1) + i,
              sizeof(DT) * para_load_num, SRAM2NRAM, sizeof(DT) * para_load_num,
              large_out_stride * sizeof(DT), large_radix - 2);

        } else {
          __memcpy_async(nram_para_load_tw.r, cur_large_twiddles + i,
                         sizeof(DT) * para_load_num, GDRAM2NRAM,
                         sizeof(DT) * para_load_num,
                         large_out_stride * sizeof(DT), large_radix - 2);
          __memcpy_async(
              nram_para_load_tw.i,
              cur_large_twiddles +
                  ((large_butterfly_num + 2) / 2) * (large_radix - 1) + i,
              sizeof(DT) * para_load_num, GDRAM2NRAM,
              sizeof(DT) * para_load_num, large_out_stride * sizeof(DT),
              large_radix - 2);
        }
      }

      // pipeline: store-stage
      if (repeat_id >= 2) {
        int i = max_para_ldst_num * (repeat_id - 2);

        int para_store_num =
            (max_para_ldst_num > (((large_butterfly_num + 2) / 2) - i))
                ? (((large_butterfly_num + 2) / 2) - i)
                : max_para_ldst_num;

        FFT_CPX_T<DT> nram_para_store =
            (repeat_id % 2 == 0) ? nram_para_store_ping : nram_para_store_pong;

        if (last_stage) {
          __memcpy_async(output + (Fout_stride + i) * 2, nram_para_store.r,
                         sizeof(DT) * 2 * para_store_num, NRAM2GDRAM,
                         large_butterfly_num * 2 * sizeof(DT),
                         sizeof(DT) * 2 * para_store_num,
                         (large_radix + 1) / 2 - 1);

          __memcpy_async(
              output +
                  (Fout_stride + large_butterfly_num - i - para_store_num + 1) *
                      2,
              nram_para_store.r + ((large_radix + 1) / 2) * 2 * para_store_num,
              sizeof(DT) * 2 * para_store_num, NRAM2GDRAM,
              large_butterfly_num * 2 * sizeof(DT),
              sizeof(DT) * 2 * para_store_num, large_radix / 2 - 1);
        } else {
          // real
          __memcpy_async(output + Fout_stride + i, nram_para_store.r,
                         para_store_num * sizeof(DT), NRAM2GDRAM,
                         large_butterfly_num * sizeof(DT),
                         sizeof(DT) * para_store_num,
                         (large_radix + 1) / 2 - 1);

          __memcpy_async(
              output + Fout_stride + large_butterfly_num - i - para_store_num +
                  1,
              nram_para_store.r + ((large_radix + 1) / 2) * para_store_num,
              para_store_num * sizeof(DT), NRAM2GDRAM,
              large_butterfly_num * sizeof(DT), sizeof(DT) * para_store_num,
              large_radix / 2 - 1);

          // imag
          __memcpy_async(output + Fout_stride + i + _nfft, nram_para_store.i,
                         para_store_num * sizeof(DT), NRAM2GDRAM,
                         large_butterfly_num * sizeof(DT),
                         sizeof(DT) * para_store_num,
                         (large_radix + 1) / 2 - 1);

          __memcpy_async(
              output + Fout_stride + large_butterfly_num - i - para_store_num +
                  1 + _nfft,
              nram_para_store.i + ((large_radix + 1) / 2) * para_store_num,
              para_store_num * sizeof(DT), NRAM2GDRAM,
              large_butterfly_num * sizeof(DT), sizeof(DT) * para_store_num,
              large_radix / 2 - 1);
        }
      }
      // pipeline: compute-stage

      if (repeat_id >= 1 && repeat_id < repeat_num + 1) {
        int i = max_para_ldst_num * (repeat_id - 1);

        FFT_CPX_T<DT> nram_para_load_in = (repeat_id % 2 != 0)
                                              ? nram_para_load_in_ping
                                              : nram_para_load_in_pong;

        FFT_CPX_T<DT> nram_para_load_tw = (repeat_id % 2 != 0)
                                              ? nram_para_load_tw_ping
                                              : nram_para_load_tw_pong;

        FFT_CPX_T<DT> nram_para_store =
            (repeat_id % 2 != 0) ? nram_para_store_ping : nram_para_store_pong;

        int para_ldst_num =
            (max_para_ldst_num > ((large_butterfly_num + 2) / 2 - i))
                ? ((large_butterfly_num + 2) / 2 - i)
                : max_para_ldst_num;

        // rotation-large
        __bang_mul(CPX_MUL_RR, nram_para_load_in.r + para_ldst_num,
                   nram_para_load_tw.r, para_ldst_num * (large_radix - 1));
        __bang_mul(CPX_MUL_II, nram_para_load_in.i + para_ldst_num,
                   nram_para_load_tw.i, para_ldst_num * (large_radix - 1));
        __bang_mul(CPX_MUL_RI, nram_para_load_in.r + para_ldst_num,
                   nram_para_load_tw.i, para_ldst_num * (large_radix - 1));
        __bang_mul(CPX_MUL_IR, nram_para_load_in.i + para_ldst_num,
                   nram_para_load_tw.r, para_ldst_num * (large_radix - 1));

        __bang_sub(nram_para_load_in.r + para_ldst_num, CPX_MUL_RR, CPX_MUL_II,
                   para_ldst_num * (large_radix - 1));
        __bang_add(nram_para_load_in.i + para_ldst_num, CPX_MUL_RI, CPX_MUL_IR,
                   para_ldst_num * (large_radix - 1));

        {
          // load real & imag

          radix = small_factors[4];
          small_section_num = small_factors[5];
          small_in_stride = small_factors[7];
          small_stage_count = _small_stage_count;

          if (ld_dft_radix != radix) {
            ld_dft_radix = radix;
            for (int entry = 0;; entry++) {
              if (dft_table[entry].radix == ld_dft_radix) {
                align_K = K_num * ((radix + K_num - 1) / K_num);

                __memcpy_async(
                    nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                    sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                __sync_move();
                break;
              }

              if (dft_table[entry].radix == -1) {
                break;
              }
            }
          }

          computeGenericButterflyFirststageMat(
              nram_out_r, nram_out_i, nram_para_load_in.r, nram_para_load_in.i,
              nram_scratch, nram_dftmtx, small_section_num * para_ldst_num,
              small_section_num * para_ldst_num, 1, 1, radix);

          small_stage_count--;
          if (small_stage_count == 0) {
            {
              nram_out_tmp = (DT *)nram_in_r;
              __bang_rotate90(nram_out_tmp, nram_out_r, para_ldst_num,
                              large_radix);
              __bang_rotate180(
                  nram_out_r,
                  nram_out_tmp + ((large_radix + 1) / 2) * para_ldst_num,
                  (large_radix / 2), para_ldst_num);
              __sync_compute();
              __memcpy_async(
                  nram_out_tmp + ((large_radix + 1) / 2) * para_ldst_num,
                  nram_out_r, sizeof(DT) * (large_radix / 2) * para_ldst_num,
                  NRAM2NRAM);
              __sync_move();
              __bang_rotate270(nram_out_r, nram_out_tmp, large_radix,
                               para_ldst_num);

              __bang_rotate90(nram_out_tmp, nram_out_i, para_ldst_num,
                              large_radix);
              __bang_mul_scalar(
                  nram_out_tmp + ((large_radix + 1) / 2) * para_ldst_num,
                  nram_out_tmp + ((large_radix + 1) / 2) * para_ldst_num, -1,
                  (large_radix / 2) * para_ldst_num);
              __bang_rotate180(
                  nram_out_i,
                  nram_out_tmp + ((large_radix + 1) / 2) * para_ldst_num,
                  (large_radix / 2), para_ldst_num);
              __sync_compute();
              __memcpy_async(
                  nram_out_tmp + ((large_radix + 1) / 2) * para_ldst_num,
                  nram_out_i, sizeof(DT) * (large_radix / 2) * para_ldst_num,
                  NRAM2NRAM);
              __sync_move();
              __bang_rotate270(nram_out_i, nram_out_tmp, large_radix,
                               para_ldst_num);
            }
            if (last_stage) {
              nram_transpose_temp = {
                  (DT *)nram_in_r,
                  (DT *)nram_in_r + large_radix * ((int)last_stage) +
                      large_radix * (1 - (int)last_stage) * max_para_ldst_num};
              __sync_compute();
              __sync_move();
              __memcpy_async(nram_transpose_temp.r, nram_out_r,
                             sizeof(DT) * large_radix, NRAM2NRAM,
                             sizeof(DT) * large_radix * 2,
                             sizeof(DT) * large_radix, para_ldst_num - 1);

              __memcpy_async(nram_transpose_temp.i, nram_out_i,
                             sizeof(DT) * large_radix, NRAM2NRAM,
                             sizeof(DT) * large_radix * 2,
                             sizeof(DT) * large_radix, para_ldst_num - 1);
              __sync_move();
              __sync_compute();
              __bang_transpose(nram_para_store.r, nram_transpose_temp.r,
                               para_ldst_num * 2, large_radix);
            } else {
              __bang_transpose(nram_para_store.r, nram_out_r, para_ldst_num,
                               large_radix);
              __bang_transpose(nram_para_store.i, nram_out_i, para_ldst_num,
                               large_radix);
            }

          } else {
            FFT_SWAP_PTR(nram_out_r, nram_in_r);
            FFT_SWAP_PTR(nram_out_i, nram_in_i);

            TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                                   small_section_num, para_ldst_num, radix, DT)

            DT *nram_tw = _nram_tw;
            value_mul = 8;

            for (; small_stage_count > 1; small_stage_count--) {
              FFT_SWAP_PTR(nram_out_r, nram_in_r);
              FFT_SWAP_PTR(nram_out_i, nram_in_i);

              // // update parameter
              radix = small_factors[value_mul++];
              small_section_num = small_factors[value_mul++];
              small_butterfly_num = small_factors[value_mul++];
              small_in_stride = small_factors[value_mul++];

              if (ld_dft_radix != radix) {
                ld_dft_radix = radix;
                for (int entry = 0;; entry++) {
                  if (dft_table[entry].radix == ld_dft_radix) {
                    align_K = K_num * ((radix + K_num - 1) / K_num);

                    __memcpy_async(
                        nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                        sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                    __sync_move();
                    break;
                  }

                  if (dft_table[entry].radix == -1) {
                    break;
                  }
                }
              }

              computeGenericButterflyOtherstagesMat(
                  nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                  nram_dftmtx, nram_tw, small_section_num, small_butterfly_num,
                  para_ldst_num, small_in_stride, 1, radix);

              nram_tw += small_butterfly_num * (radix - 1) * 2;
            }  // for (stage_count)

            // last stage
            {
              FFT_SWAP_PTR(nram_out_r, nram_in_r);
              FFT_SWAP_PTR(nram_out_i, nram_in_i);

              // update parameter
              radix = small_factors[value_mul++];
              small_section_num = small_factors[value_mul++];
              small_butterfly_num = small_factors[value_mul++];
              small_in_stride = small_factors[value_mul];

              if (ld_dft_radix != radix) {
                ld_dft_radix = radix;
                for (int entry = 0;; entry++) {
                  if (dft_table[entry].radix == ld_dft_radix) {
                    align_K = K_num * ((radix + K_num - 1) / K_num);

                    __memcpy_async(
                        nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                        sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                    __sync_move();
                    break;
                  }

                  if (dft_table[entry].radix == -1) {
                    break;
                  }
                }
              }

              computeGenericButterflyLaststageMat(
                  nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                  nram_dftmtx, nram_tw, small_section_num, small_butterfly_num,
                  para_ldst_num, small_in_stride, 1, radix);

              {
                nram_out_tmp = (DT *)nram_in_r;
                __bang_rotate90(nram_out_tmp, nram_out_r, para_ldst_num,
                                large_radix);
                __bang_rotate180(
                    nram_out_r,
                    nram_out_tmp + ((large_radix + 1) / 2) * para_ldst_num,
                    (large_radix / 2), para_ldst_num);
                __sync_compute();
                __memcpy_async(
                    nram_out_tmp + ((large_radix + 1) / 2) * para_ldst_num,
                    nram_out_r, sizeof(DT) * (large_radix / 2) * para_ldst_num,
                    NRAM2NRAM);
                __sync_move();
                __bang_rotate270(nram_out_r, nram_out_tmp, large_radix,
                                 para_ldst_num);

                __bang_rotate90(nram_out_tmp, nram_out_i, para_ldst_num,
                                large_radix);
                __bang_mul_scalar(
                    nram_out_tmp + ((large_radix + 1) / 2) * para_ldst_num,
                    nram_out_tmp + ((large_radix + 1) / 2) * para_ldst_num, -1,
                    (large_radix / 2) * para_ldst_num);
                __bang_rotate180(
                    nram_out_i,
                    nram_out_tmp + ((large_radix + 1) / 2) * para_ldst_num,
                    (large_radix / 2), para_ldst_num);
                __sync_compute();
                __memcpy_async(
                    nram_out_tmp + ((large_radix + 1) / 2) * para_ldst_num,
                    nram_out_i, sizeof(DT) * (large_radix / 2) * para_ldst_num,
                    NRAM2NRAM);
                __sync_move();
                __bang_rotate270(nram_out_i, nram_out_tmp, large_radix,
                                 para_ldst_num);
              }

              if (last_stage) {
                nram_transpose_temp = {(DT *)nram_in_r,
                                       (DT *)nram_in_r +
                                           large_radix * ((int)last_stage) +
                                           large_radix * (1 - (int)last_stage) *
                                               max_para_ldst_num};
                __memcpy(nram_transpose_temp.r, nram_out_r,
                         sizeof(DT) * large_radix, NRAM2NRAM,
                         sizeof(DT) * large_radix * 2, sizeof(DT) * large_radix,
                         para_ldst_num - 1);
                __memcpy(nram_transpose_temp.i, nram_out_i,
                         sizeof(DT) * large_radix, NRAM2NRAM,
                         sizeof(DT) * large_radix * 2, sizeof(DT) * large_radix,
                         para_ldst_num - 1);

                __bang_transpose(nram_para_store.r, nram_transpose_temp.r,
                                 para_ldst_num * 2, large_radix);
              } else {
                __bang_transpose(nram_para_store.r, nram_out_r, para_ldst_num,
                                 large_radix);
                __bang_transpose(nram_para_store.i, nram_out_i, para_ldst_num,
                                 large_radix);
              }
            }
          }
        }
      }
      __sync();
    }
    Fin_stride += (large_butterfly_num + 2) / 2;
    Fout_stride += large_radix * large_butterfly_num / 2 + 1;
  }
}

// Compute the large butterfly for the last stage from real to complex (R2C)
template <typename DT>
__mlu_func__ void computeLargeButterflyLaststageR2C(
    DT *output, DT *input, const int large_radix, const DT *cur_large_twiddles,
    const DT *small_twiddles, const int small_twiddles_size,
    const DT *dft_matrix, const int large_section_num,
    const int large_butterfly_num, const int large_in_stride, void *nram_buf,
    const int *small_factors, const int nfft, const int load_once_twiddles) {
  computeLargeButterflyOtherstagesR2C(
      output, input, large_radix, cur_large_twiddles, small_twiddles,
      small_twiddles_size, dft_matrix, large_section_num, large_butterfly_num,
      large_in_stride, nram_buf, small_factors, nfft, 1, load_once_twiddles);
}
