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

// Compute the large butterfly for the first stage on the primary chip
template <typename DT>
__mlu_func__ void computeLargeButterflyFirststage(
    DT *output, DT *input, const int large_radix, const int large_in_stride,
    const int section_num, const DT *twiddles, const DT *dft_matrix,
    void *nram_buf, const int *small_factors, const int dir, const int nfft,
    const int last_stage) {
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;

  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;
  int tw_offset;

  const int K_num = 64 / sizeof(DT);
  int align_K = 0;

  _small_stage_count = small_factors[0];
  tw_offset = small_factors[1];

  int max_para_ldst_num = (4096 + large_radix - 1) / large_radix;
  max_para_ldst_num =
      (section_num < max_para_ldst_num) ? section_num : max_para_ldst_num;

  const DT *small_twiddles = twiddles + tw_offset * 2;

  int nram_buf_offset = 0;

  DT *nram_in_r = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  DT *nram_in_i = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  DT *nram_out_r = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  DT *nram_out_i = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  DT *nram_para_load_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;

  DT *nram_para_load_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;

  DT *nram_para_store_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;

  DT *nram_para_store_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;

  DT *_nram_tw = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * 2;

  int ld_dft_radix = -1;
  const int max_radix = 64;
  DT *nram_dftmtx = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += max_radix * max_radix * 2;

  DT *nram_scratch = (DT *)nram_buf + nram_buf_offset;

  __memcpy_async(_nram_tw, small_twiddles, large_radix * sizeof(DT) * 2,
                 SRAM2NRAM);

  int repeat_num = (section_num + max_para_ldst_num - 1) / max_para_ldst_num;

  for (int repeat_id = 0; repeat_id < repeat_num + 2; ++repeat_id) {
    if (repeat_id < repeat_num) {
      int i = max_para_ldst_num * repeat_id;
      DT *nram_para_load =
          (repeat_id % 2 == 0) ? nram_para_load_ping : nram_para_load_pong;

      int para_load_num = (max_para_ldst_num > (section_num - i))
                              ? (section_num - i)
                              : max_para_ldst_num;
      if (section_num == 1) {
        __memcpy_async(nram_para_load, input, sizeof(DT) * 2 * large_radix,
                       GDRAM2NRAM);
      } else {
        __memcpy_async(nram_para_load, input + i * 2,
                       sizeof(DT) * 2 * para_load_num, GDRAM2NRAM,
                       sizeof(DT) * 2 * para_load_num,
                       large_in_stride * sizeof(DT) * 2, large_radix - 1);
      }
    }

    if (repeat_id >= 2) {
      int i = max_para_ldst_num * (repeat_id - 2);

      int para_store_num = (max_para_ldst_num > (section_num - i))
                               ? (section_num - i)
                               : max_para_ldst_num;

      DT *nram_para_store =
          (repeat_id % 2 == 0) ? nram_para_store_ping : nram_para_store_pong;

      if (last_stage) {
        if (section_num == 1) {
          __memcpy_async(output, nram_para_store, sizeof(DT) * 2 * large_radix,
                         NRAM2GDRAM);
        } else {
          __memcpy_async(output + i * large_radix * 2, nram_para_store,
                         sizeof(DT) * 2 * para_store_num * large_radix,
                         NRAM2GDRAM);
        }
      } else {
        __memcpy_async(output + i * large_radix, nram_para_store,
                       para_store_num * large_radix * sizeof(DT), NRAM2GDRAM);
        __memcpy_async(output + i * large_radix + nfft,
                       nram_para_store + max_para_ldst_num * large_radix,
                       para_store_num * large_radix * sizeof(DT), NRAM2GDRAM);
      }
    }

    if (repeat_id >= 1 && repeat_id < repeat_num + 1) {
      int i = max_para_ldst_num * (repeat_id - 1);

      DT *nram_para_load =
          (repeat_id % 2 != 0) ? nram_para_load_ping : nram_para_load_pong;
      DT *nram_para_store =
          (repeat_id % 2 != 0) ? nram_para_store_ping : nram_para_store_pong;

      int para_ldst_num = (max_para_ldst_num > (section_num - i))
                              ? (section_num - i)
                              : max_para_ldst_num;
      __bang_transpose(nram_in_r, nram_para_load, large_radix * para_ldst_num,
                       2);

      {
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
            }

            if (dft_table[entry].radix == -1) {
              break;
            }
          }
        }

        computeGenericButterflyFirststageMat(
            nram_out_r, nram_out_i, nram_in_r,
            nram_in_r + large_radix * para_ldst_num, nram_scratch, nram_dftmtx,
            small_section_num * para_ldst_num,
            small_section_num * para_ldst_num, 1, dir, radix);

        small_stage_count--;
        if (small_stage_count == 0) {
          if (last_stage) {
            __bang_transpose(nram_para_store, nram_out_r, 2,
                             max_para_ldst_num * large_radix);
          } else {
            __memcpy_async(nram_para_store, nram_out_r,
                           para_ldst_num * large_radix * sizeof(DT), NRAM2NRAM);
            __memcpy_async(nram_para_store + max_para_ldst_num * large_radix,
                           nram_out_i, para_ldst_num * large_radix * sizeof(DT),
                           NRAM2NRAM);
            __sync_compute();
          }
        } else {
          FFT_SWAP_PTR(nram_out_r, nram_in_r);
          FFT_SWAP_PTR(nram_out_i, nram_in_i);

          TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                                 small_section_num, para_ldst_num, radix, DT)

          value_mul = 8;
          DT *nram_tw = _nram_tw;

          for (; small_stage_count > 1; small_stage_count--) {
            FFT_SWAP_PTR(nram_out_r, nram_in_r);
            FFT_SWAP_PTR(nram_out_i, nram_in_i);

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
                para_ldst_num, small_in_stride, dir, radix);

            nram_tw += small_butterfly_num * (radix - 1) * 2;
          }

          {
            FFT_SWAP_PTR(nram_out_r, nram_in_r);
            FFT_SWAP_PTR(nram_out_i, nram_in_i);

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
                para_ldst_num, small_in_stride, dir, radix);

            if (last_stage) {
              __bang_transpose(nram_para_store, nram_out_r, 2,
                               max_para_ldst_num * large_radix);
            } else {
              __memcpy(nram_para_store, nram_out_r,
                       para_ldst_num * large_radix * sizeof(DT), NRAM2NRAM);
              __memcpy(nram_para_store + max_para_ldst_num * large_radix,
                       nram_out_i, para_ldst_num * large_radix * sizeof(DT),
                       NRAM2NRAM);
            }
          }
        }
      }
    }
    __sync();
  }
}

// Compute the large butterfly for the first stage using batch ping-pong
// processing on the primary chip
template <typename DT>
__mlu_func__ void computeLargeButterflyFirststageBatchPingpong(
    DT *output, DT *input, const int large_radix, const int large_in_stride,
    const int section_num, const DT *small_twiddles,
    const int small_twiddles_size, const DT *dft_matrix, void *nram_buf,
    const int *small_factors, const int dir, const int nfft, int last_stage,
    const int t_start, const int t_end, const int load_once_twiddles) {
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;
  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;
  int tw_offset, max_para_ldst_num;

  const int K_num = 64 / sizeof(DT);
  int align_K = 0;

  _small_stage_count = small_factors[0];
  tw_offset = small_factors[1];

  max_para_ldst_num =
      (section_num < small_factors[3]) ? section_num : small_factors[3];
  int nram_buf_offset = 0;

  DT *nram_para_load_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;

  DT *nram_para_load_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;

  DT *nram_para_store_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;

  DT *nram_para_store_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;

  DT *_nram_tw = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += small_twiddles_size;

  int ld_dft_radix[2] = {-1, -1};
  const int max_radix = 64;
  DT *nram_dftmtx[2] = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + max_radix * max_radix * 2};
  nram_buf_offset += max_radix * max_radix * 4;

  DT *nram_scratch = (DT *)nram_buf + nram_buf_offset;

  if (small_twiddles_size) {
    if (load_once_twiddles) {
      __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, SRAM2NRAM);
    } else {
      __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, GDRAM2NRAM);
    }
  }
  int repeat_num = (t_end - t_start);

  input += t_start * (nfft << 1);
  output += t_start * (nfft << 1);

  for (int sec_id = 0; sec_id < section_num; sec_id += max_para_ldst_num) {
    DT *output_batch = output;
    DT *input_batch = input;
    int para_num = (max_para_ldst_num > (section_num - sec_id))
                       ? (section_num - sec_id)
                       : max_para_ldst_num;

    for (int repeat_id = 0; repeat_id < repeat_num + 2;
         ++repeat_id, input_batch += (nfft << 1), output_batch += (nfft << 1)) {
      if (repeat_id < repeat_num) {
        if (section_num == 1) {
          __memcpy_async(nram_para_load_ping, input_batch,
                         sizeof(DT) * 2 * large_radix, GDRAM2NRAM);
        } else {
          __memcpy_async(nram_para_load_ping, input_batch + sec_id * 2,
                         sizeof(DT) * 2 * para_num, GDRAM2NRAM,
                         sizeof(DT) * 2 * para_num,
                         large_in_stride * sizeof(DT) * 2, large_radix - 1);
        }
      }

      if (repeat_id >= 2) {
        if (last_stage) {
          if (section_num == 1) {
            __memcpy_async(output_batch - (nfft << 2), nram_para_store_ping,
                           sizeof(DT) * 2 * large_radix, NRAM2GDRAM);
          } else {
            __memcpy_async(
                output_batch - (nfft << 2) + sec_id * large_radix * 2,
                nram_para_store_ping, sizeof(DT) * 2 * para_num * large_radix,
                NRAM2GDRAM);
          }
        } else {
          __memcpy_async(output_batch - (nfft << 2) + sec_id * large_radix,
                         nram_para_store_ping,
                         para_num * large_radix * sizeof(DT), NRAM2GDRAM);
          __memcpy_async(
              output_batch - (nfft << 2) + sec_id * large_radix + nfft,
              nram_para_store_ping + max_para_ldst_num * large_radix,
              para_num * large_radix * sizeof(DT), NRAM2GDRAM);
        }
      }

      if (repeat_id >= 1 && repeat_id < repeat_num + 1) {
        DT *nram_in_r = nram_para_store_pong;
        DT *nram_in_i = nram_para_store_pong + large_radix * max_para_ldst_num;

        DT *nram_out_r = nram_para_load_pong;
        DT *nram_out_i = nram_para_load_pong + large_radix * max_para_ldst_num;

        __bang_transpose(nram_in_r, nram_para_load_pong, large_radix * para_num,
                         2);
        {
          radix = small_factors[4];
          small_section_num = small_factors[5];
          small_in_stride = small_factors[7];
          small_stage_count = _small_stage_count;

          if (ld_dft_radix[0] != radix && ld_dft_radix[1] != radix) {
            ld_dft_radix[1] = ld_dft_radix[0];
            FFT_SWAP_PTR(nram_dftmtx[0], nram_dftmtx[1]);
            ld_dft_radix[0] = radix;
            for (int entry = 0;; entry++) {
              if (dft_table[entry].radix == ld_dft_radix[0]) {
                align_K = K_num * ((radix + K_num - 1) / K_num);
                __memcpy_async(nram_dftmtx[0],
                               &dft_matrix[dft_table[entry].offset * 2],
                               sizeof(DT) * 2 * radix * align_K, SRAM2NRAM);
                __sync_move();
                break;
              }

              if (dft_table[entry].radix == -1) {
                break;
              }
            }
          } else if (ld_dft_radix[1] == radix) {
            ld_dft_radix[1] = ld_dft_radix[0];
            ld_dft_radix[0] = radix;
            FFT_SWAP_PTR(nram_dftmtx[0], nram_dftmtx[1]);
          }

          computeGenericButterflyFirststageMat(
              nram_out_r, nram_out_i, nram_in_r,
              nram_in_r + large_radix * para_num, nram_scratch, nram_dftmtx[0],
              small_section_num * para_num, small_section_num * para_num, 1,
              dir, radix);

          small_stage_count--;
          if (small_stage_count == 0) {
            if (last_stage) {
              if (nram_out_r == nram_para_store_pong) {
                FFT_SWAP_PTR(nram_para_load_pong, nram_para_store_pong)
              }
              __bang_transpose(nram_para_store_pong, nram_out_r, 2,
                               max_para_ldst_num * large_radix);
            } else {
              __memcpy_async(nram_para_store_pong, nram_out_r,
                             para_num * large_radix * sizeof(DT), NRAM2NRAM);
              __memcpy_async(
                  nram_para_store_pong + max_para_ldst_num * large_radix,
                  nram_out_i, para_num * large_radix * sizeof(DT), NRAM2NRAM);
              __sync_compute();
            }

          } else {
            FFT_SWAP_PTR(nram_out_r, nram_in_r);
            FFT_SWAP_PTR(nram_out_i, nram_in_i);

            TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                                   small_section_num, para_num, radix, DT)

            value_mul = 8;
            DT *nram_tw = _nram_tw;

            for (; small_stage_count > 1; small_stage_count--) {
              FFT_SWAP_PTR(nram_out_r, nram_in_r);
              FFT_SWAP_PTR(nram_out_i, nram_in_i);

              radix = small_factors[value_mul++];
              small_section_num = small_factors[value_mul++];
              small_butterfly_num = small_factors[value_mul++];
              small_in_stride = small_factors[value_mul++];

              if (ld_dft_radix[0] != radix && ld_dft_radix[1] != radix) {
                ld_dft_radix[1] = ld_dft_radix[0];
                FFT_SWAP_PTR(nram_dftmtx[0], nram_dftmtx[1]);
                ld_dft_radix[0] = radix;
                for (int entry = 0;; entry++) {
                  if (dft_table[entry].radix == ld_dft_radix[0]) {
                    align_K = K_num * ((radix + K_num - 1) / K_num);
                    __memcpy_async(nram_dftmtx[0],
                                   &dft_matrix[dft_table[entry].offset * 2],
                                   sizeof(DT) * 2 * radix * align_K, SRAM2NRAM);
                    __sync_move();
                    break;
                  }

                  if (dft_table[entry].radix == -1) {
                    break;
                  }
                }
              } else if (ld_dft_radix[1] == radix) {
                ld_dft_radix[1] = ld_dft_radix[0];
                ld_dft_radix[0] = radix;
                FFT_SWAP_PTR(nram_dftmtx[0], nram_dftmtx[1]);
              }

              computeGenericButterflyOtherstagesMat(
                  nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                  nram_dftmtx[0], nram_tw, small_section_num,
                  small_butterfly_num, para_num, small_in_stride, dir, radix);

              nram_tw += small_butterfly_num * (radix - 1) * 2;
            }

            {
              FFT_SWAP_PTR(nram_out_r, nram_in_r);
              FFT_SWAP_PTR(nram_out_i, nram_in_i);

              radix = small_factors[value_mul++];
              small_section_num = small_factors[value_mul++];
              small_butterfly_num = small_factors[value_mul++];
              small_in_stride = small_factors[value_mul];

              if (ld_dft_radix[0] != radix && ld_dft_radix[1] != radix) {
                ld_dft_radix[1] = ld_dft_radix[0];
                FFT_SWAP_PTR(nram_dftmtx[0], nram_dftmtx[1]);
                ld_dft_radix[0] = radix;
                for (int entry = 0;; entry++) {
                  if (dft_table[entry].radix == ld_dft_radix[0]) {
                    align_K = K_num * ((radix + K_num - 1) / K_num);
                    __memcpy_async(nram_dftmtx[0],
                                   &dft_matrix[dft_table[entry].offset * 2],
                                   sizeof(DT) * 2 * radix * align_K, SRAM2NRAM);
                    __sync_move();
                    break;
                  }

                  if (dft_table[entry].radix == -1) {
                    break;
                  }
                }
              } else if (ld_dft_radix[1] == radix) {
                ld_dft_radix[1] = ld_dft_radix[0];
                ld_dft_radix[0] = radix;
                FFT_SWAP_PTR(nram_dftmtx[0], nram_dftmtx[1]);
              }

              if (last_stage) {
                computeGenericButterflyLaststageMat(
                    nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                    nram_dftmtx[0], nram_tw, small_section_num,
                    small_butterfly_num, para_num, small_in_stride, dir, radix);
              } else {
                computeGenericButterflyLaststageMat(
                    nram_para_store_pong,
                    nram_para_store_pong + max_para_ldst_num * large_radix,
                    nram_in_r, nram_in_i, nram_scratch, nram_dftmtx[0], nram_tw,
                    small_section_num, small_butterfly_num, para_num,
                    small_in_stride, dir, radix);
              }

              if (last_stage) {
                if (nram_out_r == nram_para_store_pong) {
                  FFT_SWAP_PTR(nram_para_load_pong, nram_para_store_pong)
                }
                __bang_transpose(nram_para_store_pong, nram_out_r, 2,
                                 max_para_ldst_num * large_radix);
              }
            }
          }
        }
      }

      __sync();
      FFT_SWAP_PTR(nram_para_load_ping, nram_para_load_pong)
      FFT_SWAP_PTR(nram_para_store_ping, nram_para_store_pong)
    }
  }
}

// Compute the large butterfly for the subsequent stages of the FFT
template <typename DT>
__mlu_func__ void computeLargeButterflyOtherstages(
    DT *output, DT *input, const int large_radix, const DT *cur_large_twiddles,
    const DT *_twiddles, const DT *dft_matrix, const int large_section_num,
    const int large_butterfly_num, const int large_in_stride, void *nram_buf,
    const int *small_factors, const int nfft, const int dir,
    const int last_stage) {
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;
  const int K_num = 64 / sizeof(DT);
  int align_K = 0;
  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;

  const int large_out_stride = large_butterfly_num;
  int tw_offset;

  _small_stage_count = small_factors[0];
  tw_offset = small_factors[1];

  const DT *small_twiddles = _twiddles + tw_offset * 2;

  const int max_para_ldst_num = (4096 + large_radix - 1) / large_radix;

  int nram_buf_offset = 0;
  DT *nram_in_r = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  DT *nram_in_i = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  DT *nram_out_r = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  DT *nram_out_i = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  FFT_CPX_T<DT> nram_para_load_in_ping = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_ldst_num};
  nram_buf_offset += large_radix * max_para_ldst_num * 2;

  FFT_CPX_T<DT> nram_para_load_in_pong = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_ldst_num};
  nram_buf_offset += large_radix * max_para_ldst_num * 2;

  FFT_CPX_T<DT> nram_para_load_tw_ping = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_ldst_num};
  nram_buf_offset += large_radix * max_para_ldst_num * 2;

  FFT_CPX_T<DT> nram_para_load_tw_pong = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_ldst_num};
  nram_buf_offset += large_radix * max_para_ldst_num * 2;

  FFT_CPX_T<DT> nram_para_store_ping = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_ldst_num};
  nram_buf_offset += large_radix * max_para_ldst_num * 2;

  FFT_CPX_T<DT> nram_para_store_pong = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_ldst_num};
  nram_buf_offset += large_radix * max_para_ldst_num * 2;

  FFT_CPX_T<DT> nram_transpose_temp;
  nram_transpose_temp = {
      (DT *)nram_in_r,
      (DT *)nram_in_r + large_radix * ((int)last_stage) +
          large_radix * (1 - (int)last_stage) * max_para_ldst_num};

  DT *_nram_tw = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * 2;

  int ld_dft_radix = -1;
  const int max_radix = 64;
  DT *nram_dftmtx = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += max_radix * max_radix * 2;

  DT *nram_scratch = (DT *)nram_buf + nram_buf_offset;

  DT *CPX_MUL_RR = nram_scratch;
  DT *CPX_MUL_RI = &CPX_MUL_RR[large_radix * max_para_ldst_num];
  DT *CPX_MUL_IR = &CPX_MUL_RI[large_radix * max_para_ldst_num];
  DT *CPX_MUL_II = &CPX_MUL_IR[large_radix * max_para_ldst_num];

  nram_buf_offset += large_radix * max_para_ldst_num * 4;

  int Fin_stride = 0, Fout_stride = 0;
  int sec_count;
  int repeat_num =
      (large_butterfly_num + max_para_ldst_num - 1) / max_para_ldst_num;
  for (sec_count = 0; sec_count < large_section_num; ++sec_count) {
    for (int repeat_id = 0; repeat_id < repeat_num + 2; ++repeat_id) {
      if (repeat_id < repeat_num) {
        int i = max_para_ldst_num * repeat_id;
        FFT_CPX_T<DT> nram_para_load_in = (repeat_id % 2 == 0)
                                              ? nram_para_load_in_ping
                                              : nram_para_load_in_pong;

        FFT_CPX_T<DT> nram_para_load_tw = (repeat_id % 2 == 0)
                                              ? nram_para_load_tw_ping
                                              : nram_para_load_tw_pong;

        int para_load_num = (max_para_ldst_num > (large_butterfly_num - i))
                                ? (large_butterfly_num - i)
                                : max_para_ldst_num;

        __memcpy_async(nram_para_load_in.r, input + Fin_stride + i,
                       sizeof(DT) * para_load_num, GDRAM2NRAM,
                       sizeof(DT) * para_load_num, large_in_stride * sizeof(DT),
                       large_radix - 1);
        __memcpy_async(nram_para_load_in.i, input + nfft + Fin_stride + i,
                       sizeof(DT) * para_load_num, GDRAM2NRAM,
                       sizeof(DT) * para_load_num, large_in_stride * sizeof(DT),
                       large_radix - 1);
        __memcpy_async(nram_para_load_tw.r, cur_large_twiddles + i,
                       sizeof(DT) * para_load_num, SRAM2NRAM,
                       sizeof(DT) * para_load_num,
                       large_out_stride * sizeof(DT), large_radix - 2);
        __memcpy_async(
            nram_para_load_tw.i,
            cur_large_twiddles + large_butterfly_num * (large_radix - 1) + i,
            sizeof(DT) * para_load_num, SRAM2NRAM, sizeof(DT) * para_load_num,
            large_out_stride * sizeof(DT), large_radix - 2);
      }

      if (repeat_id >= 2) {
        int i = max_para_ldst_num * (repeat_id - 2);

        int para_store_num = (max_para_ldst_num > (large_butterfly_num - i))
                                 ? (large_butterfly_num - i)
                                 : max_para_ldst_num;

        FFT_CPX_T<DT> nram_para_store =
            (repeat_id % 2 == 0) ? nram_para_store_ping : nram_para_store_pong;

        if (last_stage) {
          __memcpy_async(output + (Fout_stride + i) * 2, nram_para_store.r,
                         sizeof(DT) * 2 * para_store_num, NRAM2GDRAM,
                         large_out_stride * 2 * sizeof(DT),
                         sizeof(DT) * 2 * para_store_num, large_radix - 1);
        } else {
          __memcpy_async(output + Fout_stride + i, nram_para_store.r,
                         para_store_num * sizeof(DT), NRAM2GDRAM,
                         large_out_stride * sizeof(DT),
                         sizeof(DT) * para_store_num, large_radix - 1);
          __memcpy_async(output + Fout_stride + i + nfft, nram_para_store.i,
                         para_store_num * sizeof(DT), NRAM2GDRAM,
                         large_out_stride * sizeof(DT),
                         sizeof(DT) * para_store_num, large_radix - 1);
        }
      }

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

        int para_ldst_num = (max_para_ldst_num > (large_butterfly_num - i))
                                ? (large_butterfly_num - i)
                                : max_para_ldst_num;

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
              small_section_num * para_ldst_num, 1, dir, radix);

          small_stage_count--;
          if (small_stage_count == 0) {
            if (last_stage) {
              __memcpy_async(nram_transpose_temp.r, nram_out_r,
                             sizeof(DT) * large_radix, NRAM2NRAM,
                             sizeof(DT) * large_radix * 2,
                             sizeof(DT) * large_radix, para_ldst_num - 1);

              __memcpy_async(nram_transpose_temp.i, nram_out_i,
                             sizeof(DT) * large_radix, NRAM2NRAM,
                             sizeof(DT) * large_radix * 2,
                             sizeof(DT) * large_radix, para_ldst_num - 1);
              __sync_move();

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

              if (sec_count == 0 && repeat_id == 1) {
                __memcpy(nram_tw, small_twiddles,
                         small_butterfly_num * (radix - 1) * sizeof(DT) * 2,
                         SRAM2NRAM);
                small_twiddles += small_butterfly_num * (radix - 1) * 2;
              }

              computeGenericButterflyOtherstagesMat(
                  nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                  nram_dftmtx, nram_tw, small_section_num, small_butterfly_num,
                  para_ldst_num, small_in_stride, dir, radix);

              nram_tw += small_butterfly_num * (radix - 1) * 2;
            }

            {
              FFT_SWAP_PTR(nram_out_r, nram_in_r);
              FFT_SWAP_PTR(nram_out_i, nram_in_i);

              radix = small_factors[value_mul++];
              small_section_num = small_factors[value_mul++];
              small_butterfly_num = small_factors[value_mul++];
              small_in_stride = small_factors[value_mul];

              if (sec_count == 0 && repeat_id == 1) {
                __memcpy_async(
                    nram_tw, small_twiddles,
                    small_butterfly_num * (radix - 1) * sizeof(DT) * 2,
                    SRAM2NRAM);
                __sync_move();
              }

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
                  para_ldst_num, small_in_stride, dir, radix);

              if (last_stage) {
                __memcpy_async(nram_transpose_temp.r, nram_out_r,
                               sizeof(DT) * large_radix, NRAM2NRAM,
                               sizeof(DT) * large_radix * 2,
                               sizeof(DT) * large_radix, para_ldst_num - 1);

                __memcpy_async(nram_transpose_temp.i, nram_out_i,
                               sizeof(DT) * large_radix, NRAM2NRAM,
                               sizeof(DT) * large_radix * 2,
                               sizeof(DT) * large_radix, para_ldst_num - 1);
                __sync_move();

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
    Fin_stride += large_butterfly_num;
    Fout_stride += large_radix * large_butterfly_num;
  }
}

template <typename DT>
__mlu_func__ void computeLargeButterflyLaststage(
    DT *output, DT *input, const int large_radix, const DT *cur_large_twiddles,
    const DT *_twiddles, const DT *dft_matrix, const int large_section_num,
    const int large_butterfly_num, const int large_in_stride, void *nram_buf,
    const int *small_factors, const int nfft, const int dir) {
  computeLargeButterflyOtherstages(
      output, input, large_radix, cur_large_twiddles, _twiddles, dft_matrix,
      large_section_num, large_butterfly_num, large_in_stride, nram_buf,
      small_factors, nfft, dir, 1);
}

// Compute the large butterfly for the last stage of the FFT
template <typename DT>
__mlu_func__ void computeLargeButterflyOtherstagesBatchPingpong(
    DT *output, DT *input, const int large_radix, const DT *cur_large_twiddles,
    const DT *small_twiddles, const int small_twiddles_size,
    const DT *dft_matrix, const int large_section_num,
    const int large_butterfly_num, const int large_in_stride, void *nram_buf,
    const int *small_factors, const int nfft, const int t_start,
    const int t_end, const int dir, const int last_stage,
    const int load_once_twiddles) {
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;

  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;

  const int large_out_stride = large_butterfly_num;
  int tw_offset;
  const int K_num = 64 / sizeof(DT);
  int align_K = 0;
  _small_stage_count = small_factors[0];
  tw_offset = small_factors[1];

  int max_para_ldst_num = (large_butterfly_num < small_factors[3])
                              ? large_butterfly_num
                              : small_factors[3];

  int nram_buf_offset = 0;

  FFT_CPX_T<DT> nram_para_load_in_ping = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_ldst_num};
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  FFT_CPX_T<DT> nram_para_load_in_pong = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_ldst_num};
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  FFT_CPX_T<DT> nram_para_load_tw = {
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
  nram_buf_offset += small_twiddles_size;  // complex

  int ld_dft_radix = -1;
  const int max_radix = 64;
  DT *nram_dftmtx = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += max_radix * max_radix * 2;  // complex

  DT *nram_scratch = (DT *)nram_buf + nram_buf_offset;

  // temp overlap with "nram_scratch"
  DT *CPX_MUL_RR = nram_scratch;
  DT *CPX_MUL_RI = &CPX_MUL_RR[large_radix * max_para_ldst_num];
  DT *CPX_MUL_IR = &CPX_MUL_RI[large_radix * max_para_ldst_num];
  DT *CPX_MUL_II = &CPX_MUL_IR[large_radix * max_para_ldst_num];

  nram_buf_offset += large_radix * max_para_ldst_num * 4;  // complex

  // overlap nram_in
  FFT_CPX_T<DT> nram_transpose_temp;
  nram_transpose_temp = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * ((int)last_stage) +
          large_radix * (1 - (int)last_stage) * max_para_ldst_num};

  if (small_twiddles_size) {
    if (load_once_twiddles) {
      __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, SRAM2NRAM);
    } else {
      __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, GDRAM2NRAM);
    }
  }

  int sec_count;

  int repeat_num = (t_end - t_start);
  input += t_start * (nfft << 1);
  output += t_start * (nfft << 1);

  for (int butterfly_id = 0; butterfly_id < large_butterfly_num;
       butterfly_id += max_para_ldst_num) {
    for (sec_count = 0; sec_count < large_section_num; ++sec_count) {
      DT *output_batch = output;
      DT *input_batch = input;
      int para_num = (max_para_ldst_num > (large_butterfly_num - butterfly_id))
                         ? (large_butterfly_num - butterfly_id)
                         : max_para_ldst_num;

      for (int repeat_id = 0; repeat_id < repeat_num + 2; ++repeat_id,
               input_batch += (nfft << 1), output_batch += (nfft << 1)) {
        // pipeline: load-stage

        if (repeat_id < repeat_num) {
          if (1) {
            __memcpy_async(
                nram_para_load_in_ping.r,
                input_batch + sec_count * large_butterfly_num + butterfly_id,
                sizeof(DT) * para_num, GDRAM2NRAM, sizeof(DT) * para_num,
                large_in_stride * sizeof(DT), large_radix - 1);
            __memcpy_async(nram_para_load_in_ping.i,
                           input_batch + nfft +
                               sec_count * large_butterfly_num + butterfly_id,
                           sizeof(DT) * para_num, GDRAM2NRAM,
                           sizeof(DT) * para_num, large_in_stride * sizeof(DT),
                           large_radix - 1);
            if (repeat_id == 0 && sec_count == 0) {
              if (load_once_twiddles) {
                __memcpy_async(
                    nram_para_load_tw.r, cur_large_twiddles + butterfly_id,
                    sizeof(DT) * para_num, SRAM2NRAM, sizeof(DT) * para_num,
                    large_out_stride * sizeof(DT), large_radix - 2);
                __memcpy_async(
                    nram_para_load_tw.i,
                    cur_large_twiddles +
                        large_butterfly_num * (large_radix - 1) + butterfly_id,
                    sizeof(DT) * para_num, SRAM2NRAM, sizeof(DT) * para_num,
                    large_out_stride * sizeof(DT), large_radix - 2);

              } else {
                __memcpy_async(
                    nram_para_load_tw.r, cur_large_twiddles + butterfly_id,
                    sizeof(DT) * para_num, GDRAM2NRAM, sizeof(DT) * para_num,
                    large_out_stride * sizeof(DT), large_radix - 2);
                __memcpy_async(
                    nram_para_load_tw.i,
                    cur_large_twiddles +
                        large_butterfly_num * (large_radix - 1) + butterfly_id,
                    sizeof(DT) * para_num, GDRAM2NRAM, sizeof(DT) * para_num,
                    large_out_stride * sizeof(DT), large_radix - 2);
              }
            }
          }
        }

        // pipeline: store-stage
        if (repeat_id >= 2) {
          if (last_stage) {
            __memcpy_async(output_batch - (nfft << 2) +
                               (sec_count * large_radix * large_butterfly_num +
                                butterfly_id) *
                                   2,
                           nram_para_store_ping.r, sizeof(DT) * 2 * para_num,
                           NRAM2GDRAM, large_out_stride * 2 * sizeof(DT),
                           sizeof(DT) * 2 * para_num, large_radix - 1);
          } else {
            // real
            __memcpy_async(output_batch - (nfft << 2) +
                               sec_count * large_radix * large_butterfly_num +
                               butterfly_id,
                           nram_para_store_ping.r, para_num * sizeof(DT),
                           NRAM2GDRAM, large_out_stride * sizeof(DT),
                           sizeof(DT) * para_num, large_radix - 1);
            // imag
            __memcpy_async(output_batch - (nfft << 2) +
                               sec_count * large_radix * large_butterfly_num +
                               butterfly_id + nfft,
                           nram_para_store_ping.i, para_num * sizeof(DT),
                           NRAM2GDRAM, large_out_stride * sizeof(DT),
                           sizeof(DT) * para_num, large_radix - 1);
          }
        }
        // pipeline: compute-stage

        if (repeat_id >= 1 && repeat_id < repeat_num + 1) {
          DT *nram_in_r = nram_para_load_in_pong.r;
          DT *nram_in_i = nram_para_load_in_pong.i;

          DT *nram_out_r = nram_para_store_pong.r;
          DT *nram_out_i = nram_para_store_pong.i;

          // rotation-large
          __bang_mul(CPX_MUL_RR, nram_para_load_in_pong.r + para_num,
                     nram_para_load_tw.r, para_num * (large_radix - 1));
          __bang_mul(CPX_MUL_II, nram_para_load_in_pong.i + para_num,
                     nram_para_load_tw.i, para_num * (large_radix - 1));
          __bang_mul(CPX_MUL_RI, nram_para_load_in_pong.r + para_num,
                     nram_para_load_tw.i, para_num * (large_radix - 1));
          __bang_mul(CPX_MUL_IR, nram_para_load_in_pong.i + para_num,
                     nram_para_load_tw.r, para_num * (large_radix - 1));

          __bang_sub(nram_para_load_in_pong.r + para_num, CPX_MUL_RR,
                     CPX_MUL_II, para_num * (large_radix - 1));
          __bang_add(nram_para_load_in_pong.i + para_num, CPX_MUL_RI,
                     CPX_MUL_IR, para_num * (large_radix - 1));

          {
            // load real & imag

            radix = small_factors[4];
            small_section_num = small_factors[5];
            small_in_stride = small_factors[7];
            small_stage_count = _small_stage_count;

            // first stage
            // if(0)
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
                nram_out_r, nram_out_i, nram_para_load_in_pong.r,
                nram_para_load_in_pong.i, nram_scratch, nram_dftmtx,
                small_section_num * para_num, small_section_num * para_num, 1,
                dir, radix);

            // [radix, small_section_num, para_num] ->
            // [small_section_num, para_num, radix] ->
            // [para_num, small_section_num, radix] ->
            // [small_section_num, radix, para_num] ==
            // [large_radix, para_num]

            small_stage_count--;
            if (small_stage_count == 0) {
              // nram to gdram

              // [nfft, 2] -> [2, nfft] -> [2, nfft] -> [nfft, 2]
              if (last_stage) {
                __memcpy(nram_transpose_temp.r, nram_out_r,
                         sizeof(DT) * large_radix, NRAM2NRAM,
                         sizeof(DT) * large_radix * 2, sizeof(DT) * large_radix,
                         para_num - 1);

                __memcpy(nram_transpose_temp.i, nram_out_i,
                         sizeof(DT) * large_radix, NRAM2NRAM,
                         sizeof(DT) * large_radix * 2, sizeof(DT) * large_radix,
                         para_num - 1);

                __bang_transpose(nram_para_store_pong.r, nram_transpose_temp.r,
                                 para_num * 2, large_radix);
              } else {
                if (nram_out_r == nram_para_store_pong.r) {
                  FFT_SWAP_PTR(nram_para_load_in_pong.r, nram_para_store_pong.r)
                  FFT_SWAP_PTR(nram_para_load_in_pong.i, nram_para_store_pong.i)
                }
                __bang_transpose(nram_para_store_pong.r, nram_out_r, para_num,
                                 large_radix);
                __bang_transpose(nram_para_store_pong.i, nram_out_i, para_num,
                                 large_radix);
              }

            } else {
              FFT_SWAP_PTR(nram_out_r, nram_in_r);
              FFT_SWAP_PTR(nram_out_i, nram_in_i);
              // DT* nram_transpose_store = nram_in_r;

              // after first stage: [butterfly_num, para_ldst_num, radix]
              // other in: [para_ldst_num, butterfly_num, radix] ==
              // [para_ldst_num, large_radix]
              TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r,
                                     nram_in_i, small_section_num, para_num,
                                     radix, DT)

              // if last-stage: stride = large_radix * 2
              //                compute_id 0 r
              //                compute_id 0 i
              //                compute_id 1 r
              //                compute_id 1 i
              // else: stride = large_radix
              //                compute_id 0 r
              //                compute_id 1 i
              //                compute_id 0 r
              //                compute_id 1 i

              // [radix, small_section_num, para_ldst_num] ->
              // [small_section_num, para_ldst_num, radix] ->
              // [para_ldst_num, small_section_num, radix] ->
              // [small_section_num, radix, para_ldst_num] ==
              // [large_radix, para_ldst_num]

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
                    nram_dftmtx, nram_tw, small_section_num,
                    small_butterfly_num, para_num, small_in_stride, dir, radix);

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
                    nram_dftmtx, nram_tw, small_section_num,
                    small_butterfly_num, para_num, small_in_stride, dir, radix);

                // if last-stage: stride = large_radix * 2
                //                compute_id 0 r
                //                compute_id 0 i
                //                compute_id 1 r
                //                compute_id 1 i
                // else: stride = large_radix
                //                compute_id 0 r
                //                compute_id 1 i
                //                compute_id 0 r
                //                compute_id 1 i

                if (last_stage) {
                  __memcpy(nram_transpose_temp.r, nram_out_r,
                           sizeof(DT) * large_radix, NRAM2NRAM,
                           sizeof(DT) * large_radix * 2,
                           sizeof(DT) * large_radix, para_num - 1);

                  __memcpy(nram_transpose_temp.i, nram_out_i,
                           sizeof(DT) * large_radix, NRAM2NRAM,
                           sizeof(DT) * large_radix * 2,
                           sizeof(DT) * large_radix, para_num - 1);

                  __bang_transpose(nram_para_store_pong.r,
                                   nram_transpose_temp.r, para_num * 2,
                                   large_radix);
                } else {
                  if (nram_out_r == nram_para_store_pong.r) {
                    FFT_SWAP_PTR(nram_para_load_in_pong.r,
                                 nram_para_store_pong.r)
                    FFT_SWAP_PTR(nram_para_load_in_pong.i,
                                 nram_para_store_pong.i)
                  }

                  __bang_transpose(nram_para_store_pong.r, nram_out_r, para_num,
                                   large_radix);
                  __bang_transpose(nram_para_store_pong.i, nram_out_i, para_num,
                                   large_radix);
                }
              }
            }
          }
        }

        __sync();
        FFT_SWAP_PTR(nram_para_load_in_ping.r, nram_para_load_in_pong.r)
        FFT_SWAP_PTR(nram_para_load_in_ping.i, nram_para_load_in_pong.i)
        FFT_SWAP_PTR(nram_para_store_ping.r, nram_para_store_pong.r)
        FFT_SWAP_PTR(nram_para_store_ping.i, nram_para_store_pong.i)
      }
    }
  }
}

template <typename DT>
__mlu_func__ void computeLargeButterflyLaststageBatchPingpong(
    DT *output, DT *input, const int large_radix, const DT *cur_large_twiddles,
    const DT *small_twiddles, const int small_twiddles_size,
    const DT *dft_matrix, const int large_section_num,
    const int large_butterfly_num, const int large_in_stride, void *nram_buf,
    const int *small_factors, const int nfft, const int t_start,
    const int t_end, const int dir, const int load_once_twiddles) {
  computeLargeButterflyOtherstagesBatchPingpong(
      output, input, large_radix, cur_large_twiddles, small_twiddles,
      small_twiddles_size, dft_matrix, large_section_num, large_butterfly_num,
      large_in_stride, nram_buf, small_factors, nfft, t_start, t_end, dir, 1,
      load_once_twiddles);
}

// Compute the large butterfly for the last stage using batch ping-pong
// processing
template <typename DT>
__mlu_func__ void computeLargeButterflyFirststageColumn(
    DT *output, DT *input, const int large_radix, const int large_in_stride,
    const int section_num, const DT *small_twiddles,
    const int small_twiddles_size, const DT *dft_matrix, void *nram_buf,
    const int *small_factors, const int dir, const int nfft,
    const int last_stage, const int para_batch, const int nb,
    const int load_once_twiddles) {
  // constant
  const int K_num = 64 / sizeof(DT);
  int align_K = 0;
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;

  // network info
  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;
  int tw_offset;
  _small_stage_count = small_factors[0];
  tw_offset = small_factors[1];

  // load compute store
  // (0)                              load 0 ping sync()
  // (1)              compute 0 ping  load 1 pong sync()
  // (2) store 0      compute 1 pong  load 2 ping sync()
  // (3) store 1      compute 2   load 3  sync()

  // compute last-large-stage (nram_out_r,nram_out_i) [2, large_radix]->
  // transpose -> [large_radix, 2]

  // complex array -> real array, imag array -> complex array
  // first-large-stage complex -> real array, imag array
  // other-large-stage none
  // last-large-stage real array, imag array -> complex
  // const DT *small_twiddles = twiddles + tw_offset * 2;  // complex

  // assign nram space
  int nram_buf_offset = 0;

  // parallel load/store space
  DT *nram_para_load_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * para_batch * 2;  // complex

  DT *nram_para_load_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * para_batch * 2;  // complex

  DT *nram_para_store_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * para_batch * 2;  // complex

  DT *nram_para_store_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * para_batch * 2;  // complex

  // if last-stage:
  //                compute_id 0 r
  //                compute_id 0 i
  //                compute_id 1 r
  //                compute_id 1 i
  // else:
  //                compute_id 0 r
  //                compute_id 1 i
  //                compute_id 0 r
  //                compute_id 1 i

  DT *_nram_tw = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * 2;  // complex

  // load dftmtx sample
  int ld_dft_radix = -1;
  const int max_radix = 64;
  DT *nram_dftmtx = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += max_radix * max_radix * 2;  // complex

  DT *nram_scratch = (DT *)nram_buf + nram_buf_offset;

  if (small_twiddles_size) {
    if (load_once_twiddles) {
      __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, SRAM2NRAM);
    } else {
      __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, GDRAM2NRAM);
    }
  }

  int repeat_num = section_num;
  for (int repeat_id = 0; repeat_id < repeat_num + 2; ++repeat_id) {
    // pipeline: load-stage

    if (repeat_id < repeat_num) {
      int i = repeat_id;

      __memcpy_async(nram_para_load_ping, input + i * 2 * nb,
                     sizeof(DT) * 2 * para_batch, GDRAM2NRAM,
                     sizeof(DT) * 2 * para_batch,
                     nb * large_in_stride * sizeof(DT) * 2, large_radix - 1);
    }

    if (repeat_id >= 2) {
      int i = (repeat_id - 2);

      if (last_stage) {
        __memcpy_async(output + i * large_radix * 2 * nb, nram_para_store_ping,
                       para_batch * sizeof(DT) * 2, NRAM2GDRAM,
                       nb * 2 * sizeof(DT), para_batch * sizeof(DT) * 2,
                       large_radix - 1);

      } else {
        // real
        __memcpy_async(output + i * large_radix * para_batch,
                       nram_para_store_ping,
                       para_batch * large_radix * sizeof(DT), NRAM2GDRAM);
        // imag
        __memcpy_async(
            output + i * large_radix * para_batch + nfft * para_batch,
            nram_para_store_ping + para_batch * large_radix,
            para_batch * large_radix * sizeof(DT), NRAM2GDRAM);
      }
    }

    // pipeline: compute-stage

    if (repeat_id >= 1 && repeat_id < repeat_num + 1) {
      DT *nram_in_r = nram_para_store_pong;
      DT *nram_in_i = nram_para_store_pong + large_radix * para_batch;

      DT *nram_out_r = nram_para_load_pong;
      DT *nram_out_i = nram_para_load_pong + large_radix * para_batch;

      __bang_transpose(nram_in_r, nram_para_load_pong, large_radix * para_batch,
                       2);

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

        computeGenericButterflyFirststageMat(
            nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
            nram_dftmtx, small_section_num * para_batch,
            small_section_num * para_batch, 1, dir, radix);

        small_stage_count--;
        if (small_stage_count == 0) {
          FFT_SWAP_PTR(nram_out_r, nram_in_r);
          FFT_SWAP_PTR(nram_out_i, nram_in_i);
          __bang_transpose(nram_out_r, nram_in_r, para_batch, large_radix);
          __bang_transpose(nram_out_i, nram_in_i, para_batch, large_radix);

          if (nram_out_r == nram_para_store_pong) {
            FFT_SWAP_PTR(nram_para_load_pong, nram_para_store_pong)
          }

          if (last_stage) {
            //  [2, para_batch, large_radix] -> [para_batch, large_radix,
            //  2]
            // DT* nram_transpose_store = nram_in_r;

            __bang_transpose(nram_para_store_pong, nram_out_r, 2,
                             para_batch * large_radix);
          } else {
            //  [2, para_batch, large_radix] -> [2, para_batch,
            //  large_radix]
            __memcpy(nram_para_store_pong, nram_out_r,
                     para_batch * large_radix * sizeof(DT), NRAM2NRAM);
            __memcpy(nram_para_store_pong + para_batch * large_radix,
                     nram_out_i, para_batch * large_radix * sizeof(DT),
                     NRAM2NRAM);
          }

        } else {
          FFT_SWAP_PTR(nram_out_r, nram_in_r);
          FFT_SWAP_PTR(nram_out_i, nram_in_i);

          TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                                 small_section_num, para_batch, radix, DT)

          value_mul = 8;
          DT *nram_tw = _nram_tw;

          for (; small_stage_count > 1; small_stage_count--) {
            FFT_SWAP_PTR(nram_out_r, nram_in_r);
            FFT_SWAP_PTR(nram_out_i, nram_in_i);

            // update parameter
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
                para_batch, small_in_stride, dir, radix);

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
                para_batch, small_in_stride, dir, radix);

            if (last_stage) {
              //  [2, para_batch, large_radix] -> [para_batch, large_radix,
              //  2]
              FFT_SWAP_PTR(nram_out_r, nram_in_r);
              FFT_SWAP_PTR(nram_out_i, nram_in_i);
              __bang_transpose(nram_out_r, nram_in_r, para_batch, large_radix);
              __bang_transpose(nram_out_i, nram_in_i, para_batch, large_radix);

              if (nram_out_r == nram_para_store_pong) {
                FFT_SWAP_PTR(nram_para_load_pong, nram_para_store_pong)
              }

              __bang_transpose(nram_para_store_pong, nram_out_r, 2,
                               para_batch * large_radix);

            } else {
              if (nram_out_r == nram_para_store_pong) {
                FFT_SWAP_PTR(nram_para_load_pong, nram_para_store_pong)
              }

              __bang_transpose(nram_para_store_pong, nram_out_r, para_batch,
                               large_radix);
              __bang_transpose(nram_para_store_pong + para_batch * large_radix,
                               nram_out_i, para_batch, large_radix);
            }
          }
        }
      }
    }

    __sync();
    FFT_SWAP_PTR(nram_para_load_ping, nram_para_load_pong)
    FFT_SWAP_PTR(nram_para_store_ping, nram_para_store_pong)
  }
}

// Compute the large butterfly for the subsequent stages using column processing
template <typename DT>
__mlu_func__ void computeLargeButterflyOtherstagesColumn(
    DT *output, DT *input, const int large_radix, const DT *cur_large_twiddles,
    const DT *small_twiddles, const int small_twiddles_size,
    const DT *dft_matrix, const int large_section_num,
    const int large_butterfly_num, const int large_in_stride, void *nram_buf,
    const int *small_factors, const int nfft, const int dir,
    const int last_stage, const int para_batch, const int nb,
    const int load_once_twiddles) {
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;

  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;

  const int large_out_stride = large_butterfly_num;
  int tw_offset;

  _small_stage_count = small_factors[0];
  tw_offset = small_factors[1];

  const int K_num = 64 / sizeof(DT);
  int align_K = 0;

  int nram_buf_offset = 0;
  DT *nram_in_r = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * para_batch;

  DT *nram_in_i = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * para_batch;

  DT *nram_out_r = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * para_batch;

  DT *nram_out_i = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * para_batch;

  // parallel load/store space
  FFT_CPX_T<DT> nram_para_load_in_ping = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * para_batch};
  nram_buf_offset += large_radix * para_batch * 2;  // complex

  FFT_CPX_T<DT> nram_para_load_in_pong = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * para_batch};
  nram_buf_offset += large_radix * para_batch * 2;  // complex

  FFT_CPX_T<DT> nram_para_load_tw_ping = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + (large_radix - 1)};
  nram_buf_offset += (large_radix - 1) * 2;  // complex

  FFT_CPX_T<DT> nram_para_load_tw_pong = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + (large_radix - 1)};
  nram_buf_offset += (large_radix - 1) * 2;  // complex

  FFT_CPX_T<DT> nram_para_store_ping = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * para_batch};
  nram_buf_offset += large_radix * para_batch * 2;  // complex

  FFT_CPX_T<DT> nram_para_store_pong = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * para_batch};
  nram_buf_offset += large_radix * para_batch * 2;  // complex

  // overlap nram_in
  FFT_CPX_T<DT> nram_transpose_temp;
  // temp out-space before transpose
  // if last-stage:
  //                compute_id 0 r
  //                compute_id 0 i
  //                compute_id 1 r
  //                compute_id 1 i
  // else:
  //                compute_id 0 r
  //                compute_id 1 i
  //                compute_id 0 r
  //                compute_id 1 i
  nram_transpose_temp = {(DT *)nram_in_r,
                         (DT *)nram_in_r + large_radix * ((int)last_stage) +
                             large_radix * (1 - (int)last_stage) * para_batch};

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

  // temp overlap with "nram_scratch"
  DT *CPX_MUL_RR = nram_scratch;
  DT *CPX_MUL_RI = &CPX_MUL_RR[(large_radix - 1) * para_batch];
  DT *CPX_MUL_IR = &CPX_MUL_RI[(large_radix - 1) * para_batch];
  DT *CPX_MUL_II = &CPX_MUL_IR[(large_radix - 1) * para_batch];

  nram_buf_offset += (large_radix - 1) * para_batch * 4;  // complex

  int Fin_stride = 0, Fout_stride = 0;
  int sec_count;
  int repeat_num = large_butterfly_num;

  for (sec_count = 0; sec_count < large_section_num; ++sec_count) {
    for (int repeat_id = 0; repeat_id < repeat_num + 2; ++repeat_id) {
      if (repeat_id < repeat_num) {
        int i = repeat_id;
        FFT_CPX_T<DT> nram_para_load_in = (repeat_id % 2 == 0)
                                              ? nram_para_load_in_ping
                                              : nram_para_load_in_pong;

        FFT_CPX_T<DT> nram_para_load_tw = (repeat_id % 2 == 0)
                                              ? nram_para_load_tw_ping
                                              : nram_para_load_tw_pong;
        {
          __memcpy_async(
              nram_para_load_in.r, input + (Fin_stride + i) * para_batch,
              sizeof(DT) * para_batch, GDRAM2NRAM, sizeof(DT) * para_batch,
              para_batch * large_in_stride * sizeof(DT), large_radix - 1);
          __memcpy_async(
              nram_para_load_in.i,
              input + para_batch * nfft + (Fin_stride + i) * para_batch,
              sizeof(DT) * para_batch, GDRAM2NRAM, sizeof(DT) * para_batch,
              para_batch * large_in_stride * sizeof(DT), large_radix - 1);

          if (load_once_twiddles) {
            __memcpy_async(nram_para_load_tw.r,
                           cur_large_twiddles + i * (large_radix - 1),
                           sizeof(DT) * (large_radix - 1) * 2, SRAM2NRAM);
            __memcpy_async(nram_para_load_tw.i,
                           cur_large_twiddles +
                               large_butterfly_num * (large_radix - 1) +
                               i * (large_radix - 1),
                           sizeof(DT) * (large_radix - 1), SRAM2NRAM);
          } else {
            __memcpy_async(nram_para_load_tw.r,
                           cur_large_twiddles + i * (large_radix - 1),
                           sizeof(DT) * (large_radix - 1) * 2, GDRAM2NRAM);
            __memcpy_async(nram_para_load_tw.i,
                           cur_large_twiddles +
                               large_butterfly_num * (large_radix - 1) +
                               i * (large_radix - 1),
                           sizeof(DT) * (large_radix - 1), GDRAM2NRAM);
          }
        }
      }

      // pipeline: store-stage
      if (repeat_id >= 2) {
        int i = (repeat_id - 2);

        FFT_CPX_T<DT> nram_para_store =
            (repeat_id % 2 == 0) ? nram_para_store_ping : nram_para_store_pong;

        if (last_stage) {
          __memcpy_async(output + (Fout_stride + i) * 2 * nb, nram_para_store.r,
                         sizeof(DT) * 2 * para_batch, NRAM2GDRAM,
                         nb * large_out_stride * 2 * sizeof(DT),
                         sizeof(DT) * 2 * para_batch, large_radix - 1);
        } else {
          // real
          __memcpy_async(output + (Fout_stride + i) * para_batch,
                         nram_para_store.r, para_batch * sizeof(DT), NRAM2GDRAM,
                         para_batch * large_out_stride * sizeof(DT),
                         sizeof(DT) * para_batch, large_radix - 1);
          // imag
          __memcpy_async(
              output + (Fout_stride + i) * para_batch + nfft * para_batch,
              nram_para_store.i, para_batch * sizeof(DT), NRAM2GDRAM,
              para_batch * large_out_stride * sizeof(DT),
              sizeof(DT) * para_batch, large_radix - 1);
        }
      }

      if (repeat_id >= 1 && repeat_id < repeat_num + 1) {
        FFT_CPX_T<DT> nram_para_load_in = (repeat_id % 2 != 0)
                                              ? nram_para_load_in_ping
                                              : nram_para_load_in_pong;

        FFT_CPX_T<DT> nram_para_load_tw = (repeat_id % 2 != 0)
                                              ? nram_para_load_tw_ping
                                              : nram_para_load_tw_pong;

        FFT_CPX_T<DT> nram_para_store =
            (repeat_id % 2 != 0) ? nram_para_store_ping : nram_para_store_pong;

        // rotation-large
        {
          for (int i = 1; i < large_radix; i++) {
            __bang_mul_scalar(&CPX_MUL_RR[(i - 1) * para_batch],
                              nram_para_load_in.r + para_batch * i,
                              nram_para_load_tw.r[(i - 1)], para_batch);
            __bang_mul_scalar(&CPX_MUL_RI[(i - 1) * para_batch],
                              nram_para_load_in.r + para_batch * i,
                              nram_para_load_tw.i[(i - 1)], para_batch);
            __bang_mul_scalar(&CPX_MUL_IR[(i - 1) * para_batch],
                              nram_para_load_in.i + para_batch * i,
                              nram_para_load_tw.r[(i - 1)], para_batch);
            __bang_mul_scalar(&CPX_MUL_II[(i - 1) * para_batch],
                              nram_para_load_in.i + para_batch * i,
                              nram_para_load_tw.i[(i - 1)], para_batch);
          }
          __bang_sub(nram_para_load_in.r + para_batch, CPX_MUL_RR, CPX_MUL_II,
                     para_batch * (large_radix - 1));
          __bang_add(nram_para_load_in.i + para_batch, CPX_MUL_RI, CPX_MUL_IR,
                     para_batch * (large_radix - 1));
        }

        {
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

          computeGenericButterflyFirststageMat(
              nram_out_r, nram_out_i, nram_para_load_in.r, nram_para_load_in.i,
              nram_scratch, nram_dftmtx, small_section_num * para_batch,
              small_section_num * para_batch, 1, dir, radix);

          // [radix, small_section_num, para_ldst_num] ->
          // [small_section_num, para_ldst_num, radix] ->
          // [para_ldst_num, small_section_num, radix] ->
          // [small_section_num, radix, para_ldst_num] ==
          // [large_radix, para_ldst_num]

          small_stage_count--;
          if (small_stage_count == 0) {
            if (last_stage) {
              __memcpy(nram_transpose_temp.r, nram_out_r,
                       sizeof(DT) * large_radix, NRAM2NRAM,
                       sizeof(DT) * large_radix * 2, sizeof(DT) * large_radix,
                       para_batch - 1);

              __memcpy(nram_transpose_temp.i, nram_out_i,
                       sizeof(DT) * large_radix, NRAM2NRAM,
                       sizeof(DT) * large_radix * 2, sizeof(DT) * large_radix,
                       para_batch - 1);

              __bang_transpose(nram_para_store.r, nram_transpose_temp.r,
                               para_batch * 2, large_radix);
            } else {
              __bang_transpose(nram_para_store.r, nram_out_r, para_batch,
                               large_radix);
              __bang_transpose(nram_para_store.i, nram_out_i, para_batch,
                               large_radix);
            }

          } else {
            FFT_SWAP_PTR(nram_out_r, nram_in_r);
            FFT_SWAP_PTR(nram_out_i, nram_in_i);

            TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                                   small_section_num, para_batch, radix, DT)

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
                  para_batch, small_in_stride, dir, radix);

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
                  para_batch, small_in_stride, dir, radix);

              if (last_stage) {
                // [2, para_batch, large_radix] -> [large_radix, para_batch, 2]
                __memcpy(nram_transpose_temp.r, nram_out_r,
                         sizeof(DT) * large_radix, NRAM2NRAM,
                         sizeof(DT) * large_radix * 2, sizeof(DT) * large_radix,
                         para_batch - 1);

                __memcpy(nram_transpose_temp.i, nram_out_i,
                         sizeof(DT) * large_radix, NRAM2NRAM,
                         sizeof(DT) * large_radix * 2, sizeof(DT) * large_radix,
                         para_batch - 1);

                __bang_transpose(nram_para_store.r, nram_transpose_temp.r,
                                 para_batch * 2, large_radix);
              } else {
                __bang_transpose(nram_para_store.r, nram_out_r, para_batch,
                                 large_radix);
                __bang_transpose(nram_para_store.i, nram_out_i, para_batch,
                                 large_radix);
              }
            }
          }
        }
      }
      __sync();
    }
    Fin_stride += large_butterfly_num;
    Fout_stride += large_radix * large_butterfly_num;
  }
}

// Compute the large butterfly for the last stage using column processing
template <typename DT>
__mlu_func__ void computeLargeButterflyLaststageColumn(
    DT *output, DT *input, const int large_radix, const DT *cur_large_twiddles,
    const DT *small_twiddles, const int small_twiddles_size,
    const DT *dft_matrix, const int large_section_num,
    const int large_butterfly_num, const int large_in_stride, void *nram_buf,
    const int *small_factors, const int nfft, const int dir,
    const int para_batch, const int nb, const int load_once_twiddles) {
  computeLargeButterflyOtherstagesColumn(
      output, input, large_radix, cur_large_twiddles, small_twiddles,
      small_twiddles_size, dft_matrix, large_section_num, large_butterfly_num,
      large_in_stride, nram_buf, small_factors, nfft, dir, 1, para_batch, nb,
      load_once_twiddles);
}
