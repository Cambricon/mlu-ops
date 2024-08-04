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

// Compute the large butterfly for the last stage using batch ping-pong
// processing from complex to real (C2R)
template <typename DT>
__mlu_func__ void computeLargeButterflyLaststageBatchPingpongC2R(
    DT *output, DT *input, const int large_radix, const int large_out_stride,
    const int large_butterfly_num, const DT *small_twiddles,
    const int small_twiddles_size, const DT *dft_matrix, void *nram_buf,
    const int *small_factors, const int nfft, const int t_start,
    const int t_end, const int load_once_twiddles) {
  const int dir = FFT_BACKWARD;
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;
  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;
  int tw_offset, max_para_num;

  const int K_num = 64 / sizeof(DT);
  int align_K = 0;

  _small_stage_count = small_factors[0];
  tw_offset = small_factors[1];

  max_para_num = (large_butterfly_num < small_factors[3]) ? large_butterfly_num
                                                          : small_factors[3];
  // load compute store
  // (0)                              load 0 ping sync()
  // (1)              compute 0 ping  load 1 pong sync()
  // (2) store 0      compute 1 pong  load 2 ping sync()
  // (3) store 1      compute 2   load 3  sync()

  // assign nram space
  int nram_buf_offset = 0;

  DT *nram_para_load_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_num * 2;  // complex

  DT *nram_para_load_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_num * 2;  // complex

  DT *nram_para_store_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_num * 2;  // complex

  DT *nram_para_store_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_num * 2;  // complex

  DT *_nram_tw = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * 2;  // complex

  int ld_dft_radix[2] = {-1, -1};
  const int max_radix = 64;
  DT *nram_dftmtx[2] = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + max_radix * max_radix * 2};
  nram_buf_offset += max_radix * max_radix * 4;  // complex

  DT *nram_scratch = (DT *)nram_buf + nram_buf_offset;

  if (small_twiddles_size) {
    if (load_once_twiddles) {
      __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, SRAM2NRAM);
    } else {
      __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, GDRAM2NRAM);
    }
  }

  int repeat_num = (t_end - t_start);

  // section_num loop
  const int odist = nfft;
  const int idist = nfft << 1;
  input += t_start * idist;
  output += t_start * odist;

  const int upper_radix = (large_radix) / 2 + 1;

  const int lower_radix = large_radix - upper_radix;

  for (int sec_id = 0; sec_id < large_butterfly_num; sec_id += max_para_num) {
    DT *output_batch = output;
    DT *input_batch = input;
    int para_num = (max_para_num > (large_butterfly_num - sec_id))
                       ? (large_butterfly_num - sec_id)
                       : max_para_num;

    for (int repeat_id = 0; repeat_id < repeat_num + 2;
         ++repeat_id, input_batch += idist, output_batch += odist) {
      // pipeline: load-stage
      if (repeat_id < repeat_num) {
        // [para_num][radix/2+1] -> [radix/2+1][para_num]

        __memcpy_async(nram_para_load_ping, input_batch + sec_id * upper_radix,
                       sizeof(DT) * para_num * upper_radix, GDRAM2NRAM);
        __memcpy_async(nram_para_load_ping + para_num * upper_radix,
                       input_batch + sec_id * upper_radix + nfft,
                       sizeof(DT) * para_num * upper_radix, GDRAM2NRAM);
      }

      // pipeline: store-stage
      if (repeat_id >= 2) {
        if (large_butterfly_num == 1) {
          // store only real part
          __memcpy_async(output_batch - 2 * odist, nram_para_store_ping,
                         sizeof(DT) * large_radix, NRAM2GDRAM);
        } else {
          // scatter-store
          __memcpy_async(output_batch - 2 * odist + sec_id,
                         nram_para_store_ping, sizeof(DT) * para_num,
                         NRAM2GDRAM, sizeof(DT) * large_out_stride,
                         sizeof(DT) * para_num, large_radix - 1);
        }
      }

      // pipeline: compute-stage
      if (repeat_id >= 1 && repeat_id < repeat_num + 1) {
        DT *nram_in_r = nram_para_store_pong;
        DT *nram_in_i = nram_para_store_pong + large_radix * para_num;

        DT *nram_out_r = nram_para_load_pong;
        DT *nram_out_i = nram_para_load_pong + large_radix * para_num;

        {
          DT *mirror_in = nram_scratch;
          DT *mirror_out = nram_scratch + lower_radix * para_num * 2;

          // 2d copy [para_num][lower_radix] from [para_num][upper_radix]
          __memcpy_async(mirror_in, nram_para_load_pong + 1,
                         sizeof(DT) * lower_radix, NRAM2NRAM,
                         sizeof(DT) * lower_radix, sizeof(DT) * upper_radix,
                         para_num - 1);

          __memcpy_async(mirror_in + lower_radix * para_num,
                         nram_para_load_pong + upper_radix * para_num + 1,
                         sizeof(DT) * lower_radix, NRAM2NRAM,
                         sizeof(DT) * lower_radix, sizeof(DT) * upper_radix,
                         para_num - 1);
          __sync_move();
          // mirror
          __bang_mirror(mirror_out, mirror_in, para_num, lower_radix);
          __bang_mirror(mirror_out + lower_radix * para_num,
                        mirror_in + lower_radix * para_num, para_num,
                        lower_radix);

          // neg
          __bang_mul_scalar(mirror_out + lower_radix * para_num,
                            mirror_out + lower_radix * para_num, -1,
                            para_num * lower_radix);

          // upper transpose
          __bang_transpose(nram_in_r, nram_para_load_pong, para_num,
                           upper_radix);
          __bang_transpose(nram_in_r + large_radix * para_num,
                           nram_para_load_pong + upper_radix * para_num,
                           para_num, upper_radix);
          // lower transpose
          __bang_transpose(nram_in_r + upper_radix * para_num, mirror_out,
                           para_num, lower_radix);
          __bang_transpose(nram_in_r + (upper_radix + large_radix) * para_num,
                           mirror_out + lower_radix * para_num, para_num,
                           lower_radix);
        }

        {
          // load real & imag
          radix = small_factors[4];
          small_section_num = small_factors[5];
          small_in_stride = small_factors[7];
          small_stage_count = _small_stage_count;

          // first stage
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

          // [radix, small_section_num, para_ldst_num] ->
          // [small_section_num, para_ldst_num, radix] -> [para_ldst_num,
          // small_section_num, radix]

          small_stage_count--;
          if (small_stage_count == 0) {
            if (nram_out_r == nram_para_store_pong) {
              FFT_SWAP_PTR(nram_para_load_pong, nram_para_store_pong)
              __bang_transpose(nram_para_store_pong, nram_out_r, para_num,
                               large_radix);
            } else {
              __bang_transpose(nram_para_store_pong, nram_out_r, para_num,
                               large_radix);
            }

          } else {
            // [small_section_num, para_ldst_num, radix] -> [para_ldst_num,
            // small_section_num, radix]
            FFT_SWAP_PTR(nram_out_r, nram_in_r);
            FFT_SWAP_PTR(nram_out_i, nram_in_i);

            TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                                   small_section_num, para_num, radix, DT)

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

              computeGenericButterflyLaststageMat(
                  nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                  nram_dftmtx[0], nram_tw, small_section_num,
                  small_butterfly_num, para_num, small_in_stride, dir, radix);

              if (nram_out_r == nram_para_store_pong) {
                FFT_SWAP_PTR(nram_para_load_pong, nram_para_store_pong)
                __bang_transpose(nram_para_store_pong, nram_out_r, para_num,
                                 large_radix);
              } else {
                __bang_transpose(nram_para_store_pong, nram_out_r, para_num,
                                 large_radix);
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

// Compute the large butterfly for the subsequent stages using batch ping-pong
// processing from complex to real (C2R)
template <typename DT>
__mlu_func__ void computeLargeButterflyOtherstagesBatchPingpongC2R(
    DT *output, DT *input, const int large_radix, const DT *cur_large_twiddles,
    const DT *small_twiddles, const int small_twiddles_size,
    const DT *dft_matrix, const int large_section_num,
    const int large_butterfly_num, const int large_out_stride, void *nram_buf,
    const int *small_factors, const int nfft, const int t_start,
    const int t_end, const int last_stage, const int load_once_twiddles) {
  const int dir = FFT_BACKWARD;
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;

  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;

  int tw_offset;
  const int K_num = 64 / sizeof(DT);
  int align_K = 0;
  _small_stage_count = small_factors[0];
  tw_offset = small_factors[1];

  const int half_butterfly_num = large_butterfly_num / 2 + 1;
  const int large_tw_stride = large_butterfly_num / 2 + 1;
  const int large_in_stride = large_butterfly_num;

  int max_para_num = (half_butterfly_num < small_factors[3])
                         ? half_butterfly_num
                         : small_factors[3];
  int nram_buf_offset = 0;

  FFT_CPX_T<DT> nram_para_load_in_ping = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_num};
  nram_buf_offset += large_radix * max_para_num * 2;  // complex

  FFT_CPX_T<DT> nram_para_load_in_pong = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_num};
  nram_buf_offset += large_radix * max_para_num * 2;  // complex

  FFT_CPX_T<DT> nram_para_load_tw = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_num};
  nram_buf_offset += large_radix * max_para_num * 2;  // complex

  FFT_CPX_T<DT> nram_para_store_ping = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_num};
  nram_buf_offset += large_radix * max_para_num * 2;  // complex

  FFT_CPX_T<DT> nram_para_store_pong = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * max_para_num};
  nram_buf_offset += large_radix * max_para_num * 2;  // complex

  DT *_nram_tw = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * 2;  // complex

  int ld_dft_radix = -1;
  const int max_radix = 64;
  DT *nram_dftmtx = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += max_radix * max_radix * 2;  // complex

  DT *nram_scratch = (DT *)nram_buf + nram_buf_offset;

  // temp overlap with "nram_scratch"
  DT *CPX_MUL_RR = nram_scratch;
  DT *CPX_MUL_RI = &CPX_MUL_RR[large_radix * max_para_num];
  DT *CPX_MUL_IR = &CPX_MUL_RI[large_radix * max_para_num];
  DT *CPX_MUL_II = &CPX_MUL_IR[large_radix * max_para_num];

  nram_buf_offset += large_radix * max_para_num * 4;  // complex

  if (small_twiddles_size) {
    if (load_once_twiddles) {
      __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, SRAM2NRAM);
    } else {
      __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, GDRAM2NRAM);
    }
  }

  // overlap nram_in
  // FFT_CPX_T<DT> nram_transpose_temp;
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
  int sec_count;

  int repeat_num = (t_end - t_start);
  const int odist = nfft << 1;
  const int idist = nfft << 1;
  input += t_start * idist;
  output += t_start * odist;

  const int upper_radix = (large_radix + 1) / 2;
  const int lower_radix = large_radix - upper_radix;

  for (int butterfly_id = 0; butterfly_id < half_butterfly_num;
       butterfly_id += max_para_num) {
    int para_num = (max_para_num > (half_butterfly_num - butterfly_id))
                       ? (half_butterfly_num - butterfly_id)
                       : max_para_num;

    for (sec_count = 0; sec_count < large_section_num; ++sec_count) {
      DT *output_batch = output - 2 * odist;
      DT *input_batch = input;

      for (int repeat_id = 0; repeat_id < repeat_num + 2;
           ++repeat_id, input_batch += idist, output_batch += odist) {
        // pipeline: load-stage
        if (repeat_id < repeat_num) {
          __memcpy_async(
              nram_para_load_in_ping.r,
              input_batch +
                  (sec_count * (large_radix * large_butterfly_num / 2 + 1) +
                   butterfly_id),
              sizeof(DT) * para_num, GDRAM2NRAM, sizeof(DT) * para_num,
              large_in_stride * sizeof(DT), upper_radix - 1);
          __memcpy_async(
              nram_para_load_in_ping.i,
              input_batch + nfft +
                  (sec_count * (large_radix * large_butterfly_num / 2 + 1) +
                   butterfly_id),
              sizeof(DT) * para_num, GDRAM2NRAM, sizeof(DT) * para_num,
              large_in_stride * sizeof(DT), upper_radix - 1);

          __memcpy_async(
              nram_para_load_in_ping.r + upper_radix * para_num,
              input_batch +
                  (sec_count * (large_radix * large_butterfly_num / 2 + 1)) +
                  (large_butterfly_num - butterfly_id - para_num + 1),
              sizeof(DT) * para_num, GDRAM2NRAM, sizeof(DT) * para_num,
              large_in_stride * sizeof(DT), lower_radix - 1);
          __memcpy_async(
              nram_para_load_in_ping.i + upper_radix * para_num,
              input_batch + nfft +
                  (sec_count * (large_radix * large_butterfly_num / 2 + 1)) +
                  (large_butterfly_num - butterfly_id - para_num + 1),
              sizeof(DT) * para_num, GDRAM2NRAM, sizeof(DT) * para_num,
              large_in_stride * sizeof(DT), lower_radix - 1);

          if (repeat_id == 0 && sec_count == 0) {
            if (load_once_twiddles) {
              __memcpy_async(
                  nram_para_load_tw.r, cur_large_twiddles + butterfly_id,
                  sizeof(DT) * para_num, SRAM2NRAM, sizeof(DT) * para_num,
                  large_tw_stride * sizeof(DT), large_radix - 2);
              __memcpy_async(
                  nram_para_load_tw.i,
                  cur_large_twiddles + large_tw_stride * (large_radix - 1) +
                      butterfly_id,
                  sizeof(DT) * para_num, SRAM2NRAM, sizeof(DT) * para_num,
                  large_tw_stride * sizeof(DT), large_radix - 2);

            } else {
              __memcpy_async(
                  nram_para_load_tw.r, cur_large_twiddles + butterfly_id,
                  sizeof(DT) * para_num, GDRAM2NRAM, sizeof(DT) * para_num,
                  large_tw_stride * sizeof(DT), large_radix - 2);
              __memcpy_async(
                  nram_para_load_tw.i,
                  cur_large_twiddles + large_tw_stride * (large_radix - 1) +
                      butterfly_id,
                  sizeof(DT) * para_num, GDRAM2NRAM, sizeof(DT) * para_num,
                  large_tw_stride * sizeof(DT), large_radix - 2);
            }
          }
        }

        // pipeline: store-stage
        if (repeat_id >= 2) {
          // real
          __memcpy_async(
              output_batch + sec_count * half_butterfly_num + butterfly_id,
              nram_para_store_ping.r, para_num * sizeof(DT), NRAM2GDRAM,
              large_out_stride * sizeof(DT), sizeof(DT) * para_num,
              large_radix - 1);
          // imag
          __memcpy_async(output_batch + sec_count * half_butterfly_num +
                             butterfly_id + nfft,
                         nram_para_store_ping.i, para_num * sizeof(DT),
                         NRAM2GDRAM, large_out_stride * sizeof(DT),
                         sizeof(DT) * para_num, large_radix - 1);
        }
        // pipeline: compute-stage

        if (repeat_id >= 1 && repeat_id < repeat_num + 1) {
          DT *nram_in_r = nram_para_load_in_pong.r;
          DT *nram_in_i = nram_para_load_in_pong.i;

          DT *nram_out_r = nram_para_store_pong.r;
          DT *nram_out_i = nram_para_store_pong.i;

          {
            // real
            __bang_rotate180(nram_out_r, nram_in_r + upper_radix * para_num, 1,
                             lower_radix * para_num);
            __memcpy_async(nram_in_r + upper_radix * para_num, nram_out_r,
                           para_num * lower_radix * sizeof(DT), NRAM2NRAM);
            // imag
            __bang_rotate180(nram_out_i, nram_in_i + upper_radix * para_num, 1,
                             lower_radix * para_num);

            __bang_mul_scalar(nram_in_i + upper_radix * para_num, nram_out_i,
                              -1, para_num * lower_radix);
            __sync_compute();
          }

          for (int compute_id = 0; compute_id < para_num;
               compute_id += para_num) {
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

            // para_num = 1
            // in: [radix, butterfly_num]
            // butterfly: [radix, radix] * [radix, butterfly_num]
            // out_butterfly: [radix, butterfly_num]
            // out: [butterfly_num, radix]

            // para_num != 1
            // in: [radix, butterfly_num, para_num] == [large_radix,
            // para_num] butterfly: [radix, radix] * [radix,
            // butterfly_num, para_num] out_butterfly: [radix,
            // butterfly_num, para_num] == [radix, butterfly_num *
            // para_num] out: [butterfly_num, para_num, radix]
            computeGenericButterflyFirststageMat(
                nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                nram_dftmtx, small_section_num * para_num,
                small_section_num * para_num, 1, dir, radix);

            // [radix, small_section_num, para_num] ->
            // [small_section_num, para_num, radix] ->
            // [para_num, small_section_num, radix] ->
            // [small_section_num, radix, para_num] ==
            // [large_radix, para_num]

            small_stage_count--;
            if (small_stage_count == 0) {
              {
                if (nram_out_r == nram_para_store_pong.r) {
                  FFT_SWAP_PTR(nram_para_load_in_pong.r, nram_para_store_pong.r)
                  FFT_SWAP_PTR(nram_para_load_in_pong.i, nram_para_store_pong.i)
                }
                __bang_transpose(nram_para_store_pong.r, nram_out_r, para_num,
                                 large_radix);
                __bang_transpose(nram_para_store_pong.i, nram_out_i, para_num,
                                 large_radix);

                // rotation-large
                __bang_mul(CPX_MUL_RR, nram_para_store_pong.r + para_num,
                           nram_para_load_tw.r, para_num * (large_radix - 1));
                __bang_mul(CPX_MUL_II, nram_para_store_pong.i + para_num,
                           nram_para_load_tw.i, para_num * (large_radix - 1));
                __bang_mul(CPX_MUL_RI, nram_para_store_pong.r + para_num,
                           nram_para_load_tw.i, para_num * (large_radix - 1));
                __bang_mul(CPX_MUL_IR, nram_para_store_pong.i + para_num,
                           nram_para_load_tw.r, para_num * (large_radix - 1));

                __bang_sub(nram_para_store_pong.r + para_num, CPX_MUL_RR,
                           CPX_MUL_II, para_num * (large_radix - 1));
                __bang_add(nram_para_store_pong.i + para_num, CPX_MUL_RI,
                           CPX_MUL_IR, para_num * (large_radix - 1));
              }

              continue;
            }

            FFT_SWAP_PTR(nram_out_r, nram_in_r);
            FFT_SWAP_PTR(nram_out_i, nram_in_i);

            // after first stage: [butterfly_num, para_ldst_num, radix]
            // other in: [para_ldst_num, butterfly_num, radix] ==
            // [para_ldst_num, large_radix]
            TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                                   small_section_num, para_num, radix, DT)

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
                  para_num, small_in_stride, dir, radix);

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
                    break;
                    __sync_move();
                  }

                  if (dft_table[entry].radix == -1) {
                    break;
                  }
                }
              }
              computeGenericButterflyLaststageMat(
                  nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                  nram_dftmtx, nram_tw, small_section_num, small_butterfly_num,
                  para_num, small_in_stride, dir, radix);

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

              {
                if (nram_out_r == nram_para_store_pong.r) {
                  FFT_SWAP_PTR(nram_para_load_in_pong.r, nram_para_store_pong.r)
                  FFT_SWAP_PTR(nram_para_load_in_pong.i, nram_para_store_pong.i)
                }

                __bang_transpose(nram_para_store_pong.r, nram_out_r, para_num,
                                 large_radix);
                __bang_transpose(nram_para_store_pong.i, nram_out_i, para_num,
                                 large_radix);

                // rotation-large
                __bang_mul(CPX_MUL_RR, nram_para_store_pong.r + para_num,
                           nram_para_load_tw.r, para_num * (large_radix - 1));
                __bang_mul(CPX_MUL_II, nram_para_store_pong.i + para_num,
                           nram_para_load_tw.i, para_num * (large_radix - 1));
                __bang_mul(CPX_MUL_RI, nram_para_store_pong.r + para_num,
                           nram_para_load_tw.i, para_num * (large_radix - 1));
                __bang_mul(CPX_MUL_IR, nram_para_store_pong.i + para_num,
                           nram_para_load_tw.r, para_num * (large_radix - 1));

                __bang_sub(nram_para_store_pong.r + para_num, CPX_MUL_RR,
                           CPX_MUL_II, para_num * (large_radix - 1));
                __bang_add(nram_para_store_pong.i + para_num, CPX_MUL_RI,
                           CPX_MUL_IR, para_num * (large_radix - 1));
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

// Compute the large butterfly for the first stage using batch ping-pong
// processing from complex to real (C2R)
template <typename DT>
__mlu_func__ void computeLargeButterflyFirststageBatchPingpongC2R(
    DT *output, DT *input, const int large_radix, const DT *cur_large_twiddles,
    const DT *small_twiddles, const int small_twiddles_size,
    const DT *dft_matrix, const int large_section_num,
    const int large_butterfly_num, const int large_out_stride, void *nram_buf,
    const int *small_factors, const int nfft, const int t_start,
    const int t_end, const int dir, const int last_stage,
    const int load_once_twiddles) {
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;
  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;
  int tw_offset, max_para_num;

  const int K_num = 64 / sizeof(DT);
  int align_K = 0;

  _small_stage_count = small_factors[0];
  tw_offset = small_factors[1];

  const int half_butterfly_num = large_butterfly_num / 2 + 1;
  const int large_tw_stride = large_butterfly_num / 2 + 1;

  max_para_num = (large_butterfly_num < small_factors[3]) ? large_butterfly_num
                                                          : small_factors[3];

  // load compute store
  // (0)                              load 0 ping sync()
  // (1)              compute 0 ping  load 1 pong sync()
  // (2) store 0      compute 1 pong  load 2 ping sync()
  // (3) store 1      compute 2   load 3  sync()

  // assign nram space
  int nram_buf_offset = 0;

  DT *nram_para_load_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_num * 2;  // complex

  DT *nram_para_load_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_num * 2;  // complex

  DT *nram_para_store_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_num * 2;  // complex

  DT *nram_para_store_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_num * 2;  // complex

  DT *nram_para_load_tw = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_num * 2;  // complex

  DT *_nram_tw = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * 2;  // complex

  int ld_dft_radix[2] = {-1, -1};
  const int max_radix = 64;
  DT *nram_dftmtx[2] = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + max_radix * max_radix * 2};
  nram_buf_offset += max_radix * max_radix * 4;  // complex

  DT *nram_scratch = (DT *)nram_buf + nram_buf_offset;

  DT *CPX_MUL_RR = nram_scratch;
  DT *CPX_MUL_RI = &CPX_MUL_RR[large_radix * max_para_num];
  DT *CPX_MUL_IR = &CPX_MUL_RI[large_radix * max_para_num];
  DT *CPX_MUL_II = &CPX_MUL_IR[large_radix * max_para_num];

  if (small_twiddles_size) {
    if (load_once_twiddles) {
      __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, SRAM2NRAM);
    } else {
      __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, GDRAM2NRAM);
    }
  }

  int repeat_num = (t_end - t_start);

  const int odist = (last_stage) ? nfft : nfft << 1;
  const int idist = (nfft / 2 + 1) << 1;
  input += t_start * idist;
  output += t_start * odist;

  const int upper_radix = (large_radix + 1) / 2;
  const int lower_radix = large_radix - upper_radix;

  for (int butterfly_id = 0; butterfly_id < half_butterfly_num;
       butterfly_id += max_para_num) {
    DT *output_batch = output;
    DT *input_batch = input;
    int para_num = (max_para_num > (half_butterfly_num - butterfly_id))
                       ? (half_butterfly_num - butterfly_id)
                       : max_para_num;

    for (int repeat_id = 0; repeat_id < repeat_num + 2;
         ++repeat_id, input_batch += idist, output_batch += odist) {
      // pipeline: load-stage
      if (repeat_id < repeat_num) {
        // [para_num][radix/2+1] -> [radix/2+1][para_num]

        __memcpy_async(nram_para_load_ping, input_batch + butterfly_id * 2,
                       sizeof(DT) * 2 * para_num, GDRAM2NRAM,
                       sizeof(DT) * 2 * para_num,
                       large_butterfly_num * sizeof(DT) * 2, upper_radix - 1);
        __memcpy_async(
            nram_para_load_ping + upper_radix * para_num * 2,
            input_batch +
                (large_butterfly_num - butterfly_id - para_num + 1) * 2,
            sizeof(DT) * 2 * para_num, GDRAM2NRAM, sizeof(DT) * 2 * para_num,
            large_butterfly_num * sizeof(DT) * 2, lower_radix - 1);

        if (!last_stage && repeat_id == 0) {
          if (load_once_twiddles) {
            __memcpy_async(nram_para_load_tw, cur_large_twiddles + butterfly_id,
                           sizeof(DT) * para_num, SRAM2NRAM,
                           sizeof(DT) * para_num, large_tw_stride * sizeof(DT),
                           large_radix - 2);
            __memcpy_async(
                nram_para_load_tw + large_radix * max_para_num,
                cur_large_twiddles + large_tw_stride * (large_radix - 1) +
                    butterfly_id,
                sizeof(DT) * para_num, SRAM2NRAM, sizeof(DT) * para_num,
                large_tw_stride * sizeof(DT), large_radix - 2);
          } else {
            __memcpy_async(nram_para_load_tw, cur_large_twiddles + butterfly_id,
                           sizeof(DT) * para_num, GDRAM2NRAM,
                           sizeof(DT) * para_num, large_tw_stride * sizeof(DT),
                           large_radix - 2);
            __memcpy_async(
                nram_para_load_tw + large_radix * max_para_num,
                cur_large_twiddles + large_tw_stride * (large_radix - 1) +
                    butterfly_id,
                sizeof(DT) * para_num, GDRAM2NRAM, sizeof(DT) * para_num,
                large_tw_stride * sizeof(DT), large_radix - 2);
          }
        }
      }

      // pipeline: store-stage
      if (repeat_id >= 2) {
        if (last_stage) {
          // store only real part
          __memcpy_async(output_batch - 2 * odist, nram_para_store_ping,
                         sizeof(DT) * large_radix, NRAM2GDRAM);

        } else {
          // scatter-store
          __memcpy_async(output_batch - 2 * odist + butterfly_id,
                         nram_para_store_ping, sizeof(DT) * para_num,
                         NRAM2GDRAM, sizeof(DT) * large_out_stride,
                         sizeof(DT) * para_num, large_radix - 1);
          __memcpy_async(output_batch - 2 * odist + butterfly_id + nfft,
                         nram_para_store_ping + large_radix * para_num,
                         sizeof(DT) * para_num, NRAM2GDRAM,
                         sizeof(DT) * large_out_stride, sizeof(DT) * para_num,
                         large_radix - 1);
        }
      }

      // pipeline: compute-stage
      if (repeat_id >= 1 && repeat_id < repeat_num + 1) {
        DT *nram_in_r = nram_para_store_pong;
        DT *nram_in_i = nram_para_store_pong + large_radix * para_num;

        DT *nram_out_r = nram_para_load_pong;
        DT *nram_out_i = nram_para_load_pong + large_radix * para_num;
        {
          __bang_transpose(nram_in_r, nram_para_load_pong,
                           large_radix * para_num, 2);
          // real
          __bang_rotate180(nram_out_r, nram_in_r + upper_radix * para_num, 1,
                           lower_radix * para_num);
          __memcpy_async(nram_in_r + upper_radix * para_num, nram_out_r,
                         para_num * lower_radix * sizeof(DT), NRAM2NRAM);
          // imag
          __bang_rotate180(
              nram_out_i,
              nram_in_r + upper_radix * para_num + para_num * large_radix, 1,
              lower_radix * para_num);

          __bang_mul_scalar(
              nram_in_r + upper_radix * para_num + para_num * large_radix,
              nram_out_i, -1, para_num * lower_radix);
        }

        {
          // load real & imag

          radix = small_factors[4];
          small_section_num = small_factors[5];
          small_in_stride = small_factors[7];
          small_stage_count = _small_stage_count;

          // first stage
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
          __sync_compute();

          computeGenericButterflyFirststageMat(
              nram_out_r, nram_out_i, nram_in_r,
              nram_in_r + large_radix * para_num, nram_scratch, nram_dftmtx[0],
              small_section_num * para_num, small_section_num * para_num, 1,
              dir, radix);

          // [radix, small_section_num, para_ldst_num] ->
          // [small_section_num, para_ldst_num, radix] -> [para_ldst_num,
          // small_section_num, radix]

          small_stage_count--;
          if (small_stage_count == 0) {
            // nram to gdram

            if (last_stage) {
              //  [2, para_ldst_num, large_radix] -> [para_ldst_num,
              //  large_radix, 2]

              if (nram_out_r != nram_para_store_pong) {
                __memcpy_async(nram_para_store_pong, nram_out_r,
                               para_num * large_radix * sizeof(DT), NRAM2NRAM);
              }
            } else {
              //  [2, para_num, large_radix] -> [2, para_num,
              //  large_radix]

              if (nram_out_r == nram_para_store_pong) {
                FFT_SWAP_PTR(nram_para_load_pong, nram_para_store_pong)
              }
              __bang_transpose(nram_para_store_pong, nram_out_r, para_num,
                               large_radix);
              __bang_transpose(nram_para_store_pong + para_num * large_radix,
                               nram_out_i, para_num, large_radix);
              __bang_mul(CPX_MUL_RR, nram_para_store_pong + para_num,
                         nram_para_load_tw, para_num * (large_radix - 1));
              __bang_mul(
                  CPX_MUL_II,
                  nram_para_store_pong + para_num * large_radix + para_num,
                  nram_para_load_tw + large_radix * max_para_num,
                  para_num * (large_radix - 1));
              __bang_mul(CPX_MUL_RI, nram_para_store_pong + para_num,
                         nram_para_load_tw + large_radix * max_para_num,
                         para_num * (large_radix - 1));
              __bang_mul(
                  CPX_MUL_IR,
                  nram_para_store_pong + para_num * large_radix + para_num,
                  nram_para_load_tw, para_num * (large_radix - 1));

              __bang_sub(nram_para_store_pong + para_num, CPX_MUL_RR,
                         CPX_MUL_II, para_num * (large_radix - 1));
              __bang_add(
                  nram_para_store_pong + para_num * large_radix + para_num,
                  CPX_MUL_RI, CPX_MUL_IR, para_num * (large_radix - 1));
            }

          } else {
            // [small_section_num, para_ldst_num, radix] -> [para_ldst_num,
            // small_section_num, radix]

            FFT_SWAP_PTR(nram_out_r, nram_in_r);
            FFT_SWAP_PTR(nram_out_i, nram_in_i);

            TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                                   small_section_num, para_num, radix, DT)

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
            }  // for (stage_count)

            // last stage
            {
              FFT_SWAP_PTR(nram_out_r, nram_in_r);
              FFT_SWAP_PTR(nram_out_i, nram_in_i);

              // copy GDRAM2SRAM

              // update parameter
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
                    nram_para_store_pong, nram_out_i, nram_in_r, nram_in_i,
                    nram_scratch, nram_dftmtx[0], nram_tw, small_section_num,
                    small_butterfly_num, para_num, small_in_stride, dir, radix);

              } else {
                computeGenericButterflyLaststageMat(
                    nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                    nram_dftmtx[0], nram_tw, small_section_num,
                    small_butterfly_num, para_num, small_in_stride, dir, radix);
              }

              if (!last_stage) {
                if (nram_out_r == nram_para_store_pong) {
                  FFT_SWAP_PTR(nram_para_store_pong, nram_para_load_pong);
                }
                __bang_transpose(nram_para_store_pong, nram_out_r, para_num,
                                 large_radix);
                __bang_transpose(nram_para_store_pong + para_num * large_radix,
                                 nram_out_i, para_num, large_radix);

                __bang_mul(CPX_MUL_RR, nram_para_store_pong + para_num,
                           nram_para_load_tw, para_num * (large_radix - 1));
                __bang_mul(
                    CPX_MUL_II,
                    nram_para_store_pong + para_num * large_radix + para_num,
                    nram_para_load_tw + large_radix * max_para_num,
                    para_num * (large_radix - 1));
                __bang_mul(CPX_MUL_RI, nram_para_store_pong + para_num,
                           nram_para_load_tw + large_radix * max_para_num,
                           para_num * (large_radix - 1));
                __bang_mul(
                    CPX_MUL_IR,
                    nram_para_store_pong + para_num * large_radix + para_num,
                    nram_para_load_tw, para_num * (large_radix - 1));

                __bang_sub(nram_para_store_pong + para_num, CPX_MUL_RR,
                           CPX_MUL_II, para_num * (large_radix - 1));
                __bang_add(
                    nram_para_store_pong + para_num * large_radix + para_num,
                    CPX_MUL_RI, CPX_MUL_IR, para_num * (large_radix - 1));
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
