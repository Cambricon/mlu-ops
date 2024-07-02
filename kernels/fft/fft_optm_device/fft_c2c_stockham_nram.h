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
#include "kernels/fft/fft_optm_device/fft_vector_butterfly.h"

template <typename DT>
__mlu_func__ void computeLargeButterflyFirststage(
    DT *output, DT *input, const int large_radix, int large_in_stride,
    int section_num, const DT *twiddles, const DT *dft_matrix, void *nram_buf,
    const int *small_factors, int dir, int nfft, int last_stage) {
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;

  // network info
  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;
  int tw_offset;

  const int K_num = 64 / sizeof(DT);
  int align_K = 0;

  _small_stage_count = small_factors[0];
  // large_radix = small_factors[1];
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

  // const int max_para_ldst_num = 1;
  int max_para_ldst_num = (4096 + large_radix - 1) / large_radix;
  max_para_ldst_num =
      (section_num < max_para_ldst_num) ? section_num : max_para_ldst_num;

  const DT *small_twiddles = twiddles + tw_offset * 2;  // complex

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

  DT *nram_para_load_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *nram_para_load_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *nram_para_store_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *nram_para_store_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  // transpose space: [radix, 2 * parrallel] -> [parrallel * 2, radix]
  // DT *nram_transpose_load = (DT *)nram_buf + nram_buf_offset;
  // nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

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

  DT *_nram_tw = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * 2;  // complex

  int ld_dft_radix = -1;
  const int max_radix = 64;
  DT *nram_dftmtx = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += max_radix * max_radix * 2;  // complex

  // nram space used:
  // sizeof(DT) * 2 * large_radix * (max_para_ldst_num * 6 + 1) + sizeof(DT) * 2
  // * (max_radix * max_radix)
  // + sizeof(DT) * 2 * large_radix * max_para_ldst_num * 4
  DT *nram_scratch = (DT *)nram_buf + nram_buf_offset;

  __memcpy_async(_nram_tw, small_twiddles, large_radix * sizeof(DT) * 2,
                 SRAM2NRAM);

  // ceil
  int repeat_num = (section_num + max_para_ldst_num - 1) / max_para_ldst_num;

  for (int repeat_id = 0; repeat_id < repeat_num + 2; ++repeat_id) {
    // pipeline: load-stage
    if (repeat_id < repeat_num) {
      // MLULOG("pipeline: load-stage.\n");
      int i = max_para_ldst_num * repeat_id;
      DT *nram_para_load =
          (repeat_id % 2 == 0) ? nram_para_load_ping : nram_para_load_pong;

      // DT *nram_dftmtx =
      //     (repeat_id % 2 == 0) ? nram_dftmtx_ping : nram_dftmtx_pong;
      int para_load_num = (max_para_ldst_num > (section_num - i))
                              ? (section_num - i)
                              : max_para_ldst_num;
      if (section_num == 1) {
        __memcpy_async(nram_para_load, input, sizeof(DT) * 2 * large_radix,
                       GDRAM2NRAM);
      } else {
        // gather load
        // 2d memcpy
        // 0 1 2 3 4 ... 1023
        // GDRAM -> NRAM
        // 8bytes radix-1024
        // 64bytes

        __memcpy_async(nram_para_load, input + i * 2,
                       sizeof(DT) * 2 * para_load_num, GDRAM2NRAM,
                       sizeof(DT) * 2 * para_load_num,
                       large_in_stride * sizeof(DT) * 2, large_radix - 1);
      }
    }

    // pipeline: store-stage
    if (repeat_id >= 2) {
      // MLULOG("pipeline: store-stage.\n");
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
          // scatter-store
          __memcpy_async(output + i * large_radix * 2, nram_para_store,
                         sizeof(DT) * 2 * para_store_num * large_radix,
                         NRAM2GDRAM);
        }
      } else {
        // real
        __memcpy_async(output + i * large_radix, nram_para_store,
                       para_store_num * large_radix * sizeof(DT), NRAM2GDRAM);
        // imag
        __memcpy_async(output + i * large_radix + nfft,
                       nram_para_store + max_para_ldst_num * large_radix,
                       para_store_num * large_radix * sizeof(DT), NRAM2GDRAM);
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
      // // [large_radix, para_ldst_num, 2] -> [para_ldst_num, 2, large_radix]
      // __bang_transpose(nram_transpose_load, nram_para_load, large_radix,
      //                  2 * para_ldst_num);

      // [large_radix, para_ldst_num, 2] -> [2, para_ldst_num, large_radix]
      // overlap nram_out_r
      // DT *nram_transpose_load = nram_out_r;
      // __bang_transpose(nram_transpose_load, nram_para_load,
      //                  large_radix * para_ldst_num, 2);
      // // [large_radix, para_ldst_num] -> [para_ldst_num, large_radix]
      // __bang_transpose(nram_in_r, nram_transpose_load, large_radix,
      //                  para_ldst_num);
      // __bang_transpose(nram_in_i,
      //                  nram_transpose_load + large_radix * para_ldst_num,
      //                  large_radix, para_ldst_num);

      // DT *nram_transpose_load = nram_in_r;
      __bang_transpose(nram_in_r, nram_para_load, large_radix * para_ldst_num,
                       2);
      // [large_radix, para_ldst_num] -> [para_ldst_num, large_radix]
      // __bang_transpose(nram_in_r, nram_transpose_load, large_radix,
      //                  para_ldst_num);
      // __bang_transpose(nram_in_i,
      //                  nram_transpose_load + large_radix * para_ldst_num,
      //                  large_radix, para_ldst_num);

      for (int compute_id = 0; compute_id < para_ldst_num;
           compute_id += para_ldst_num) {
        // load real & imag

        radix = small_factors[4];
        small_section_num = small_factors[5];
        small_in_stride = small_factors[7];
        small_stage_count = _small_stage_count;

        // __memcpy(nram_in_r,
        //         nram_transpose_load + compute_id * large_radix * 2,
        //          large_radix * sizeof(DT) * 2, NRAM2NRAM);

        // first stage

        if (ld_dft_radix != radix) {
          ld_dft_radix = radix;
          for (int entry = 0;; entry++) {
            if (dft_table[entry].radix == ld_dft_radix) {
              align_K = K_num * ((radix + K_num - 1) / K_num);
              __memcpy(nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                       sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
            }

            if (dft_table[entry].radix == -1) {
              break;
            }
          }
        }

        switch (radix) {
          default:
            MLULOG("computeGenericButterflyFirststageMat: %d.\n", radix);
            computeGenericButterflyFirststageMat(
                nram_out_r, nram_out_i, nram_in_r,
                nram_in_r + large_radix * para_ldst_num, nram_scratch,
                nram_dftmtx, small_section_num * para_ldst_num,
                small_section_num * para_ldst_num, 1, dir, radix);
            break;
        }

        // [radix, small_section_num, para_ldst_num] ->
        // [small_section_num, para_ldst_num, radix] -> [para_ldst_num,
        // small_section_num, radix]

        small_stage_count--;
        if (small_stage_count == 0) {
          // nram to gdram

          if (last_stage) {
            //  [2, para_ldst_num, large_radix] -> [para_ldst_num, large_radix,
            //  2]
            // DT* nram_transpose_store = nram_in_r;

            __bang_transpose(nram_para_store, nram_out_r, 2,
                             max_para_ldst_num * large_radix);
          } else {
            //  [2, para_ldst_num, large_radix] -> [2, para_ldst_num,
            //  large_radix]
            // TODO(zrg): redundant move
            __memcpy(nram_para_store, nram_out_r,
                     para_ldst_num * large_radix * sizeof(DT), NRAM2NRAM);
            __memcpy(nram_para_store + max_para_ldst_num * large_radix,
                     nram_out_i, para_ldst_num * large_radix * sizeof(DT),
                     NRAM2NRAM);
          }

          continue;
        }

        // [small_section_num, para_ldst_num, radix] -> [para_ldst_num,
        // small_section_num, radix]

        FFT_SWAP_PTR(nram_out_r, nram_in_r);
        FFT_SWAP_PTR(nram_out_i, nram_in_i);

        TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                               small_section_num, para_ldst_num, radix, DT)

        value_mul = 8;
        // DT *sram_tw = (DT *)sram_buffer;
        DT *nram_tw = _nram_tw;

        for (; small_stage_count > 1; small_stage_count--) {
          FFT_SWAP_PTR(nram_out_r, nram_in_r);
          FFT_SWAP_PTR(nram_out_i, nram_in_i);

          // value_mul = (_small_stage_count - small_stage_count + 1) << 2;

          // // update parameter
          radix = small_factors[value_mul++];
          small_section_num = small_factors[value_mul++];
          small_butterfly_num = small_factors[value_mul++];
          small_in_stride = small_factors[value_mul++];
          // value_mul += 4;
          // copy GDRAM2SRAM

          // if (compute_id == 0 && repeat_id == 1 && 0) {
          //   __memcpy(nram_tw, small_twiddles,
          //            small_butterfly_num * (radix - 1) * sizeof(DT) * 2,
          //            GDRAM2NRAM);
          //   small_twiddles += small_butterfly_num * (radix - 1) * 2;
          // }

          if (ld_dft_radix != radix) {
            ld_dft_radix = radix;
            for (int entry = 0;; entry++) {
              if (dft_table[entry].radix == ld_dft_radix) {
                align_K = K_num * ((radix + K_num - 1) / K_num);
                __memcpy(nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                         sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                break;
              }

              if (dft_table[entry].radix == -1) {
                break;
              }
            }
          }

          switch (radix) {
            case 2:
              // computeRadix2ButterflyOtherstages(Fout, Fin, section_num,
              // section_num, 1, dir);
              break;
            case 3:
              computeRadix3ButterflyOtherstages(
                  nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                  nram_tw, small_section_num, small_butterfly_num,
                  small_in_stride, dir);
              break;

            default:
              // computeGenericButterflyOtherstages(Fout, buffer, twiddles,
              // radix, section_num, butterfly_num, in_stride, 0, dir);
              MLULOG("computeGenericButterflyOtherstagesMat: %d.\n", radix);
              computeGenericButterflyOtherstagesMat(
                  nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                  nram_dftmtx, nram_tw, small_section_num, small_butterfly_num,
                  para_ldst_num, small_in_stride, dir, radix);
              break;
          }

          nram_tw += small_butterfly_num * (radix - 1) * 2;
        }  // for (stage_count)

        //           for (int j = 0; j < large_radix; j++) {
        //   MLULOG("output i: (%f, %f).\n", nram_out_r[j], nram_out_i[j]);
        // }

        // MLULOG("butterfly id: %d\n", i);
        // last stage
        {
          FFT_SWAP_PTR(nram_out_r, nram_in_r);
          FFT_SWAP_PTR(nram_out_i, nram_in_i);

          // copy GDRAM2SRAM

          // update parameter
          // value_mul = _small_stage_count << 2;
          radix = small_factors[value_mul++];
          small_section_num = small_factors[value_mul++];
          small_butterfly_num = small_factors[value_mul++];
          small_in_stride = small_factors[value_mul];

          // if (compute_id == 0 && repeat_id == 1 && 0) {
          //   __memcpy(nram_tw, small_twiddles,
          //            small_butterfly_num * (radix - 1) * sizeof(DT) * 2,
          //            GDRAM2NRAM);
          // }
          if (ld_dft_radix != radix) {
            ld_dft_radix = radix;
            for (int entry = 0;; entry++) {
              if (dft_table[entry].radix == ld_dft_radix) {
                align_K = K_num * ((radix + K_num - 1) / K_num);
                __memcpy(nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                         sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                break;
              }

              if (dft_table[entry].radix == -1) {
                break;
              }
            }
          }

          switch (radix) {
            case 2:
              break;

            default:
              MLULOG("computeGenericButterflyLaststageMat: %d.\n", radix);
              computeGenericButterflyLaststageMat(
                  nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                  nram_dftmtx, nram_tw, small_section_num, small_butterfly_num,
                  para_ldst_num, small_in_stride, dir, radix);
              MLULOG("computeGenericButterflyLaststageMat: %d End.\n", radix);
              break;
          }

          if (last_stage) {
            //  [2, para_ldst_num, large_radix] -> [para_ldst_num, large_radix,
            //  2]
            // DT* nram_transpose_store = nram_in_r;

            __bang_transpose(nram_para_store, nram_out_r, 2,
                             max_para_ldst_num * large_radix);
          } else {
            //  [2, para_ldst_num, large_radix] -> [2, para_ldst_num,
            //  large_radix]
            // TODO(zrg): redundant move
            __memcpy(nram_para_store, nram_out_r,
                     para_ldst_num * large_radix * sizeof(DT), NRAM2NRAM);
            __memcpy(nram_para_store + max_para_ldst_num * large_radix,
                     nram_out_i, para_ldst_num * large_radix * sizeof(DT),
                     NRAM2NRAM);
          }

          // if (last_stage) {
          //   // MLULOG("last_stage. \n");

          //   // __memcpy(nram_transpose_temp + (compute_id * 2) * large_radix,
          //   //          nram_out_r, large_radix * sizeof(DT), NRAM2NRAM);
          //   // __memcpy(nram_transpose_temp
          //   // + (compute_id * 2 + 1) * large_radix,
          //   //          nram_out_i, large_radix * sizeof(DT), NRAM2NRAM);

          //   __memcpy(nram_transpose_temp + (compute_id * 2) * large_radix,
          //            nram_out_r, large_radix * sizeof(DT) * 2, NRAM2NRAM);
          //   __bang_transpose(
          //       nram_para_store + (compute_id * 2) * large_radix,
          //       nram_transpose_temp + (compute_id * 2) * large_radix, 2,
          //       large_radix);

          // } else {
          //   // MLULOG("not last_stage. \n");
          //   __memcpy(nram_para_store + compute_id * large_radix, nram_out_r,
          //            large_radix * sizeof(DT), NRAM2NRAM);
          //   __memcpy(nram_para_store +
          //                (compute_id + max_para_ldst_num) * large_radix,
          //            nram_out_i, large_radix * sizeof(DT), NRAM2NRAM);
          // }
          // MLULOG("last_stage. \n");
        }
      }
    }

    __sync();
  }
}

template <typename DT>
__mlu_func__ void computeLargeButterflyFirststageBatchPingpong(
    DT *output, DT *input, const int large_radix, int large_in_stride,
    int section_num, const DT *small_twiddles, const int small_twiddles_size,
    const DT *dft_matrix, void *nram_buf, const int *small_factors, int dir,
    int nfft, int last_stage, const int t_start, const int t_end) {
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;
  // DT *input_batch = input + t * (nfft << 1);
  // DT *output_batch = output + t * (nfft << 1);
  // network info
  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;
  int tw_offset, max_para_ldst_num;

  const int K_num = 64 / sizeof(DT);
  int align_K = 0;

  _small_stage_count = small_factors[0];
  // large_radix = small_factors[1];
  tw_offset = small_factors[1];
  // printf("tw_offset: %d, large_radix: %d\n", tw_offset, large_radix);

  max_para_ldst_num =
      (section_num < small_factors[3]) ? section_num : small_factors[3];
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

  // const int max_para_ldst_num = (5900 + large_radix - 1) / large_radix;

  // unsigned int max_para_ldst_num =
  //     ((7232) / large_radix > 0) ? (7232) / large_radix : 1;
  // max_para_ldst_num =
  //     (section_num < max_para_ldst_num) ? section_num : max_para_ldst_num;

  // max_para_ldst_num = ((7232) / large_radix > 0) ? (7232) / large_radix : 1;
  // const DT *small_twiddles = twiddles + tw_offset * 2;  // complex

  // assign nram space
  int nram_buf_offset = 0;

  DT *nram_para_load_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *nram_para_load_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *nram_para_store_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *nram_para_store_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  // nram_buf_offset += large_radix * max_para_ldst_num;

  // transpose space: [radix, 2 * parrallel] -> [parrallel * 2, radix]
  // DT *nram_transpose_load = (DT *)nram_buf + nram_buf_offset;
  // nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

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

  DT *_nram_tw = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * 2;  // complex

  int ld_dft_radix[2] = {-1, -1};
  const int max_radix = 64;
  DT *nram_dftmtx[2] = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + max_radix * max_radix * 2};
  nram_buf_offset += max_radix * max_radix * 4;  // complex

  // nram space used:
  // sizeof(DT) * 2 * large_radix * (max_para_ldst_num * 6 + 1) + sizeof(DT) * 2
  // * (max_radix * max_radix)
  // + sizeof(DT) * 2 * large_radix * max_para_ldst_num * 4
  DT *nram_scratch = (DT *)nram_buf + nram_buf_offset;

  // DT *nram_temp_r = (DT *)nram_buf + nram_buf_offset;
  // nram_buf_offset += large_radix * max_para_ldst_num;

  // DT *nram_temp_i = (DT *)nram_buf + nram_buf_offset;
  // nram_buf_offset += large_radix * max_para_ldst_num;

  if (small_twiddles_size) {
    __memcpy_async(_nram_tw, small_twiddles, small_twiddles_size, SRAM2NRAM);
  }

  // ceil
  // int repeat_num = (section_num + max_para_ldst_num - 1) / max_para_ldst_num;
  // int repeat_num = ((t_end - t_start) + max_para_ldst_num - 1) /
  // max_para_ldst_num;
  int repeat_num = (t_end - t_start);

  // section_num loop
  input += t_start * (nfft << 1);
  output += t_start * (nfft << 1);

  for (int sec_id = 0; sec_id < section_num; sec_id += max_para_ldst_num) {
    // input + (t_start + repeat_id) * (nfft << 1)
    DT *output_batch = output;
    DT *input_batch = input;
    int para_num = (max_para_ldst_num > (section_num - sec_id))
                       ? (section_num - sec_id)
                       : max_para_ldst_num;

    for (int repeat_id = 0; repeat_id < repeat_num + 2;
         ++repeat_id, input_batch += (nfft << 1), output_batch += (nfft << 1)) {
      // pipeline: load-stage
      if (repeat_id < repeat_num) {
        if (section_num == 1) {
          __memcpy_async(nram_para_load_ping, input_batch,
                         sizeof(DT) * 2 * large_radix, GDRAM2NRAM);
        } else {
          // gather load
          // 2d memcpy
          // 0 1 2 3 4 ... 1023
          // GDRAM -> NRAM
          // 8bytes radix-1024
          // 64bytes

          __memcpy_async(nram_para_load_ping, input_batch + sec_id * 2,
                         sizeof(DT) * 2 * para_num, GDRAM2NRAM,
                         sizeof(DT) * 2 * para_num,
                         large_in_stride * sizeof(DT) * 2, large_radix - 1);
        }
      }

      // pipeline: store-stage
      if (repeat_id >= 2) {
        if (last_stage) {
          if (section_num == 1) {
            __memcpy_async(output_batch - (nfft << 2), nram_para_store_ping,
                           sizeof(DT) * 2 * large_radix, NRAM2GDRAM);
          } else {
            // scatter-store
            __memcpy_async(
                output_batch - (nfft << 2) + sec_id * large_radix * 2,
                nram_para_store_ping, sizeof(DT) * 2 * para_num * large_radix,
                NRAM2GDRAM);
          }
        } else {
          // real
          __memcpy_async(output_batch - (nfft << 2) + sec_id * large_radix,
                         nram_para_store_ping,
                         para_num * large_radix * sizeof(DT), NRAM2GDRAM);
          // imag
          __memcpy_async(
              output_batch - (nfft << 2) + sec_id * large_radix + nfft,
              nram_para_store_ping + max_para_ldst_num * large_radix,
              para_num * large_radix * sizeof(DT), NRAM2GDRAM);
        }
      }

      // pipeline: compute-stage

      if (repeat_id >= 1 && repeat_id < repeat_num + 1) {
        // int i = max_para_ldst_num * (repeat_id - 1);

        // DT *nram_para_load = nram_para_load_pong;
        // DT *nram_para_store = nram_para_store_pong;

        // int para_ldst_num = (max_para_ldst_num > (section_num - i))
        //                         ? (section_num - i)
        //                         : max_para_ldst_num;
        // // [large_radix, para_ldst_num, 2] -> [para_ldst_num, 2, large_radix]
        // __bang_transpose(nram_transpose_load, nram_para_load, large_radix,
        //                  2 * para_ldst_num);

        // [large_radix, para_ldst_num, 2] -> [2, para_ldst_num, large_radix]
        // overlap nram_out_r
        // DT *nram_transpose_load = nram_out_r;
        // __bang_transpose(nram_transpose_load, nram_para_load,
        //                  large_radix * para_ldst_num, 2);
        // // [large_radix, para_ldst_num] -> [para_ldst_num, large_radix]
        // __bang_transpose(nram_in_r, nram_transpose_load, large_radix,
        //                  para_ldst_num);
        // __bang_transpose(nram_in_i,
        //                  nram_transpose_load + large_radix * para_ldst_num,
        //                  large_radix, para_ldst_num);

        // DT *nram_transpose_load = nram_in_r;
        // nram_in_r = nram_para_store_pong;
        // nram_in_r = nram_para_load_pong;

        DT *nram_in_r = nram_para_store_pong;
        DT *nram_in_i = nram_para_store_pong + large_radix * max_para_ldst_num;

        DT *nram_out_r = nram_para_load_pong;
        DT *nram_out_i = nram_para_load_pong + large_radix * max_para_ldst_num;

        __bang_transpose(nram_in_r, nram_para_load_pong, large_radix * para_num,
                         2);
        // [large_radix, para_ldst_num] -> [para_ldst_num, large_radix]
        // __bang_transpose(nram_in_r, nram_transpose_load, large_radix,
        //                  para_ldst_num);
        // __bang_transpose(nram_in_i,
        //                  nram_transpose_load + large_radix * para_ldst_num,
        //                  large_radix, para_ldst_num);

        for (int compute_id = 0; compute_id < para_num;
             compute_id += para_num) {
          // load real & imag

          radix = small_factors[4];
          small_section_num = small_factors[5];
          small_in_stride = small_factors[7];
          small_stage_count = _small_stage_count;

          // __memcpy(nram_in_r,
          //         nram_transpose_load + compute_id * large_radix * 2,
          //          large_radix * sizeof(DT) * 2, NRAM2NRAM);

          // first stage
          if (ld_dft_radix[0] != radix && ld_dft_radix[1] != radix) {
            ld_dft_radix[1] = ld_dft_radix[0];
            FFT_SWAP_PTR(nram_dftmtx[0], nram_dftmtx[1]);
            ld_dft_radix[0] = radix;
            for (int entry = 0;; entry++) {
              if (dft_table[entry].radix == ld_dft_radix[0]) {
                align_K = K_num * ((radix + K_num - 1) / K_num);
                __memcpy(nram_dftmtx[0],
                         &dft_matrix[dft_table[entry].offset * 2],
                         sizeof(DT) * 2 * radix * align_K, SRAM2NRAM);
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

          switch (radix) {
              // case 4:
              //   computeRadix4ButterflyFirststage(
              //       nram_out_r, nram_out_i, nram_in_r, nram_in_i,
              //       nram_scratch, small_section_num * para_num,
              //       small_section_num * para_num, 1, dir);
              //   break;

            default:
              MLULOG("computeGenericButterflyFirststageMat: %d.\n", radix);
              computeGenericButterflyFirststageMat(
                  nram_out_r, nram_out_i, nram_in_r,
                  nram_in_r + large_radix * para_num, nram_scratch,
                  nram_dftmtx[0], small_section_num * para_num,
                  small_section_num * para_num, 1, dir, radix);
              break;
          }

          // [radix, small_section_num, para_ldst_num] ->
          // [small_section_num, para_ldst_num, radix] -> [para_ldst_num,
          // small_section_num, radix]

          small_stage_count--;
          if (small_stage_count == 0) {
            // nram to gdram

            if (last_stage) {
              //  [2, para_ldst_num, large_radix] -> [para_ldst_num,
              //  large_radix, 2]
              // DT* nram_transpose_store = nram_in_r;
              if (nram_out_r == nram_para_store_pong) {
                FFT_SWAP_PTR(nram_para_load_pong, nram_para_store_pong)
              }
              __bang_transpose(nram_para_store_pong, nram_out_r, 2,
                               max_para_ldst_num * large_radix);
            } else {
              //  [2, para_ldst_num, large_radix] -> [2, para_ldst_num,
              //  large_radix]
              // TODO(zrg): redundant move
              __memcpy(nram_para_store_pong, nram_out_r,
                       para_num * large_radix * sizeof(DT), NRAM2NRAM);
              __memcpy(nram_para_store_pong + max_para_ldst_num * large_radix,
                       nram_out_i, para_num * large_radix * sizeof(DT),
                       NRAM2NRAM);
            }

            continue;
          }

          // [small_section_num, para_ldst_num, radix] -> [para_ldst_num,
          // small_section_num, radix]

          FFT_SWAP_PTR(nram_out_r, nram_in_r);
          FFT_SWAP_PTR(nram_out_i, nram_in_i);

          // DT *trans_r = (DT *)nram_buf + nram_buf_offset;
          // nram_buf_offset += large_radix * max_para_ldst_num;

          // DT *nram_out_i = (DT *)nram_buf + nram_buf_offset;
          // nram_buf_offset += large_radix * max_para_ldst_num;

          TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                                 small_section_num, para_num, radix, DT)

          value_mul = 8;
          // DT *sram_tw = (DT *)sram_buffer;
          DT *nram_tw = _nram_tw;

          for (; small_stage_count > 1; small_stage_count--) {
            FFT_SWAP_PTR(nram_out_r, nram_in_r);
            FFT_SWAP_PTR(nram_out_i, nram_in_i);

            // value_mul = (_small_stage_count - small_stage_count + 1) << 2;

            // // update parameter
            radix = small_factors[value_mul++];
            small_section_num = small_factors[value_mul++];
            small_butterfly_num = small_factors[value_mul++];
            small_in_stride = small_factors[value_mul++];
            // value_mul += 4;
            // copy GDRAM2SRAM

            // if (compute_id == 0 && repeat_id == 1 && 0) {
            //   __memcpy(nram_tw, small_twiddles,
            //            small_butterfly_num * (radix - 1) * sizeof(DT) * 2,
            //            GDRAM2NRAM);
            //   small_twiddles += small_butterfly_num * (radix - 1) * 2;
            // }

            if (ld_dft_radix[0] != radix && ld_dft_radix[1] != radix) {
              ld_dft_radix[1] = ld_dft_radix[0];
              FFT_SWAP_PTR(nram_dftmtx[0], nram_dftmtx[1]);
              ld_dft_radix[0] = radix;
              for (int entry = 0;; entry++) {
                if (dft_table[entry].radix == ld_dft_radix[0]) {
                  align_K = K_num * ((radix + K_num - 1) / K_num);
                  __memcpy(nram_dftmtx[0],
                           &dft_matrix[dft_table[entry].offset * 2],
                           sizeof(DT) * 2 * radix * align_K, SRAM2NRAM);
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

            switch (radix) {
              case 2:
                // computeRadix2ButterflyOtherstages(Fout, Fin, section_num,
                // section_num, 1, dir);
                break;
              case 3:
                computeRadix3ButterflyOtherstages(
                    nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                    nram_tw, small_section_num, small_butterfly_num,
                    small_in_stride, dir);
                break;

              default:
                // computeGenericButterflyOtherstages(Fout, buffer, twiddles,
                // radix, section_num, butterfly_num, in_stride, 0, dir);
                MLULOG("computeGenericButterflyOtherstagesMat: %d.\n", radix);
                computeGenericButterflyOtherstagesMat(
                    nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                    nram_dftmtx[0], nram_tw, small_section_num,
                    small_butterfly_num, para_num, small_in_stride, dir, radix);
                break;
            }

            nram_tw += small_butterfly_num * (radix - 1) * 2;
          }  // for (stage_count)

          //           for (int j = 0; j < large_radix; j++) {
          //   MLULOG("output i: (%f, %f).\n", nram_out_r[j], nram_out_i[j]);
          // }

          // MLULOG("butterfly id: %d\n", i);
          // last stage
          {
            FFT_SWAP_PTR(nram_out_r, nram_in_r);
            FFT_SWAP_PTR(nram_out_i, nram_in_i);

            // copy GDRAM2SRAM

            // update parameter
            // value_mul = _small_stage_count << 2;
            radix = small_factors[value_mul++];
            small_section_num = small_factors[value_mul++];
            small_butterfly_num = small_factors[value_mul++];
            small_in_stride = small_factors[value_mul];

            // if (compute_id == 0 && repeat_id == 1 && 0) {
            //   __memcpy(nram_tw, small_twiddles,
            //            small_butterfly_num * (radix - 1) * sizeof(DT) * 2,
            //            GDRAM2NRAM);
            // }
            if (ld_dft_radix[0] != radix && ld_dft_radix[1] != radix) {
              ld_dft_radix[1] = ld_dft_radix[0];
              FFT_SWAP_PTR(nram_dftmtx[0], nram_dftmtx[1]);
              ld_dft_radix[0] = radix;
              for (int entry = 0;; entry++) {
                if (dft_table[entry].radix == ld_dft_radix[0]) {
                  align_K = K_num * ((radix + K_num - 1) / K_num);
                  __memcpy(nram_dftmtx[0],
                           &dft_matrix[dft_table[entry].offset * 2],
                           sizeof(DT) * 2 * radix * align_K, SRAM2NRAM);
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

            switch (radix) {
              case 2:
                break;

              default:
                if (last_stage) {
                  computeGenericButterflyLaststageMat(
                      nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                      nram_scratch, nram_dftmtx[0], nram_tw, small_section_num,
                      small_butterfly_num, para_num, small_in_stride, dir,
                      radix);
                } else {
                  // TODO(zrg): check
                  computeGenericButterflyLaststageMat(
                      nram_para_store_pong,
                      nram_para_store_pong + max_para_ldst_num * large_radix,
                      nram_in_r, nram_in_i, nram_scratch, nram_dftmtx[0],
                      nram_tw, small_section_num, small_butterfly_num, para_num,
                      small_in_stride, dir, radix);
                }
                break;
            }

            if (last_stage) {
              //  [2, para_ldst_num, large_radix] -> [para_ldst_num,
              //  large_radix, 2]
              // DT* nram_transpose_store = nram_in_r;
              if (nram_out_r == nram_para_store_pong) {
                FFT_SWAP_PTR(nram_para_load_pong, nram_para_store_pong)
              }
              __bang_transpose(nram_para_store_pong, nram_out_r, 2,
                               max_para_ldst_num * large_radix);
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

template <typename DT>
__mlu_func__ void computeLargeButterflyOtherstages(
    DT *output, DT *input, const int large_radix, const DT *cur_large_twiddles,
    const DT *_twiddles, const DT *dft_matrix, int large_section_num,
    int large_butterfly_num, int large_in_stride, void *nram_buf,
    const int *small_factors, int nfft, int dir, int last_stage) {
  // return;
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;
  const int K_num = 64 / sizeof(DT);
  int align_K = 0;
  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;

  const int large_out_stride = large_butterfly_num;
  int tw_offset;

  _small_stage_count = small_factors[0];
  // large_radix = small_factors[1];
  tw_offset = small_factors[1];

  const DT *small_twiddles = _twiddles + tw_offset * 2;  // complex

  const int max_para_ldst_num = (4096 + large_radix - 1) / large_radix;

  // int para_ldst_num;
  // TODO(zrg): save nram space.
  // __nram__ DT nram_space[MAX_BUTTERFLY_ON_CHIP * 2];
  // 0 1 2 3 4 5
  // 0   1   3
  // DT *nram_buf_end = (DT*)&((uint8_t*)nram_buf)[NRAM_BUFFER_SIZE];
  // FFT_CPX_T<DT> *nram_in = (FFT_CPX_T<DT> *)nram_buffer;
  // FFT_CPX_T<DT> *nram_out = &nram_in[large_radix];
  // FFT_CPX_T<DT> *nram_buf = &nram_in[large_radix * 2];
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
  nram_transpose_temp = {
      (DT *)nram_in_r,
      (DT *)nram_in_r + large_radix * ((int)last_stage) +
          large_radix * (1 - (int)last_stage) * max_para_ldst_num};
  // nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *_nram_tw = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * 2;  // complex

  // transpose space: [radix, 2 * parrallel] -> [parrallel * 2, radix]
  // FFT_CPX_T<DT> nram_transpose_load = {
  //     (DT *)nram_buf + nram_buf_offset,
  //     (DT *)nram_buf + nram_buf_offset + large_radix * max_para_ldst_num};
  // nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

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

  // size: (large_radix - 1) * max_para_ldst_num
  // DT *scratch_tw_r = &CPX_MUL_II[large_radix * max_para_ldst_num];
  // DT *scratch_tw_i = &scratch_tw_r[(large_radix - 1) * max_para_ldst_num];

  int Fin_stride = 0, Fout_stride = 0;
  int sec_count;
  int repeat_num =
      (large_butterfly_num + max_para_ldst_num - 1) / max_para_ldst_num;
  for (sec_count = 0; sec_count < large_section_num; ++sec_count) {
    for (int repeat_id = 0; repeat_id < repeat_num + 2; ++repeat_id) {
      // small_twiddles = _small_twiddles;

      // pipeline: load-stage

      if (repeat_id < repeat_num) {
        // MLULOG("pipeline: load-stage.\n");
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

      // pipeline: store-stage
      if (repeat_id >= 2) {
        // MLULOG("pipeline: store-stage.\n");
        int i = max_para_ldst_num * (repeat_id - 2);

        int para_store_num = (max_para_ldst_num > (large_butterfly_num - i))
                                 ? (large_butterfly_num - i)
                                 : max_para_ldst_num;

        FFT_CPX_T<DT> nram_para_store =
            (repeat_id % 2 == 0) ? nram_para_store_ping : nram_para_store_pong;

        if (last_stage) {
          // __memcpy_async(
          //     output + (Fout_stride + i * large_radix) * 2,
          //     nram_para_store.r,
          //     sizeof(DT) * 2 * para_store_num * large_radix, NRAM2GDRAM);

          __memcpy_async(output + (Fout_stride + i) * 2, nram_para_store.r,
                         sizeof(DT) * 2 * para_store_num, NRAM2GDRAM,
                         large_out_stride * 2 * sizeof(DT),
                         sizeof(DT) * 2 * para_store_num, large_radix - 1);
        } else {
          // // real
          // __memcpy_async(output + Fout_stride + i * large_radix,
          //                nram_para_store.r,
          //                para_store_num * large_radix * sizeof(DT),
          //                NRAM2GDRAM);
          // // imag
          // __memcpy_async(output + Fout_stride + i * large_radix + nfft,
          //                nram_para_store.i,
          //                para_store_num * large_radix * sizeof(DT),
          //                NRAM2GDRAM);
          // real
          __memcpy_async(output + Fout_stride + i, nram_para_store.r,
                         para_store_num * sizeof(DT), NRAM2GDRAM,
                         large_out_stride * sizeof(DT),
                         sizeof(DT) * para_store_num, large_radix - 1);
          // imag
          __memcpy_async(output + Fout_stride + i + nfft, nram_para_store.i,
                         para_store_num * sizeof(DT), NRAM2GDRAM,
                         large_out_stride * sizeof(DT),
                         sizeof(DT) * para_store_num, large_radix - 1);
        }
      }
      // __sync();
      // pipeline: compute-stage

      if (repeat_id >= 1 && repeat_id < repeat_num + 1) {
        // MLULOG("pipeline: compute-stage.\n");
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

        // __bang_transpose(nram_transpose_load, nram_para_load, large_radix,
        //                  2 * para_ldst_num);

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

        // __bang_transpose(nram_transpose_load.r, nram_para_load_in.r,
        //                  large_radix, para_ldst_num);
        // __bang_transpose(nram_transpose_load.i, nram_para_load_in.i,
        //                  large_radix, para_ldst_num);

        // for (int compute_id = 0; compute_id < para_ldst_num; compute_id++) {
        for (int compute_id = 0; compute_id < para_ldst_num;
             compute_id += para_ldst_num) {
          // load real & imag

          radix = small_factors[4];
          small_section_num = small_factors[5];
          small_in_stride = small_factors[7];
          small_stage_count = _small_stage_count;

          // __memcpy(nram_in_r,
          //         nram_transpose_load + compute_id * large_radix * 2,
          //          large_radix * sizeof(DT) * 2, NRAM2NRAM);

          // first stage
          // if(0)
          if (ld_dft_radix != radix) {
            ld_dft_radix = radix;
            for (int entry = 0;; entry++) {
              if (dft_table[entry].radix == ld_dft_radix) {
                align_K = K_num * ((radix + K_num - 1) / K_num);
                __memcpy(nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                         sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                break;
              }

              if (dft_table[entry].radix == -1) {
                break;
              }
            }
          }

          switch (radix) {
            default:
              // computeGenericButterflyFirststage(Fout, buffer, twiddles,
              // radix, section_num, butterfly_num, in_stride, 0, dir);
              MLULOG("computeGenericButterflyFirststageMat: %d.\n", radix);

              // para_ldst_num = 1
              // in: [radix, butterfly_num]
              // butterfly: [radix, radix] * [radix, butterfly_num]
              // out_butterfly: [radix, butterfly_num]
              // out: [butterfly_num, radix]

              // para_ldst_num != 1
              // in: [radix, butterfly_num, para_ldst_num] == [large_radix,
              // para_ldst_num] butterfly: [radix, radix] * [radix,
              // butterfly_num, para_ldst_num] out_butterfly: [radix,
              // butterfly_num, para_ldst_num] == [radix, butterfly_num *
              // para_ldst_num] out: [butterfly_num, para_ldst_num, radix]

              computeGenericButterflyFirststageMat(
                  nram_out_r, nram_out_i, nram_para_load_in.r,
                  nram_para_load_in.i, nram_scratch, nram_dftmtx,
                  small_section_num * para_ldst_num,
                  small_section_num * para_ldst_num, 1, dir, radix);
              break;
          }

          //           for (int j = 0; j < large_radix; j++) {
          //   MLULOG("output i: (%f, %f).\n", nram_out_r[j], nram_out_i[j]);
          // }

          // [radix, small_section_num, para_ldst_num] ->
          // [small_section_num, para_ldst_num, radix] ->
          // [para_ldst_num, small_section_num, radix] ->
          // [small_section_num, radix, para_ldst_num] ==
          // [large_radix, para_ldst_num]

          small_stage_count--;
          if (small_stage_count == 0) {
            // nram to gdram

            // if (last_stage) {
            //   //  [2, para_ldst_num, large_radix] -> [para_ldst_num,
            //   large_radix,
            //   //  2]
            //   // DT* nram_transpose_store = nram_in_r;

            //   __bang_transpose(nram_in_r, nram_out_r, 2,
            //                    max_para_ldst_num * large_radix);

            // } else {
            //   //  [2, para_ldst_num, large_radix] -> [2, para_ldst_num,
            //   //  large_radix]
            //   // TODO(zrg): redundant move
            //   __memcpy(nram_in_r, nram_out_r,
            //            para_ldst_num * large_radix * sizeof(DT), NRAM2NRAM);
            //   __memcpy(nram_in_i,
            //            nram_out_i, para_ldst_num * large_radix * sizeof(DT),
            //            NRAM2NRAM);
            // }

            // [nfft, 2] -> [2, nfft] -> [2, nfft] -> [nfft, 2]
            if (last_stage) {
              __memcpy(nram_transpose_temp.r +
                           compute_id * large_radix * (1 + (int)last_stage),
                       nram_out_r, sizeof(DT) * large_radix, NRAM2NRAM,
                       sizeof(DT) * large_radix * 2, sizeof(DT) * large_radix,
                       para_ldst_num - 1);

              __memcpy(nram_transpose_temp.i +
                           compute_id * large_radix * (1 + (int)last_stage),
                       nram_out_i, sizeof(DT) * large_radix, NRAM2NRAM,
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

            continue;
          }

          FFT_SWAP_PTR(nram_out_r, nram_in_r);
          FFT_SWAP_PTR(nram_out_i, nram_in_i);
          // DT* nram_transpose_store = nram_in_r;

          // for (int para_ldst_id = 0; para_ldst_id < para_ldst_num;
          //      para_ldst_id++) {
          //   __memcpy(nram_out_r + para_ldst_id * small_section_num * radix,
          //            nram_in_r + para_ldst_id * radix, sizeof(DT) * radix,
          //            NRAM2NRAM, sizeof(DT) * radix,
          //            para_ldst_num * radix * sizeof(DT), small_section_num -
          //            1);

          //   __memcpy(nram_out_i + para_ldst_id * small_section_num * radix,
          //            nram_in_i + para_ldst_id * radix, sizeof(DT) * radix,
          //            NRAM2NRAM, sizeof(DT) * radix,
          //            para_ldst_num * radix * sizeof(DT), small_section_num -
          //            1);
          // }

          // after first stage: [butterfly_num, para_ldst_num, radix]
          // other in: [para_ldst_num, butterfly_num, radix] == [para_ldst_num,
          // large_radix]
          TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                                 small_section_num, para_ldst_num, radix, DT)

          // TODO(zrg) : add not last-stage
          // if (small_stage_count == 0) {
          //   // if last-stage: stride = large_radix * 2
          //   //                compute_id 0 r
          //   //                compute_id 0 i
          //   //                compute_id 1 r
          //   //                compute_id 1 i
          //   // else: stride = large_radix
          //   //                compute_id 0 r
          //   //                compute_id 1 i
          //   //                compute_id 0 r
          //   //                compute_id 1 i

          //   // [radix, small_section_num, para_ldst_num] ->
          //   // [small_section_num, para_ldst_num, radix] ->
          //   // [para_ldst_num, small_section_num, radix] ->
          //   // [small_section_num, radix, para_ldst_num] ==
          //   // [large_radix, para_ldst_num]

          //   __memcpy(nram_transpose_temp.r +
          //                compute_id * large_radix * (1 + (int)last_stage),
          //            nram_out_r, sizeof(DT) * large_radix, NRAM2NRAM,
          //            sizeof(DT) * large_radix * 2, sizeof(DT) * large_radix,
          //            para_ldst_num - 1);

          //   __memcpy(nram_transpose_temp.i +
          //                compute_id * large_radix * (1 + (int)last_stage),
          //            nram_out_i, sizeof(DT) * large_radix, NRAM2NRAM,
          //            sizeof(DT) * large_radix * 2, sizeof(DT) * large_radix,
          //            para_ldst_num - 1);

          //   // __memcpy(nram_transpose_temp.r +
          //   //              compute_id * large_radix * (1 + (int)last_stage),
          //   //          nram_out_r, large_radix * sizeof(DT), NRAM2NRAM);
          //   // __memcpy(nram_transpose_temp.i +
          //   //              compute_id * large_radix * (1 + (int)last_stage),
          //   //          nram_out_i, large_radix * sizeof(DT), NRAM2NRAM);

          //   // __bang_transpose(nram_transpose_temp.r, nram_transpose_temp.r,
          //   //                  max_para_ldst_num * 2, large_radix);
          //   continue;
          // }

          // DT *sram_tw = (DT *)sram_buffer;
          DT *nram_tw = _nram_tw;
          value_mul = 8;

          for (; small_stage_count > 1; small_stage_count--) {
            FFT_SWAP_PTR(nram_out_r, nram_in_r);
            FFT_SWAP_PTR(nram_out_i, nram_in_i);

            // value_mul = (_small_stage_count - small_stage_count + 1) * 4;
            // // update parameter
            radix = small_factors[value_mul++];
            small_section_num = small_factors[value_mul++];
            small_butterfly_num = small_factors[value_mul++];
            small_in_stride = small_factors[value_mul++];
            // copy GDRAM2SRAM

            if (ld_dft_radix != radix) {
              ld_dft_radix = radix;
              for (int entry = 0;; entry++) {
                if (dft_table[entry].radix == ld_dft_radix) {
                  align_K = K_num * ((radix + K_num - 1) / K_num);
                  __memcpy(nram_dftmtx,
                           &dft_matrix[dft_table[entry].offset * 2],
                           sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                  break;
                }

                if (dft_table[entry].radix == -1) {
                  break;
                }
              }
            }

            if (sec_count == 0 && compute_id == 0 && repeat_id == 1) {
              __memcpy(nram_tw, small_twiddles,
                       small_butterfly_num * (radix - 1) * sizeof(DT) * 2,
                       SRAM2NRAM);
              small_twiddles += small_butterfly_num * (radix - 1) * 2;
            }

            switch (radix) {
              // case 2:
              //   // computeRadix2ButterflyOtherstages(Fout, Fin, section_num,
              //   // section_num, 1, dir);
              //   break;
              // case 3:
              //   computeRadix3ButterflyOtherstages(
              //       nram_out_r, nram_out_i, nram_in_r, nram_in_i,
              //       nram_scratch, nram_tw, small_section_num,
              //       small_butterfly_num, small_in_stride, dir);
              //   break;
              // case 9:
              //   computeRadix9ButterflyOtherstages(
              //       nram_out_r, nram_out_i, nram_in_r, nram_in_i,
              //       nram_scratch, nram_tw, small_section_num,
              //       small_butterfly_num, small_in_stride, dir);
              //   break;
              default:
                // computeGenericButterflyOtherstages(Fout, buffer, twiddles,
                // radix, section_num, butterfly_num, in_stride, 0, dir);
                computeGenericButterflyOtherstagesMat(
                    nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                    nram_dftmtx, nram_tw, small_section_num,
                    small_butterfly_num, para_ldst_num, small_in_stride, dir,
                    radix);
                break;
            }

            nram_tw += small_butterfly_num * (radix - 1) * 2;
          }  // for (stage_count)

          //           for (int j = 0; j < large_radix; j++) {
          //   MLULOG("output i: (%f, %f).\n", nram_out_r[j], nram_out_i[j]);
          // }

          // MLULOG("butterfly id: %d\n", i);
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

            if (sec_count == 0 && compute_id == 0 && repeat_id == 1) {
              __memcpy(nram_tw, small_twiddles,
                       small_butterfly_num * (radix - 1) * sizeof(DT) * 2,
                       SRAM2NRAM);
            }

            if (ld_dft_radix != radix) {
              ld_dft_radix = radix;
              for (int entry = 0;; entry++) {
                if (dft_table[entry].radix == ld_dft_radix) {
                  align_K = K_num * ((radix + K_num - 1) / K_num);
                  __memcpy(nram_dftmtx,
                           &dft_matrix[dft_table[entry].offset * 2],
                           sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                  break;
                }

                if (dft_table[entry].radix == -1) {
                  break;
                }
              }
            }
            switch (radix) {
              // case 2:
              //   break;
              // case 3:
              //   computeRadix3ButterflyLaststage(
              //       nram_out_r, nram_out_i, nram_in_r, nram_in_i,
              //       nram_scratch, nram_tw, small_section_num,
              //       small_butterfly_num, small_in_stride, dir);
              //   break;
              // case 9:
              //   computeRadix9ButterflyLaststage(
              //       nram_out_r, nram_out_i, nram_in_r, nram_in_i,
              //       nram_scratch, nram_tw, small_section_num,
              //       small_butterfly_num, small_in_stride, dir);
              //   break;
              default:
                // computeGenericButterflyLaststage(Fout, buffer, twiddles,
                // radix, section_num, butterfly_num, in_stride, 0, dir);
                computeGenericButterflyLaststageMat(
                    nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                    nram_dftmtx, nram_tw, small_section_num,
                    small_butterfly_num, para_ldst_num, small_in_stride, dir,
                    radix);
                break;
            }

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
            // __memcpy(nram_transpose_temp.r +
            //              compute_id * large_radix * (1 + (int)last_stage),
            //          nram_out_r, large_radix * sizeof(DT), NRAM2NRAM);
            // __memcpy(nram_transpose_temp.i +
            //              compute_id * large_radix * (1 + (int)last_stage),
            //          nram_out_i, large_radix * sizeof(DT), NRAM2NRAM);

            if (last_stage) {
              __memcpy(nram_transpose_temp.r +
                           compute_id * large_radix * (1 + (int)last_stage),
                       nram_out_r, sizeof(DT) * large_radix, NRAM2NRAM,
                       sizeof(DT) * large_radix * 2, sizeof(DT) * large_radix,
                       para_ldst_num - 1);

              __memcpy(nram_transpose_temp.i +
                           compute_id * large_radix * (1 + (int)last_stage),
                       nram_out_i, sizeof(DT) * large_radix, NRAM2NRAM,
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

            // __bang_transpose(nram_para_store, nram_transpose_temp.r,
            //                  max_para_ldst_num * 2, large_radix);
          }
        }
      }
      // __sync();

      __sync();
    }
    Fin_stride += large_butterfly_num;
    Fout_stride += large_radix * large_butterfly_num;
  }
}

template <typename DT>
__mlu_func__ void computeLargeButterflyLaststage(
    DT *output, DT *input, const int large_radix, const DT *cur_large_twiddles,
    const DT *_twiddles, const DT *dft_matrix, int large_section_num,
    int large_butterfly_num, int large_in_stride, void *nram_buf,
    const int *small_factors, int nfft, int dir) {
  computeLargeButterflyOtherstages(
      output, input, large_radix, cur_large_twiddles, _twiddles, dft_matrix,
      large_section_num, large_butterfly_num, large_in_stride, nram_buf,
      small_factors, nfft, dir, 1);
}

template <typename DT>
__mlu_func__ void computeLargeButterflyOtherstagesBatchPingpong(
    DT *output, DT *input, const int large_radix, const DT *cur_large_twiddles,
    const DT *small_twiddles, const int small_twiddles_size,
    const DT *dft_matrix, int large_section_num, int large_butterfly_num,
    int large_in_stride, void *nram_buf, const int *small_factors, int nfft,
    const int t_start, const int t_end, int dir, int last_stage) {
  // return;
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;

  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;

  const int large_out_stride = large_butterfly_num;
  int tw_offset;
  const int K_num = 64 / sizeof(DT);
  int align_K = 0;
  _small_stage_count = small_factors[0];
  // large_radix = small_factors[1];
  tw_offset = small_factors[1];

  // const int max_para_ldst_num = (6144 + large_radix - 1) / large_radix;
  // int max_para_ldst_num = (6400 + large_radix - 1) / large_radix;
  int max_para_ldst_num = (large_butterfly_num < small_factors[3])
                              ? large_butterfly_num
                              : small_factors[3];

  // int para_ldst_num;
  // TODO(zrg): save nram space.
  // __nram__ DT nram_space[MAX_BUTTERFLY_ON_CHIP * 2];
  // 0 1 2 3 4 5
  // 0   1   3

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

  // nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *_nram_tw = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * 2;  // complex

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
  nram_transpose_temp = {
      (DT *)nram_buf + nram_buf_offset,
      (DT *)nram_buf + nram_buf_offset + large_radix * ((int)last_stage) +
          large_radix * (1 - (int)last_stage) * max_para_ldst_num};

  // size: (large_radix - 1) * max_para_ldst_num
  // DT *scratch_tw_r = &CPX_MUL_II[large_radix * max_para_ldst_num];
  // DT *scratch_tw_i = &scratch_tw_r[(large_radix - 1) * max_para_ldst_num];

  // int Fin_stride = 0, Fout_stride = 0;
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
        // small_twiddles = _small_twiddles;

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
              __memcpy_async(
                  nram_para_load_tw.r, cur_large_twiddles + butterfly_id,
                  sizeof(DT) * para_num, SRAM2NRAM, sizeof(DT) * para_num,
                  large_out_stride * sizeof(DT), large_radix - 2);
              __memcpy_async(
                  nram_para_load_tw.i,
                  cur_large_twiddles + large_butterfly_num * (large_radix - 1) +
                      butterfly_id,
                  sizeof(DT) * para_num, SRAM2NRAM, sizeof(DT) * para_num,
                  large_out_stride * sizeof(DT), large_radix - 2);
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
        // __sync();
        // pipeline: compute-stage

        if (repeat_id >= 1 && repeat_id < repeat_num + 1) {
          // MLULOG("pipeline: compute-stage.\n");
          // int i = max_para_ldst_num * (repeat_id - 1);

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

          // __bang_transpose(nram_transpose_load.r, nram_para_load_in.r,
          //                  large_radix, para_num);
          // __bang_transpose(nram_transpose_load.i, nram_para_load_in.i,
          //                  large_radix, para_num);

          // for (int compute_id = 0; compute_id < para_num; compute_id++)
          // {
          for (int compute_id = 0; compute_id < para_num;
               compute_id += para_num) {
            // load real & imag

            radix = small_factors[4];
            small_section_num = small_factors[5];
            small_in_stride = small_factors[7];
            small_stage_count = _small_stage_count;

            // __memcpy(nram_in_r,
            //         nram_transpose_load + compute_id * large_radix * 2,
            //          large_radix * sizeof(DT) * 2, NRAM2NRAM);

            // first stage
            // if(0)
            if (ld_dft_radix != radix) {
              ld_dft_radix = radix;
              for (int entry = 0;; entry++) {
                if (dft_table[entry].radix == ld_dft_radix) {
                  align_K = K_num * ((radix + K_num - 1) / K_num);
                  __memcpy(nram_dftmtx,
                           &dft_matrix[dft_table[entry].offset * 2],
                           sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                  break;
                }

                if (dft_table[entry].radix == -1) {
                  break;
                }
              }
            }

            switch (radix) {
              default:

                computeGenericButterflyFirststageMat(
                    nram_out_r, nram_out_i, nram_para_load_in_pong.r,
                    nram_para_load_in_pong.i, nram_scratch, nram_dftmtx,
                    small_section_num * para_num, small_section_num * para_num,
                    1, dir, radix);
                break;
            }

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
                __memcpy(nram_transpose_temp.r +
                             compute_id * large_radix * (1 + (int)last_stage),
                         nram_out_r, sizeof(DT) * large_radix, NRAM2NRAM,
                         sizeof(DT) * large_radix * 2, sizeof(DT) * large_radix,
                         para_num - 1);

                __memcpy(nram_transpose_temp.i +
                             compute_id * large_radix * (1 + (int)last_stage),
                         nram_out_i, sizeof(DT) * large_radix, NRAM2NRAM,
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

              continue;
            }

            FFT_SWAP_PTR(nram_out_r, nram_in_r);
            FFT_SWAP_PTR(nram_out_i, nram_in_i);
            // DT* nram_transpose_store = nram_in_r;

            // after first stage: [butterfly_num, para_ldst_num, radix]
            // other in: [para_ldst_num, butterfly_num, radix] ==
            // [para_ldst_num, large_radix]
            TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                                   small_section_num, para_num, radix, DT)

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

            // DT *sram_tw = (DT *)sram_buffer;
            DT *nram_tw = _nram_tw;
            value_mul = 8;

            for (; small_stage_count > 1; small_stage_count--) {
              FFT_SWAP_PTR(nram_out_r, nram_in_r);
              FFT_SWAP_PTR(nram_out_i, nram_in_i);

              // value_mul = (_small_stage_count - small_stage_count + 1) * 4;
              // // update parameter
              radix = small_factors[value_mul++];
              small_section_num = small_factors[value_mul++];
              small_butterfly_num = small_factors[value_mul++];
              small_in_stride = small_factors[value_mul++];
              // copy GDRAM2SRAM

              if (ld_dft_radix != radix) {
                ld_dft_radix = radix;
                for (int entry = 0;; entry++) {
                  if (dft_table[entry].radix == ld_dft_radix) {
                    align_K = K_num * ((radix + K_num - 1) / K_num);
                    __memcpy(
                        nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                        sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                    break;
                  }

                  if (dft_table[entry].radix == -1) {
                    break;
                  }
                }
              }

              if (sec_count == 0 && compute_id == 0 && repeat_id == 1) {
                __memcpy(nram_tw, small_twiddles,
                         small_butterfly_num * (radix - 1) * sizeof(DT) * 2,
                         SRAM2NRAM);
                small_twiddles += small_butterfly_num * (radix - 1) * 2;
              }

              switch (radix) {
                default:

                  computeGenericButterflyOtherstagesMat(
                      nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                      nram_scratch, nram_dftmtx, nram_tw, small_section_num,
                      small_butterfly_num, para_num, small_in_stride, dir,
                      radix);
                  break;
              }

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

              if (sec_count == 0 && compute_id == 0 && repeat_id == 1) {
                __memcpy(nram_tw, small_twiddles,
                         small_butterfly_num * (radix - 1) * sizeof(DT) * 2,
                         SRAM2NRAM);
              }

              if (ld_dft_radix != radix) {
                ld_dft_radix = radix;
                for (int entry = 0;; entry++) {
                  if (dft_table[entry].radix == ld_dft_radix) {
                    align_K = K_num * ((radix + K_num - 1) / K_num);
                    __memcpy(
                        nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                        sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                    break;
                  }

                  if (dft_table[entry].radix == -1) {
                    break;
                  }
                }
              }
              switch (radix) {
                default:

                  computeGenericButterflyLaststageMat(
                      nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                      nram_scratch, nram_dftmtx, nram_tw, small_section_num,
                      small_butterfly_num, para_num, small_in_stride, dir,
                      radix);
                  break;
              }

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
                __memcpy(nram_transpose_temp.r +
                             compute_id * large_radix * (1 + (int)last_stage),
                         nram_out_r, sizeof(DT) * large_radix, NRAM2NRAM,
                         sizeof(DT) * large_radix * 2, sizeof(DT) * large_radix,
                         para_num - 1);

                __memcpy(nram_transpose_temp.i +
                             compute_id * large_radix * (1 + (int)last_stage),
                         nram_out_i, sizeof(DT) * large_radix, NRAM2NRAM,
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
            }
          }
        }
        // __sync();

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
    const DT *dft_matrix, int large_section_num, int large_butterfly_num,
    int large_in_stride, void *nram_buf, const int *small_factors, int nfft,
    const int t_start, const int t_end, int dir) {
  computeLargeButterflyOtherstagesBatchPingpong(
      output, input, large_radix, cur_large_twiddles, small_twiddles,
      small_twiddles_size, dft_matrix, large_section_num, large_butterfly_num,
      large_in_stride, nram_buf, small_factors, nfft, t_start, t_end, dir, 1);
}

template <typename DT>
__mlu_func__ void computeLargeButterflyFirststageColumn(
    DT *output, DT *input, const int large_radix, int large_in_stride,
    int section_num, const DT *twiddles, const DT *dft_matrix, void *nram_buf,
    const int *small_factors, int dir, int nfft, int last_stage,
    const int para_batch, const int nb) {
  // constant
  // const int para_batch = 3;
  const int K_num = 64 / sizeof(DT);
  int align_K = 0;
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;
  // test

  // network info
  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;
  int tw_offset;
  // int max_radix = small_factors[4];
  _small_stage_count = small_factors[0];
  // large_radix = small_factors[1];
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
  const DT *small_twiddles = twiddles + tw_offset * 2;  // complex

  // assign nram space
  int nram_buf_offset = 0;

  // parallel load/store space
  // sizeof(DT) * 2 * large_radix * para_batch * 4
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

  __memcpy_async(_nram_tw, small_twiddles, large_radix * sizeof(DT) * 2,
                 SRAM2NRAM);

  // return;
  // ceil
  int repeat_num = section_num;
  // MLULOG("repeat_num column: %d\n", repeat_num);
  for (int repeat_id = 0; repeat_id < repeat_num + 2; ++repeat_id) {
    // pipeline: load-stage

    if (repeat_id < repeat_num) {
      // MLULOG("pipeline: load-stage.\n");
      int i = repeat_id;

      __memcpy_async(nram_para_load_ping, input + i * 2 * nb,
                     sizeof(DT) * 2 * para_batch, GDRAM2NRAM,
                     sizeof(DT) * 2 * para_batch,
                     nb * large_in_stride * sizeof(DT) * 2, large_radix - 1);
    }

    // pipeline: store-stage

    if (repeat_id >= 2) {
      // MLULOG("pipeline: store-stage.\n");
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

      // DT *nram_transpose_load = nram_in_r;
      __bang_transpose(nram_in_r, nram_para_load_pong, large_radix * para_batch,
                       2);

      for (int compute_id = 0; compute_id < para_batch;
           compute_id += para_batch) {
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
              __memcpy(nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                       sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
              break;
            }

            if (dft_table[entry].radix == -1) {
              break;
            }
          }
        }

        switch (radix) {
          default:
            MLULOG("computeGenericButterflyFirststageMat: %d.\n", radix);
            computeGenericButterflyFirststageMat(
                nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                nram_dftmtx, small_section_num * para_batch,
                small_section_num * para_batch, 1, dir, radix);
            break;
        }

        small_stage_count--;
        if (small_stage_count == 0) {
          // nram to gdram
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
            // TODO(zrg): redundant move
            __memcpy(nram_para_store_pong, nram_out_r,
                     para_batch * large_radix * sizeof(DT), NRAM2NRAM);
            __memcpy(nram_para_store_pong + para_batch * large_radix,
                     nram_out_i, para_batch * large_radix * sizeof(DT),
                     NRAM2NRAM);
          }

          continue;
        }

        FFT_SWAP_PTR(nram_out_r, nram_in_r);
        FFT_SWAP_PTR(nram_out_i, nram_in_i);

        TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                               small_section_num, para_batch, radix, DT)

        value_mul = 8;
        // DT *sram_tw = (DT *)sram_buffer;
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
                __memcpy(nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                         sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                break;
              }

              if (dft_table[entry].radix == -1) {
                break;
              }
            }
          }

          switch (radix) {
            default:

              computeGenericButterflyOtherstagesMat(
                  nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                  nram_dftmtx, nram_tw, small_section_num, small_butterfly_num,
                  para_batch, small_in_stride, dir, radix);
              break;
          }

          nram_tw += small_butterfly_num * (radix - 1) * 2;
        }  // for (stage_count)

        // last stage
        {
          FFT_SWAP_PTR(nram_out_r, nram_in_r);
          FFT_SWAP_PTR(nram_out_i, nram_in_i);

          // copy GDRAM2SRAM

          // update parameter
          // value_mul = _small_stage_count << 2;
          radix = small_factors[value_mul++];
          small_section_num = small_factors[value_mul++];
          small_butterfly_num = small_factors[value_mul++];
          small_in_stride = small_factors[value_mul];

          if (ld_dft_radix != radix) {
            ld_dft_radix = radix;
            for (int entry = 0;; entry++) {
              if (dft_table[entry].radix == ld_dft_radix) {
                align_K = K_num * ((radix + K_num - 1) / K_num);
                __memcpy(nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                         sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                break;
              }

              if (dft_table[entry].radix == -1) {
                break;
              }
            }
          }

          switch (radix) {
            default:
              MLULOG("computeGenericButterflyLaststageMat: %d.\n", radix);
              computeGenericButterflyLaststageMat(
                  nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                  nram_dftmtx, nram_tw, small_section_num, small_butterfly_num,
                  para_batch, small_in_stride, dir, radix);
              MLULOG("computeGenericButterflyLaststageMat: %d End.\n", radix);
              break;
          }

          // [2, para_batch, large_radix] -> [2, large_radix, para_batch]

          if (last_stage) {
            //  [2, para_batch, large_radix] -> [para_batch, large_radix,
            //  2]
            // DT* nram_transpose_store = nram_in_r;

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
            //  [2, para_batch, large_radix] -> [2, para_batch,
            //  large_radix]
            // TODO(zrg): test
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

    __sync();
    FFT_SWAP_PTR(nram_para_load_ping, nram_para_load_pong)
    FFT_SWAP_PTR(nram_para_store_ping, nram_para_store_pong)
  }
}

template <typename DT>
__mlu_func__ void computeLargeButterflyOtherstagesColumn(
    DT *output, DT *input, const int large_radix, const DT *cur_large_twiddles,
    const DT *_twiddles, const DT *dft_matrix, int large_section_num,
    int large_butterfly_num, int large_in_stride, void *nram_buf,
    const int *small_factors, int nfft, int dir, int last_stage, int para_batch,
    int nb) {
  // return;
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;

  int radix, small_in_stride, small_stage_count, _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;

  const int large_out_stride = large_butterfly_num;
  int tw_offset;

  _small_stage_count = small_factors[0];
  // large_radix = small_factors[1];
  tw_offset = small_factors[1];

  const int K_num = 64 / sizeof(DT);
  int align_K = 0;

  const DT *small_twiddles = _twiddles + tw_offset * 2;  // complex
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
      // small_twiddles = _small_twiddles;

      // pipeline: load-stage

      if (repeat_id < repeat_num) {
        // MLULOG("pipeline: load-stage.\n");
        int i = repeat_id;
        FFT_CPX_T<DT> nram_para_load_in = (repeat_id % 2 == 0)
                                              ? nram_para_load_in_ping
                                              : nram_para_load_in_pong;

        FFT_CPX_T<DT> nram_para_load_tw = (repeat_id % 2 == 0)
                                              ? nram_para_load_tw_ping
                                              : nram_para_load_tw_pong;

        if (para_batch != 1 || 1) {
          __memcpy_async(
              nram_para_load_in.r, input + (Fin_stride + i) * para_batch,
              sizeof(DT) * para_batch, GDRAM2NRAM, sizeof(DT) * para_batch,
              para_batch * large_in_stride * sizeof(DT), large_radix - 1);
          __memcpy_async(
              nram_para_load_in.i,
              input + para_batch * nfft + (Fin_stride + i) * para_batch,
              sizeof(DT) * para_batch, GDRAM2NRAM, sizeof(DT) * para_batch,
              para_batch * large_in_stride * sizeof(DT), large_radix - 1);

          __memcpy_async(nram_para_load_tw.r,
                         cur_large_twiddles + i * (large_radix - 1),
                         sizeof(DT) * (large_radix - 1) * 2, SRAM2NRAM);
          __memcpy_async(nram_para_load_tw.i,
                         cur_large_twiddles +
                             large_butterfly_num * (large_radix - 1) +
                             i * (large_radix - 1),
                         sizeof(DT) * (large_radix - 1), SRAM2NRAM);
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

        // for (int compute_id = 0; compute_id < para_batch; compute_id++) {
        for (int compute_id = 0; compute_id < para_batch;
             compute_id += para_batch) {
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
                __memcpy(nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                         sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                break;
              }

              if (dft_table[entry].radix == -1) {
                break;
              }
            }
          }

          switch (radix) {
            default:

              computeGenericButterflyFirststageMat(
                  nram_out_r, nram_out_i, nram_para_load_in.r,
                  nram_para_load_in.i, nram_scratch, nram_dftmtx,
                  small_section_num * para_batch,
                  small_section_num * para_batch, 1, dir, radix);
              break;
          }

          // [radix, small_section_num, para_ldst_num] ->
          // [small_section_num, para_ldst_num, radix] ->
          // [para_ldst_num, small_section_num, radix] ->
          // [small_section_num, radix, para_ldst_num] ==
          // [large_radix, para_ldst_num]

          small_stage_count--;
          if (small_stage_count == 0) {
            if (last_stage) {
              __memcpy(nram_transpose_temp.r + compute_id * large_radix * 2,
                       nram_out_r, sizeof(DT) * large_radix, NRAM2NRAM,
                       sizeof(DT) * large_radix * 2, sizeof(DT) * large_radix,
                       para_batch - 1);

              __memcpy(nram_transpose_temp.i + compute_id * large_radix * 2,
                       nram_out_i, sizeof(DT) * large_radix, NRAM2NRAM,
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

            continue;
          }

          FFT_SWAP_PTR(nram_out_r, nram_in_r);
          FFT_SWAP_PTR(nram_out_i, nram_in_i);

          TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                                 small_section_num, para_batch, radix, DT)

          // DT *sram_tw = (DT *)sram_buffer;
          DT *nram_tw = _nram_tw;
          value_mul = 8;

          for (; small_stage_count > 1; small_stage_count--) {
            FFT_SWAP_PTR(nram_out_r, nram_in_r);
            FFT_SWAP_PTR(nram_out_i, nram_in_i);

            // value_mul = (_small_stage_count - small_stage_count + 1) * 4;
            // // update parameter
            radix = small_factors[value_mul++];
            small_section_num = small_factors[value_mul++];
            small_butterfly_num = small_factors[value_mul++];
            small_in_stride = small_factors[value_mul++];
            // copy GDRAM2SRAM

            if (ld_dft_radix != radix) {
              ld_dft_radix = radix;
              for (int entry = 0;; entry++) {
                if (dft_table[entry].radix == ld_dft_radix) {
                  align_K = K_num * ((radix + K_num - 1) / K_num);
                  __memcpy(nram_dftmtx,
                           &dft_matrix[dft_table[entry].offset * 2],
                           sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                  break;
                }

                if (dft_table[entry].radix == -1) {
                  break;
                }
              }
            }

            if (sec_count == 0 && compute_id == 0 && repeat_id == 1) {
              __memcpy(nram_tw, small_twiddles,
                       small_butterfly_num * (radix - 1) * sizeof(DT) * 2,
                       SRAM2NRAM);
              small_twiddles += small_butterfly_num * (radix - 1) * 2;
            }

            switch (radix) {
              default:

                computeGenericButterflyOtherstagesMat(
                    nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                    nram_dftmtx, nram_tw, small_section_num,
                    small_butterfly_num, para_batch, small_in_stride, dir,
                    radix);
                break;
            }

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

            if (sec_count == 0 && compute_id == 0 && repeat_id == 1) {
              __memcpy(nram_tw, small_twiddles,
                       small_butterfly_num * (radix - 1) * sizeof(DT) * 2,
                       SRAM2NRAM);
            }

            if (ld_dft_radix != radix) {
              ld_dft_radix = radix;
              for (int entry = 0;; entry++) {
                if (dft_table[entry].radix == ld_dft_radix) {
                  align_K = K_num * ((radix + K_num - 1) / K_num);
                  __memcpy(nram_dftmtx,
                           &dft_matrix[dft_table[entry].offset * 2],
                           sizeof(DT) * 2 * ld_dft_radix * align_K, SRAM2NRAM);
                  break;
                }

                if (dft_table[entry].radix == -1) {
                  break;
                }
              }
            }
            switch (radix) {
              default:

                computeGenericButterflyLaststageMat(
                    nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                    nram_dftmtx, nram_tw, small_section_num,
                    small_butterfly_num, para_batch, small_in_stride, dir,
                    radix);
                break;
            }

            if (last_stage) {
              // [2, para_batch, large_radix] -> [large_radix, para_batch, 2]
              __memcpy(nram_transpose_temp.r + compute_id * large_radix * 2,
                       nram_out_r, sizeof(DT) * large_radix, NRAM2NRAM,
                       sizeof(DT) * large_radix * 2, sizeof(DT) * large_radix,
                       para_batch - 1);

              __memcpy(nram_transpose_temp.i + compute_id * large_radix * 2,
                       nram_out_i, sizeof(DT) * large_radix, NRAM2NRAM,
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

            // __bang_transpose(nram_para_store, nram_transpose_temp.r,
            //                  max_para_ldst_num * 2, large_radix);
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
__mlu_func__ void computeLargeButterflyLaststageColumn(
    DT *output, DT *input, const int large_radix, const DT *cur_large_twiddles,
    const DT *_twiddles, const DT *dft_matrix, int large_section_num,
    int large_butterfly_num, int large_in_stride, void *nram_buf,
    const int *small_factors, int nfft, int dir, int para_batch, int nb) {
  computeLargeButterflyOtherstagesColumn(
      output, input, large_radix, cur_large_twiddles, _twiddles, dft_matrix,
      large_section_num, large_butterfly_num, large_in_stride, nram_buf,
      small_factors, nfft, dir, 1, para_batch, nb);
}
