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
#include "kernels/fft/fft_optm_device/fft_c2c_stockham_nram.h"
#include "kernels/fft/fft_optm_device/fft_sram_allocate.h"

extern __nram__ char nram_buffer[MAX_NRAM_SIZE + REM_FOR_STACK - 32 * 1024];
extern __wram__ char wram_buffer[MAX_WRAM_SIZE];

// Perform C2C in-place network allocation for the row slice on the chip
template <typename DT>
__mlu_func__ void computeMutiStageOnchip(DT *input, DT *output, int *factors,
                                         DT *twiddles, DT *twiddles_end,
                                         const DT *dft_matrix, DT *buffer,
                                         const int batch, const int fft_flag,
                                         const int direction) {
  int total_num = batch;
  int repeat_num = total_num / taskDim;
  int remain_num = total_num % taskDim;

  char *nram_buf = nram_buffer + FFT_MAXFACTORS * sizeof(int);
  int *nram_factors = (int *)nram_buffer;

  int t_len = repeat_num + ((remain_num > 0 && taskId < remain_num) ? 1 : 0);
  int t_start = taskId - remain_num <= 0 ? taskId * (repeat_num + 1)
                                         : (remain_num * (repeat_num + 1) +
                                            (taskId - remain_num) * repeat_num);
  int t_end = (t_start + t_len);

  MLULOG(
      "taskId: %d, repeat_num: %d, "
      "remain_num: %d, t_len: %d, t_start: %d, t_end: %d\n",
      taskId, repeat_num, remain_num, t_len, t_start, t_end);

  int radix, section_num, butterfly_num, in_stride, stage_count, value_mul,
      small_factors_offset;

  int *small_factors;
  int last_stage;

  int sram_offset = 0;
  int *sram_factors = (int *)(sram_buffer + sram_offset);
  sram_offset += FFT_MAXFACTORS * sizeof(int);
  DT *sram_dftmtx = (DT *)(sram_buffer + sram_offset);
  sram_offset += DFT_TABLE_SIZE * sizeof(DT);

  DT *sram_twiddles = (DT *)(sram_buffer + sram_offset);
  const int twiddles_size = twiddles_end - twiddles;

  int load_once_twiddles =
      ((MAX_SRAM_SIZE - FFT_MAXFACTORS * sizeof(int) -
        DFT_TABLE_SIZE * sizeof(DT)) >= twiddles_size * sizeof(DT));

  const int _stage_count = factors[0];
  const int nfft = factors[1];

  radix = factors[5 + 0];
  section_num = factors[5 + 1];
  in_stride = factors[5 + 3];
  small_factors_offset = factors[5 + 4];

  stage_count = _stage_count;
  last_stage = (stage_count == 1);

  if (__is_mpu()) {
    __memcpy_async(sram_factors, factors, FFT_MAXFACTORS * sizeof(int),
                   GDRAM2SRAM);
    if (twiddles_size) {
      if (load_once_twiddles) {
        __memcpy_async(sram_twiddles, twiddles, twiddles_size * sizeof(DT),
                       GDRAM2SRAM);
      }
    }

    const dft_table_entry *dft_table_gdram =
        (const dft_table_entry *)dft_matrix;
    int dft_matrix_offset = dft_table_gdram[0].offset;

    if (dft_matrix_offset != -1) {
      __memcpy(sram_dftmtx, dft_matrix, sizeof(DT) * 2 * dft_matrix_offset,
               GDRAM2SRAM);
      const dft_table_entry *dft_table = (const dft_table_entry *)sram_dftmtx;

      for (int entry = 0;; entry++) {
        if (dft_table[entry + 1].radix == -1) {
          int last_radix = dft_table[entry].radix;
          int last_offset = dft_table[entry].offset;

          const int K_num = 64 / sizeof(DT);
          int align_K = K_num * ((last_radix + K_num - 1) / K_num);
          __memcpy_async(sram_dftmtx, dft_matrix,
                         sizeof(DT) * 2 * (last_radix * align_K + last_offset),
                         GDRAM2SRAM);
          break;
        }
      }
    }
  }

  __sync_cluster();
  if (__is_ipu()) {
    __memcpy(nram_factors, sram_factors, FFT_MAXFACTORS * sizeof(int),
             SRAM2NRAM);
    factors = nram_factors;
    if (load_once_twiddles) twiddles = sram_twiddles;
  }

  if (__is_mpu()) {
    return;
  }

  DT *_twiddles = twiddles;
  DT *odd_extra_buffer;
  {
    odd_extra_buffer = buffer + batch * (nfft << 1);

    if (_stage_count != 1) FFT_SWAP_PTR(buffer, output);
    small_factors = factors + small_factors_offset;
    int tw_offset = factors[small_factors_offset + 1];
    int small_twiddles_size = factors[small_factors_offset + 2];
    const DT *small_twiddles = _twiddles + tw_offset * 2;
    if (repeat_num > 0 || taskId < remain_num) {
      computeLargeButterflyFirststageBatchPingpong<DT>(
          output, input, radix, in_stride, section_num, small_twiddles,
          small_twiddles_size, sram_dftmtx, (void *)nram_buf, small_factors,
          direction, nfft, last_stage, t_start, t_end, load_once_twiddles);
    }
  }

  stage_count--;
  if (stage_count == 0) {
    return;
  }

  value_mul = 10;
  for (; stage_count > 1; stage_count--) {
    FFT_SWAP_PTR(buffer, output);

    if (stage_count == 2 && _stage_count % 2) {
      FFT_SWAP_PTR(odd_extra_buffer, output);
    }

    radix = factors[value_mul++];
    section_num = factors[value_mul++];
    butterfly_num = factors[value_mul++];
    in_stride = factors[value_mul++];
    small_factors_offset = factors[value_mul++];

    small_factors = factors + small_factors_offset;
    int tw_offset = factors[small_factors_offset + 1];
    int small_twiddles_size = factors[small_factors_offset + 2];
    const DT *small_twiddles = _twiddles + tw_offset * 2;

    if (repeat_num > 0 || taskId < remain_num) {
      computeLargeButterflyOtherstagesBatchPingpong<DT>(
          output, buffer, radix, (DT *)twiddles, small_twiddles,
          small_twiddles_size, sram_dftmtx, section_num, butterfly_num,
          in_stride, (void *)nram_buf, small_factors, nfft, t_start, t_end,
          direction, 0, load_once_twiddles);
    }
    twiddles += butterfly_num * (radix - 1) * 2;
  }

  {
    if ((_stage_count % 2 == 1)) {
      FFT_SWAP_PTR(odd_extra_buffer, buffer);
    }

    FFT_SWAP_PTR(buffer, output);

    radix = factors[value_mul++];
    section_num = factors[value_mul++];
    butterfly_num = factors[value_mul++];
    in_stride = factors[value_mul++];
    small_factors_offset = factors[value_mul];

    small_factors = factors + small_factors_offset;
    int tw_offset = factors[small_factors_offset + 1];
    int small_twiddles_size = factors[small_factors_offset + 2];
    const DT *small_twiddles = _twiddles + tw_offset * 2;
    if (repeat_num > 0 || taskId < remain_num) {
      computeLargeButterflyLaststageBatchPingpong(
          output, buffer, radix, (DT *)twiddles, small_twiddles,
          small_twiddles_size, sram_dftmtx, section_num, butterfly_num,
          in_stride, (void *)nram_buf, small_factors, nfft, t_start, t_end,
          direction, load_once_twiddles);
    }
  }
}

// Perform C2C in-place network allocation for the column slice on the chip
template <typename DT>
__mlu_func__ void computeMutiStageOnchipColumn(
    DT *input, DT *output, int *factors, DT *twiddles, DT *twiddles_end,
    const DT *dft_matrix, DT *buffer, const int batch, const int fft_flag,
    const int direction, const int nb) {
  int total_num = batch;
  int repeat_num = total_num / taskDim;
  int remain_num = total_num % taskDim;

  char *nram_buf = nram_buffer + FFT_MAXFACTORS * sizeof(int);
  int *nram_factors = (int *)nram_buffer;

  int t_len = repeat_num + ((remain_num > 0 && taskId < remain_num) ? 1 : 0);
  int t_start = taskId - remain_num <= 0 ? taskId * (repeat_num + 1)
                                         : (remain_num * (repeat_num + 1) +
                                            (taskId - remain_num) * repeat_num);
  int t_end = (t_start + t_len);

  MLULOG(
      "taskId: %d, repeat_num: %d, "
      "remain_num: %d, t_len: %d, t_start: %d, t_end: %d\n",
      taskId, repeat_num, remain_num, t_len, t_start, t_end);

  int radix, section_num, butterfly_num, in_stride, stage_count, value_mul,
      small_factors_offset;

  int *small_factors;
  int last_stage;

  int sram_offset = 0;
  int *sram_factors = (int *)(sram_buffer + sram_offset);
  sram_offset += FFT_MAXFACTORS * sizeof(int);
  DT *sram_dftmtx = (DT *)(sram_buffer + sram_offset);
  sram_offset += DFT_TABLE_SIZE * sizeof(DT);
  DT *sram_twiddles = (DT *)(sram_buffer + sram_offset);
  const int twiddles_size = twiddles_end - twiddles;

  int load_once_twiddles = 0;

  const int _stage_count = factors[0];
  const int nfft = factors[1];

  radix = factors[5 + 0];
  section_num = factors[5 + 1];
  in_stride = factors[5 + 3];
  small_factors_offset = factors[5 + 4];

  stage_count = _stage_count;
  last_stage = (stage_count == 1);

  if (__is_mpu()) {
    __memcpy_async(sram_factors, factors, FFT_MAXFACTORS * sizeof(int),
                   GDRAM2SRAM);
    if (twiddles_size) {
      if (load_once_twiddles) {
        __memcpy_async(sram_twiddles, twiddles, twiddles_size * sizeof(DT),
                       GDRAM2SRAM);
      }
    }

    const dft_table_entry *dft_table_gdram =
        (const dft_table_entry *)dft_matrix;
    int dft_matrix_offset = dft_table_gdram[0].offset;

    if (dft_matrix_offset != -1) {
      __memcpy(sram_dftmtx, dft_matrix, sizeof(DT) * 2 * dft_matrix_offset,
               GDRAM2SRAM);
      const dft_table_entry *dft_table = (const dft_table_entry *)sram_dftmtx;

      for (int entry = 0;; entry++) {
        if (dft_table[entry + 1].radix == -1) {
          int last_radix = dft_table[entry].radix;
          int last_offset = dft_table[entry].offset;
          const int K_num = 64 / sizeof(DT);
          int align_K = K_num * ((last_radix + K_num - 1) / K_num);
          __memcpy_async(sram_dftmtx, dft_matrix,
                         sizeof(DT) * 2 * (last_radix * align_K + last_offset),
                         GDRAM2SRAM);
          break;
        }
      }
    }
  }

  __sync_cluster();
  if (__is_ipu()) {
    __memcpy(nram_factors, sram_factors, FFT_MAXFACTORS * sizeof(int),
             SRAM2NRAM);
    factors = nram_factors;
    if (load_once_twiddles) twiddles = sram_twiddles;
  }

  if (__is_mpu()) {
    return;
  }

  DT *_twiddles = twiddles;

  DT *extra_buffer;

  int max_para_batch;

  {
    extra_buffer = buffer + batch * (nfft << 1);

    if (_stage_count == 1) buffer = output;

    if (repeat_num > 0 || taskId < remain_num) {
      small_factors = factors + small_factors_offset;
      int tw_offset = factors[small_factors_offset + 1];
      int small_twiddles_size = factors[small_factors_offset + 2];
      const DT *small_twiddles = _twiddles + tw_offset * 2;
      int min_max_para_batch = __INT_MAX__;
      for (int stage_id = 1; stage_id <= _stage_count; stage_id++) {
        int cur_para_num = factors[factors[5 * stage_id + 4] + 3];
        min_max_para_batch = (min_max_para_batch < cur_para_num)
                                 ? min_max_para_batch
                                 : cur_para_num;
      }

      max_para_batch = min_max_para_batch > (t_end - t_start)
                           ? (t_end - t_start)
                           : min_max_para_batch;

      for (int t = t_start; t < t_end; t += max_para_batch) {
        int para_batch =
            (max_para_batch < (t_end - t)) ? max_para_batch : (t_end - t);
        DT *input_batch = input + t * 2;
        DT *output_batch;
        if (last_stage) {
          output_batch = buffer + t * 2;
        } else {
          output_batch = buffer + t * nfft * 2;
        }

        computeLargeButterflyFirststageColumn<DT>(
            output_batch, input_batch, radix, in_stride, section_num,
            small_twiddles, small_twiddles_size, sram_dftmtx, (void *)nram_buf,
            small_factors, direction, nfft, last_stage, para_batch, nb,
            load_once_twiddles);
      }
    }
    stage_count--;
    if (stage_count == 0) {
      return;
    }
  }

  value_mul = 10;
  for (; stage_count > 1; stage_count--) {
    radix = factors[value_mul++];
    section_num = factors[value_mul++];
    butterfly_num = factors[value_mul++];
    in_stride = factors[value_mul++];
    small_factors_offset = factors[value_mul++];

    small_factors = factors + small_factors_offset;
    int tw_offset = factors[small_factors_offset + 1];
    int small_twiddles_size = factors[small_factors_offset + 2];
    const DT *small_twiddles = _twiddles + tw_offset * 2;

    if (repeat_num > 0 || taskId < remain_num) {
      for (int t = t_start; t < t_end; t += max_para_batch) {
        DT *output_batch = extra_buffer + t * (nfft * 2);
        DT *buffer_batch = buffer + t * (nfft * 2);

        int para_batch =
            (max_para_batch < (t_end - t)) ? max_para_batch : (t_end - t);

        computeLargeButterflyOtherstagesColumn<DT>(
            output_batch, buffer_batch, radix, (DT *)twiddles, small_twiddles,
            small_twiddles_size, sram_dftmtx, section_num, butterfly_num,
            in_stride, (void *)nram_buf, small_factors, nfft, direction, 0,
            para_batch, nb, load_once_twiddles);
      }
    }
    twiddles += butterfly_num * (radix - 1) * 2;
    FFT_SWAP_PTR(extra_buffer, buffer);
  }

  {
    radix = factors[value_mul++];
    section_num = factors[value_mul++];
    butterfly_num = factors[value_mul++];
    in_stride = factors[value_mul++];
    small_factors_offset = factors[value_mul];

    small_factors = factors + small_factors_offset;
    int tw_offset = factors[small_factors_offset + 1];
    int small_twiddles_size = factors[small_factors_offset + 2];
    const DT *small_twiddles = _twiddles + tw_offset * 2;

    if (repeat_num > 0 || taskId < remain_num) {
      for (int t = t_start; t < t_end; t += max_para_batch) {
        DT *output_batch = output + t * 2;
        DT *buffer_batch = buffer + t * (nfft * 2);

        int para_batch =
            (max_para_batch < (t_end - t)) ? max_para_batch : (t_end - t);

        computeLargeButterflyLaststageColumn<DT>(
            output_batch, buffer_batch, radix, (DT *)twiddles, small_twiddles,
            small_twiddles_size, sram_dftmtx, section_num, butterfly_num,
            in_stride, (void *)nram_buf, small_factors, nfft, direction,
            para_batch, nb, load_once_twiddles);
      }
    }
  }
}
