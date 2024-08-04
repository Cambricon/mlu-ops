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
#include "kernels/fft/fft_optm_device/fft_butterfly_ops.h"

extern __wram__ char wram_buffer[MAX_WRAM_SIZE];

template <typename DT>
__mlu_func__ void computeGenericButterflyFirststageMat_v1(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_in_i,
    DT *nram_scratch, DT *nram_dftmtx, const int section_num,
    const int butterfly_num, const int in_stride, const int dir,
    const int radix) {
  // outplace(nram)

  // origin: M = radix, K = radix, N =butterfly_num
  // pad_up:
  const int para_num = butterfly_num;
  const int align_M = radix;  // no align
  const int K_num = 64 / sizeof(DT);
  const int align_K = K_num * ((radix + K_num - 1) / K_num);
  const int align_N = 64 * ((para_num + 64 - 1) / 64);

  int nram_scratch_offset = 0;
  int wram_scratch_offset = 0;
  DT *wram_sratch = (DT *)wram_buffer;

  FFT_CPX_T<DT> in_wram = {
      &wram_sratch[wram_scratch_offset],
      &wram_sratch[wram_scratch_offset + align_N * align_K]};

  wram_scratch_offset += (align_N * align_K * 2);

  // overlap
  FFT_CPX_T<DT> in_align = {
      &nram_scratch[nram_scratch_offset],
      &nram_scratch[nram_scratch_offset + align_N * align_K]};
  FFT_CPX_T<DT> out_trans = {
      &nram_scratch[nram_scratch_offset],
      &nram_scratch[nram_scratch_offset + align_M * align_N]};

  nram_scratch_offset += (align_N * align_K * 2);

  // overlap
  FFT_CPX_T<DT> in_align2 = {
      &nram_scratch[nram_scratch_offset],
      &nram_scratch[nram_scratch_offset + align_N * align_K]};
  FFT_CPX_T<DT> out = {&nram_scratch[nram_scratch_offset],
                       &nram_scratch[nram_scratch_offset + align_M * align_N]};

  nram_scratch_offset += (align_N * align_K * 2);
  FFT_CPX_T<DT> in_trans = {nram_out_r, nram_out_i};

  FFT_CPX_T<DT> dftmtx = {nram_dftmtx, &nram_dftmtx[radix * align_K]};

  DT *RR = &nram_scratch[nram_scratch_offset];
  DT *RI = &nram_scratch[nram_scratch_offset + align_K * align_N];
  DT *IR = &nram_scratch[nram_scratch_offset + align_K * align_N * 2];
  DT *II = &nram_scratch[nram_scratch_offset + align_K * align_N * 3];

  nram_scratch_offset += (align_K * 4 * align_N);

  __bang_transpose(in_trans.r, nram_in_r, radix, butterfly_num);
  __bang_transpose(in_trans.i, nram_in_i, radix, butterfly_num);

  if (align_K == radix) {
    if (para_num != 1 && align_N != para_num) {
      __bang_pad(in_align.r, in_trans.r, 1, para_num, radix, 0,
                 align_N - para_num, 0, 0);
      __bang_pad(in_align.i, in_trans.i, 1, para_num, radix, 0,
                 align_N - para_num, 0, 0);

    } else {
      in_align = in_trans;
    }
  } else {
    if (para_num != 1) {
      __bang_pad(in_align.r, in_trans.r, 1, para_num, radix, 0,
                 align_N - para_num, 0, align_K - radix);
      __bang_pad(in_align.i, in_trans.i, 1, para_num, radix, 0,
                 align_N - para_num, 0, align_K - radix);

    } else {
      __bang_pad(in_align.r, in_trans.r, 1, radix, 1, 0, align_K - radix, 0, 0);
      __bang_pad(in_align.i, in_trans.i, 1, radix, 1, 0, align_K - radix, 0, 0);
    }
  }

  __bang_reshape_filter(in_align2.r, in_align.r, align_N, 1, 1, align_K);
  __bang_reshape_filter(in_align2.i, in_align.i, align_N, 1, 1, align_K);

  __memcpy(in_wram.r, in_align2.r, align_N * align_K * sizeof(DT), NRAM2WRAM);
  __memcpy(in_wram.i, in_align2.i, align_N * align_K * sizeof(DT), NRAM2WRAM);

  __bang_matmul((float *)RR, (float *)dftmtx.r, (float *)in_wram.r, align_M,
                align_K, align_N);
  __bang_matmul((float *)II, (float *)dftmtx.i, (float *)in_wram.i, align_M,
                align_K, align_N);

  __bang_sub(out.r, RR, II, align_M * align_N);
  __bang_transpose(out_trans.r, out.r, align_M, align_N);
  __memcpy(nram_out_r, out_trans.r, radix * butterfly_num * sizeof(DT),
           NRAM2NRAM);

  __bang_matmul((float *)RI, (float *)dftmtx.r, (float *)in_wram.i, align_M,
                align_K, align_N);
  __bang_matmul((float *)IR, (float *)dftmtx.i, (float *)in_wram.r, align_M,
                align_K, align_N);

  __bang_add(out.i, RI, IR, align_M * align_N);
  __bang_transpose(out_trans.i, out.i, align_M, align_N);
  __memcpy(nram_out_i, out_trans.i, radix * butterfly_num * sizeof(DT),
           NRAM2NRAM);
}

// Compute the generic butterfly for the first stage using matrix operations
template <typename DT>
__mlu_func__ void computeGenericButterflyFirststageMat(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_in_i,
    DT *nram_scratch, DT *nram_dftmtx, const int section_num,
    const int butterfly_num, const int in_stride, const int dir,
    const int radix) {
  // outplace(nram)

#define in_trans_r (nram_out_r)
#define in_trans_i (&nram_out_r[radix * butterfly_num])

  __bang_transpose(in_trans_r, nram_in_r, radix, butterfly_num);
  __bang_transpose(in_trans_i, nram_in_i, radix, butterfly_num);

  // origin: M = radix, K = radix, N = butterfly_num * 2
  // pad_up:
  const int para_num = butterfly_num * 2;
  const int align_M = radix;  // no align
  const int K_num = 64 / sizeof(DT);
  const int align_K = K_num * ((radix + K_num - 1) / K_num);
  const int align_N = 64 * ((para_num + 64 - 1) / 64);

  int nram_scratch_offset = 0;
  DT *in_wram = (DT *)wram_buffer;

  // overlap
  DT *RR_RI_trans = &nram_scratch[nram_scratch_offset];
  DT *IR_II_trans = &nram_scratch[nram_scratch_offset + align_N * align_M];

  DT *in_align = &nram_scratch[nram_scratch_offset];

  nram_scratch_offset +=
      ((align_M * 2 > align_K) ? (align_M * 2) : align_K) * align_N;

  // overlap
  DT *in_align2 = &nram_scratch[nram_scratch_offset];

  DT *RR_RI = &nram_scratch[nram_scratch_offset];
  DT *IR_II = &nram_scratch[nram_scratch_offset + align_M * align_N];
  nram_scratch_offset +=
      ((align_M * 2 > align_K) ? (align_M * 2) : align_K) * align_N;

  // overlap
  FFT_CPX_T<DT> dftmtx = {nram_dftmtx, &nram_dftmtx[radix * align_K]};

  if (align_K == radix && align_N == para_num) {
    in_align = in_trans_r;
  } else {
    __bang_pad(in_align, in_trans_r, 1, para_num, radix, 0, align_N - para_num,
               0, align_K - radix);
  }

  __bang_reshape_filter(in_align2, in_align, align_N, 1, 1, align_K);
  __sync_compute();
  __memcpy_async(in_wram, in_align2, align_N * align_K * sizeof(DT), NRAM2WRAM);
  __sync_move();

  __bang_matmul((float *)RR_RI, (float *)dftmtx.r, (float *)in_wram, align_M,
                align_K, align_N);
  __bang_transpose(RR_RI_trans, RR_RI, align_M, align_N);
  __bang_matmul((float *)IR_II, (float *)dftmtx.i, (float *)in_wram, align_M,
                align_K, align_N);
  __bang_transpose(IR_II_trans, IR_II, align_M, align_N);

  __bang_sub(nram_out_r, RR_RI_trans, &IR_II_trans[butterfly_num * radix],
             radix * butterfly_num);
  __bang_add(nram_out_i, &RR_RI_trans[butterfly_num * radix], &IR_II_trans[0],
             radix * butterfly_num);
}

// Compute the generic butterfly for other stages using matrix operations
template <typename DT>
__mlu_func__ void computeGenericButterflyOtherstagesMat(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_in_i,
    DT *nram_scratch, DT *nram_dftmtx, DT *nram_tw, const int section_num,
    const int butterfly_num, const int para_large_butterfly,
    const int in_stride, const int dir, const int radix) {
  const int para_num = butterfly_num * section_num * para_large_butterfly;
  const int align_M = radix;  // no align
  const int K_num = 64 / sizeof(DT);
  const int align_K = K_num * ((radix + K_num - 1) / K_num);
  const int align_N = 64 * ((para_num + 64 - 1) / 64);

  FFT_CPX_T<DT> scratch_tw = {nram_tw, &nram_tw[butterfly_num * (radix - 1)]};

  int nram_scratch_offset = 0;
  int wram_scratch_offset = 0;

  // overlap
  FFT_CPX_T<DT> in_align2 = {
      &nram_scratch[nram_scratch_offset],
      &nram_scratch[nram_scratch_offset + align_N * align_K]};
  FFT_CPX_T<DT> out = {&nram_scratch[nram_scratch_offset],
                       &nram_scratch[nram_scratch_offset + align_M * align_N]};

  nram_scratch_offset += (align_N * align_K * 2);

  FFT_CPX_T<DT> Fin = in_align2;

  // [para_large_butterfly, large_radix]
  // [para_large_butterfly, radix, section_num, butterfly_num]

  // [radix, para_large_butterfly, section_num, butterfly_num]
  // [radix, para_num]

  // butterfly: [radix, radix] * [radix, para_num]
  TRANSPOSE_XYZ2YXZ_PAIR(Fin.r, Fin.i, nram_in_r, nram_in_i,
                         para_large_butterfly, radix,
                         butterfly_num * section_num, DT)

  DT *wram_sratch = (DT *)wram_buffer;
  FFT_CPX_T<DT> in_wram = {
      &wram_sratch[wram_scratch_offset],
      &wram_sratch[wram_scratch_offset + align_N * align_K]};
  wram_scratch_offset += (align_N * align_K * 2);

  FFT_CPX_T<DT> in_align = {
      &nram_scratch[nram_scratch_offset],
      &nram_scratch[nram_scratch_offset + align_N * align_K]};
  nram_scratch_offset += (align_N * align_K * 2);

  FFT_CPX_T<DT> in_trans = {
      &nram_scratch[nram_scratch_offset],
      &nram_scratch[nram_scratch_offset + para_num * radix]};
  nram_scratch_offset += (para_num * radix * 2);

  FFT_CPX_T<DT> dftmtx = {nram_dftmtx, &nram_dftmtx[radix * align_K]};

  DT *RR = &nram_scratch[nram_scratch_offset];
  DT *RI = &nram_scratch[nram_scratch_offset + align_K * align_N];
  DT *IR = &nram_scratch[nram_scratch_offset + align_K * align_N * 2];
  DT *II = &nram_scratch[nram_scratch_offset + align_K * align_N * 3];

  nram_scratch_offset += (align_K * 4 * align_N);

  // [para_large_butterfly, radix, butterfly_num * section_num]
  //  -> [radix, para_large_butterfly, butterfly_num * section_num]

  int nram_in_offset = para_num;

  for (int i = 1; i < radix; i++, nram_in_offset += para_num) {
    __bang_cycle_mul(&RR[(i - 1) * para_num], &Fin.r[nram_in_offset],
                     &scratch_tw.r[(i - 1) * butterfly_num], para_num,
                     butterfly_num);
    __bang_cycle_mul(&RI[(i - 1) * para_num], &Fin.r[nram_in_offset],
                     &scratch_tw.i[(i - 1) * butterfly_num], para_num,
                     butterfly_num);
    __bang_cycle_mul(&IR[(i - 1) * para_num], &Fin.i[nram_in_offset],
                     &scratch_tw.r[(i - 1) * butterfly_num], para_num,
                     butterfly_num);
    __bang_cycle_mul(&II[(i - 1) * para_num], &Fin.i[nram_in_offset],
                     &scratch_tw.i[(i - 1) * butterfly_num], para_num,
                     butterfly_num);
  }

  __bang_sub(&Fin.r[para_num], RR, II, para_num * (radix - 1));
  __bang_transpose(in_trans.r, Fin.r, radix, para_num);

  __bang_add(&Fin.i[para_num], RI, IR, para_num * (radix - 1));
  __bang_transpose(in_trans.i, Fin.i, radix, para_num);

  if (align_K == radix) {
    if (para_num != 1 && align_N != para_num) {
      __bang_pad(in_align.r, in_trans.r, 1, para_num, radix, 0,
                 align_N - para_num, 0, 0);
      __bang_pad(in_align.i, in_trans.i, 1, para_num, radix, 0,
                 align_N - para_num, 0, 0);

    } else {
      in_align = in_trans;
    }
  } else {
    if (para_num != 1) {
      __bang_pad(in_align.r, in_trans.r, 1, para_num, radix, 0,
                 align_N - para_num, 0, align_K - radix);
      __bang_pad(in_align.i, in_trans.i, 1, para_num, radix, 0,
                 align_N - para_num, 0, align_K - radix);

    } else {
      __bang_pad(in_align.r, in_trans.r, 1, radix, 1, 0, align_K - radix, 0, 0);
      __bang_pad(in_align.i, in_trans.i, 1, radix, 1, 0, align_K - radix, 0, 0);
    }
  }

  __bang_reshape_filter(in_align2.r, in_align.r, align_N, 1, 1, align_K);
  __bang_reshape_filter(in_align2.i, in_align.i, align_N, 1, 1, align_K);
  __sync_compute();
  __memcpy_async(in_wram.r, in_align2.r, align_N * align_K * sizeof(DT),
                 NRAM2WRAM);
  __memcpy_async(in_wram.i, in_align2.i, align_N * align_K * sizeof(DT),
                 NRAM2WRAM);

  __sync_move();

  __bang_matmul((float *)RR, (float *)dftmtx.r, (float *)in_wram.r, align_M,
                align_K, align_N);
  __bang_matmul((float *)II, (float *)dftmtx.i, (float *)in_wram.i, align_M,
                align_K, align_N);
  __bang_sub(out.r, RR, II, align_M * align_N);

  __bang_matmul((float *)RI, (float *)dftmtx.r, (float *)in_wram.i, align_M,
                align_K, align_N);
  __bang_matmul((float *)IR, (float *)dftmtx.i, (float *)in_wram.r, align_M,
                align_K, align_N);
  __bang_add(out.i, RI, IR, align_M * align_N);

  // [small_section_num, para_ldst_num, radix] -> [para_ldst_num,
  // small_section_num, radix]
  {
    int src_stride0 = butterfly_num * sizeof(DT);
    int src_segnum1 = para_large_butterfly * section_num - 1;
    int src_stride1 = align_N * sizeof(DT);
    int src_segnum2 = radix - 1;

    int dst_stride0 = radix * butterfly_num * sizeof(DT);
    int dst_segnum1 = para_large_butterfly * section_num - 1;
    int dst_stride1 = butterfly_num * sizeof(DT);
    int dst_segnum2 = radix - 1;
    __sync_compute();
    __memcpy_async(nram_out_r, out.r, sizeof(DT) * butterfly_num, NRAM2NRAM,
                   dst_stride0, dst_segnum1, dst_stride1, dst_segnum2,
                   src_stride0, src_segnum1, src_stride1, src_segnum2);
    __memcpy_async(nram_out_i, out.i, sizeof(DT) * butterfly_num, NRAM2NRAM,
                   dst_stride0, dst_segnum1, dst_stride1, dst_segnum2,
                   src_stride0, src_segnum1, src_stride1, src_segnum2);
    __sync_move();
  }
}

// Compute the generic butterfly for the last stage using matrix operations
template <typename DT>
__mlu_func__ void computeGenericButterflyLaststageMat(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_in_i,
    DT *nram_scratch, DT *nram_dftmtx, DT *nram_tw, const int section_num,
    const int butterfly_num, const int para_large_butterfly,
    const int in_stride, const int dir, const int radix) {
  computeGenericButterflyOtherstagesMat(
      nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch, nram_dftmtx,
      nram_tw, section_num, butterfly_num, para_large_butterfly, in_stride, dir,
      radix);
}

// Compute the generic butterfly for the first stage using matrix operations
// from real to complex (R2C)
template <typename DT>
__mlu_func__ void computeGenericButterflyFirststageMatR2C(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_scratch,
    DT *nram_dftmtx, const int section_num, const int butterfly_num,
    const int in_stride, const int radix) {
  // outplace(nram)

  const int para_num = butterfly_num;
  const int M = radix;
  const int align_M = M / 2 + 1;  // height of dftmtx to be computed
  const int K_num = 64 / sizeof(DT);
  const int align_K = K_num * ((radix + K_num - 1) / K_num);
  const int align_N = 64 * ((para_num + 64 - 1) / 64);

  int nram_scratch_offset = 0;
  int wram_scratch_offset = 0;
  DT *wram_scratch = (DT *)wram_buffer;

  DT *in_wram = &wram_scratch[wram_scratch_offset];
  wram_scratch_offset += (align_N * align_K);

  // overlap
  DT *in_align = &nram_scratch[nram_scratch_offset];

  FFT_CPX_T<DT> out_trans = {
      &nram_scratch[nram_scratch_offset],
      &nram_scratch[nram_scratch_offset + align_M * align_N]};
  // align_K+2 >= radix+2 >= 2M >= radix+1
  nram_scratch_offset += (align_N * (align_K + 2));

  // overlap
  DT *in_align2 = &nram_scratch[nram_scratch_offset];
  FFT_CPX_T<DT> out = {&nram_scratch[nram_scratch_offset],
                       &nram_scratch[nram_scratch_offset + M * align_N]};

  nram_scratch_offset += (align_N * align_K * 2);
  DT *in_trans = nram_out_r;
  FFT_CPX_T<DT> dftmtx = {nram_dftmtx, &nram_dftmtx[radix * align_K]};

  __bang_transpose(in_trans, nram_in_r, radix, butterfly_num);

  if (align_K == radix) {
    if (para_num != 1 && align_N != para_num) {
      __bang_pad(in_align, in_trans, 1, para_num, radix, 0, align_N - para_num,
                 0, 0);
    } else {
      in_align = in_trans;
    }
  } else {
    if (para_num != 1) {
      __bang_pad(in_align, in_trans, 1, para_num, radix, 0, align_N - para_num,
                 0, align_K - radix);
    } else {
      __bang_pad(in_align, in_trans, 1, radix, 1, 0, align_K - radix, 0, 0);
    }
  }

  __bang_reshape_filter(in_align2, in_align, align_N, 1, 1, align_K);
  __sync_compute();

  __memcpy_async(in_wram, in_align2, align_N * align_K * sizeof(DT), NRAM2WRAM);
  __sync_move();

  __bang_matmul((float *)out.r, (float *)dftmtx.r, (float *)in_wram, align_M,
                align_K, align_N);
  __bang_matmul((float *)out.i, (float *)dftmtx.i, (float *)in_wram, align_M,
                align_K, align_N);

  __bang_transpose(out_trans.r, out.r, align_M, align_N);
  __bang_transpose(out_trans.i, out.i, align_M, align_N);
  __sync_compute();

  __memcpy_async(nram_out_r, out_trans.r, align_M * butterfly_num * sizeof(DT),
                 NRAM2NRAM);
  __memcpy_async(nram_out_i, out_trans.i, align_M * butterfly_num * sizeof(DT),
                 NRAM2NRAM);
  __sync_move();
}

// Compute the generic butterfly for other stages using matrix operations from
// real to complex (R2C)
template <typename DT>
__mlu_func__ void computeGenericButterflyOtherstagesMatR2C(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_in_i,
    DT *nram_scratch, DT *nram_dftmtx, DT *nram_tw, const int section_num,
    const int butterfly_num, const int para_large_butterfly,
    const int in_stride, const int radix) {
  const int compute_butterfly_num = ((butterfly_num + 2) >> 1);
  const int para_num =
      compute_butterfly_num * section_num * para_large_butterfly;
  const int align_M = radix;  // no align
  const int K_num = 64 / sizeof(DT);
  const int align_K = K_num * ((radix + K_num - 1) / K_num);
  const int align_N = 64 * ((para_num + 64 - 1) / 64);

  FFT_CPX_T<DT> scratch_tw = {nram_tw,
                              &nram_tw[compute_butterfly_num * (radix - 1)]};

  int nram_scratch_offset = 0;
  int wram_scratch_offset = 0;

  // overlap
  FFT_CPX_T<DT> in_align2 = {
      &nram_scratch[nram_scratch_offset],
      &nram_scratch[nram_scratch_offset + align_N * align_K]};
  FFT_CPX_T<DT> out = {&nram_scratch[nram_scratch_offset],
                       &nram_scratch[nram_scratch_offset + align_M * align_N]};
  DT *out_r_sym =
      &nram_scratch[nram_scratch_offset + ((align_M + 1) / 2) * align_N];
  DT *out_i_sym = &nram_scratch[nram_scratch_offset + align_M * align_N +
                                ((align_M + 1) / 2) * align_N];

  nram_scratch_offset += (align_N * align_K * 2);

  FFT_CPX_T<DT> Fin = in_align2;

  TRANSPOSE_XYZ2YXZ_PAIR(Fin.r, Fin.i, nram_in_r, nram_in_i,
                         para_large_butterfly, radix,
                         compute_butterfly_num * section_num, DT)

  DT *wram_sratch = (DT *)wram_buffer;
  FFT_CPX_T<DT> in_wram = {
      &wram_sratch[wram_scratch_offset],
      &wram_sratch[wram_scratch_offset + align_N * align_K]};
  wram_scratch_offset += (align_N * align_K * 2);

  FFT_CPX_T<DT> in_align = {
      &nram_scratch[nram_scratch_offset],
      &nram_scratch[nram_scratch_offset + align_N * align_K]};

  DT *out_r_sratch = &nram_scratch[nram_scratch_offset];
  DT *out_i_sratch =
      &nram_scratch[nram_scratch_offset + ((align_M + 1) / 2) * align_N];
  nram_scratch_offset += (align_N * align_K * 2);

  FFT_CPX_T<DT> in_trans = {
      &nram_scratch[nram_scratch_offset],
      &nram_scratch[nram_scratch_offset + para_num * radix]};
  nram_scratch_offset += (para_num * radix * 2);

  FFT_CPX_T<DT> dftmtx = {nram_dftmtx, &nram_dftmtx[radix * align_K]};

  DT *RR = &nram_scratch[nram_scratch_offset];
  DT *RI = &nram_scratch[nram_scratch_offset + align_K * align_N];
  DT *IR = &nram_scratch[nram_scratch_offset + align_K * align_N * 2];
  DT *II = &nram_scratch[nram_scratch_offset + align_K * align_N * 3];

  nram_scratch_offset += (align_K * 4 * align_N);

  int nram_in_offset = para_num;

  for (int i = 1; i < radix; i++, nram_in_offset += para_num) {
    __bang_cycle_mul(&RR[(i - 1) * para_num], &Fin.r[nram_in_offset],
                     &scratch_tw.r[(i - 1) * compute_butterfly_num], para_num,
                     compute_butterfly_num);
    __bang_cycle_mul(&RI[(i - 1) * para_num], &Fin.r[nram_in_offset],
                     &scratch_tw.i[(i - 1) * compute_butterfly_num], para_num,
                     compute_butterfly_num);
    __bang_cycle_mul(&IR[(i - 1) * para_num], &Fin.i[nram_in_offset],
                     &scratch_tw.r[(i - 1) * compute_butterfly_num], para_num,
                     compute_butterfly_num);
    __bang_cycle_mul(&II[(i - 1) * para_num], &Fin.i[nram_in_offset],
                     &scratch_tw.i[(i - 1) * compute_butterfly_num], para_num,
                     compute_butterfly_num);
  }

  __bang_sub(&Fin.r[para_num], RR, II, para_num * (radix - 1));

  __bang_transpose(in_trans.r, Fin.r, radix, para_num);

  __bang_add(&Fin.i[para_num], RI, IR, para_num * (radix - 1));
  __bang_transpose(in_trans.i, Fin.i, radix, para_num);
  if (align_K == radix) {
    if (para_num != 1 && align_N != para_num) {
      __bang_pad(in_align.r, in_trans.r, 1, para_num, radix, 0,
                 align_N - para_num, 0, 0);
      __bang_pad(in_align.i, in_trans.i, 1, para_num, radix, 0,
                 align_N - para_num, 0, 0);

    } else {
      in_align = in_trans;
    }
  } else {
    if (para_num != 1) {
      __bang_pad(in_align.r, in_trans.r, 1, para_num, radix, 0,
                 align_N - para_num, 0, align_K - radix);
      __bang_pad(in_align.i, in_trans.i, 1, para_num, radix, 0,
                 align_N - para_num, 0, align_K - radix);

    } else {
      __bang_pad(in_align.r, in_trans.r, 1, radix, 1, 0, align_K - radix, 0, 0);
      __bang_pad(in_align.i, in_trans.i, 1, radix, 1, 0, align_K - radix, 0, 0);
    }
  }

  __bang_reshape_filter(in_align2.r, in_align.r, align_N, 1, 1, align_K);
  __bang_reshape_filter(in_align2.i, in_align.i, align_N, 1, 1, align_K);

  __sync_compute();
  __memcpy_async(in_wram.r, in_align2.r, align_N * align_K * sizeof(DT),
                 NRAM2WRAM);
  __memcpy_async(in_wram.i, in_align2.i, align_N * align_K * sizeof(DT),
                 NRAM2WRAM);
  __sync_move();

  __bang_matmul((float *)RR, (float *)dftmtx.r, (float *)in_wram.r, align_M,
                align_K, align_N);
  __bang_matmul((float *)II, (float *)dftmtx.i, (float *)in_wram.i, align_M,
                align_K, align_N);
  __bang_sub(out.r, RR, II, align_M * align_N);

  __bang_matmul((float *)RI, (float *)dftmtx.r, (float *)in_wram.i, align_M,
                align_K, align_N);
  __bang_matmul((float *)IR, (float *)dftmtx.i, (float *)in_wram.r, align_M,
                align_K, align_N);
  __bang_add(out.i, RI, IR, align_M * align_N);

  __bang_mul_scalar(out_i_sym, out_i_sym, -1, (align_M / 2) * align_N);

  const int matr_width =
      ((align_N % compute_butterfly_num) == 0) ? align_N : para_num;

  if (align_N % compute_butterfly_num != 0) {
    __sync_compute();
    __memcpy_async(out_r_sym, out_r_sym, para_num * sizeof(DT), NRAM2NRAM,
                   para_num * sizeof(DT), align_N * sizeof(DT),
                   (align_M / 2) - 1);
    __memcpy_async(out_i_sym, out_i_sym, para_num * sizeof(DT), NRAM2NRAM,
                   para_num * sizeof(DT), align_N * sizeof(DT),
                   (align_M / 2) - 1);
    __sync_move();
  }

  __bang_mirror(out_r_sratch, out_r_sym,
                (align_M / 2) * (matr_width / compute_butterfly_num),
                compute_butterfly_num);
  __bang_mirror(out_i_sratch, out_i_sym,
                (align_M / 2) * (matr_width / compute_butterfly_num),
                compute_butterfly_num);

  __bang_rotate90(out_r_sym, out_r_sratch, (align_M / 2), matr_width);
  __bang_rotate90(out_i_sym, out_i_sratch, (align_M / 2), matr_width);
  __bang_mirror(out_r_sratch, out_r_sym, matr_width, (align_M / 2));
  __bang_mirror(out_i_sratch, out_i_sym, matr_width, (align_M / 2));
  __bang_rotate270(out_r_sym, out_r_sratch, matr_width, (align_M / 2));
  __bang_rotate270(out_i_sym, out_i_sratch, matr_width, (align_M / 2));
  {
    int src_stride0 = compute_butterfly_num * sizeof(DT);
    int src_segnum1 = para_large_butterfly * section_num - 1;
    int src_stride1_sym1 = align_N * sizeof(DT);
    int src_stride1_sym2 = matr_width * sizeof(DT);
    int src_segnum2_sym1 = (radix - 1) / 2;
    int src_segnum2_sym2 = radix / 2 - 1;

    int dst_stride0 = (radix * butterfly_num / 2 + 1) * sizeof(DT);
    int dst_segnum1 = para_large_butterfly * section_num - 1;
    int dst_stride1 = butterfly_num * sizeof(DT);
    int dst_segnum2_sym1 = (radix - 1) / 2;
    int dst_segnum2_sym2 = radix / 2 - 1;

    __sync_move();
    __sync_compute();
    __memcpy_async(nram_out_r, out.r, sizeof(DT) * compute_butterfly_num,
                   NRAM2NRAM, dst_stride0, dst_segnum1, dst_stride1,
                   dst_segnum2_sym1, src_stride0, src_segnum1, src_stride1_sym1,
                   src_segnum2_sym1);
    __memcpy_async(nram_out_r + ((butterfly_num + 1) / 2), out_r_sym,
                   sizeof(DT) * compute_butterfly_num, NRAM2NRAM, dst_stride0,
                   dst_segnum1, dst_stride1, dst_segnum2_sym2, src_stride0,
                   src_segnum1, src_stride1_sym2, src_segnum2_sym2);

    __memcpy_async(nram_out_i, out.i, sizeof(DT) * compute_butterfly_num,
                   NRAM2NRAM, dst_stride0, dst_segnum1, dst_stride1,
                   dst_segnum2_sym1, src_stride0, src_segnum1, src_stride1_sym1,
                   src_segnum2_sym1);
    __memcpy_async(nram_out_i + ((butterfly_num + 1) / 2), out_i_sym,
                   sizeof(DT) * compute_butterfly_num, NRAM2NRAM, dst_stride0,
                   dst_segnum1, dst_stride1, dst_segnum2_sym2, src_stride0,
                   src_segnum1, src_stride1_sym2, src_segnum2_sym2);
    __sync_move();
  }
}

// Compute the generic butterfly for the last stage using matrix operations from
// real to complex (R2C)
template <typename DT>
__mlu_func__ void computeGenericButterflyLaststageMatR2C(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_in_i,
    DT *nram_scratch, DT *nram_dftmtx, DT *nram_tw, const int section_num,
    const int butterfly_num, const int para_large_butterfly,
    const int in_stride, const int radix) {
  computeGenericButterflyOtherstagesMatR2C(
      nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch, nram_dftmtx,
      nram_tw, section_num, butterfly_num, para_large_butterfly, in_stride,
      radix);
}
