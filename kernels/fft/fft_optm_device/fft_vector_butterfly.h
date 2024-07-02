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

template <typename DT>
__mlu_func__ void computeRadix3ButterflyFirststage(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_in_i,
    DT *nram_scratch, int section_num, int butterfly_num, int in_stride,
    int dir) {
  // outplace(nram)

  const int sign = (dir == FFT_FORWARD) ? 1 : -1;
  const DT tw3_1i = sign * TW3_1I_F;

  FFT_CPX_T<DT> scratch[4];
  int nram_scratch_offset = 0;
  for (int i = 0; i < 4; i++) {
    scratch[i].r = &nram_scratch[nram_scratch_offset];
    nram_scratch_offset += butterfly_num;
    scratch[i].i = &nram_scratch[nram_scratch_offset];
    nram_scratch_offset += butterfly_num;
  }

  FFT_CPX_T<DT> Fin[3];
  int nram_in_offset = 0;
  for (int i = 0; i < 3; i++) {
    Fin[i].r = &nram_in_r[nram_in_offset];
    Fin[i].i = &nram_in_i[nram_in_offset];
    nram_in_offset += butterfly_num;
  }

  FFT_CPX_T<DT> Fout[3];

  // seperate the space for: Fout[i].r  Fout[i].i
  for (int i = 0; i < 3; i++) {
    Fout[i].r = &nram_scratch[nram_scratch_offset];
    Fout[i].i = &nram_scratch[nram_scratch_offset + butterfly_num * 3];
    nram_scratch_offset += butterfly_num;
  }
  nram_scratch_offset += (butterfly_num * 3);

  // __sync();
  FFT_CPX_T<DT> _A = {&nram_scratch[nram_scratch_offset],
                      &nram_scratch[nram_scratch_offset + butterfly_num]};
  FFT_CPX_T<DT> _B = {&nram_scratch[nram_scratch_offset + butterfly_num * 2],
                      &nram_scratch[nram_scratch_offset + butterfly_num * 3]};

  nram_scratch_offset += (butterfly_num * 6);

  MLU_CPX_ADD(scratch[0], Fin[1], Fin[2], butterfly_num);
  MLU_CPX_SUB(scratch[1], Fin[1], Fin[2], butterfly_num);

  MLU_CPX_ADD(Fout[0], scratch[0], Fin[0], butterfly_num);

  MLU_CPX_MLA_OUTPLACE(_A, Fin[0], scratch[0], TW3_1R_F, butterfly_num);
  MLU_CPX_MUL_S(_B, scratch[1], tw3_1i, butterfly_num);
  MLU_CPX_ODD_OUT(Fout[1], Fout[2], _A, _B, butterfly_num);

  // output

  __bang_transpose(nram_out_r, Fout[0].r, 3, butterfly_num);
  __bang_transpose(nram_out_i, Fout[0].i, 3, butterfly_num);
}

template <typename DT>
__mlu_func__ void computeRadix3ButterflyOtherstages(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_in_i,
    DT *nram_scratch, DT *nram_tw, int section_num, int butterfly_num,
    int in_stride, int dir) {
  const int sign = (dir == FFT_FORWARD) ? 1 : -1;
  const DT tw3_1i = sign * TW3_1I_F;

  int Fin_stride = 0, Fout_stride = 0;
  int sec_count;

  DT *scratch_r[4] = {nram_scratch, &nram_scratch[butterfly_num],
                      &nram_scratch[butterfly_num * 2],
                      &nram_scratch[butterfly_num * 3]};
  DT *scratch_i[4] = {
      &nram_scratch[butterfly_num * 4], &nram_scratch[butterfly_num * 5],
      &nram_scratch[butterfly_num * 6], &nram_scratch[butterfly_num * 7]};

  DT *_A_r = &nram_scratch[butterfly_num * 8];
  DT *_A_i = &nram_scratch[butterfly_num * 9];
  DT *_B_r = &nram_scratch[butterfly_num * 10];
  DT *_B_i = &nram_scratch[butterfly_num * 11];

  DT *scratch_tw_r = nram_tw;  // (3-1)*butterfly_num
  DT *scratch_tw_i = &nram_tw[butterfly_num * (3 - 1)];

  DT *CPX_MUL_RR = &nram_scratch[butterfly_num * 16];  // (3-1)*butterfly_num
  DT *CPX_MUL_RI = &nram_scratch[butterfly_num * 18];  // (3-1)*butterfly_num
  DT *CPX_MUL_IR = &nram_scratch[butterfly_num * 20];  // (3-1)*butterfly_num
  DT *CPX_MUL_II = &nram_scratch[butterfly_num * 22];  // (3-1)*butterfly_num

  DT *scratch_in_r;
  DT *scratch_in_i;

  for (sec_count = 0; sec_count < section_num; ++sec_count) {
    DT *Fout_r[3] = {&nram_out_r[Fout_stride],
                     &nram_out_r[butterfly_num + Fout_stride],
                     &nram_out_r[butterfly_num * 2 + Fout_stride]};
    DT *Fout_i[3] = {&nram_out_i[Fout_stride],
                     &nram_out_i[butterfly_num + Fout_stride],
                     &nram_out_i[butterfly_num * 2 + Fout_stride]};

    if (section_num == 1) {
      scratch_in_r = nram_in_r;
      scratch_in_i = nram_in_i;

    } else {
      // nram_scratch
      scratch_in_r = &nram_scratch[butterfly_num * 24];
      scratch_in_i = &nram_scratch[butterfly_num * (24 + 3)];
      __memcpy(scratch_in_r, nram_in_r + Fin_stride, sizeof(DT) * butterfly_num,
               NRAM2NRAM, sizeof(DT) * butterfly_num, in_stride * sizeof(DT),
               3 - 1);
      __memcpy(scratch_in_i, nram_in_i + Fin_stride, sizeof(DT) * butterfly_num,
               NRAM2NRAM, sizeof(DT) * butterfly_num, in_stride * sizeof(DT),
               3 - 1);
    }

    DT *Fin_r[3] = {scratch_in_r, &scratch_in_r[butterfly_num],
                    &scratch_in_r[butterfly_num * 2]};
    DT *Fin_i[3] = {scratch_in_i, &scratch_in_i[butterfly_num],
                    &scratch_in_i[butterfly_num * 2]};

    // rotate
    __bang_mul(CPX_MUL_RR, Fin_r[1], scratch_tw_r, butterfly_num * (3 - 1));
    __bang_mul(CPX_MUL_II, Fin_i[1], scratch_tw_i, butterfly_num * (3 - 1));
    __bang_mul(CPX_MUL_RI, Fin_r[1], scratch_tw_i, butterfly_num * (3 - 1));
    __bang_mul(CPX_MUL_IR, Fin_i[1], scratch_tw_r, butterfly_num * (3 - 1));

    __bang_sub(Fin_r[1], CPX_MUL_RR, CPX_MUL_II, butterfly_num * (3 - 1));
    __bang_add(Fin_i[1], CPX_MUL_RI, CPX_MUL_IR, butterfly_num * (3 - 1));

    // butterfly compute

    // CPX_ADD
    __bang_add(scratch_r[0], Fin_r[1], Fin_r[2], butterfly_num);
    __bang_add(scratch_i[0], Fin_i[1], Fin_i[2], butterfly_num);

    // CPX_SUB
    __bang_sub(scratch_r[1], Fin_r[1], Fin_r[2], butterfly_num);
    __bang_sub(scratch_i[1], Fin_i[1], Fin_i[2], butterfly_num);

    // CPX_ADD
    __bang_add(Fout_r[0], scratch_r[0], Fin_r[0], butterfly_num);
    __bang_add(Fout_i[0], scratch_i[0], Fin_i[0], butterfly_num);

    // CPX_MLA_OUTPLACE
    __bang_fusion(FUSION_FMA, _A_r, scratch_r[0], TW3_1R_F, Fin_r[0],
                  butterfly_num, butterfly_num);
    __bang_fusion(FUSION_FMA, _A_i, scratch_i[0], TW3_1R_F, Fin_i[0],
                  butterfly_num, butterfly_num);

    // CPX_MUL_S
    __bang_mul_scalar(_B_r, scratch_r[1], tw3_1i, butterfly_num);
    __bang_mul_scalar(_B_i, scratch_i[1], tw3_1i, butterfly_num);

    // OPENFFT_CPX_ODD_OUT
    __bang_sub(Fout_r[1], _A_r, _B_i, butterfly_num);
    __bang_add(Fout_i[1], _A_i, _B_r, butterfly_num);
    __bang_add(Fout_r[2], _A_r, _B_i, butterfly_num);
    __bang_sub(Fout_i[2], _A_i, _B_r, butterfly_num);

    Fin_stride += butterfly_num;
    Fout_stride += 3 * butterfly_num;
  }
}

template <typename DT>
__mlu_func__ void computeRadix3ButterflyLaststage(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_in_i,
    DT *nram_scratch, DT *nram_tw, int section_num, int butterfly_num,
    int in_stride, int dir) {
  computeRadix3ButterflyOtherstages(nram_out_r, nram_out_i, nram_in_r,
                                    nram_in_i, nram_scratch, nram_tw,
                                    section_num, butterfly_num, in_stride, dir);
}

template <typename DT>
__mlu_func__ void computeRadix4ButterflyFirststage(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_in_i,
    DT *nram_scratch, int section_num, int butterfly_num, int in_stride,
    int dir) {
  // outplace(nram)

  // const int sign = (dir == FFT_FORWARD) ? 1 : -1;

  FFT_CPX_T<DT> scratch01[2];
  int nram_scratch_offset = 0;
  for (int i = 0; i < 2; i++) {
    scratch01[i].r = &nram_scratch[nram_scratch_offset];
    nram_scratch_offset += butterfly_num;
    scratch01[i].i = &nram_scratch[nram_scratch_offset];
    nram_scratch_offset += butterfly_num;
  }

  FFT_CPX_T<DT> scratch1[2];

  for (int i = 0; i < 2; i++) {
    scratch1[i].r = &nram_scratch[nram_scratch_offset];
    nram_scratch_offset += butterfly_num;
    scratch1[i].i = &nram_scratch[nram_scratch_offset];
    nram_scratch_offset += butterfly_num;
  }

  FFT_CPX_T<DT> Fin[4];
  int nram_in_offset = 0;
  for (int i = 0; i < 4; i++) {
    Fin[i].r = &nram_in_r[nram_in_offset];
    Fin[i].i = &nram_in_i[nram_in_offset];
    nram_in_offset += butterfly_num;
  }

  FFT_CPX_T<DT> Fout[4];

  // seperate the space for: Fout[i].r  Fout[i].i
  for (int i = 0; i < 4; i++) {
    Fout[i].r = &nram_scratch[nram_scratch_offset];
    Fout[i].i = &nram_scratch[nram_scratch_offset + butterfly_num * 4];
    nram_scratch_offset += butterfly_num;
  }
  nram_scratch_offset += (butterfly_num * 4);

  MLU_CPX_ADD(scratch01[0], Fin[0], Fin[2], butterfly_num);
  MLU_CPX_SUB(scratch01[1], Fin[0], Fin[2], butterfly_num);

  MLU_CPX_ADD(scratch1[0], Fin[1], Fin[3], butterfly_num);
  MLU_CPX_SUB(scratch1[1], Fin[1], Fin[3], butterfly_num);

  // 1
  // 1/3
  MLU_CPX_ADD_NEG_I(Fout[1], scratch01[1], scratch1[1], butterfly_num);
  MLU_CPX_ADD_I(Fout[3], scratch01[1], scratch1[1], butterfly_num);

  // 0
  // 0/2
  MLU_CPX_ADD(Fout[0], scratch01[0], scratch1[0], butterfly_num);
  MLU_CPX_SUB(Fout[2], scratch01[0], scratch1[0], butterfly_num);

  // output

  __bang_transpose(nram_out_r, Fout[0].r, 4, butterfly_num);
  __bang_transpose(nram_out_i, Fout[0].i, 4, butterfly_num);
}

template <typename DT>
__mlu_func__ void computeRadix9ButterflyFirststage(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_in_i,
    DT *nram_scratch, int section_num, int butterfly_num, int in_stride,
    int dir) {
  // outplace(nram)

  const int sign = (dir == FFT_FORWARD) ? 1 : -1;
  const DT tw9_1i = sign * TW9_1I_F;
  const DT tw9_2i = sign * TW9_2I_F;
  const DT tw9_3i = sign * TW9_3I_F;
  const DT tw9_4i = sign * TW9_4I_F;

  FFT_CPX_T<DT> scratch[10];
  int nram_scratch_offset = 0;
  for (int i = 0; i < 10; i++) {
    scratch[i].r = &nram_scratch[nram_scratch_offset];
    nram_scratch_offset += butterfly_num;
    scratch[i].i = &nram_scratch[nram_scratch_offset];
    nram_scratch_offset += butterfly_num;
  }

  FFT_CPX_T<DT> Fin[9];
  int nram_in_offset = 0;
  for (int i = 0; i < 9; i++) {
    Fin[i].r = &nram_in_r[nram_in_offset];
    Fin[i].i = &nram_in_i[nram_in_offset];
    nram_in_offset += butterfly_num;
  }

  FFT_CPX_T<DT> Fout[9];

  // seperate the space for: Fout[i].r  Fout[i].i
  for (int i = 0; i < 9; i++) {
    Fout[i].r = &nram_scratch[nram_scratch_offset];
    Fout[i].i = &nram_scratch[nram_scratch_offset + butterfly_num * 9];
    nram_scratch_offset += butterfly_num;
  }
  nram_scratch_offset += (butterfly_num * 9);

  FFT_CPX_T<DT> _A = {&nram_scratch[nram_scratch_offset],
                      &nram_scratch[nram_scratch_offset + butterfly_num]};
  FFT_CPX_T<DT> _B = {&nram_scratch[nram_scratch_offset + butterfly_num * 2],
                      &nram_scratch[nram_scratch_offset + butterfly_num * 3]};

  FFT_CPX_T<DT> in_0 = {&nram_scratch[nram_scratch_offset + butterfly_num * 4],
                        &nram_scratch[nram_scratch_offset + butterfly_num * 5]};
  nram_scratch_offset += (butterfly_num * 6);

  FFT_CPX_T<DT> _A_TEMP = {&nram_scratch[nram_scratch_offset],
                           &nram_scratch[nram_scratch_offset + butterfly_num]};
  FFT_CPX_T<DT> _B_TEMP = {
      &nram_scratch[nram_scratch_offset + butterfly_num * 2],
      &nram_scratch[nram_scratch_offset + butterfly_num * 3]};

  __bang_move(in_0.r, Fin[0].r, butterfly_num * sizeof(DT));
  __bang_move(in_0.i, Fin[0].i, butterfly_num * sizeof(DT));

  MLU_CPX_ADD(scratch[0], Fin[1], Fin[8], butterfly_num);
  MLU_CPX_SUB(scratch[1], Fin[1], Fin[8], butterfly_num);
  MLU_CPX_ADD(scratch[2], Fin[2], Fin[7], butterfly_num);
  MLU_CPX_SUB(scratch[3], Fin[2], Fin[7], butterfly_num);
  MLU_CPX_ADD(scratch[4], Fin[3], Fin[6], butterfly_num);
  MLU_CPX_SUB(scratch[5], Fin[3], Fin[6], butterfly_num);
  MLU_CPX_ADD(scratch[6], Fin[4], Fin[5], butterfly_num);
  MLU_CPX_SUB(scratch[7], Fin[4], Fin[5], butterfly_num);

  MLU_CPX_ADD(scratch[9], scratch[0], scratch[2], butterfly_num);
  MLU_CPX_ADD(scratch[9], scratch[4], scratch[9], butterfly_num);
  MLU_CPX_ADD(scratch[9], scratch[6], scratch[9], butterfly_num);
  MLU_CPX_ADD(Fout[0], scratch[9], in_0, butterfly_num);

  MLU_CPX_MLA_OUTPLACE(_A, in_0, scratch[0], TW9_1R_F, butterfly_num);
  MLU_CPX_MUL_S(_B, scratch[1], tw9_1i, butterfly_num);
  MLU_CPX_MLA_INPLACE(_A, scratch[2], TW9_2R_F, _A_TEMP, butterfly_num);
  MLU_CPX_MLA_INPLACE(_B, scratch[3], tw9_2i, _B_TEMP, butterfly_num);
  MLU_CPX_MLA_INPLACE(_A, scratch[4], TW9_3R_F, _A_TEMP, butterfly_num);
  MLU_CPX_MLA_INPLACE(_B, scratch[5], tw9_3i, _B_TEMP, butterfly_num);
  MLU_CPX_MLA_INPLACE(_A, scratch[6], TW9_4R_F, _A_TEMP, butterfly_num);
  MLU_CPX_MLA_INPLACE(_B, scratch[7], tw9_4i, _B_TEMP, butterfly_num);
  MLU_CPX_ODD_OUT(Fout[1], Fout[8], _A, _B, butterfly_num);

  MLU_CPX_MLA_OUTPLACE(_A, in_0, scratch[0], TW9_2R_F, butterfly_num);
  MLU_CPX_MUL_S(_B, scratch[1], tw9_2i, butterfly_num);
  MLU_CPX_MLA_INPLACE(_A, scratch[2], TW9_4R_F, _A_TEMP, butterfly_num);
  MLU_CPX_MLA_INPLACE(_B, scratch[3], tw9_4i, _B_TEMP, butterfly_num);
  MLU_CPX_MLA_INPLACE(_A, scratch[4], TW9_3R_F, _A_TEMP, butterfly_num);
  MLU_CPX_MLA_INPLACE(_B, scratch[5], -tw9_3i, _B_TEMP, butterfly_num);
  MLU_CPX_MLA_INPLACE(_A, scratch[6], TW9_1R_F, _A_TEMP, butterfly_num);
  MLU_CPX_MLA_INPLACE(_B, scratch[7], -tw9_1i, _B_TEMP, butterfly_num);
  MLU_CPX_ODD_OUT(Fout[2], Fout[7], _A, _B, butterfly_num);

  MLU_CPX_ADD(_A, in_0, scratch[4], butterfly_num);
  MLU_CPX_ADD(_B, scratch[0], scratch[2], butterfly_num);
  MLU_CPX_ADD(_B, scratch[6], _B, butterfly_num);
  MLU_CPX_MLA_OUTPLACE(_A, _A, _B, TW9_3R_F, butterfly_num);
  MLU_CPX_SUB(_B, scratch[1], scratch[3], butterfly_num);
  MLU_CPX_ADD(_B, scratch[7], _B, butterfly_num);
  MLU_CPX_MUL_S(_B, _B, tw9_3i, butterfly_num);
  MLU_CPX_ODD_OUT(Fout[3], Fout[6], _A, _B, butterfly_num);

  MLU_CPX_MLA_OUTPLACE(_A, in_0, scratch[0], TW9_4R_F, butterfly_num);
  MLU_CPX_MUL_S(_B, scratch[1], tw9_4i, butterfly_num);
  MLU_CPX_MLA_INPLACE(_A, scratch[2], TW9_1R_F, _A_TEMP, butterfly_num);
  MLU_CPX_MLA_INPLACE(_B, scratch[3], -tw9_1i, _B_TEMP, butterfly_num);
  MLU_CPX_MLA_INPLACE(_A, scratch[4], TW9_3R_F, _A_TEMP, butterfly_num);
  MLU_CPX_MLA_INPLACE(_B, scratch[5], tw9_3i, _B_TEMP, butterfly_num);
  MLU_CPX_MLA_INPLACE(_A, scratch[6], TW9_2R_F, _A_TEMP, butterfly_num);
  MLU_CPX_MLA_INPLACE(_B, scratch[7], -tw9_2i, _B_TEMP, butterfly_num);
  MLU_CPX_ODD_OUT(Fout[4], Fout[5], _A, _B, butterfly_num);

  // output
  __bang_transpose(nram_out_r, Fout[0].r, 9, butterfly_num);
  __bang_transpose(nram_out_i, Fout[0].i, 9, butterfly_num);
}

template <typename DT>
__mlu_func__ void computeRadix9ButterflyOtherstages(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_in_i,
    DT *nram_scratch, DT *nram_tw, int section_num, int butterfly_num,
    int in_stride, int dir) {
  const int radix = 9;
  const int sign = (dir == FFT_FORWARD) ? 1 : -1;
  const DT tw9_1i = sign * TW9_1I_F;
  const DT tw9_2i = sign * TW9_2I_F;
  const DT tw9_3i = sign * TW9_3I_F;
  const DT tw9_4i = sign * TW9_4I_F;

  // const int out_stride = butterfly_num;
  int Fin_stride = 0, Fout_stride = 0;
  int sec_count;

  FFT_CPX_T<DT> scratch[10];
  int nram_scratch_offset = 0;
  for (int i = 0; i < 10; i++) {
    scratch[i].r = &nram_scratch[nram_scratch_offset];
    nram_scratch_offset += butterfly_num;
    scratch[i].i = &nram_scratch[nram_scratch_offset];
    nram_scratch_offset += butterfly_num;
  }

  FFT_CPX_T<DT> _A = {&nram_scratch[nram_scratch_offset],
                      &nram_scratch[nram_scratch_offset + butterfly_num]};
  FFT_CPX_T<DT> _B = {&nram_scratch[nram_scratch_offset + butterfly_num * 2],
                      &nram_scratch[nram_scratch_offset + butterfly_num * 3]};

  FFT_CPX_T<DT> in_0 = {&nram_scratch[nram_scratch_offset + butterfly_num * 4],
                        &nram_scratch[nram_scratch_offset + butterfly_num * 5]};
  nram_scratch_offset += (butterfly_num * 6);

  FFT_CPX_T<DT> _A_TEMP = {&nram_scratch[nram_scratch_offset],
                           &nram_scratch[nram_scratch_offset + butterfly_num]};
  FFT_CPX_T<DT> _B_TEMP = {
      &nram_scratch[nram_scratch_offset + butterfly_num * 2],
      &nram_scratch[nram_scratch_offset + butterfly_num * 3]};
  nram_scratch_offset += (butterfly_num * 4);
  FFT_CPX_T<DT> scratch_tw = {nram_tw, &nram_tw[butterfly_num * (radix - 1)]};

  DT *RR = &nram_scratch[nram_scratch_offset];  // (9-1)*butterfly_num
  DT *RI = &nram_scratch[nram_scratch_offset +
                         butterfly_num * (radix - 1)];  // (9-1)*butterfly_num
  DT *IR = &nram_scratch[nram_scratch_offset + butterfly_num * (radix - 1) *
                                                   2];  // (9-1)*butterfly_num
  DT *II = &nram_scratch[nram_scratch_offset + butterfly_num * (radix - 1) *
                                                   3];  // (9-1)*butterfly_num
  nram_scratch_offset += (butterfly_num * 4 * (radix - 1));

  FFT_CPX_T<DT> scratch_in;
  FFT_CPX_T<DT> Fin[radix];
  FFT_CPX_T<DT> Fout[radix];

  for (sec_count = 0; sec_count < section_num; ++sec_count) {
    for (int i = 0; i < radix; i++) {
      Fout[i].r = &nram_out_r[Fout_stride + butterfly_num * i];
      Fout[i].i = &nram_out_i[Fout_stride + butterfly_num * i];
    }

    if (section_num == 1) {
      scratch_in = FFT_CPX_T<DT>{nram_in_r, nram_in_i};
    } else {
      // nram_scratch
      scratch_in.r = &nram_scratch[nram_scratch_offset];
      scratch_in.i = &nram_scratch[nram_scratch_offset + radix * butterfly_num];
      __memcpy(scratch_in.r, nram_in_r + Fin_stride, sizeof(DT) * butterfly_num,
               NRAM2NRAM, sizeof(DT) * butterfly_num, in_stride * sizeof(DT),
               radix - 1);
      __memcpy(scratch_in.i, nram_in_i + Fin_stride, sizeof(DT) * butterfly_num,
               NRAM2NRAM, sizeof(DT) * butterfly_num, in_stride * sizeof(DT),
               radix - 1);
    }

    int nram_in_offset = 0;
    for (int i = 0; i < radix; i++) {
      Fin[i].r = &scratch_in.r[nram_in_offset];
      Fin[i].i = &scratch_in.i[nram_in_offset];
      nram_in_offset += butterfly_num;
    }

    // rotate
    MLU_CPX_MUL(Fin[1], Fin[1], scratch_tw, RR, II, RI, IR,
                butterfly_num * (radix - 1));

    // butterfly compute

    __bang_move(in_0.r, Fin[0].r, butterfly_num * sizeof(DT));
    __bang_move(in_0.i, Fin[0].i, butterfly_num * sizeof(DT));

    MLU_CPX_ADD(scratch[0], Fin[1], Fin[8], butterfly_num);
    MLU_CPX_SUB(scratch[1], Fin[1], Fin[8], butterfly_num);
    MLU_CPX_ADD(scratch[2], Fin[2], Fin[7], butterfly_num);
    MLU_CPX_SUB(scratch[3], Fin[2], Fin[7], butterfly_num);
    MLU_CPX_ADD(scratch[4], Fin[3], Fin[6], butterfly_num);
    MLU_CPX_SUB(scratch[5], Fin[3], Fin[6], butterfly_num);
    MLU_CPX_ADD(scratch[6], Fin[4], Fin[5], butterfly_num);
    MLU_CPX_SUB(scratch[7], Fin[4], Fin[5], butterfly_num);

    MLU_CPX_ADD(scratch[9], scratch[0], scratch[2], butterfly_num);
    MLU_CPX_ADD(scratch[9], scratch[4], scratch[9], butterfly_num);
    MLU_CPX_ADD(scratch[9], scratch[6], scratch[9], butterfly_num);
    MLU_CPX_ADD(Fout[0], scratch[9], in_0, butterfly_num);

    MLU_CPX_MLA_OUTPLACE(_A, in_0, scratch[0], TW9_1R_F, butterfly_num);
    MLU_CPX_MUL_S(_B, scratch[1], tw9_1i, butterfly_num);
    MLU_CPX_MLA_INPLACE(_A, scratch[2], TW9_2R_F, _A_TEMP, butterfly_num);
    MLU_CPX_MLA_INPLACE(_B, scratch[3], tw9_2i, _B_TEMP, butterfly_num);
    MLU_CPX_MLA_INPLACE(_A, scratch[4], TW9_3R_F, _A_TEMP, butterfly_num);
    MLU_CPX_MLA_INPLACE(_B, scratch[5], tw9_3i, _B_TEMP, butterfly_num);
    MLU_CPX_MLA_INPLACE(_A, scratch[6], TW9_4R_F, _A_TEMP, butterfly_num);
    MLU_CPX_MLA_INPLACE(_B, scratch[7], tw9_4i, _B_TEMP, butterfly_num);
    MLU_CPX_ODD_OUT(Fout[1], Fout[8], _A, _B, butterfly_num);

    MLU_CPX_MLA_OUTPLACE(_A, in_0, scratch[0], TW9_2R_F, butterfly_num);
    MLU_CPX_MUL_S(_B, scratch[1], tw9_2i, butterfly_num);
    MLU_CPX_MLA_INPLACE(_A, scratch[2], TW9_4R_F, _A_TEMP, butterfly_num);
    MLU_CPX_MLA_INPLACE(_B, scratch[3], tw9_4i, _B_TEMP, butterfly_num);
    MLU_CPX_MLA_INPLACE(_A, scratch[4], TW9_3R_F, _A_TEMP, butterfly_num);
    MLU_CPX_MLA_INPLACE(_B, scratch[5], -tw9_3i, _B_TEMP, butterfly_num);
    MLU_CPX_MLA_INPLACE(_A, scratch[6], TW9_1R_F, _A_TEMP, butterfly_num);
    MLU_CPX_MLA_INPLACE(_B, scratch[7], -tw9_1i, _B_TEMP, butterfly_num);
    MLU_CPX_ODD_OUT(Fout[2], Fout[7], _A, _B, butterfly_num);

    MLU_CPX_ADD(_A, in_0, scratch[4], butterfly_num);
    MLU_CPX_ADD(_B, scratch[0], scratch[2], butterfly_num);
    MLU_CPX_ADD(_B, scratch[6], _B, butterfly_num);
    MLU_CPX_MLA_OUTPLACE(_A, _A, _B, TW9_3R_F, butterfly_num);
    MLU_CPX_SUB(_B, scratch[1], scratch[3], butterfly_num);
    MLU_CPX_ADD(_B, scratch[7], _B, butterfly_num);
    MLU_CPX_MUL_S(_B, _B, tw9_3i, butterfly_num);
    MLU_CPX_ODD_OUT(Fout[3], Fout[6], _A, _B, butterfly_num);

    MLU_CPX_MLA_OUTPLACE(_A, in_0, scratch[0], TW9_4R_F, butterfly_num);
    MLU_CPX_MUL_S(_B, scratch[1], tw9_4i, butterfly_num);
    MLU_CPX_MLA_INPLACE(_A, scratch[2], TW9_1R_F, _A_TEMP, butterfly_num);
    MLU_CPX_MLA_INPLACE(_B, scratch[3], -tw9_1i, _B_TEMP, butterfly_num);
    MLU_CPX_MLA_INPLACE(_A, scratch[4], TW9_3R_F, _A_TEMP, butterfly_num);
    MLU_CPX_MLA_INPLACE(_B, scratch[5], tw9_3i, _B_TEMP, butterfly_num);
    MLU_CPX_MLA_INPLACE(_A, scratch[6], TW9_2R_F, _A_TEMP, butterfly_num);
    MLU_CPX_MLA_INPLACE(_B, scratch[7], -tw9_2i, _B_TEMP, butterfly_num);
    MLU_CPX_ODD_OUT(Fout[4], Fout[5], _A, _B, butterfly_num);

    Fin_stride += butterfly_num;
    Fout_stride += radix * butterfly_num;
  }
}

template <typename DT>
__mlu_func__ void computeRadix9ButterflyLaststage(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_in_i,
    DT *nram_scratch, DT *nram_tw, int section_num, int butterfly_num,
    int in_stride, int dir) {
  computeRadix9ButterflyOtherstages(nram_out_r, nram_out_i, nram_in_r,
                                    nram_in_i, nram_scratch, nram_tw,
                                    section_num, butterfly_num, in_stride, dir);
}
