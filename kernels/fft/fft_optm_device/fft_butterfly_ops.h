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

template <typename DT>
struct FFT_CPX_T {
  DT *r;
  DT *i;
};

#define FFT_SWAP_PTR(X, Y)                     \
  {                                            \
    X = (DT *)((intptr_t)(X) ^ (intptr_t)(Y)); \
    Y = (DT *)((intptr_t)(X) ^ (intptr_t)(Y)); \
    X = (DT *)((intptr_t)(X) ^ (intptr_t)(Y)); \
  }

#define MLU_CPX_ADD(Z, A, B, VL)   \
  {                                \
    __bang_add(Z.r, A.r, B.r, VL); \
    __bang_add(Z.i, A.i, B.i, VL); \
  }

#define MLU_CPX_SUB(Z, A, B, VL)   \
  {                                \
    __bang_sub(Z.r, A.r, B.r, VL); \
    __bang_sub(Z.i, A.i, B.i, VL); \
  }

#define MLU_CPX_MLA_INPLACE(OUT, IN, TWI, TEMP, VL) \
  {                                                 \
    MLU_CPX_MUL_S(TEMP, IN, TWI, VL)                \
    MLU_CPX_ADD(OUT, OUT, TEMP, VL)                 \
  }

#define MLU_CPX_MLA_OUTPLACE(OUT, IN1, IN2, TWR, VL)             \
  {                                                              \
    __bang_fusion(FUSION_FMA, OUT.r, IN2.r, TWR, IN1.r, VL, VL); \
    __bang_fusion(FUSION_FMA, OUT.i, IN2.i, TWR, IN1.i, VL, VL); \
  }

#define MLU_CPX_MUL_S(OUT, IN, TWI, VL)      \
  {                                          \
    __bang_mul_scalar(OUT.r, IN.r, TWI, VL); \
    __bang_mul_scalar(OUT.i, IN.i, TWI, VL); \
  }

#define MLU_CPX_ODD_OUT(HEAD, TAIL, _A, _B, VL) \
  {                                             \
    __bang_sub(HEAD.r, _A.r, _B.i, VL);         \
    __bang_add(HEAD.i, _A.i, _B.r, VL);         \
    __bang_add(TAIL.r, _A.r, _B.i, VL);         \
    __bang_sub(TAIL.i, _A.i, _B.r, VL);         \
  }

#define MLU_CPX_MUL(Z, A, B, RR, II, RI, IR, VL) \
  {                                              \
    __bang_mul(RR, A.r, B.r, VL);                \
    __bang_mul(II, A.i, B.i, VL);                \
    __bang_mul(RI, A.r, B.i, VL);                \
    __bang_mul(IR, A.i, B.r, VL);                \
    __bang_sub(Z.r, RR, II, VL);                 \
    __bang_add(Z.i, RI, IR, VL);                 \
  }

#define MLU_CPX_ADD_NEG_I(Z, A, B, VL) \
  {                                    \
    __bang_add(Z.r, A.r, B.i, VL);     \
    __bang_sub(Z.i, A.i, B.r, VL);     \
  }

#define MLU_CPX_ADD_I(Z, A, B, VL) \
  {                                \
    __bang_sub(Z.r, A.r, B.i, VL); \
    __bang_add(Z.i, A.i, B.r, VL); \
  }

// #define TRANSPOSE_XYZ2YXZ(out, in, X, Y, Z, DT) \
//         {
//           FFT_SWAP_PTR(nram_out_r, nram_in_r);
//           FFT_SWAP_PTR(nram_out_i, nram_in_i);

//           // [X, Y, Z] -> [Y,
//           // X, Z]

//           int src_stride0 = Z * sizeof(DT);
//           int src_segnum1 = Y - 1;
//           int src_stride1 = Y * Z * sizeof(DT);
//           int src_segnum2 = X - 1;

//           int dst_stride0 = X * Z * sizeof(DT);
//           int dst_segnum1 = Y - 1;
//           int dst_stride1 = Z * sizeof(DT);
//           int dst_segnum2 = X - 1;

//           __memcpy(nram_out_r, nram_in_r, sizeof(DT) * Z, NRAM2NRAM,
//                    dst_stride0, dst_segnum1, dst_stride1, dst_segnum2,
//                    src_stride0, src_segnum1, src_stride1, src_segnum2);
//           __memcpy(nram_out_i, nram_in_i, sizeof(DT) * Z, NRAM2NRAM,
//                    dst_stride0, dst_segnum1, dst_stride1, dst_segnum2,
//                    src_stride0, src_segnum1, src_stride1, src_segnum2);
//         }

// // [X, Y, Z] -> [Y, X, Z]
// #define TRANSPOSE_XYZ2YXZ_PAIR(out1, out2, in1, in2, X, Y, Z, DT) \
//   { \
//     int src_stride0 = Z * sizeof(DT); \
//     int src_segnum1 = Y - 1; \
//     int src_stride1 = Y * Z * sizeof(DT); \
//     int src_segnum2 = X - 1; \
//                                                                               \
//     int dst_stride0 = X * Z * sizeof(DT); \
//     int dst_segnum1 = Y - 1; \
//     int dst_stride1 = Z * sizeof(DT); \
//     int dst_segnum2 = X - 1; \
//                                                                               \
//     __memcpy(out1, in1, sizeof(DT) * Z, NRAM2NRAM, dst_stride0, dst_segnum1,
//     \
//              dst_stride1, dst_segnum2, src_stride0, src_segnum1, src_stride1,
//              \
//              src_segnum2); \
//     __memcpy(out2, in2, sizeof(DT) * Z, NRAM2NRAM, dst_stride0, dst_segnum1,
//     \
//              dst_stride1, dst_segnum2, src_stride0, src_segnum1, src_stride1,
//              \
//              src_segnum2); \
//   }

// [X, Y, Z] -> [Y, X, Z]
#define TRANSPOSE_XYZ2YXZ_PAIR(out1, out2, in1, in2, X, Y, Z, DT)          \
  {                                                                        \
    int stride0 = (Z) * sizeof(DT);                                        \
    int segnum1 = (Y)-1;                                                   \
    int src_stride1 = (Y)*stride0;                                         \
    int segnum2 = (X)-1;                                                   \
    int dst_stride0 = (X)*stride0;                                         \
                                                                           \
    __memcpy(out1, in1, stride0, NRAM2NRAM, dst_stride0, segnum1, stride0, \
             segnum2, stride0, segnum1, src_stride1, segnum2);             \
    __memcpy(out2, in2, stride0, NRAM2NRAM, dst_stride0, segnum1, stride0, \
             segnum2, stride0, segnum1, src_stride1, segnum2);             \
  }
