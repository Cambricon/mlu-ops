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

#define FFT_SWAP_VALUE(X, Y) \
  do {                       \
    int temp = (X);          \
    (X) = (Y);               \
    (Y) = temp;              \
  } while (0)

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
