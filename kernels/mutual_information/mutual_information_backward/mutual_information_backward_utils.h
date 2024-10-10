/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
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
#ifndef KERNELS_MUTUAL_INFORMATION_BACKWARD_MUTUAL_INFORMATION_BACKWARD_UTILS_H_
#define KERNELS_MUTUAL_INFORMATION_BACKWARD_MUTUAL_INFORMATION_BACKWARD_UTILS_H_

#include "mlu_op.h"

__nram__ int8_t nram_buffer[MAX_NRAM_SIZE];

__mlu_func__ void setNanInfToZero(float *src, float *mask, const int num) {
  // band with 0x7F800000, exp bits are not all 1, mask -> 0xffffffff
  __asm__ volatile(
      "fuse.nram.s32 [%[dst]], %[size], [%[src0]],"
      ".and(%[src1]), .ne(%[src2]), .mul(%[src3]);\n" ::[dst] "r"(
          (int32_t *)mask),
      [ size ] "r"(num), [ src0 ] "r"((int32_t *)src), [ src1 ] "r"(0x7f800000),
      [ src2 ] "r"(0x7f800000), [ src3 ] "r"(-1));
  __bang_band((int8_t *)src, (int8_t *)src, (int8_t *)mask,
              num * sizeof(float));
}

__mlu_func__ void safeExp(float *dst, float *src, float *mask, const int num) {
  setNanInfToZero(src, mask, num);
  __mluop_exp(dst, src, NULL, 0, num);
  // erase exp(0) to 0 with mask
  __bang_band((int8_t *)dst, (int8_t *)dst, (int8_t *)mask,
              num * sizeof(float));
  setNanInfToZero(dst, mask, num);
}

#endif  // KERNELS_MUTUAL_INFORMATION_BACKWARD_MUTUAL_INFORMATION_BACKWARD_UTILS_H_  // NOLINT
