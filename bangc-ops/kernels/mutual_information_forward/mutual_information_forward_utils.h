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
#ifndef KERNELS_MUTUAL_INFORMATION_FORWARD_MUTUAL_INFORMATION_FORWARD_UTILS_H_
#define KERNELS_MUTUAL_INFORMATION_FORWARD_MUTUAL_INFORMATION_FORWARD_UTILS_H_

#include "mlu_op.h"

#define MIN_LOG_DIFF_FLOAT -15.9423847198486328125f

__nram__ char nram_buffer[MAX_NRAM_SIZE];

__mlu_func__ void logAddVector(float *dst, float *src1, float *src2,
                               float *max_value, float *mask, float *temp,
                               int data_num) {
  __bang_nan_minimum(dst, src1, src2, data_num);
  __bang_maximum(max_value, src1, src2, data_num);

  // If src1 is nan, then max_value = src1 = nan
  // use band with exp and mantissa bits, then compare ge with 0x7f800001
  __asm__ volatile(
      "fuse.nram.s32 [%[dst]], %[size], [%[src0]],"
      ".and(%[src1]), .ge(%[src2]), .mul(%[src3]),"
      ".and([%[src4]]);\n" :: [dst] "r"((int32_t *)temp),
      [ size ] "r"(data_num),
      [ src0 ] "r"((int32_t *)src1),
      [ src1 ] "r"(0x7fffffff),
      [ src2 ] "r"(0x7f800001),
      [ src3 ] "r"(-1),
      [ src4 ] "r"((int32_t *)src1));
  __bang_add(max_value, max_value, temp, data_num);

  // Compute log sum exp: max_value + log1p(exp(min_value - max_value))
  __bang_sub(dst, dst, max_value, data_num);  // min_value - max_value
  __bang_ge_scalar(mask, dst, MIN_LOG_DIFF_FLOAT, data_num);
  __mluop_exp(dst, dst, nullptr, 0, data_num);
  __bang_add_scalar(dst, dst, 1.f, data_num);
  __mluop_log(dst, dst, nullptr, 0, data_num);
  __bang_add(dst, dst, max_value, data_num);

  // If min_value - max_value < MIN_LOG_DIFF_FLOAT, return the larger one
  // mask eq with 0x3f800000(float32(1.0)), -> 0xffffffff
  __asm__ volatile(
      "fuse.nram.s32 [%[dst]], %[size], [%[src0]],"
      ".eq(%[src1]), .mul(%[src2]);\n" :: [dst] "r"((int32_t *)mask),
      [ size ] "r"(data_num),
      [ src0 ] "r"((int32_t *)mask),
      [ src1 ] "r"(0x3f800000),
      [ src2 ] "r"(-1));
  __bang_band((char *)dst, (char *)dst, (char *)mask, data_num * sizeof(float));

  // Reverse the mask bits, ((int)mask+1)*(-1), 0->-1, -1->0
  __bang_fusion(FUSION_FAM, (int *)mask, (int *)mask, 1, -1, data_num);
  __bang_band((char *)max_value, (char *)max_value, (char *)mask,
              data_num * sizeof(float));
  __bang_add(dst, dst, max_value, data_num);
}

#endif  // KERNELS_MUTUAL_INFORMATION_FORWARD_MUTUAL_INFORMATION_FORWARD_UTILS_H_  // NOLINT
