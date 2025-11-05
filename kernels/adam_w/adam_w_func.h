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
#ifndef KERNELS_ADAM_W_ADAM_W_FUNC_H_
#define KERNELS_ADAM_W_ADAM_W_FUNC_H_

#include <mlu.h>
#include <bang_fusor.h>
#include <limits>
template <typename SrcT>
using bang_fusor = bang::experimental::fusor<SrcT>;

#include "kernels/debug.h"
#include "kernels/kernel.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define SIZE_NRAM_PER_REGION PAD_DOWN((MAX_NRAM_SIZE / 12), NFU_ALIGN_SIZE)
#define HIGH_PRECISION_MODE 1

__nram__ int8_t nbuf_head[MAX_NRAM_SIZE];

// The place of nan/inf wil filled by 1, others will be filled by 0.
template <typename T>
__mlu_func__ void get_nan_inf_mask(T *dst, const T *src, size_t size) {
  __asm__ volatile(
      "fuse.nram.u32 [%[dst]], %[size], [%[src0]],"
      ".and(%[src1]),"
      ".ge(%[src2]);\n" ::[dst] "r"((uint32_t *)dst),
      [size] "r"(size), [src0] "r"((uint32_t *)src), [src1] "i"(0x7fffffff), [src2] "i"(0x7f800000));
}

// for AdamW
template <typename T>
__mlu_func__ void loadData(T *nbuf_paramh, T *nbuf_grad, float *nbuf_param,
                           float *nbuf_momentum, float *nbuf_velocity,
                           T *ddr_paramh, T *ddr_grad, float *ddr_param,
                           float *ddr_momentum, float *ddr_velocity,
                           const int data_num, const int offset) {
  __memcpy_async(nbuf_paramh + offset, ddr_paramh, data_num * sizeof(T),
                  GDRAM2NRAM);
  __memcpy_async(nbuf_param + offset, ddr_param, data_num * sizeof(float),
                  GDRAM2NRAM);
  __memcpy_async(nbuf_grad + offset * 2, ddr_grad, data_num * sizeof(T),
                 GDRAM2NRAM);
  __memcpy_async(nbuf_momentum + offset, ddr_momentum, data_num * sizeof(float),
                 GDRAM2NRAM);
  __memcpy_async(nbuf_velocity + offset, ddr_velocity, data_num * sizeof(float),
                 GDRAM2NRAM);
}

// for AdamWRemoveNan
__mlu_func__ void loadData(float *nbuf_grad, float *nbuf_param,
                           float *nbuf_momentum, float *nbuf_velocity,
                           float *ddr_grad, float *ddr_param,
                           float *ddr_momentum, float *ddr_velocity,
                           const int data_num, const int offset) {
  __memcpy_async(nbuf_param + offset, ddr_param, data_num * sizeof(float),
                  GDRAM2NRAM);
  __memcpy_async(nbuf_grad + offset, ddr_grad, data_num * sizeof(float),
                 GDRAM2NRAM);
  __memcpy_async(nbuf_momentum + offset, ddr_momentum, data_num * sizeof(float),
                 GDRAM2NRAM);
  __memcpy_async(nbuf_velocity + offset, ddr_velocity, data_num * sizeof(float),
                 GDRAM2NRAM);
}

// for AdamW
template <typename T>
__mlu_func__ void storeData(T *ddr_paramh, float *ddr_param,
                            float *ddr_momentum, float *ddr_velocity,
                            T *nbuf_paramh, float *nbuf_param,
                            float *nbuf_momentum, float *nbuf_velocity,
                            const int data_num, const int offset) {
  __memcpy_async(ddr_paramh, nbuf_paramh + offset, data_num * sizeof(T),
                  NRAM2GDRAM);
  __memcpy_async(ddr_param, nbuf_param + offset, data_num * sizeof(float),
                  NRAM2GDRAM);
  __memcpy_async(ddr_momentum, nbuf_momentum + offset, data_num * sizeof(float),
                 NRAM2GDRAM);
  __memcpy_async(ddr_velocity, nbuf_velocity + offset, data_num * sizeof(float),
                 NRAM2GDRAM);
}

// for AdamWRemoveNan
__mlu_func__ void storeData(float *ddr_param,
                            float *ddr_momentum, float *ddr_velocity,
                            float *nbuf_param,
                            float *nbuf_momentum, float *nbuf_velocity,
                            const int data_num, const int offset) {
  __memcpy_async(ddr_param, nbuf_param + offset, data_num * sizeof(float),
                  NRAM2GDRAM);
  __memcpy_async(ddr_momentum, nbuf_momentum + offset, data_num * sizeof(float),
                 NRAM2GDRAM);
  __memcpy_async(ddr_velocity, nbuf_velocity + offset, data_num * sizeof(float),
                 NRAM2GDRAM);
}

#endif  // KERNELS_ADAM_W_ADAM_W_FUNC_H_
