/*************************************************************************
 * Copyright (C) [2022] by Cambricon, Inc.
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
#ifndef KERNELS_UNARY_OP_UNARY_OP_3PIPELINE_H_
#define KERNELS_UNARY_OP_UNARY_OP_3PIPELINE_H_

#include "kernels/kernel.h"
#define UNARY_ALIGN_NUM 64

#define UNARY_OP_KERNEL_3PIPELINE_DECLARE(Op, DType, Prefer)           \
  __mlu_global__ void MLUBlockKernel3StagePipeline##Op##DType##Prefer( \
      void *x, void *y, uint32_t num_total, float coef);

#define UNARY_OP_KERNEL_3PIPELINE_IMPLE(Op, DType, Prefer)                 \
  __mlu_global__ void MLUBlockKernel3StagePipeline##Op##DType##Prefer(     \
      void *x, void *y, uint32_t num_total, float coef) {                  \
    int32_t num_deal = 0, num_pong = 0;                                    \
    int32_t offset_half = 0, offset_aux_a = 0, offset_aux_b = 0;           \
    get3Offset##Op##Prefer<DType>(offset_half, offset_aux_a, offset_aux_b, \
                                  num_deal, num_pong);                     \
    block3Unary<DType, compute##Op##Prefer>(                               \
        (DType *)x, (DType *)y, nram_buffer, num_total, offset_half,       \
        offset_aux_a, offset_aux_b, num_deal, num_pong, coef);             \
  }

template <typename T, void (*OpFunc)(T *, T *, T *, T *, int, int, float)>
__mlu_func__ void block3Unary(T *x, T *y, char *nram_buffer, int32_t num_total,
                              int32_t offset_x_half, int32_t offset_aux_a,
                              int32_t offset_aux_b, int32_t num_deal,
                              int32_t num_pong, float coef) {
  if (coreId == 0x80) {
    return;
  }
  int32_t num_per_core = num_total / taskDim;
  int32_t num_rem = num_total % taskDim;
  T *addr_x = (T *)x + taskId * num_per_core;
  T *addr_y = (T *)y + taskId * num_per_core;
  if (num_rem > 0 && taskId == taskDim - 1) {
    num_per_core = num_per_core + num_rem;
  }
  int32_t repeat = num_per_core / num_deal;
  int32_t rem = num_per_core % num_deal;
  int32_t align_rem = CEIL_ALIGN(rem, UNARY_ALIGN_NUM);

  T *nram_x = (T *)nram_buffer;
  T *nram_x_half = (T *)nram_buffer + offset_x_half;
  T *nram_aux_a = (T *)nram_buffer + offset_aux_a;
  T *nram_aux_b = (T *)nram_buffer + offset_aux_b;
  int32_t span_handle_size = num_deal * sizeof(T);

  // 3 level pipeline.
  if (repeat > 0) {
    __memcpy_async(nram_x_half, addr_x, span_handle_size, GDRAM2NRAM);
    __asm__ volatile("sync;");
  }

  if (repeat > 1) {
    __memcpy_async(nram_x_half + num_pong, addr_x + num_deal, span_handle_size,
                   GDRAM2NRAM);
    OpFunc(nram_x, nram_x_half, nram_aux_a, nram_aux_b, num_deal, num_deal,
           coef);
    __asm__ volatile("sync;");
  }

  for (int i = 0; i < repeat - 2; i++) {
    pvLock();
    __memcpy_async(addr_y + i * num_deal, nram_x + (i % 2) * num_pong,
                   span_handle_size, NRAM2GDRAM);
    pvUnlock();

    __memcpy_async(nram_x_half + (i % 2) * num_pong,
                   addr_x + (i + 2) * num_deal, span_handle_size, GDRAM2NRAM);
    OpFunc(nram_x + ((i + 1) % 2) * num_pong,
           nram_x_half + ((i + 1) % 2) * num_pong, nram_aux_a, nram_aux_b,
           num_deal, num_deal, coef);
    __asm__ volatile("sync;");
  }

  if (repeat > 1) {
    pvLock();
    __memcpy_async(addr_y + (repeat - 2) * num_deal,
                   nram_x + ((repeat - 2) % 2) * num_pong, span_handle_size,
                   NRAM2GDRAM);
    pvUnlock();
  }

  if (rem > 0) {
    __memcpy_async(nram_x_half + (repeat % 2) * num_pong,
                   addr_x + repeat * num_deal, rem * sizeof(T), GDRAM2NRAM);
  }

  if (repeat > 0) {
    OpFunc(nram_x + ((repeat - 1) % 2) * num_pong,
           nram_x_half + ((repeat - 1) % 2) * num_pong, nram_aux_a, nram_aux_b,
           num_deal, num_deal, coef);
  }
  __asm__ volatile("sync;");

  if (repeat > 0) {
    pvLock();
    __memcpy_async(addr_y + (repeat - 1) * num_deal,
                   nram_x + ((repeat - 1) % 2) * num_pong, span_handle_size,
                   NRAM2GDRAM);
    pvUnlock();
  }

  if (rem > 0) {
    OpFunc(nram_x + (repeat % 2) * num_pong,
           nram_x_half + (repeat % 2) * num_pong, nram_aux_a, nram_aux_b,
           align_rem, rem, coef);
    __asm__ volatile("sync;");

    pvLock();
    __memcpy_async(addr_y + repeat * num_deal, nram_x + (repeat % 2) * num_pong,
                   rem * sizeof(T), NRAM2GDRAM);
    pvUnlock();
  }
}
#endif  // KERNELS_UNARY_OP_UNARY_OP_3PIPELINE_H_
