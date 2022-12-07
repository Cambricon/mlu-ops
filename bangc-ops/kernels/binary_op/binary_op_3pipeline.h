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
#ifndef KERNELS_BINARY_OP_BINARY_OP_3PIPELINE_H_
#define KERNELS_BINARY_OP_BINARY_OP_3PIPELINE_H_

#include "kernels/kernel.h"
#include "kernels/utils/common.h"
#define BINARY_ALIGN_NUM 64

#define BINARY_OP_3PIPELINE_DECLARE(Op, Dtype, Prefer)                 \
  __mlu_global__ void MLUBlockKernel3StagePipeline##Op##Dtype##Prefer( \
      void *a, void *b, void *c, int32_t data_num)

#define BINARY_OP_3PIPELINE_IMPLE(Op, Dtype, Prefer)                      \
  __mlu_global__ void MLUBlockKernel3StagePipeline##Op##Dtype##Prefer(    \
      void *x, void *y, void *z, int32_t data_num) {                      \
    int32_t nram_limit = 0;                                               \
    int32_t pong_x = 0;                                                   \
    int32_t pong_y = 0;                                                   \
    Dtype *nram_x = NULL;                                                 \
    Dtype *nram_y = NULL;                                                 \
    Dtype *nram_aux1 = NULL;                                              \
    Dtype *nram_aux2 = NULL;                                              \
    Dtype *nram_aux3 = NULL;                                              \
    get3Offset##Op##Prefer(nram_limit, pong_x, pong_y, nram_x, nram_y,    \
                           nram_aux1, nram_aux2, nram_aux3, nram_buffer); \
    processBinaryPipe3<Dtype, compute##Op##Prefer>(                       \
        (Dtype *)x, (Dtype *)y, (Dtype *)z, nram_buffer, (Dtype *)nram_x, \
        (Dtype *)nram_y, (Dtype *)nram_aux1, (Dtype *)nram_aux2,          \
        (Dtype *)nram_aux3, nram_limit, pong_x, pong_y, data_num);        \
  }

template <typename Dtype, void (*OpFunc)(Dtype *, Dtype *, Dtype *, Dtype *,
                                         Dtype *, int32_t, int32_t)>
__mlu_func__ void processBinaryPipe3(const Dtype *x, const Dtype *y, Dtype *z,
                                     char *nram_buffer, Dtype *nram_x,
                                     Dtype *nram_y, Dtype *nram_aux1,
                                     Dtype *nram_aux2, Dtype *nram_aux3,
                                     const int32_t nram_limit,
                                     const int32_t pong_x, const int32_t pong_y,
                                     const int32_t data_num) {
  if (coreId == 0x80) {
    return;
  }
  // split data by cores
  // Dtype just use for POWN y inDtype6_t
  int32_t num_per_core = data_num / taskDim;
  int32_t rem_for_all = data_num % taskDim;
  Dtype *base_addr_x = (Dtype *)x + taskId * num_per_core;
  Dtype *base_addr_y = (Dtype *)y + taskId * num_per_core;
  Dtype *base_addr_z = (Dtype *)z + taskId * num_per_core;
  if (rem_for_all > 0 && taskId == (taskDim - 1)) {
    num_per_core = num_per_core + rem_for_all;
  }

  int32_t repeat = num_per_core / nram_limit;
  int32_t rem = num_per_core % nram_limit;
  int32_t align_rem = CEIL_ALIGN(rem, BINARY_ALIGN_NUM);

  int32_t span_handle_size = nram_limit * sizeof(Dtype);
  int32_t rem_size = rem * sizeof(Dtype);

  if (repeat > 0) {
    // L
    __memcpy_async(nram_x, base_addr_x, span_handle_size, GDRAM2NRAM);
    __memcpy_async(nram_y, base_addr_y, span_handle_size, GDRAM2NRAM);
    __asm__ volatile("sync;");
  }
  if (repeat > 1) {
    // L
    __memcpy_async(nram_x + pong_x, base_addr_x + nram_limit, span_handle_size,
                   GDRAM2NRAM);
    __memcpy_async(nram_y + pong_y, base_addr_y + nram_limit, span_handle_size,
                   GDRAM2NRAM);
    // C
    OpFunc(nram_x, nram_y, nram_aux1, nram_aux2, nram_aux3, nram_limit,
           nram_limit);
    __asm__ volatile("sync;");
  }

  for (int32_t i = 0; i < repeat - 2; i++) {
    // S
    pvLock();
    __memcpy_async(base_addr_z + i * nram_limit, nram_x + (i % 2) * pong_x,
                   span_handle_size, NRAM2GDRAM);
    pvUnlock();
    // L
    __memcpy_async(nram_x + (i % 2) * pong_x,
                   base_addr_x + (i + 2) * nram_limit, span_handle_size,
                   GDRAM2NRAM);
    __memcpy_async(nram_y + (i % 2) * pong_y,
                   base_addr_y + (i + 2) * nram_limit, span_handle_size,
                   GDRAM2NRAM);
    // C
    OpFunc(nram_x + ((i + 1) % 2) * pong_x, nram_y + ((i + 1) % 2) * pong_y,
           nram_aux1, nram_aux2, nram_aux3, nram_limit, nram_limit);
    __asm__ volatile("sync;");
  }

  if (repeat >= 2) {
    // S
    pvLock();
    __memcpy_async(base_addr_z + (repeat - 2) * nram_limit,
                   nram_x + (repeat % 2) * pong_x, span_handle_size,
                   NRAM2GDRAM);
    pvUnlock();
  }
  if (rem > 0) {
    // L
    __memcpy_async(nram_x + (repeat % 2) * pong_x,
                   base_addr_x + repeat * nram_limit, rem_size, GDRAM2NRAM);
    __memcpy_async(nram_y + (repeat % 2) * pong_y,
                   base_addr_y + repeat * nram_limit, rem_size, GDRAM2NRAM);
  }
  if (repeat > 0) {
    // C
    OpFunc(nram_x + ((repeat - 1) % 2) * pong_x,
           nram_y + ((repeat - 1) % 2) * pong_y, nram_aux1, nram_aux2,
           nram_aux3, nram_limit, nram_limit);
  }
  __asm__ volatile("sync;");

  if (repeat > 0) {
    // S
    pvLock();
    __memcpy_async(base_addr_z + (repeat - 1) * nram_limit,
                   nram_x + ((repeat - 1) % 2) * pong_x, span_handle_size,
                   NRAM2GDRAM);
    pvUnlock();
  }
  if (rem > 0) {
    // C
    OpFunc(nram_x + (repeat % 2) * pong_x, nram_y + (repeat % 2) * pong_y,
           nram_aux1, nram_aux2, nram_aux3, rem, align_rem);
    __asm__ volatile("sync;");
    // S
    pvLock();
    __memcpy_async(base_addr_z + repeat * nram_limit,
                   nram_x + (repeat % 2) * pong_x, rem_size, NRAM2GDRAM);
    pvUnlock();
  }
}

#endif  // KERNELS_BINARY_OP_BINARY_OP_3PIPELINE_H_
