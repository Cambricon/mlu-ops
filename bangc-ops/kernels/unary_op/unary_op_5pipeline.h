/*************************************************************************
 * Copyright (C) 2021 by Cambricon, Inc. All rights reserved.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef KERNELS_UNARY_OP_UNARY_OP_5PIPELINE_H_
#define KERNELS_UNARY_OP_UNARY_OP_5PIPELINE_H_

#include "kernels/kernel.h"
#define UNARY_ALIGN_NUM 64

#define UNARY_OP_KERNEL_5PIPELINE_DECLARE(Op, DType, Prefer)           \
  __mlu_global__ void MLUBlockKernel5StagePipeline##Op##DType##Prefer( \
      void *x, void *y, uint32_t num_total, float coef);

#define UNARY_OP_KERNEL_5PIPELINE_IMPLE(Op, DType, Prefer)                     \
  __mlu_global__ void MLUBlockKernel5StagePipeline##Op##DType##Prefer(         \
      void *x, void *y, uint32_t num_total, float coef) {                      \
    int32_t num_deal = 0, offset_half = 0, offset_aux_a = 0, offset_aux_b = 0; \
    get5Offset##Op##Prefer<DType>(offset_half, offset_aux_a, offset_aux_b,     \
                                  num_deal);                                   \
    block5Unary<DType, compute##Op##Prefer>(                                   \
        (DType *)x, (DType *)y, nram_buffer, sram_buffer, num_total,           \
        offset_half, offset_aux_a, offset_aux_b, num_deal, coef);              \
  }

template <typename T, void (*OpFunc)(T *, T *, T *, T *, int, int, float)>
__mlu_func__ void moveAndCompute(T *sram_x, T *nram_x, T *nram_x_half,
                                 T *nram_aux_a, T *nram_aux_b, int deal_num,
                                 int offset, int cur_num, float coef) {
  __memcpy_async(nram_x_half, sram_x + offset, cur_num * sizeof(T), SRAM2NRAM);
  __asm__ volatile("sync;\n\t");
  OpFunc(nram_x, nram_x_half, nram_aux_a, nram_aux_b, deal_num, cur_num, coef);
  __asm__ volatile("sync;\n\t");
  __memcpy_async(sram_x + offset, nram_x, cur_num * sizeof(T), NRAM2SRAM);
}

template <typename T, void (*OpFunc)(T *, T *, T *, T *, int, int, float)>
__mlu_func__ void block5Unary(T *x, T *y, char *nram_buffer, char *sram_buffer,
                              int32_t num_total, int32_t offset_x_half,
                              int32_t offset_aux_a, int32_t offset_aux_b,
                              int32_t num_deal, float coef) {
  // split data_num by clusters
  int32_t num_per_cluster = num_total / taskDimY;
  int32_t remain_cluster = num_total % taskDimY;
  // ddr ram space
  T *addr_x = (T *)x + taskIdY * num_per_cluster;
  T *addr_y = (T *)y + taskIdY * num_per_cluster;
  if (remain_cluster > 0 && taskIdY == taskDimY - 1) {
    num_per_cluster += remain_cluster;
  }

  // onchip ran space
  T *sram_x = (T *)sram_buffer;
  T *nram_x = (T *)nram_buffer;
  T *nram_x_half = (T *)nram_buffer + offset_x_half;
  T *nram_aux_a = (T *)nram_buffer + offset_aux_a;
  T *nram_aux_b = (T *)nram_buffer + offset_aux_b;

  int32_t num_pong = num_deal * CORE_DIM;
  int32_t repeat = num_per_cluster / num_pong;
  int32_t rem = num_per_cluster % num_pong;

  // split rem num by cores
  int32_t rem_per_core = rem / coreDim;
  int32_t remain_core = rem % coreDim;
  int32_t rem_core_offset = taskIdX * rem_per_core;
  if (remain_core > 0 && coreId == coreDim - 1) {
    rem_per_core += remain_core;
  }
  int32_t align_rem_per_core = CEIL_ALIGN(rem_per_core, UNARY_ALIGN_NUM);
  int32_t span_hanld_size = num_pong * sizeof(T);

  // 5 level pipeline.
  if (repeat > 0) {
    __memcpy_async(sram_x, addr_x, span_hanld_size, GDRAM2SRAM);
    __sync_cluster();
  }

  if (repeat > 1) {
    __memcpy_async(sram_x + num_pong, addr_x + num_pong, span_hanld_size,
                   GDRAM2SRAM);
    moveAndCompute<T, OpFunc>(sram_x, nram_x, nram_x_half, nram_aux_a,
                              nram_aux_b, num_deal, coreId * num_deal, num_deal,
                              coef);
    __sync_cluster();
  }

  for (int i = 0; i < repeat - 2; i++) {
    __memcpy_async(addr_y + i * num_pong, sram_x + (i % 2) * num_pong,
                   span_hanld_size, SRAM2GDRAM);
    __memcpy_async(sram_x + (i % 2) * num_pong, addr_x + (i + 2) * num_pong,
                   span_hanld_size, GDRAM2SRAM);
    moveAndCompute<T, OpFunc>(sram_x + ((i + 1) % 2) * num_pong, nram_x,
                              nram_x_half, nram_aux_a, nram_aux_b, num_deal,
                              coreId * num_deal, num_deal, coef);
    __sync_cluster();
  }

  if (repeat > 1) {
    __memcpy_async(addr_y + (repeat - 2) * num_pong,
                   sram_x + ((repeat - 2) % 2) * num_pong, span_hanld_size,
                   SRAM2GDRAM);
  }

  if (rem > 0) {
    __memcpy_async(sram_x + (repeat % 2) * num_pong, addr_x + repeat * num_pong,
                   rem * sizeof(T), GDRAM2SRAM);
  }

  if (repeat > 0) {
    moveAndCompute<T, OpFunc>(sram_x + ((repeat - 1) % 2) * num_pong, nram_x,
                              nram_x_half, nram_aux_a, nram_aux_b, num_deal,
                              coreId * num_deal, num_deal, coef);
  }
  __sync_cluster();

  if (repeat > 0) {
    __memcpy_async(addr_y + (repeat - 1) * num_pong,
                   sram_x + ((repeat - 1) % 2) * num_pong, span_hanld_size,
                   SRAM2GDRAM);
  }

  if (rem > 0) {
    if (rem_per_core > 0) {
      moveAndCompute<T, OpFunc>(
          sram_x + (repeat % 2) * num_pong, nram_x, nram_x_half, nram_aux_a,
          nram_aux_b, align_rem_per_core, rem_core_offset, rem_per_core, coef);
    }
    __sync_cluster();
    __memcpy_async(addr_y + repeat * num_pong, sram_x + (repeat % 2) * num_pong,
                   rem * sizeof(T), SRAM2GDRAM);
  }
}

#endif  // KERNELS_UNARY_OP_UNARY_OP_5PIPELINE_H_
