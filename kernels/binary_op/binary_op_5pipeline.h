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
#ifndef KERNELS_BINARY_OP_BINARY_OP_5PIPELINE_H_
#define KERNELS_BINARY_OP_BINARY_OP_5PIPELINE_H_

#include "mlu_op.h"
#include "kernels/kernel.h"
#include "kernels/debug.h"
#include "binary_op_host.h"
#include "kernels/utils/common.h"
#include "kernels/tensor_stride_process/tensor_stride_process_common.h"

#define BINARY_ALIGN_NUM 64
#define BINARY_NRAM_SIZE (MAX_NRAM_SIZE + REM_FOR_STACK - 112 * 1024)
#define BINARY_SRAM_SIZE (CORE_DIM * BINARY_NRAM_SIZE)

#define BINARY_OP_KERNEL_5PIPELINE_DECLARE(Op, Prefer)                  \
  template <typename DType_in1, typename DType_in2, typename DType_out, \
            typename... Args>                                           \
  __mlu_global__ void MLUBlockKernel5StagePipeline##Op##Prefer(         \
      char *x, char *y, char *z, size_t data_num, Args... args)

/****************************************************************************
 * GDRAM2SRAM: io pipeline
 * SRAM2NRAM ：mo pipeline
 * In Cambricon hardware, The io, compute, and move pipeline have their own
 * instruction queues and can be launched in parallel,
 * so the time of io, compute and move can cover each other.
 * The five pipleine is:
 * GDRAM2SRAM: io load data form gdram
 * SRAM2NRAM : mv load data from sram
 * Compute : compute pipleine
 * NRAM2SRAM : mv store data to sram
 * SRAM2GDRAM ： io store data to gdram
 */

#if __BANG_ARCH__ != 520  // TODO(sram): tp_520
#define BINARY_OP_KERNEL_5PIPELINE(Op, Prefer)                                 \
  template <typename DType_in1, typename DType_in2, typename DType_out,        \
            typename... Args>                                                  \
  __mlu_global__ void MLUBlockKernel5StagePipeline##Op##Prefer(                \
      char *x, char *y, char *z, size_t data_num, Args... args) {              \
    size_t span_num_deal = 0;                                                  \
    size_t output_input1_gap = 0, output_input2_gap = 0;                       \
    size_t auxiliary_a_gap = 0, auxiliary_b_gap = 0, auxiliary_c_gap = 0;      \
    size_t align_num = BINARY_ALIGN_NUM;                                       \
    auxFunc5##Op##Prefer<DType_in1, DType_in2, DType_out>(                     \
        span_num_deal, output_input1_gap, output_input2_gap, auxiliary_a_gap,  \
        auxiliary_b_gap, auxiliary_c_gap, align_num, args...);                 \
    size_t num_per_cluster = data_num / taskDimY;                              \
    size_t num_rem_cluster = data_num % taskDimY;                              \
    size_t cluster_offset = taskIdY * num_per_cluster;                         \
    DType_in1 *base_cluster_in1 = (DType_in1 *)x + cluster_offset;             \
    DType_in2 *base_cluster_in2 = (DType_in2 *)y + cluster_offset;             \
    DType_out *base_cluster_out = (DType_out *)z + cluster_offset;             \
    if (num_rem_cluster > 0 && taskIdY == taskDimY - 1) {                      \
      num_per_cluster = num_per_cluster + num_rem_cluster;                     \
    }                                                                          \
    size_t sram_num = span_num_deal * coreDim;                                 \
    size_t sram_pong_gap = sram_num;                                           \
    DType_in1 *sram_in1 = (DType_in1 *)sram_buffer;                            \
    DType_in2 *sram_in2 = (DType_in2 *)(sram_in1 + sram_num * 2);              \
    int32_t repeat = num_per_cluster / sram_num;                               \
    size_t cluster_rem = num_per_cluster % sram_num;                           \
    size_t cluster_rem_to_core = cluster_rem / coreDim;                        \
    size_t remain_core = cluster_rem % coreDim;                                \
    size_t rem_core_offset = taskIdX * cluster_rem_to_core;                    \
    if (remain_core > 0 && coreId == coreDim - 1) {                            \
      cluster_rem_to_core = cluster_rem_to_core + remain_core;                 \
    }                                                                          \
    size_t align_cluster_rem_to_core = PAD_UP(cluster_rem_to_core, align_num); \
                                                                               \
    char *nram_out = nram_buffer;                                              \
    char *nram_in1 = nram_buffer + output_input1_gap;                          \
    char *nram_in2 = nram_buffer + output_input2_gap;                          \
    char *nram_aux1 = nram_buffer + auxiliary_a_gap;                           \
    char *nram_aux2 = nram_buffer + auxiliary_b_gap;                           \
    char *nram_aux3 = nram_buffer + auxiliary_c_gap;                           \
                                                                               \
    if (repeat > 0) {                                                          \
      __memcpy_async(sram_in1, base_cluster_in1, sram_num * sizeof(DType_in1), \
                     GDRAM2SRAM);                                              \
      __memcpy_async(sram_in2, base_cluster_in2, sram_num * sizeof(DType_in2), \
                     GDRAM2SRAM);                                              \
      __sync_cluster();                                                        \
    }                                                                          \
    if (repeat > 1) {                                                          \
      __memcpy_async(sram_in1 + sram_pong_gap, base_cluster_in1 + sram_num,    \
                     sram_num * sizeof(DType_in1), GDRAM2SRAM);                \
      __memcpy_async(sram_in2 + sram_pong_gap, base_cluster_in2 + sram_num,    \
                     sram_num * sizeof(DType_in2), GDRAM2SRAM);                \
      __memcpy_async(nram_in1, sram_in1 + taskIdX * span_num_deal,             \
                     span_num_deal * sizeof(DType_in1), SRAM2NRAM);            \
      __memcpy_async(nram_in2, sram_in2 + taskIdX * span_num_deal,             \
                     span_num_deal * sizeof(DType_in2), SRAM2NRAM);            \
      __asm__ volatile("sync;\n\t");                                           \
      compute##Op##Prefer<DType_in1, DType_in2, DType_out>(                    \
          nram_out, nram_in1, nram_in2, nram_aux1, nram_aux2, nram_aux3,       \
          span_num_deal, span_num_deal, args...);                              \
      __asm__ volatile("sync;\n\t");                                           \
                                                                               \
      __memcpy_async(sram_in1 + taskIdX * span_num_deal, nram_out,             \
                     span_num_deal * sizeof(DType_out), NRAM2SRAM);            \
      __sync_cluster();                                                        \
    }                                                                          \
                                                                               \
    for (int32_t i = 0; i < repeat - 2; i++) {                                 \
      __memcpy_async(base_cluster_out + i * sram_num,                          \
                     sram_in1 + (i % 2) * sram_pong_gap,                       \
                     sram_num * sizeof(DType_out), SRAM2GDRAM);                \
      __memcpy_async(sram_in1 + (i % 2) * sram_pong_gap,                       \
                     base_cluster_in1 + (i + 2) * sram_num,                    \
                     sram_num * sizeof(DType_in1), GDRAM2SRAM);                \
      __memcpy_async(sram_in2 + (i % 2) * sram_pong_gap,                       \
                     base_cluster_in2 + (i + 2) * sram_num,                    \
                     sram_num * sizeof(DType_in2), GDRAM2SRAM);                \
      __memcpy_async(                                                          \
          nram_in1,                                                            \
          sram_in1 + ((i + 1) % 2) * sram_pong_gap + taskIdX * span_num_deal,  \
          span_num_deal * sizeof(DType_in1), SRAM2NRAM);                       \
      __memcpy_async(                                                          \
          nram_in2,                                                            \
          sram_in2 + ((i + 1) % 2) * sram_pong_gap + taskIdX * span_num_deal,  \
          span_num_deal * sizeof(DType_in2), SRAM2NRAM);                       \
      __asm__ volatile("sync;\n\t");                                           \
      compute##Op##Prefer<DType_in1, DType_in2, DType_out>(                    \
          nram_out, nram_in1, nram_in2, nram_aux1, nram_aux2, nram_aux3,       \
          span_num_deal, span_num_deal, args...);                              \
      __asm__ volatile("sync;\n\t");                                           \
      __memcpy_async(                                                          \
          sram_in1 + ((i + 1) % 2) * sram_pong_gap + taskIdX * span_num_deal,  \
          nram_out, span_num_deal * sizeof(DType_out), NRAM2SRAM);             \
      __sync_cluster();                                                        \
    }                                                                          \
                                                                               \
    if (repeat >= 2) {                                                         \
      __memcpy_async(base_cluster_out + (repeat - 2) * sram_num,               \
                     sram_in1 + (repeat % 2) * sram_pong_gap,                  \
                     sram_num * sizeof(DType_out), SRAM2GDRAM);                \
    }                                                                          \
    if (cluster_rem > 0) {                                                     \
      __memcpy_async(sram_in1 + (repeat % 2) * sram_pong_gap,                  \
                     base_cluster_in1 + repeat * sram_num,                     \
                     cluster_rem * sizeof(DType_in1), GDRAM2SRAM);             \
      __memcpy_async(sram_in2 + (repeat % 2) * sram_pong_gap,                  \
                     base_cluster_in2 + repeat * sram_num,                     \
                     cluster_rem * sizeof(DType_in2), GDRAM2SRAM);             \
    }                                                                          \
    if (repeat > 0) {                                                          \
      __memcpy_async(nram_in1,                                                 \
                     sram_in1 + ((repeat - 1) % 2) * sram_pong_gap +           \
                         taskIdX * span_num_deal,                              \
                     span_num_deal * sizeof(DType_in1), SRAM2NRAM);            \
      __memcpy_async(nram_in2,                                                 \
                     sram_in2 + ((repeat - 1) % 2) * sram_pong_gap +           \
                         taskIdX * span_num_deal,                              \
                     span_num_deal * sizeof(DType_in2), SRAM2NRAM);            \
      __asm__ volatile("sync;\n\t");                                           \
      compute##Op##Prefer<DType_in1, DType_in2, DType_out>(                    \
          nram_out, nram_in1, nram_in2, nram_aux1, nram_aux2, nram_aux3,       \
          span_num_deal, span_num_deal, args...);                              \
      __asm__ volatile("sync;\n\t");                                           \
      __memcpy_async(sram_in1 + ((repeat - 1) % 2) * sram_pong_gap +           \
                         taskIdX * span_num_deal,                              \
                     nram_out, span_num_deal * sizeof(DType_out), NRAM2SRAM);  \
    }                                                                          \
    __sync_cluster();                                                          \
                                                                               \
    if (repeat > 0) {                                                          \
      __memcpy_async(base_cluster_out + (repeat - 1) * sram_num,               \
                     sram_in1 + ((repeat - 1) % 2) * sram_pong_gap,            \
                     sram_num * sizeof(DType_out), SRAM2GDRAM);                \
    }                                                                          \
                                                                               \
    if (cluster_rem > 0) {                                                     \
      if (cluster_rem_to_core) {                                               \
        __memcpy_async(                                                        \
            nram_in1,                                                          \
            sram_in1 + (repeat % 2) * sram_pong_gap + rem_core_offset,         \
            cluster_rem_to_core * sizeof(DType_in1), SRAM2NRAM);               \
        __memcpy_async(                                                        \
            nram_in2,                                                          \
            sram_in2 + (repeat % 2) * sram_pong_gap + rem_core_offset,         \
            cluster_rem_to_core * sizeof(DType_in2), SRAM2NRAM);               \
        __asm__ volatile("sync;\n\t");                                         \
        compute##Op##Prefer<DType_in1, DType_in2, DType_out>(                  \
            nram_out, nram_in1, nram_in2, nram_aux1, nram_aux2, nram_aux3,     \
            align_cluster_rem_to_core, cluster_rem_to_core, args...);          \
        __asm__ volatile("sync;\n\t");                                         \
        __memcpy_async(                                                        \
            sram_in1 + (repeat % 2) * sram_pong_gap + rem_core_offset,         \
            nram_out, cluster_rem_to_core * sizeof(DType_out), NRAM2SRAM);     \
      }                                                                        \
      __sync_cluster();                                                        \
      __memcpy_async(base_cluster_out + repeat * sram_num,                     \
                     sram_in1 + (repeat % 2) * sram_pong_gap,                  \
                     cluster_rem * sizeof(DType_out), SRAM2GDRAM);             \
      __sync_cluster();                                                        \
    }                                                                          \
  }
#else
#define BINARY_OP_KERNEL_5PIPELINE(Op, Prefer)                          \
  template <typename DType_in1, typename DType_in2, typename DType_out, \
            typename... Args>                                           \
  __mlu_global__ void MLUBlockKernel5StagePipeline##Op##Prefer(         \
      char *x, char *y, char *z, size_t data_num, Args... args) {}
#endif
#endif  // KERNELS_BINARY_OP_BINARY_OP_5PIPELINE_H_
