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

#ifndef KERNELS_UNARY_OP_UNARY_OP_4PIPELINE_H_
#define KERNELS_UNARY_OP_UNARY_OP_4PIPELINE_H_

#include "kernels/kernel.h"
#include "kernels/debug.h"
#include "kernels/utils/common.h"

#define UNARY_OP_4PIPELINE_DECLARE(Op, Prefer)                       \
  template <typename DType_in, typename DType_out, typename... Args> \
  __mlu_global__ void MLUBlockKernel4StagePipeline##Op##Prefer(      \
      char *x, char *y, size_t element_num, Args... args);

#define PRINTF(fmt, ...)                 \
  if (false) {                           \
    __bang_printf("taskId%d: ", taskId); \
    __bang_printf(fmt, ##__VA_ARGS__);   \
  }

__mlu_func__ void strategyOfPartitionCluster(size_t element_num,
                                             size_t &cluster_num_deal,
                                             size_t &cluster_offset) {
  cluster_num_deal = element_num / taskDimY;
  cluster_offset = taskIdY * cluster_num_deal;
  size_t cluster_loop_remain = element_num % taskDimY;
  if (taskIdY == taskDimY - 1 && cluster_loop_remain > 0) {
    cluster_num_deal += cluster_loop_remain;
  }
}

__mlu_func__ void strategyOfPartitionCore(size_t remain_num,
                                          size_t &core_remain_num_deal,
                                          size_t &core_remain_offset) {
  core_remain_num_deal = remain_num / taskDimX;
  core_remain_offset = taskIdX * core_remain_num_deal;
  size_t loop_left_num = remain_num % taskDimX;
  if (taskIdX == taskDimX - 1 && loop_left_num > 0) {
    core_remain_num_deal += loop_left_num;
  }
}

#if __BANG_ARCH__ != 520  // TODO(sram): tp_520
#define UNARY_OP_4PIPELINE_IMPLEMENTATION(Op, Prefer)                          \
  template <typename DType_in, typename DType_out, typename... Args>           \
  __mlu_global__ void MLUBlockKernel4StagePipeline##Op##Prefer(                \
      char *input_gdram, char *output_gdram, size_t element_num,               \
      Args... args) {                                                          \
    size_t output_input_gap = 0, ping_pong_gap = 0;                            \
    size_t auxiliary_a_gap = 0, auxiliary_b_gap = 0;                           \
    size_t span_num_deal = 0;                                                  \
    size_t align_num = 1;                                                      \
    auxFunc3##Op##Prefer<DType_in, DType_out>(                                 \
        output_input_gap, ping_pong_gap, auxiliary_a_gap, auxiliary_b_gap,     \
        span_num_deal, align_num, args...);                                    \
    size_t cluster_num_deal = 0, cluster_offset = 0;                           \
    strategyOfPartitionCluster(element_num, cluster_num_deal, cluster_offset); \
    char *load_start = input_gdram + cluster_offset * sizeof(DType_in);        \
    char *store_start = output_gdram + cluster_offset * sizeof(DType_out);     \
    size_t cluster_span_deal = span_num_deal * coreDim;                        \
    int32_t repeat = cluster_num_deal / cluster_span_deal;                     \
    size_t cluster_remain = cluster_num_deal % cluster_span_deal;              \
    size_t cluster_load_size = cluster_span_deal * sizeof(DType_in);           \
    size_t cluster_store_size = cluster_span_deal * sizeof(DType_out);         \
    size_t core_load_size = span_num_deal * sizeof(DType_in);                  \
    size_t core_store_size = span_num_deal * sizeof(DType_out);                \
    size_t sram_pong_gap = cluster_load_size;                                  \
    size_t sram_load_offset = taskIdX * span_num_deal * sizeof(DType_in);      \
    size_t sram_store_offset = taskIdX * span_num_deal * sizeof(DType_out);    \
    size_t core_remain_num_deal = 0, core_remain_offset = 0;                   \
    strategyOfPartitionCore(cluster_remain, core_remain_num_deal,              \
                            core_remain_offset);                               \
    size_t align_remain_num = PAD_UP(core_remain_num_deal, align_num);         \
    char *sram_ping = sram_buffer;                                             \
    char *ping_output = nram_buffer;                                           \
    char *ping_input = nram_buffer + output_input_gap;                         \
    char *auxiliary_a = nram_buffer + auxiliary_a_gap;                         \
    char *auxiliary_b = nram_buffer + auxiliary_b_gap;                         \
    if (repeat > 0) {                                                          \
      __memcpy_async(sram_ping, load_start, cluster_load_size, GDRAM2SRAM);    \
      __sync_cluster();                                                        \
    }                                                                          \
    if (repeat > 1) {                                                          \
      __memcpy_async(sram_ping + sram_pong_gap,                                \
                     load_start + cluster_load_size, cluster_load_size,        \
                     GDRAM2SRAM);                                              \
      __memcpy_async(ping_input, sram_ping + sram_load_offset, core_load_size, \
                     SRAM2NRAM);                                               \
      __sync_copy_sram_to_nram();                                              \
      compute##Op##Prefer<DType_in, DType_out>(                                \
          ping_output, ping_input, auxiliary_a, auxiliary_b, span_num_deal,    \
          span_num_deal, args...);                                             \
      __sync_cluster();                                                        \
    }                                                                          \
    for (int i = 0; i < repeat - 2; i++) {                                     \
      int ping_flag = i % 2, pong_flag = (i + 1) % 2;                          \
      pvLock();                                                                \
      __memcpy_async(store_start + i * cluster_store_size + sram_store_offset, \
                     ping_output + ping_flag * ping_pong_gap, core_store_size, \
                     NRAM2GDRAM);                                              \
      pvUnlock();                                                              \
      __memcpy_async(sram_ping + ping_flag * sram_pong_gap,                    \
                     load_start + (i + 2) * cluster_load_size,                 \
                     cluster_load_size, GDRAM2SRAM);                           \
      __memcpy_async(ping_input + pong_flag * ping_pong_gap,                   \
                     sram_ping + pong_flag * sram_pong_gap + sram_load_offset, \
                     core_load_size, SRAM2NRAM);                               \
      __sync_copy_sram_to_nram();                                              \
      compute##Op##Prefer<DType_in, DType_out>(                                \
          ping_output + pong_flag * ping_pong_gap,                             \
          ping_input + pong_flag * ping_pong_gap, auxiliary_a, auxiliary_b,    \
          span_num_deal, span_num_deal, args...);                              \
      __sync_cluster();                                                        \
    }                                                                          \
    if (repeat > 1) {                                                          \
      pvLock();                                                                \
      __memcpy_async(                                                          \
          store_start + (repeat - 2) * cluster_store_size + sram_store_offset, \
          ping_output + ((repeat - 2) % 2) * ping_pong_gap, core_store_size,   \
          NRAM2GDRAM);                                                         \
      pvUnlock();                                                              \
    }                                                                          \
    if (cluster_remain > 0) {                                                  \
      __memcpy_async(sram_ping + (repeat % 2) * sram_pong_gap,                 \
                     load_start + repeat * cluster_load_size,                  \
                     cluster_remain * sizeof(DType_in), GDRAM2SRAM);           \
    }                                                                          \
    if (repeat > 0) {                                                          \
      int ping_pong_flag = (repeat - 1) % 2;                                   \
      __memcpy_async(                                                          \
          ping_input + ping_pong_flag * ping_pong_gap,                         \
          sram_ping + ping_pong_flag * sram_pong_gap + sram_load_offset,       \
          core_load_size, SRAM2NRAM);                                          \
      __sync_copy_sram_to_nram();                                              \
      compute##Op##Prefer<DType_in, DType_out>(                                \
          ping_output + ping_pong_flag * ping_pong_gap,                        \
          ping_input + ping_pong_flag * ping_pong_gap, auxiliary_a,            \
          auxiliary_b, span_num_deal, span_num_deal, args...);                 \
    }                                                                          \
    __sync_cluster();                                                          \
    if (repeat > 0) {                                                          \
      pvLock();                                                                \
      __memcpy_async(                                                          \
          store_start + (repeat - 1) * cluster_store_size + sram_store_offset, \
          ping_output + ((repeat - 1) % 2) * ping_pong_gap, core_store_size,   \
          NRAM2GDRAM);                                                         \
      pvUnlock();                                                              \
    }                                                                          \
    if (core_remain_num_deal > 0) {                                            \
      int ping_pong_flag = repeat % 2;                                         \
      __memcpy_async(ping_input + ping_pong_flag * ping_pong_gap,              \
                     sram_ping + ping_pong_flag * sram_pong_gap +              \
                         core_remain_offset * sizeof(DType_in),                \
                     core_remain_num_deal * sizeof(DType_in), SRAM2NRAM);      \
      __sync_copy_sram_to_nram();                                              \
      compute##Op##Prefer<DType_in, DType_out>(                                \
          ping_output + ping_pong_flag * ping_pong_gap,                        \
          ping_input + ping_pong_flag * ping_pong_gap, auxiliary_a,            \
          auxiliary_b, align_remain_num, core_remain_num_deal, args...);       \
      __asm__ volatile("sync;");                                               \
      pvLock();                                                                \
      __memcpy_async(store_start + repeat * cluster_store_size +               \
                         core_remain_offset * sizeof(DType_out),               \
                     ping_output + ping_pong_flag * ping_pong_gap,             \
                     core_remain_num_deal * sizeof(DType_out), NRAM2GDRAM);    \
      pvUnlock();                                                              \
    }                                                                          \
  }
#else
#define UNARY_OP_4PIPELINE_IMPLEMENTATION(Op, Prefer)                \
  template <typename DType_in, typename DType_out, typename... Args> \
  __mlu_global__ void MLUBlockKernel4StagePipeline##Op##Prefer(      \
      char *input_gdram, char *output_gdram, size_t element_num,     \
      Args... args) {}
#endif
#endif  // KERNELS_UNARY_OP_UNARY_OP_4PIPELINE_H_
