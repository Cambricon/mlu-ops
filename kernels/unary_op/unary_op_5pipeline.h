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

#ifndef KERNELS_UNARY_OP_UNARY_OP_5PIPELINE_H_
#define KERNELS_UNARY_OP_UNARY_OP_5PIPELINE_H_

#include "kernels/kernel.h"
#include "kernels/debug.h"
#include "kernels/utils/common.h"

#define ALIGN_NUM 64
#define UNARY_NRAM_SIZE (MAX_NRAM_SIZE + REM_FOR_STACK - 148 * 1024)
#define UNARY_SRAM_SIZE (CORE_DIM * UNARY_NRAM_SIZE)

#define UNARY_OP_KERNEL_5PIPELINE_DECLARE(Op, Prefer)                \
  template <typename DType_in, typename DType_out, typename... Args> \
  __mlu_global__ void MLUBlockKernel5StagePipeline##Op##Prefer(      \
      char *x, char *y, size_t element_num, Args... args);

#define UNARY_OP_KERNEL_5PIPELINE_MVCVT_DECLARE(Op, ...)                  \
  template <typename DType_in, typename DType_out, typename... Args>      \
  __mlu_global__ void MLUBlockKernel5StageMvCvtPipeline##Op##__VA_ARGS__( \
      const void *x, void *y, size_t element_num, Args... args);

__mlu_func__ void strategyOfPartitionCluster(size_t data_num,
                                             size_t &num_per_cluster,
                                             size_t &offset_cluster) {
  num_per_cluster = data_num / taskDimY;
  size_t num_rem = data_num % taskDimY;
  num_per_cluster += (taskIdY < num_rem);
  offset_cluster = taskIdY * num_per_cluster + (taskIdY >= num_rem) * num_rem;
}

__mlu_func__ void strategyOfPartitionCore(size_t data_num, size_t &num_per_core,
                                          size_t &offset_core) {
  if (coreId != 0x80) {
    num_per_core = data_num / taskDimX;
    size_t num_rem = data_num % taskDimX;
    num_per_core += (taskIdX < num_rem);
    offset_core = taskIdX * num_per_core + (taskIdX >= num_rem) * num_rem;
  }
}

#if __BANG_ARCH__ != 520  // TODO(sram): tp_520
#define UNARY_OP_KERNEL_5PIPELINE_IMPLE(Op, Prefer)                            \
  template <typename DType_in, typename DType_out, typename... Args>           \
  __mlu_global__ void MLUBlockKernel5StagePipeline##Op##Prefer(                \
      char *input_gdram, char *output_gdram, size_t element_num,               \
      Args... args) {                                                          \
    /* The gap of input and output.*/                                          \
    size_t output_input_gap = 0;                                               \
    /* The gap of two auxiliary pointers.*/                                    \
    size_t auxiliary_a_gap = 0, auxiliary_b_gap = 0;                           \
    /* The number(not size) of data to be dealt once.*/                        \
    size_t num_deal = 0;                                                       \
    size_t align_num = 1;                                                      \
                                                                               \
    auxFunc5##Op##Prefer<DType_in, DType_out>(                                 \
        output_input_gap, auxiliary_a_gap, auxiliary_b_gap, num_deal,          \
        align_num, args...);                                                   \
                                                                               \
    /* The total data number(not size) of each cluster.*/                      \
    size_t num_per_cluster = 0;                                                \
    /* The data offset of each cluster.*/                                      \
    size_t offset_cluster = 0;                                                 \
    strategyOfPartitionCluster(element_num, num_per_cluster, offset_cluster);  \
                                                                               \
    size_t cluster_num_deal = num_deal * taskDimX;                             \
    int repeat = num_per_cluster / cluster_num_deal;                           \
    size_t cluster_rem = num_per_cluster % cluster_num_deal;                   \
    size_t cluster_load_size = cluster_num_deal * sizeof(DType_in);            \
    size_t cluster_store_size = cluster_num_deal * sizeof(DType_out);          \
    size_t core_load_size = num_deal * sizeof(DType_in);                       \
    size_t core_store_size = num_deal * sizeof(DType_out);                     \
    size_t sram_pong_gap = cluster_load_size >= cluster_store_size             \
                               ? cluster_load_size                             \
                               : cluster_store_size;                           \
    size_t sram_load_offset = taskIdX * core_load_size;                        \
    size_t sram_store_offset = taskIdX * core_store_size;                      \
                                                                               \
    size_t cluster_rem_per_core = 0;                                           \
    size_t offset_core = 0;                                                    \
    strategyOfPartitionCore(cluster_rem, cluster_rem_per_core, offset_core);   \
    size_t cluster_rem_per_core_align =                                        \
        PAD_UP(cluster_rem_per_core, align_num);                               \
                                                                               \
    char *load_start =                                                         \
        (char *)input_gdram + offset_cluster * sizeof(DType_in);               \
    char *store_start =                                                        \
        (char *)output_gdram + offset_cluster * sizeof(DType_out);             \
                                                                               \
    char *sram_ping = (char *)sram_buffer;                                     \
    char *nram_output = (char *)nram_buffer;                                   \
    char *nram_input = (char *)nram_buffer + output_input_gap;                 \
    char *nram_aux_a = (char *)nram_buffer + auxiliary_a_gap;                  \
    char *nram_aux_b = (char *)nram_buffer + auxiliary_b_gap;                  \
                                                                               \
    if (repeat > 0) {                                                          \
      __memcpy_async(sram_ping, load_start, cluster_load_size, GDRAM2SRAM);    \
      __sync_cluster();                                                        \
    }                                                                          \
                                                                               \
    if (repeat > 1) {                                                          \
      __memcpy_async(sram_ping + sram_pong_gap,                                \
                     load_start + cluster_load_size, cluster_load_size,        \
                     GDRAM2SRAM);                                              \
      __memcpy_async(nram_input, sram_ping + sram_load_offset, core_load_size, \
                     SRAM2NRAM);                                               \
      __sync_copy_sram_to_nram();                                              \
      compute##Op##Prefer<DType_in, DType_out>(nram_output, nram_input,        \
                                               nram_aux_a, nram_aux_b,         \
                                               num_deal, num_deal, args...);   \
      __sync_compute();                                                        \
      __memcpy_async(sram_ping + sram_store_offset, nram_output,               \
                     core_store_size, NRAM2SRAM);                              \
      __sync_cluster();                                                        \
    }                                                                          \
                                                                               \
    for (int i = 0; i < repeat - 2; i++) {                                     \
      __memcpy_async(store_start + i * cluster_store_size,                     \
                     sram_ping + (i % 2) * sram_pong_gap, cluster_store_size,  \
                     SRAM2GDRAM);                                              \
      __memcpy_async(sram_ping + (i % 2) * sram_pong_gap,                      \
                     load_start + (i + 2) * cluster_load_size,                 \
                     cluster_load_size, GDRAM2SRAM);                           \
      __memcpy_async(                                                          \
          nram_input,                                                          \
          sram_ping + ((i + 1) % 2) * sram_pong_gap + sram_load_offset,        \
          core_load_size, SRAM2NRAM);                                          \
      __sync_copy_sram_to_nram();                                              \
      compute##Op##Prefer<DType_in, DType_out>(nram_output, nram_input,        \
                                               nram_aux_a, nram_aux_b,         \
                                               num_deal, num_deal, args...);   \
      __sync_compute();                                                        \
      __memcpy_async(                                                          \
          sram_ping + ((i + 1) % 2) * sram_pong_gap + sram_store_offset,       \
          nram_output, core_store_size, NRAM2SRAM);                            \
      __sync_cluster();                                                        \
    }                                                                          \
                                                                               \
    if (repeat > 1) {                                                          \
      __memcpy_async(store_start + (repeat - 2) * cluster_store_size,          \
                     sram_ping + ((repeat - 2) % 2) * sram_pong_gap,           \
                     cluster_store_size, SRAM2GDRAM);                          \
    }                                                                          \
                                                                               \
    if (cluster_rem > 0) {                                                     \
      __memcpy_async(sram_ping + (repeat % 2) * sram_pong_gap,                 \
                     load_start + repeat * cluster_load_size,                  \
                     cluster_rem * sizeof(DType_in), GDRAM2SRAM);              \
    }                                                                          \
                                                                               \
    if (repeat > 0) {                                                          \
      __memcpy_async(                                                          \
          nram_input,                                                          \
          sram_ping + ((repeat - 1) % 2) * sram_pong_gap + sram_load_offset,   \
          core_load_size, SRAM2NRAM);                                          \
      __sync_copy_sram_to_nram();                                              \
      compute##Op##Prefer<DType_in, DType_out>(nram_output, nram_input,        \
                                               nram_aux_a, nram_aux_b,         \
                                               num_deal, num_deal, args...);   \
      __sync_compute();                                                        \
      __memcpy_async(                                                          \
          sram_ping + ((repeat - 1) % 2) * sram_pong_gap + sram_store_offset,  \
          nram_output, core_store_size, NRAM2SRAM);                            \
    }                                                                          \
    __sync_cluster();                                                          \
                                                                               \
    if (repeat > 0) {                                                          \
      __memcpy_async(store_start + (repeat - 1) * cluster_store_size,          \
                     sram_ping + ((repeat - 1) % 2) * sram_pong_gap,           \
                     cluster_store_size, SRAM2GDRAM);                          \
    }                                                                          \
                                                                               \
    if (cluster_rem > 0) {                                                     \
      if (cluster_rem_per_core > 0) {                                          \
        __memcpy_async(nram_input,                                             \
                       sram_ping + (repeat % 2) * sram_pong_gap +              \
                           offset_core * sizeof(DType_in),                     \
                       cluster_rem_per_core * sizeof(DType_in), SRAM2NRAM);    \
        __sync_copy_sram_to_nram();                                            \
        compute##Op##Prefer<DType_in, DType_out>(                              \
            nram_output, nram_input, nram_aux_a, nram_aux_b,                   \
            cluster_rem_per_core_align, cluster_rem_per_core, args...);        \
        __sync_compute();                                                      \
        __memcpy_async(sram_ping + (repeat % 2) * sram_pong_gap +              \
                           offset_core * sizeof(DType_out),                    \
                       nram_output, cluster_rem_per_core * sizeof(DType_out),  \
                       NRAM2SRAM);                                             \
      }                                                                        \
      __sync_cluster();                                                        \
      __memcpy_async(store_start + repeat * cluster_store_size,                \
                     sram_ping + (repeat % 2) * sram_pong_gap,                 \
                     cluster_rem * sizeof(DType_out), SRAM2GDRAM);             \
    }                                                                          \
  }
#else
#define UNARY_OP_KERNEL_5PIPELINE_IMPLE(Op, Prefer)                  \
  template <typename DType_in, typename DType_out, typename... Args> \
  __mlu_global__ void MLUBlockKernel5StagePipeline##Op##Prefer(      \
      char *input_gdram, char *output_gdram, size_t element_num,     \
      Args... args) {}
#endif

#endif  // KERNELS_UNARY_OP_UNARY_OP_5PIPELINE_H_
