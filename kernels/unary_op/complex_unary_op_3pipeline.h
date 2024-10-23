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
#ifndef KERNELS_UNARY_OP_COMPLEX_UNARY_OP_3PIPELINE_H_
#define KERNELS_UNARY_OP_COMPLEX_UNARY_OP_3PIPELINE_H_

#include "kernels/debug.h"
#include "kernels/utils/common.h"

#define ALIGN_NUM 64
#define UNARY_NRAM_SIZE (MAX_NRAM_SIZE + REM_FOR_STACK - 148 * 1024)
#define UNARY_SRAM_SIZE (CORE_DIM * UNARY_NRAM_SIZE)

// Instructions for use are available at wiki:76990366 in red domain
#define COMPLEX_UNARY_OP_KERNEL_3PIPELINE_DECLARE(Op, Prefer)          \
  template <typename DType_in, typename DType_out, typename... Args>   \
  __mlu_global__ void MLUBlockKernel3StagePipelineComplex##Op##Prefer( \
      int8_t *x, int8_t *y, size_t element_num, Args... args);

#define COMPLEX_UNARY_OP_KERNEL_3PIPELINE_IMPLE(Op, Prefer)                    \
  template <typename DType_in, typename DType_out, typename... Args>           \
  __mlu_global__ void MLUBlockKernel3StagePipelineComplex##Op##Prefer(         \
      int8_t *input_gdram, int8_t *output_gdram, size_t element_num,           \
      Args... args) {                                                          \
    if (__is_mpu()) {                                                          \
      return;                                                                  \
    }                                                                          \
    size_t output_input_gap = 0, ping_pong_gap = 0;                            \
    size_t auxiliary_a_gap = 0, auxiliary_b_gap = 0;                           \
    size_t span_num_deal = 0;                                                  \
    size_t align_num = 1;                                                      \
    auxComplexFunc3##Op##Prefer<DType_in, DType_out>(                          \
        output_input_gap, ping_pong_gap, auxiliary_a_gap, auxiliary_b_gap,     \
        span_num_deal, align_num, args...);                                    \
    size_t num_per_core = element_num / taskDim;                               \
    size_t num_rem = element_num % taskDim;                                    \
    int8_t *input_start =                                                      \
        input_gdram + taskId * num_per_core * sizeof(DType_in);                \
    int8_t *output_start =                                                     \
        output_gdram + taskId * num_per_core * sizeof(DType_out);              \
    if (num_rem > 0 && taskId == taskDim - 1) {                                \
      num_per_core = num_per_core + num_rem;                                   \
    }                                                                          \
    int repeat = num_per_core / span_num_deal;                                 \
    size_t rem = num_per_core % span_num_deal;                                 \
    size_t align_rem = CEIL_ALIGN(rem, align_num);                             \
    int8_t *ping_output = nram_buffer;                                         \
    int8_t *ping_input = nram_buffer + output_input_gap;                       \
    int8_t *auxiliary_a = nram_buffer + auxiliary_a_gap;                       \
    int8_t *auxiliary_b = nram_buffer + auxiliary_b_gap;                       \
    size_t span_load_size = span_num_deal * sizeof(DType_in);                  \
    size_t span_store_size = span_num_deal * sizeof(DType_out);                \
    if (repeat > 0) {                                                          \
      __memcpy_async(ping_input, input_start, span_load_size, GDRAM2NRAM);     \
      __asm__ volatile("sync;");                                               \
    }                                                                          \
    if (repeat > 1) {                                                          \
      __memcpy_async(ping_input + ping_pong_gap, input_start + span_load_size, \
                     span_load_size, GDRAM2NRAM);                              \
      compute##Op##Prefer<DType_in, DType_out>(                                \
          ping_output, ping_input, auxiliary_a, auxiliary_b, span_num_deal,    \
          span_num_deal, args...);                                             \
      __asm__ volatile("sync;");                                               \
    }                                                                          \
    for (int i = 0; i < repeat - 2; i++) {                                     \
      __memcpy_async(output_start + i * span_store_size,                       \
                     ping_output + (i % 2) * ping_pong_gap, span_store_size,   \
                     NRAM2GDRAM);                                              \
      __memcpy_async(ping_input + (i % 2) * ping_pong_gap,                     \
                     input_start + (i + 2) * span_load_size, span_load_size,   \
                     GDRAM2NRAM);                                              \
      compute##Op##Prefer<DType_in, DType_out>(                                \
          ping_output + ((i + 1) % 2) * ping_pong_gap,                         \
          ping_input + ((i + 1) % 2) * ping_pong_gap, auxiliary_a,             \
          auxiliary_b, span_num_deal, span_num_deal, args...);                 \
      __asm__ volatile("sync;");                                               \
    }                                                                          \
    if (repeat > 1) {                                                          \
      __memcpy_async(output_start + (repeat - 2) * span_store_size,            \
                     ping_output + ((repeat - 2) % 2) * ping_pong_gap,         \
                     span_store_size, NRAM2GDRAM);                             \
    }                                                                          \
    if (rem > 0) {                                                             \
      __memcpy_async(ping_input + (repeat % 2) * ping_pong_gap,                \
                     input_start + repeat * span_load_size,                    \
                     rem * sizeof(DType_in), GDRAM2NRAM);                      \
    }                                                                          \
    if (repeat > 0) {                                                          \
      compute##Op##Prefer<DType_in, DType_out>(                                \
          ping_output + ((repeat - 1) % 2) * ping_pong_gap,                    \
          ping_input + ((repeat - 1) % 2) * ping_pong_gap, auxiliary_a,        \
          auxiliary_b, span_num_deal, span_num_deal, args...);                 \
    }                                                                          \
    __asm__ volatile("sync;");                                                 \
    if (repeat > 0) {                                                          \
      __memcpy_async(output_start + (repeat - 1) * span_store_size,            \
                     ping_output + ((repeat - 1) % 2) * ping_pong_gap,         \
                     span_store_size, NRAM2GDRAM);                             \
    }                                                                          \
    if (rem > 0) {                                                             \
      compute##Op##Prefer<DType_in, DType_out>(                                \
          ping_output + (repeat % 2) * ping_pong_gap,                          \
          ping_input + (repeat % 2) * ping_pong_gap, auxiliary_a, auxiliary_b, \
          align_rem, rem, args...);                                            \
      __asm__ volatile("sync;");                                               \
      __memcpy_async(output_start + repeat * span_store_size,                  \
                     ping_output + (repeat % 2) * ping_pong_gap,               \
                     rem * sizeof(DType_out), NRAM2GDRAM);                     \
    }                                                                          \
  }

#endif  // KERNELS_UNARY_OP_COMPLEX_UNARY_OP_3PIPELINE_H_
