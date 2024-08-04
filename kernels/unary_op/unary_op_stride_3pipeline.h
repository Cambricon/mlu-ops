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

#ifndef KERNELS_UNARY_OP_UNARY_OP_STRIDE_3PIPELINE_H_
#define KERNELS_UNARY_OP_UNARY_OP_STRIDE_3PIPELINE_H_

#include "kernels/kernel.h"
#include "kernels/utils/common.h"
#include "kernels/tensor_stride_process/tensor_stride_process_common.h"
#include "kernels/tensor_stride_process/tensor_stride_process_host.h"
#include "kernels/tensor_stride_process/tensor_stride_process_mlu.h"

#define ALIGN_NUM 64
#define UNARY_NRAM_SIZE (MAX_NRAM_SIZE + REM_FOR_STACK - 148 * 1024)
#define UNARY_SRAM_SIZE (CORE_DIM * UNARY_NRAM_SIZE)

#define UNARY_OP_KERNEL_3PIPELINE_WITH_STRIDE_DECLARE(Op, Prefer)         \
  template <typename DType_in, typename DType_out, typename... Args>      \
  __mlu_global__ void MLUBlockKernel3StagePipelineWithStride##Op##Prefer( \
      char *x, mluop::TensorShape x_shape, char *y,                       \
      mluop::TensorShape y_shape, size_t element_num, Args... args);

#define UNARY_OP_KERNEL_3PIPELINE_WITH_STRIDE_IMPLE(Op, Prefer)                \
  template <typename DType_in, typename DType_out, typename... Args>           \
  __mlu_global__ void MLUBlockKernel3StagePipelineWithStride##Op##Prefer(      \
      char *input_gdram, mluop::TensorShape x_shape, char *output_gdram,       \
      mluop::TensorShape y_shape, size_t element_num, Args... args) {          \
    if (__is_mpu()) {                                                          \
      return;                                                                  \
    }                                                                          \
    /* The gap of input and output.*/                                          \
    size_t output_input_gap = 0;                                               \
    /* The gap of ping and pong.*/                                             \
    size_t ping_pong_gap = 0;                                                  \
    /* The gap of two auxiliary pointers.*/                                    \
    size_t auxiliary_a_gap = 0, auxiliary_b_gap = 0;                           \
    /* The number(not size) of data to be dealt once.*/                        \
    size_t num_deal = 0;                                                       \
    size_t align_num = 1;                                                      \
                                                                               \
    auxFunc3##Op##Prefer<DType_in, DType_out>(                                 \
        output_input_gap, ping_pong_gap, auxiliary_a_gap, auxiliary_b_gap,     \
        num_deal, align_num, args...);                                         \
                                                                               \
    size_t num_per_core = element_num / taskDim;                               \
    size_t num_rem = element_num % taskDim;                                    \
    size_t task_offset = taskId * num_per_core;                                \
    if (num_rem && taskId == taskDim - 1) {                                    \
      num_per_core += num_rem;                                                 \
    }                                                                          \
                                                                               \
    int repeat = num_per_core / num_deal;                                      \
    size_t rem = num_per_core % num_deal;                                      \
    size_t rem_align = CEIL_ALIGN(rem, align_num);                             \
                                                                               \
    char *ping_output = nram_buffer;                                           \
    char *ping_input = nram_buffer + output_input_gap;                         \
    /* Two auxiliary pointers.*/                                               \
    char *auxiliary_a = nram_buffer + auxiliary_a_gap;                         \
    char *auxiliary_b = nram_buffer + auxiliary_b_gap;                         \
                                                                               \
    if (repeat > 0) {                                                          \
      tensorStrideLoad<DType_in>(ping_input, input_gdram, task_offset,         \
                                 num_deal, sizeof(DType_in), x_shape);         \
      __asm__ volatile("sync;");                                               \
    }                                                                          \
    if (repeat > 1) {                                                          \
      tensorStrideLoad<DType_in>(ping_input + ping_pong_gap, input_gdram,      \
                                 task_offset + num_deal, num_deal,             \
                                 sizeof(DType_in), x_shape);                   \
      compute##Op##Prefer<DType_in, DType_out>(ping_output, ping_input,        \
                                               auxiliary_a, auxiliary_b,       \
                                               num_deal, num_deal, args...);   \
      __asm__ volatile("sync;");                                               \
    }                                                                          \
    for (int i = 0; i < repeat - 2; i++) {                                     \
      tensorStrideStore<DType_out>(output_gdram, task_offset + i * num_deal,   \
                                   ping_output + (i % 2) * ping_pong_gap,      \
                                   num_deal, sizeof(DType_out), y_shape);      \
      tensorStrideLoad<DType_in>(ping_input + (i % 2) * ping_pong_gap,         \
                                 input_gdram,                                  \
                                 task_offset + (i + 2) * num_deal, num_deal,   \
                                 sizeof(DType_in), x_shape);                   \
      compute##Op##Prefer<DType_in, DType_out>(                                \
          ping_output + ((i + 1) % 2) * ping_pong_gap,                         \
          ping_input + ((i + 1) % 2) * ping_pong_gap, auxiliary_a,             \
          auxiliary_b, num_deal, num_deal, args...);                           \
      __asm__ volatile("sync;");                                               \
    }                                                                          \
    if (repeat > 1) {                                                          \
      tensorStrideStore<DType_out>(                                            \
          output_gdram, task_offset + (repeat - 2) * num_deal,                 \
          ping_output + ((repeat - 2) % 2) * ping_pong_gap, num_deal,          \
          sizeof(DType_out), y_shape);                                         \
    }                                                                          \
    if (rem) {                                                                 \
      tensorStrideLoad<DType_in>(ping_input + (repeat % 2) * ping_pong_gap,    \
                                 input_gdram, task_offset + repeat * num_deal, \
                                 rem, sizeof(DType_in), x_shape);              \
    }                                                                          \
    if (repeat > 0) {                                                          \
      compute##Op##Prefer<DType_in, DType_out>(                                \
          ping_output + ((repeat - 1) % 2) * ping_pong_gap,                    \
          ping_input + ((repeat - 1) % 2) * ping_pong_gap, auxiliary_a,        \
          auxiliary_b, num_deal, num_deal, args...);                           \
    }                                                                          \
    __asm__ volatile("sync;");                                                 \
    if (repeat > 0) {                                                          \
      tensorStrideStore<DType_out>(                                            \
          output_gdram, task_offset + (repeat - 1) * num_deal,                 \
          ping_output + ((repeat - 1) % 2) * ping_pong_gap, num_deal,          \
          sizeof(DType_out), y_shape);                                         \
    }                                                                          \
    if (rem) {                                                                 \
      compute##Op##Prefer<DType_in, DType_out>(                                \
          ping_output + (repeat % 2) * ping_pong_gap,                          \
          ping_input + (repeat % 2) * ping_pong_gap, auxiliary_a, auxiliary_b, \
          rem_align, rem, args...);                                            \
      __asm__ volatile("sync;");                                               \
      tensorStrideStore<DType_out>(output_gdram,                               \
                                   task_offset + repeat * num_deal,            \
                                   ping_output + (repeat % 2) * ping_pong_gap, \
                                   rem, sizeof(DType_out), y_shape);           \
    }                                                                          \
  }

#endif  // KERNELS_UNARY_OP_UNARY_OP_STRIDE_3PIPELINE_H_
