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
#ifndef KERNELS_BINARY_OP_BINARY_OP_STRIDE_3PIPELINE_H_
#define KERNELS_BINARY_OP_BINARY_OP_STRIDE_3PIPELINE_H_

#include "mlu_op.h"
#include "kernels/kernel.h"
#include "kernels/debug.h"
#include "binary_op_host.h"
#include "kernels/utils/common.h"
#include "kernels/tensor_stride_process/tensor_stride_process_host.h"
#include "kernels/tensor_stride_process/tensor_stride_process_common.h"

#define BINARY_ALIGN_NUM 64
#define BINARY_NRAM_SIZE (MAX_NRAM_SIZE + REM_FOR_STACK - 112 * 1024)
#define BINARY_SRAM_SIZE (CORE_DIM * BINARY_NRAM_SIZE)

#define BINARY_OP_PIP3_WITH_STRIDE_DECLARE(Op, Prefer)                     \
  template <typename DType_in1, typename DType_in2, typename DType_out,    \
            typename... Args>                                              \
  __mlu_global__ void MLUBlockKernelBinaryPipe3WithStride##Op##Prefer(     \
      char *x, TensorShape x_shape, char *y, TensorShape y_shape, char *z, \
      TensorShape z_shape, size_t element_num,                             \
      mluOpComputationPreference_t prefer, Args... args);

#define BINARY_OP_PIP3_WITH_STRIDE_KERNEL(Op, Prefer)                         \
  template <typename DType_in1, typename DType_in2, typename DType_out,       \
            typename... Args>                                                 \
  __mlu_global__ void MLUBlockKernelBinaryPipe3WithStride##Op##Prefer(        \
      char *x, TensorShape x_shape, char *y, TensorShape y_shape, char *z,    \
      TensorShape z_shape, size_t element_num,                                \
      mluOpComputationPreference_t prefer, Args... args) {                    \
    if (__is_mpu()) {                                                         \
      return;                                                                 \
    }                                                                         \
    size_t output_input1_gap = 0, output_input2_gap = 0, ping_pong_gap = 0;   \
    size_t auxiliary_a_gap = 0, auxiliary_b_gap = 0, auxiliary_c_gap = 0;     \
    size_t span_num_deal = 0;                                                 \
    size_t align_num = BINARY_ALIGN_NUM;                                      \
    auxFunc3##Op##Prefer<DType_in1, DType_in2, DType_out>(                    \
        output_input1_gap, output_input2_gap, ping_pong_gap, auxiliary_a_gap, \
        auxiliary_b_gap, auxiliary_c_gap, span_num_deal, align_num, args...); \
    size_t num_per_core = element_num / taskDim;                              \
    size_t num_rem = element_num % taskDim;                                   \
    size_t core_offset = taskId * num_per_core;                               \
    size_t offset = 0;                                                        \
    if (num_rem > 0 && taskId == (taskDim - 1)) {                             \
      num_per_core = num_per_core + num_rem;                                  \
    }                                                                         \
    int repeat = num_per_core / span_num_deal;                                \
    size_t rem = num_per_core % span_num_deal;                                \
    int32_t align_rem = PAD_UP(rem, align_num);                               \
    char *ping_output = nram_buffer;                                          \
    char *ping_input1 = nram_buffer + output_input1_gap;                      \
    char *ping_input2 = nram_buffer + output_input2_gap;                      \
    char *auxiliary_a = nram_buffer + auxiliary_a_gap;                        \
    char *auxiliary_b = nram_buffer + auxiliary_b_gap;                        \
    char *auxiliary_c = nram_buffer + auxiliary_c_gap;                        \
                                                                              \
    if (repeat > 0) {                                                         \
      offset = core_offset;                                                   \
      TENSOR_STRIDE_LOAD(DType_in1, ping_input1, x, offset, span_num_deal,    \
                         sizeof(DType_in1), x_shape);                         \
      TENSOR_STRIDE_LOAD(DType_in2, ping_input2, y, offset, span_num_deal,    \
                         sizeof(DType_in2), y_shape);                         \
      __asm__ volatile("sync;");                                              \
    }                                                                         \
    if (repeat > 1) {                                                         \
      offset = core_offset + span_num_deal;                                   \
      TENSOR_STRIDE_LOAD(DType_in1, ping_input1 + ping_pong_gap, x, offset,   \
                         span_num_deal, sizeof(DType_in1), x_shape);          \
      TENSOR_STRIDE_LOAD(DType_in2, ping_input2 + ping_pong_gap, y, offset,   \
                         span_num_deal, sizeof(DType_in2), y_shape);          \
      compute##Op##Prefer<DType_in1, DType_in2, DType_out>(                   \
          ping_output, ping_input1, ping_input2, auxiliary_a, auxiliary_b,    \
          auxiliary_c, span_num_deal, span_num_deal, args...);                \
      __asm__ volatile("sync;");                                              \
    }                                                                         \
                                                                              \
    for (int32_t i = 0; i < repeat - 2; i++) {                                \
      offset = core_offset + i * span_num_deal;                               \
      TENSOR_STRIDE_STORE(DType_out, z, offset,                               \
                          ping_output + (i % 2) * ping_pong_gap,              \
                          span_num_deal, sizeof(DType_out), z_shape);         \
      offset += 2 * span_num_deal;                                            \
      TENSOR_STRIDE_LOAD(DType_in1, ping_input1 + (i % 2) * ping_pong_gap, x, \
                         offset, span_num_deal, sizeof(DType_in1), x_shape);  \
      TENSOR_STRIDE_LOAD(DType_in2, ping_input2 + (i % 2) * ping_pong_gap, y, \
                         offset, span_num_deal, sizeof(DType_in2), y_shape);  \
      compute##Op##Prefer<DType_in1, DType_in2, DType_out>(                   \
          ping_output + ((i + 1) % 2) * ping_pong_gap,                        \
          ping_input1 + ((i + 1) % 2) * ping_pong_gap,                        \
          ping_input2 + ((i + 1) % 2) * ping_pong_gap, auxiliary_a,           \
          auxiliary_b, auxiliary_c, span_num_deal, span_num_deal, args...);   \
      __asm__ volatile("sync;");                                              \
    }                                                                         \
                                                                              \
    if (repeat >= 2) {                                                        \
      offset = core_offset + (repeat - 2) * span_num_deal;                    \
      TENSOR_STRIDE_STORE(DType_out, z, offset,                               \
                          ping_output + (repeat % 2) * ping_pong_gap,         \
                          span_num_deal, sizeof(DType_out), z_shape);         \
    }                                                                         \
    if (rem > 0) {                                                            \
      offset = core_offset + repeat * span_num_deal;                          \
      TENSOR_STRIDE_LOAD(DType_in1,                                           \
                         ping_input1 + (repeat % 2) * ping_pong_gap, x,       \
                         offset, rem, sizeof(DType_in1), x_shape);            \
      TENSOR_STRIDE_LOAD(DType_in2,                                           \
                         ping_input2 + (repeat % 2) * ping_pong_gap, y,       \
                         offset, rem, sizeof(DType_in2), y_shape);            \
    }                                                                         \
    if (repeat > 0) {                                                         \
      compute##Op##Prefer<DType_in1, DType_in2, DType_out>(                   \
          ping_output + ((repeat - 1) % 2) * ping_pong_gap,                   \
          ping_input1 + ((repeat - 1) % 2) * ping_pong_gap,                   \
          ping_input2 + ((repeat - 1) % 2) * ping_pong_gap, auxiliary_a,      \
          auxiliary_b, auxiliary_c, span_num_deal, span_num_deal, args...);   \
    }                                                                         \
    __asm__ volatile("sync;");                                                \
                                                                              \
    if (repeat > 0) {                                                         \
      offset = core_offset + (repeat - 1) * span_num_deal;                    \
      TENSOR_STRIDE_STORE(DType_out, z, offset,                               \
                          ping_output + ((repeat - 1) % 2) * ping_pong_gap,   \
                          span_num_deal, sizeof(DType_out), z_shape);         \
    }                                                                         \
    if (rem > 0) {                                                            \
      compute##Op##Prefer<DType_in1, DType_in2, DType_out>(                   \
          ping_output + (repeat % 2) * ping_pong_gap,                         \
          ping_input1 + (repeat % 2) * ping_pong_gap,                         \
          ping_input2 + (repeat % 2) * ping_pong_gap, auxiliary_a,            \
          auxiliary_b, auxiliary_c, align_rem, rem, args...);                 \
      __asm__ volatile("sync;");                                              \
      offset = core_offset + repeat * span_num_deal;                          \
      TENSOR_STRIDE_STORE(DType_out, z, offset,                               \
                          ping_output + (repeat % 2) * ping_pong_gap, rem,    \
                          sizeof(DType_out), z_shape);                        \
      __asm__ volatile("sync;");                                              \
    }                                                                         \
  }
#endif  // KERNELS_BINARY_OP_BINARY_OP_STRIDE_3PIPELINE_H_
