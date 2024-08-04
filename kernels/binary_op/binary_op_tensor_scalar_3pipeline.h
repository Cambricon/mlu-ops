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
#ifndef KERNELS_BINARY_OP_BINARY_OP_TENSOR_SCALAR_3PIPELINE_H_
#define KERNELS_BINARY_OP_BINARY_OP_TENSOR_SCALAR_3PIPELINE_H_

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

#define TENSOR_LOAD_ASYNC(T, dst_nram, src_gdram, src_offset, data_num,     \
                          dtype_size)                                       \
  __memcpy_async(dst_nram,                                                  \
                 (int8_t *)src_gdram + OFFSET_SHIFT(src_offset, sizeof(T)), \
                 data_num * dtype_size, GDRAM2NRAM);

#define TENSOR_STORE_ASYNC(T, dst_gdram, dst_offset, src_nram, data_num,    \
                           dtype_size)                                      \
  __memcpy_async((int8_t *)dst_gdram + OFFSET_SHIFT(dst_offset, sizeof(T)), \
                 src_nram, data_num * dtype_size, NRAM2GDRAM);

#define BINARY_OP_TENSOR_SCALAR_PIP3_DECLARE(Op, Prefer)                  \
  template <typename DType_in, typename DType_scalar, typename DType_out, \
            typename... Args>                                             \
  __mlu_global__ void MLUBlockKernelBinaryTensorScalarPipe3##Op##Prefer(  \
      char *input_tensor, char *input_scalar, char *output,               \
      uint32_t host_scalar, mluOpPointerMode_t pointer_mode,              \
      size_t element_num, Args... args);

#define BINARY_OP_TENSOR_SCALAR_PIP3_KERNEL(Op, Prefer)                        \
  template <typename DType_in, typename DType_scalar, typename DType_out,      \
            typename... Args>                                                  \
  __mlu_global__ void MLUBlockKernelBinaryTensorScalarPipe3##Op##Prefer(       \
      char *input_tensor, char *input_scalar, char *output_tensor,             \
      uint32_t host_scalar, mluOpPointerMode_t pointer_mode,                   \
      size_t element_num, Args... args) {                                      \
    if (__is_mpu()) {                                                          \
      return;                                                                  \
    }                                                                          \
    size_t output_input_gap = 0, ping_pong_gap = 0;                            \
    size_t auxiliary_a_gap = 0, auxiliary_b_gap = 0, auxiliary_c_gap = 0;      \
    size_t span_num_deal = 0;                                                  \
    size_t dtsize_in = sizeof(DType_in);                                       \
    size_t dtsize_out = sizeof(DType_out);                                     \
    size_t align_num =                                                         \
        (sizeof(DType_out) == 1) ? NFU_ALIGN_SIZE : BINARY_ALIGN_NUM;          \
    auxFunc3##Op##Prefer<DType_in, DType_scalar, DType_out>(                   \
        output_input_gap, ping_pong_gap, auxiliary_a_gap, auxiliary_b_gap,     \
        auxiliary_c_gap, span_num_deal, align_num, args...);                   \
                                                                               \
    size_t num_per_core = element_num / taskDim;                               \
    size_t num_rem = element_num % taskDim;                                    \
    size_t core_offset = taskId * num_per_core;                                \
    size_t offset = 0;                                                         \
    if (num_rem > 0 && taskId == (taskDim - 1)) {                              \
      num_per_core = num_per_core + num_rem;                                   \
    }                                                                          \
                                                                               \
    int32_t repeat = num_per_core / span_num_deal;                             \
    size_t rem = num_per_core % span_num_deal;                                 \
    size_t align_rem = PAD_UP(rem, align_num);                                 \
    char *ping_output = nram_buffer;                                           \
    char *ping_input = nram_buffer + output_input_gap;                         \
    char *auxiliary_a = nram_buffer + auxiliary_a_gap;                         \
    char *auxiliary_b = nram_buffer + auxiliary_b_gap;                         \
    char *auxiliary_c = nram_buffer + auxiliary_c_gap;                         \
    char *scalar = nram_buffer + BINARY_NRAM_SIZE;                             \
    if (pointer_mode == MLUOP_POINTER_MODE_HOST) {                             \
      ((DType_scalar *)scalar)[0] = *((DType_scalar *)&host_scalar);           \
    } else {                                                                   \
      ((DType_scalar *)scalar)[0] = ((DType_scalar *)input_scalar)[0];         \
    }                                                                          \
                                                                               \
    if (repeat > 0) {                                                          \
      offset = core_offset;                                                    \
      TENSOR_LOAD_ASYNC(DType_in, ping_input, input_tensor, offset,            \
                        span_num_deal, dtsize_in);                             \
      __asm__ volatile("sync;");                                               \
    }                                                                          \
    if (repeat > 1) {                                                          \
      offset = core_offset + span_num_deal;                                    \
      TENSOR_LOAD_ASYNC(DType_in, ping_input + ping_pong_gap, input_tensor,    \
                        offset, span_num_deal, dtsize_in);                     \
      compute##Op##Prefer<DType_in, DType_scalar, DType_out>(                  \
          ping_output, ping_input, scalar, auxiliary_a, auxiliary_b,           \
          auxiliary_c, span_num_deal, span_num_deal, args...);                 \
      __asm__ volatile("sync;");                                               \
    }                                                                          \
    for (int32_t i = 0; i < repeat - 2; ++i) {                                 \
      offset = core_offset + i * span_num_deal;                                \
      TENSOR_STORE_ASYNC(DType_out, output_tensor, offset,                     \
                         ping_output + (i % 2) * ping_pong_gap, span_num_deal, \
                         dtsize_out);                                          \
      offset += 2 * span_num_deal;                                             \
      TENSOR_LOAD_ASYNC(DType_in, ping_input + (i % 2) * ping_pong_gap,        \
                        input_tensor, offset, span_num_deal, dtsize_in);       \
      compute##Op##Prefer<DType_in, DType_scalar, DType_out>(                  \
          ping_output + ((i + 1) % 2) * ping_pong_gap,                         \
          ping_input + ((i + 1) % 2) * ping_pong_gap, scalar, auxiliary_a,     \
          auxiliary_b, auxiliary_c, span_num_deal, span_num_deal, args...);    \
      __asm__ volatile("sync;");                                               \
    }                                                                          \
    if (repeat > 1) {                                                          \
      offset = core_offset + (repeat - 2) * span_num_deal;                     \
      TENSOR_STORE_ASYNC(DType_out, output_tensor, offset,                     \
                         ping_output + (repeat % 2) * ping_pong_gap,           \
                         span_num_deal, dtsize_out);                           \
    }                                                                          \
    if (rem > 0) {                                                             \
      offset = core_offset + repeat * span_num_deal;                           \
      TENSOR_LOAD_ASYNC(DType_in, ping_input + (repeat % 2) * ping_pong_gap,   \
                        input_tensor, offset, rem, dtsize_in);                 \
    }                                                                          \
    if (repeat > 0) {                                                          \
      compute##Op##Prefer<DType_in, DType_scalar, DType_out>(                  \
          ping_output + ((repeat - 1) % 2) * ping_pong_gap,                    \
          ping_input + ((repeat - 1) % 2) * ping_pong_gap, scalar,             \
          auxiliary_a, auxiliary_b, auxiliary_c, span_num_deal, span_num_deal, \
          args...);                                                            \
    }                                                                          \
    __asm__ volatile("sync;");                                                 \
    if (repeat > 0) {                                                          \
      offset = core_offset + (repeat - 1) * span_num_deal;                     \
      TENSOR_STORE_ASYNC(DType_out, output_tensor, offset,                     \
                         ping_output + ((repeat - 1) % 2) * ping_pong_gap,     \
                         span_num_deal, dtsize_out);                           \
    }                                                                          \
    if (rem > 0) {                                                             \
      compute##Op##Prefer<DType_in, DType_scalar, DType_out>(                  \
          ping_output + (repeat % 2) * ping_pong_gap,                          \
          ping_input + (repeat % 2) * ping_pong_gap, scalar, auxiliary_a,      \
          auxiliary_b, auxiliary_c, align_rem, rem, args...);                  \
      __asm__ volatile("sync;");                                               \
      offset = core_offset + repeat * span_num_deal;                           \
      TENSOR_STORE_ASYNC(DType_out, output_tensor, offset,                     \
                         ping_output + (repeat % 2) * ping_pong_gap, rem,      \
                         dtsize_out);                                          \
      __asm__ volatile("sync;");                                               \
    }                                                                          \
  }

#define BINARY_OP_STRIDE_TENSOR_SCALAR_PIP3_DECLARE(Op, Prefer)                \
  template <typename DType_in, typename DType_scalar, typename DType_out,      \
            typename... Args>                                                  \
  __mlu_global__ void MLUBlockKernelBinaryStrideTensorScalarPipe3##Op##Prefer( \
      char *input_tensor, TensorShape input_tensor_shape, char *input_scalar,  \
      char *output, TensorShape output_shape, uint32_t host_scalar,            \
      mluOpPointerMode_t pointer_mode, size_t element_num, Args... args);

#define BINARY_OP_STRIDE_TENSOR_SCALAR_PIP3_KERNEL(Op, Prefer)                 \
  template <typename DType_in, typename DType_scalar, typename DType_out,      \
            typename... Args>                                                  \
  __mlu_global__ void MLUBlockKernelBinaryStrideTensorScalarPipe3##Op##Prefer( \
      char *input_tensor, TensorShape input_tensor_shape, char *input_scalar,  \
      char *output_tensor, TensorShape output_tensor_shape,                    \
      uint32_t host_scalar, mluOpPointerMode_t pointer_mode,                   \
      size_t element_num, Args... args) {                                      \
    if (__is_mpu()) {                                                          \
      return;                                                                  \
    }                                                                          \
    size_t output_input_gap = 0, ping_pong_gap = 0;                            \
    size_t auxiliary_a_gap = 0, auxiliary_b_gap = 0, auxiliary_c_gap = 0;      \
    size_t span_num_deal = 0;                                                  \
    size_t dtsize_in = sizeof(DType_in);                                       \
    size_t dtsize_out = sizeof(DType_out);                                     \
    size_t align_num = BINARY_ALIGN_NUM;                                       \
    auxFunc3##Op##Prefer<DType_in, DType_scalar, DType_out>(                   \
        output_input_gap, ping_pong_gap, auxiliary_a_gap, auxiliary_b_gap,     \
        auxiliary_c_gap, span_num_deal, align_num, args...);                   \
                                                                               \
    size_t num_per_core = element_num / taskDim;                               \
    size_t num_rem = element_num % taskDim;                                    \
    size_t core_offset = taskId * num_per_core;                                \
    size_t offset = 0;                                                         \
    if (num_rem > 0 && taskId == (taskDim - 1)) {                              \
      num_per_core = num_per_core + num_rem;                                   \
    }                                                                          \
                                                                               \
    int32_t repeat = num_per_core / span_num_deal;                             \
    size_t rem = num_per_core % span_num_deal;                                 \
    size_t align_rem = PAD_UP(rem, align_num);                                 \
    char *ping_output = nram_buffer;                                           \
    char *ping_input = nram_buffer + output_input_gap;                         \
    char *auxiliary_a = nram_buffer + auxiliary_a_gap;                         \
    char *auxiliary_b = nram_buffer + auxiliary_b_gap;                         \
    char *auxiliary_c = nram_buffer + auxiliary_c_gap;                         \
    char *scalar = nram_buffer + BINARY_NRAM_SIZE;                             \
    if (pointer_mode == MLUOP_POINTER_MODE_HOST) {                             \
      ((DType_scalar *)scalar)[0] = *((DType_scalar *)&host_scalar);           \
    } else {                                                                   \
      ((DType_scalar *)scalar)[0] = ((DType_scalar *)input_scalar)[0];         \
    }                                                                          \
                                                                               \
    if (repeat > 0) {                                                          \
      offset = core_offset;                                                    \
      TENSOR_STRIDE_LOAD(DType_in, ping_input, input_tensor, offset,           \
                         span_num_deal, dtsize_in, input_tensor_shape);        \
      __asm__ volatile("sync;");                                               \
    }                                                                          \
    if (repeat > 1) {                                                          \
      offset = core_offset + span_num_deal;                                    \
      TENSOR_STRIDE_LOAD(DType_in, ping_input + ping_pong_gap, input_tensor,   \
                         offset, span_num_deal, dtsize_in,                     \
                         input_tensor_shape);                                  \
      compute##Op##Prefer<DType_in, DType_scalar, DType_out>(                  \
          ping_output, ping_input, scalar, auxiliary_a, auxiliary_b,           \
          auxiliary_c, span_num_deal, span_num_deal, args...);                 \
      __asm__ volatile("sync;");                                               \
    }                                                                          \
    for (int32_t i = 0; i < repeat - 2; ++i) {                                 \
      offset = core_offset + i * span_num_deal;                                \
      TENSOR_STRIDE_STORE(DType_out, output_tensor, offset,                    \
                          ping_output + (i % 2) * ping_pong_gap,               \
                          span_num_deal, dtsize_out, output_tensor_shape);     \
      offset += 2 * span_num_deal;                                             \
      TENSOR_STRIDE_LOAD(DType_in, ping_input + (i % 2) * ping_pong_gap,       \
                         input_tensor, offset, span_num_deal, dtsize_in,       \
                         input_tensor_shape);                                  \
      compute##Op##Prefer<DType_in, DType_scalar, DType_out>(                  \
          ping_output + ((i + 1) % 2) * ping_pong_gap,                         \
          ping_input + ((i + 1) % 2) * ping_pong_gap, scalar, auxiliary_a,     \
          auxiliary_b, auxiliary_c, span_num_deal, span_num_deal, args...);    \
      __asm__ volatile("sync;");                                               \
    }                                                                          \
    if (repeat > 1) {                                                          \
      offset = core_offset + (repeat - 2) * span_num_deal;                     \
      TENSOR_STRIDE_STORE(DType_out, output_tensor, offset,                    \
                          ping_output + (repeat % 2) * ping_pong_gap,          \
                          span_num_deal, dtsize_out, output_tensor_shape);     \
    }                                                                          \
    if (rem > 0) {                                                             \
      offset = core_offset + repeat * span_num_deal;                           \
      TENSOR_STRIDE_LOAD(DType_in, ping_input + (repeat % 2) * ping_pong_gap,  \
                         input_tensor, offset, rem, dtsize_in,                 \
                         input_tensor_shape);                                  \
    }                                                                          \
    if (repeat > 0) {                                                          \
      compute##Op##Prefer<DType_in, DType_scalar, DType_out>(                  \
          ping_output + ((repeat - 1) % 2) * ping_pong_gap,                    \
          ping_input + ((repeat - 1) % 2) * ping_pong_gap, scalar,             \
          auxiliary_a, auxiliary_b, auxiliary_c, span_num_deal, span_num_deal, \
          args...);                                                            \
    }                                                                          \
    __asm__ volatile("sync;");                                                 \
    if (repeat > 0) {                                                          \
      offset = core_offset + (repeat - 1) * span_num_deal;                     \
      TENSOR_STRIDE_STORE(DType_out, output_tensor, offset,                    \
                          ping_output + ((repeat - 1) % 2) * ping_pong_gap,    \
                          span_num_deal, dtsize_out, output_tensor_shape);     \
    }                                                                          \
    if (rem > 0) {                                                             \
      compute##Op##Prefer<DType_in, DType_scalar, DType_out>(                  \
          ping_output + (repeat % 2) * ping_pong_gap,                          \
          ping_input + (repeat % 2) * ping_pong_gap, scalar, auxiliary_a,      \
          auxiliary_b, auxiliary_c, align_rem, rem, args...);                  \
      __asm__ volatile("sync;");                                               \
      offset = core_offset + repeat * span_num_deal;                           \
      TENSOR_STRIDE_STORE(DType_out, output_tensor, offset,                    \
                          ping_output + (repeat % 2) * ping_pong_gap, rem,     \
                          dtsize_out, output_tensor_shape);                    \
      __asm__ volatile("sync;");                                               \
    }                                                                          \
  }
#endif  // KERNELS_BINARY_OP_BINARY_OP_TENSOR_SCALAR_3PIPELINE_H_
