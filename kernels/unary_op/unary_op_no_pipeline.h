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

#ifndef KERNELS_UNARY_OP_UNARY_OP_NO_PIPELINE_H_
#define KERNELS_UNARY_OP_UNARY_OP_NO_PIPELINE_H_

#include "kernels/kernel.h"
#include "kernels/utils/common.h"

#define UNARY_OP_NO_PIPELINE_DECLARE(Op, Prefer)                     \
  template <typename DType_in, typename DType_out, typename... Args> \
  __mlu_global__ void MLUBlockKernelNoPipeline##Op##Prefer(          \
      char *x, char *y, size_t element_num, Args... args);

#define UNARY_OP_NO_PIPELINE_IMPLE(Op, Prefer)                               \
  template <typename DType_in, typename DType_out, typename... Args>         \
  __mlu_global__ void MLUBlockKernelNoPipeline##Op##Prefer(                  \
      char *input_gdram, char *output_gdram, size_t element_num,             \
      Args... args) {                                                        \
    if (__is_mpu()) {                                                        \
      return;                                                                \
    }                                                                        \
    size_t output_input_gap = 0, auxiliary_a_gap = 0, auxiliary_b_gap = 0;   \
    size_t span_num_deal = 0;                                                \
    size_t align_num = 1;                                                    \
    auxFuncNoPipe##Op##Prefer<DType_in, DType_out>(                          \
        output_input_gap, auxiliary_a_gap, auxiliary_b_gap, span_num_deal,   \
        align_num, args...);                                                 \
    size_t num_per_core = element_num / taskDim;                             \
    size_t num_rem = element_num % taskDim;                                  \
    char *input_start =                                                      \
        input_gdram + taskId * num_per_core * sizeof(DType_in);              \
    char *output_start =                                                     \
        output_gdram + taskId * num_per_core * sizeof(DType_out);            \
    if (num_rem > 0 && taskId == taskDim - 1) {                              \
      num_per_core = num_per_core + num_rem;                                 \
    }                                                                        \
    int repeat = num_per_core / span_num_deal;                               \
    size_t rem = num_per_core % span_num_deal;                               \
    size_t align_rem = CEIL_ALIGN(rem, align_num);                           \
    char *output = nram_buffer;                                              \
    char *input = nram_buffer + output_input_gap;                            \
    char *auxiliary_a = nram_buffer + auxiliary_a_gap;                       \
    char *auxiliary_b = nram_buffer + auxiliary_b_gap;                       \
    size_t span_load_size = span_num_deal * sizeof(DType_in);                \
    size_t span_store_size = span_num_deal * sizeof(DType_out);              \
    for (int i = 0; i < repeat; ++i) {                                       \
      __memcpy(input, input_start + i * span_load_size, span_load_size,      \
               GDRAM2NRAM);                                                  \
      compute##Op##Prefer<DType_in, DType_out>(output, input, auxiliary_a,   \
                                               auxiliary_b, span_num_deal,   \
                                               span_num_deal, args...);      \
      __memcpy(output_start + i * span_store_size, output, span_store_size,  \
               NRAM2GDRAM);                                                  \
    }                                                                        \
    if (rem > 0) {                                                           \
      __memcpy(input, input_start + repeat * span_load_size,                 \
               rem * sizeof(DType_in), GDRAM2NRAM);                          \
      compute##Op##Prefer<DType_in, DType_out>(                              \
          output, input, auxiliary_a, auxiliary_b, align_rem, rem, args...); \
      __memcpy(output_start + repeat * span_store_size, output,              \
               rem * sizeof(DType_out), NRAM2GDRAM);                         \
    }                                                                        \
  }

#endif  // KERNELS_UNARY_OP_UNARY_OP_NO_PIPELINE_H_
