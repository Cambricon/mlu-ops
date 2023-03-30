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
#ifndef KERNELS_EXPAND_EXPAND_H
#define KERNELS_EXPAND_EXPAND_H

#include "mlu_op.h"

void MLUOP_WIN_API KernelExpandTensor(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, void *output, const int32_t input_1,
    const int32_t input_2, const int32_t input_3, const int32_t input_4,
    const int32_t input_5, const int32_t input_6, const int32_t input_7,
    const int32_t input_8, const int32_t output_1, const int32_t output_2,
    const int32_t output_3, const int32_t output_4, const int32_t output_5,
    const int32_t output_6, const int32_t output_7, const int32_t output_8,
    const int dtype_size);

void MLUOP_WIN_API KernelExpandOneDim(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, void *output, const int32_t high_num,
    const int32_t expand_num, const int32_t low_num, const int dtype_size);

#endif  // KERNELS_EXPAND_EXPAND_H
