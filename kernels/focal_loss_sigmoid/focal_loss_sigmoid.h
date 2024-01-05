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
#ifndef KERNELS_FOCAL_LOSS_SIGMOID_FOCAL_LOSS_SIGMOID_H
#define KERNELS_FOCAL_LOSS_SIGMOID_FOCAL_LOSS_SIGMOID_H

#include "mlu_op.h"

typedef enum {
  COMPUTATION_FAST = 0,           /* fastest algorithm. */
  COMPUTATION_HIGH_PRECISION = 1, /* high-precision algorithm. */
} focalLossSigmoidPreference_t;

mluOpStatus_t MLUOP_WIN_API mluOpBlockKernelFocalLossSigmoidForwardHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const focalLossSigmoidPreference_t prefer, const void *input,
    const void *target, const void *weight, const int32_t N, const int32_t C,
    const float alpha, const float gamma, void *output);

mluOpStatus_t MLUOP_WIN_API mluOpBlockKernelFocalLossSigmoidForwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const focalLossSigmoidPreference_t prefer, const void *input,
    const void *target, const void *weight, const int32_t N, const int32_t C,
    const float alpha, const float gamma, void *output);

mluOpStatus_t MLUOP_WIN_API mluOpBlockKernelFocalLossSigmoidBackwardHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, const void *target, const void *weight,
    const float gamma, const float alpha, const int32_t dim_n,
    const int32_t deal_n, const int32_t dim_c, void *output);

mluOpStatus_t MLUOP_WIN_API mluOpBlockKernelFocalLossSigmoidBackwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, const void *target, const void *weight,
    const float gamma, const float alpha, const int32_t dim_n,
    const int32_t deal_n, const int32_t dim_c, void *output);

#endif  // KERNELS_FOCAL_LOSS_SIGMOID_FOCAL_LOSS_SIGMOID_H
