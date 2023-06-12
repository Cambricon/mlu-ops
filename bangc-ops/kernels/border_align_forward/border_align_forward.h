/*******************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
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
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *******************************************************************************/
#ifndef KERNELS_BORDER_ALIGN_FORWARD_BORDER_ALIGN_FORWARD_H_
#define KERNELS_BORDER_ALIGN_FORWARD_BORDER_ALIGN_FORWARD_H_

#include "mlu_op.h"
#include "kernels/debug.h"
#include "kernels/kernel.h"

void MLUOP_WIN_API KernelBorderAlignForward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, const void *input, const void *boxes,
    const int32_t pool_size, const int32_t N, const int32_t H, const int32_t W,
    const int32_t C, const int32_t K, void *output, int32_t *argmax_idx_nram);

#endif  // KERNELS_BORDER_ALIGN_FORWARD_BORDER_ALIGN_FORWARD_H_
