/*************************************************************************
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
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef KERNELS_MASKED_IM2COL_FORWARD_MASKED_IM2COL_FORWARD_H_
#define KERNELS_MASKED_IM2COL_FORWARD_MASKED_IM2COL_FORWARD_H_

#include "mlu_op.h"

// decare func
mluOpStatus_t MLUOP_WIN_API KernelMaskedIm2colForward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const mluOpDataType_t data_dtype, const void *feature, const int height,
    const int width, const int channels, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const void *mask_h_idx,
    const void *mask_w_idx, const int mask_cnt, void *data_col);

#endif  // KERNELS_MASKED_IM2COL_FORWARD_MASKED_IM2COL_FORWARD_H_
