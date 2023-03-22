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
#ifndef KERNELS_ROTATED_FEATURE_ALIGN_ROTATED_FEATURE_ALIGN_H
#define KERNELS_ROTATED_FEATURE_ALIGN_ROTATED_FEATURE_ALIGN_H

#include "mlu_op.h"

void MLUOP_WIN_API KernelRotatedFeatureAlignForward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, const void *input, const void *bboxes,
    const int batches, const int height, const int width, const int channels,
    const int offset_rois, const float spatial_scale, const int points,
    void *output);

void MLUOP_WIN_API KernelRotatedFeatureAlignBackward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, const void *top_output, const void *bboxes,
    const int batches, const int height, const int width, const int channels,
    const int offset_rois, const float spatial_scale, const int points,
    void *bottom_input);

#endif  // KERNELS_ROTATED_FEATURE_ALIGN_ROTATED_FEATURE_ALIGN_H
