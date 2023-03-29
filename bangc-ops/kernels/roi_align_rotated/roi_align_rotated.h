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
#ifndef KERNELS_ROI_ALIGN_ROTATED_ROI_ALIGN_ROTATED_H
#define KERNELS_ROI_ALIGN_ROTATED_ROI_ALIGN_ROTATED_H

#include "mlu_op.h"

struct mluOpRoiAlignRotatedParams {
  int pooled_height;
  int pooled_width;
  int sample_ratio;
  float spatial_scale;
  bool aligned;
  bool clockwise;
};

void MLUOP_WIN_API KernelRoiAlignRotatedForward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, const void *features, const void *rois,
    const int batch, const int height, const int width, const int channel,
    const int rois_num, const mluOpRoiAlignRotatedParams rroiAlignParams,
    void *output);

void MLUOP_WIN_API KernelRoiAlignRotatedBackward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, const void *top_grad, const void *rois,
    const int batch, const int height, const int width, const int channel,
    const int rois_num, const mluOpRoiAlignRotatedParams rroiAlignParams,
    void *bottom_grad);

#endif  // KERNELS_ROI_ALIGN_ROTATED_ROI_ALIGN_ROTATED_H
