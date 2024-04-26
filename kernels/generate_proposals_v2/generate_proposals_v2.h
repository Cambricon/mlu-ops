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
#ifndef KERNELS_GENERATE_PROPOSALS_V2_GENERATE_PROPOSALS_V2_H
#define KERNELS_GENERATE_PROPOSALS_V2_GENERATE_PROPOSALS_V2_H

#include "mlu_op.h"

mluOpStatus_t MLUOP_WIN_API KernelGenerateProposalsV2(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const float *scores, const int32_t *scores_index, const float *bbox_deltas,
    const float *im_shape, const float *anchors, const float *variances,
    float *workspace, float *rpn_rois, float *rpn_roi_probs, int *rpn_rois_num,
    int *rpn_rois_batch_size, const int pre_nms_top_n, const int post_nms_top_n,
    const float nms_thresh, const float min_size, const float eta,
    const bool pixel_offset, const int batch_size, const int Anchors_num,
    const int H, const int W);

mluOpStatus_t MLUOP_WIN_API KernelGenerateProposalsV2_Default(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const float *scores, const float *bbox_deltas, const float *im_shape,
    const float *anchors, const float *variances, float *workspace,
    float *rpn_rois, float *rpn_roi_probs, int *rpn_rois_num,
    int *rpn_rois_batch_size, const int pre_nms_top_n, const int post_nms_top_n,
    const float nms_thresh, const float min_size, const float eta,
    const bool pixel_offset, const int batch_size, const int Anchors_num,
    const int H, const int W);

#endif  // KERNELS_GENERATE_PROPOSALS_V2_GENERATE_PROPOSALS_V2_H
