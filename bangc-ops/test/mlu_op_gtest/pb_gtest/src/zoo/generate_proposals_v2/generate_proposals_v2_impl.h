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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_GENERATE_PROPOSALS_v2_GENERATE_PROPOSALS_v2_IMPL_H_  // NOLINT
#define TEST_MLU_OP_GTEST_SRC_ZOO_GENERATE_PROPOSALS_v2_GENERATE_PROPOSALS_v2_IMPL_H_  // NOLINT

#include <vector>

#include "executor.h"

namespace GenerateProposalsV2 {
void generateProposalsV2CPUImpl(
    float* scores, float* bbox_deltas, float* im_shape, float* anchors,
    float* variances, const int pre_nms_top_n, const int post_nms_top_n,
    const float nms_thresh, const float min_size, const float eta,
    bool pixel_offset, const int N, const int H, const int W, const int A,
    float* rpn_rois, float* rpn_roi_probs, float* rpn_rois_num,
    float* rpn_rois_batch_size);
}  // namespace GenerateProposalsV2

#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_GENERATE_PROPOSALS_v2_GENERATE_PROPOSALS_v2_IMPL_H_  // NOLINT
