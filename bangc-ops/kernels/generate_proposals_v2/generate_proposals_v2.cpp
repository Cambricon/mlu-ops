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
#include <algorithm>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"

mluOpStatus_t MLUOP_WIN_API
mluOpGetGenerateProposalsV2WorkspaceSize(mluOpHandle_t handle,
                             const mluOpTensorDescriptor_t scores_desc,
                             size_t *size){
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API 
mluOpGenerateProposalsV2(mluOpHandle_t handle,
                        const int pre_nms_top_n,
                        const int post_nms_top_n,
                        const float nms_thresh,
                        const float min_size,
                        const float eta,
                        bool pixel_offset,
                        const mluOpTensorDescriptor_t scores_desc,
                        const void *scores,
                        const mluOpTensorDescriptor_t bbox_deltas_desc,
                        const void *bbox_deltas,
                        const mluOpTensorDescriptor_t im_shape_desc,
                        const void *im_shape,
                        const mluOpTensorDescriptor_t anchors_desc,
                        const void *anchors,
                        const mluOpTensorDescriptor_t variances_desc,
                        const void *variances,
                        void *workspace,
                        size_t workspace_size,
                        const mluOpTensorDescriptor_t rpn_rois_desc,
                        void *rpn_rois,
                        const mluOpTensorDescriptor_t rpn_roi_probs_desc,
                        void *rpn_roi_probs,
                        const mluOpTensorDescriptor_t rpn_rois_num_desc,
                        void *rpn_rois_num,
                        void *rpn_rois_batch_size){

  return MLUOP_STATUS_SUCCESS;
}
