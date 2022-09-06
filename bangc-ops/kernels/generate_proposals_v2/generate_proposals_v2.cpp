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
#include "kernels/kernel.h"

static void policyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type, const int AHW)
{
  int job = mluop::runtime::getJobLimitCapability(handle);

  int per_core_num = AHW / job;
  const int min_per_core_num = 256;


  while (per_core_num < min_per_core_num  && job >= 4) {
    per_core_num *= 2;
    job /= 2;
  }
  k_dim->y = 1;
  k_dim->z = 1;
  // *k_type = CNRT_FUNC_TYPE_UNION1;
  // k_dim->x = handle->core_num_per_cluster;
  // //   *k_type = CNRT_FUNC_TYPE_BLOCK;
  // // k_dim->x = 1;
  // return;

  if(job < 4){
    k_dim->x = 1;
    *k_type = CNRT_FUNC_TYPE_BLOCK;
  } else if (job == 4) {
    *k_type = CNRT_FUNC_TYPE_UNION1;
    k_dim->x = handle->core_num_per_cluster;
  } else
  {
    *k_type = (cnrtFunctionType_t)job;
    k_dim->x = job;
  }
  return;
  switch (static_cast<KernelClass>(job))
  {
  case CN_KERNEL_CLASS_BLOCK:
    *k_type = CNRT_FUNC_TYPE_BLOCK;
    k_dim->x = 1;
    break;
  case CN_KERNEL_CLASS_UNION:
    *k_type = CNRT_FUNC_TYPE_UNION1;
    k_dim->x = handle->core_num_per_cluster;
    break;
  case CN_KERNEL_CLASS_UNION2:
    *k_type = CNRT_FUNC_TYPE_UNION2;
    k_dim->x = handle->core_num_per_cluster * 2;
    break;
  case CN_KERNEL_CLASS_UNION4:
    *k_type = CNRT_FUNC_TYPE_UNION4;
    k_dim->x = handle->core_num_per_cluster * 4;
    break;
  case CNRT_FUNC_TYPE_UNION8:
    *k_type = CNRT_FUNC_TYPE_UNION8;
    k_dim->x = handle->core_num_per_cluster * 8;
    break;
  case CNRT_FUNC_TYPE_UNION16:
    *k_type = CNRT_FUNC_TYPE_UNION16;
    k_dim->x = handle->core_num_per_cluster * 16;
    break;
  default:
    *k_type = CNRT_FUNC_TYPE_MUTABLE;
    k_dim->x = handle->core_num_per_cluster * handle->capability_job_limit;
    break;
  }

  k_dim->y = 1;
  k_dim->z = 1;
  return;
}

mluOpStatus_t MLUOP_WIN_API
mluOpGetGenerateProposalsV2WorkspaceSize(mluOpHandle_t handle,
                                         const mluOpTensorDescriptor_t scores_desc,
                                         size_t *size)
{
  int batch_size = scores_desc->dims[0];
  int Anchors_num = scores_desc->dims[1];
  int H = scores_desc->dims[2];
  int W = scores_desc->dims[3];
  *size = 6 * Anchors_num * H * W * 4 + 128 * 4;
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
                         void *rpn_rois_batch_size)
{
  int batch_size = scores_desc->dims[0];
  int Anchors_num = scores_desc->dims[1];
  int H = scores_desc->dims[2];
  int W = scores_desc->dims[3];
  int AHW = Anchors_num * H * W;
  cnrtDim3_t k_dim;
  cnrtJobType_t k_type;
  policyFunc(handle, &k_dim, &k_type, AHW);

  VLOG(5) << "Launch Kernel mluOpUBestKernelGenerateProposalsV2Float <<<k_dim: " << k_type << ", "
          << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>";
  // return MLUOP_STATUS_SUCCESS;
  KERNEL_CHECK(mluOpUBestKernelGenerateProposalsV2Float(k_dim, k_type, handle->queue, (float *)scores, (float *)bbox_deltas, (float *)im_shape, (float *)anchors, (float *)variances, (float *)workspace, (float *)rpn_rois, (float *)rpn_roi_probs, (int *)rpn_rois_num, (int *)rpn_rois_batch_size, pre_nms_top_n, post_nms_top_n, nms_thresh, min_size, eta, pixel_offset, batch_size, Anchors_num, H, W));

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
