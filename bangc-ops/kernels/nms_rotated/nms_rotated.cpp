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
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"

// each box data contains 5 number: x, y, w, h, a
#define SINGLE_BOX_DIM (5)

static void policyFunc(const mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type, const int box_num) {
  // When current MLU arch only support Block type job
  if (mluop::runtime::getJobLimitCapability(handle) == CN_KERNEL_CLASS_BLOCK) {
    *k_type = CNRT_FUNC_TYPE_BLOCK;
    k_dim->x = 1;
    k_dim->y = 1;
    k_dim->z = 1;
    VLOG(5) << "Launch Kernel MLUKernelNmsRotated in BLOCK type";
    return;
  }
  // union1 policy func
  *k_type = CNRT_FUNC_TYPE_UNION1;
  // dimx equals to num of ipu cores in each cluster
  k_dim->x = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  k_dim->y = 1;
  k_dim->z = 1;
  VLOG(5) << "Launch Kernel MLUKernelNmsRotated in UNION1 type";
}

mluOpStatus_t MLUOP_WIN_API mluOpGetNmsRotatedWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t boxes_desc,
    size_t *workspace_size) {
  PARAM_CHECK("[mluOpGetNmsRotatedWorkspaceSize]", handle != nullptr);
  PARAM_CHECK("[mluOpGetNmsRotatedWorkspaceSize]", boxes_desc != nullptr);
  PARAM_CHECK("[mluOpGetNmsRotatedWorkspaceSize]", workspace_size != nullptr);
  const uint64_t box_num = boxes_desc->dims[0];
  const uint64_t box_dim = boxes_desc->dims[1];
  uint64_t total_num = box_num * box_dim + box_num;
  *workspace_size = total_num * mluop::getSizeOfDataType(boxes_desc->dtype);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpNmsRotated(mluOpHandle_t handle, const float iou_threshold,
                const mluOpTensorDescriptor_t boxes_desc,
                const void *boxes, const mluOpTensorDescriptor_t scores_desc,
                const void *scores, void *workspace, size_t workspace_size,
                const mluOpTensorDescriptor_t output_desc, void *output,
                int32_t *result_num) {
  // desc null pointer check
  PARAM_CHECK("[mluOpNmsRotated]", handle != NULL);
  PARAM_CHECK("[mluOpNmsRotated]", boxes_desc != NULL);
  PARAM_CHECK("[mluOpNmsRotated]", scores_desc != NULL);
  PARAM_CHECK("[mluOpNmsRotated]", output_desc != NULL);

  // datatype check
  PARAM_CHECK("[mluOpNmsRotated]", boxes_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK_EQ("[mluOpNmsRotated]", boxes_desc->dtype, scores_desc->dtype);
  PARAM_CHECK("[mluOpNmsRotated]", output_desc->dtype == MLUOP_DTYPE_INT32);

  // dims and shape check
  PARAM_CHECK_EQ("[mluOpNmsRotated]", boxes_desc->dim, 2);
  PARAM_CHECK_EQ("[mluOpNmsRotated]", scores_desc->dim, 1);
  PARAM_CHECK_EQ("[mluOpNmsRotated]", output_desc->dim, 1);

  PARAM_CHECK("[mluOpNmsRotated]", boxes_desc->dims[0] == scores_desc->dims[0]);
  PARAM_CHECK("[mluOpNmsRotated]", boxes_desc->dims[0] == output_desc->dims[0]);
  if (boxes_desc->dims[1] != SINGLE_BOX_DIM &&
        boxes_desc->dims[1] != (SINGLE_BOX_DIM + 1)) {
    LOG(ERROR)
      << "[mluOpNmsRotated] Check failed: The Boxes' last dimenstion "
          "should be 5 or 6. Now is " << boxes_desc->dims[1] << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  const uint64_t tensor_boxes_num = mluOpGetTensorElementNum(boxes_desc);
  TENSOR_NUM_CHECK("[mluOpNmsRotated]", tensor_boxes_num, LARGE_TENSOR_NUM, "");
  // 0-element check, after dim and shape check
  if (boxes_desc->dims[0] == 0) {
    VLOG(5) << "[mluOpNmsRotated] Skip zero element boxes.";
    return MLUOP_STATUS_SUCCESS;
  }

  // data nullptr should check after 0-element check
  PARAM_CHECK("[mluOpNmsRotated]", boxes != NULL);
  PARAM_CHECK("[mluOpNmsRotated]", scores != NULL);
  PARAM_CHECK("[mluOpNmsRotated]", output != NULL);
  PARAM_CHECK("[mluOpNmsRotated]", result_num != NULL);
  if (workspace_size != 0) {
    PARAM_CHECK("[mluOpNmsRotated]", workspace != NULL);
  }

  // generate prototxt
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("nms_rotated");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA_REAL(true, "input1", boxes, boxes_desc);
    GEN_CASE_DATA_REAL(true, "input2", scores, scores_desc);
    GEN_CASE_DATA_REAL(false, "output1", output, output_desc);
    GEN_CASE_DATA_UNFOLD(false, "output2", result_num, 1, {1},
                MLUOP_DTYPE_INT32, MLUOP_LAYOUT_ARRAY, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "nms_rotated", "iou_threshold", iou_threshold);
    GEN_CASE_TEST_PARAM_NEW(false, false, true, 3e-3, 3e-3, 0);
  }

  float p = iou_threshold;
  if (std::isnan(iou_threshold)) {
    p = INFINITY;
  }

  int32_t box_num = boxes_desc->dims[0];
  int32_t box_dim = boxes_desc->dims[1];
  // Choose the best task dimension.
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, &k_dim, &k_type, box_num);

  // transpose box [N, box_dim] -> [box_dim, N]
  char *box_workspace = (char*)workspace;
  char *scores_workspace = box_workspace +
        mluop::getSizeOfDataType(boxes_desc->dtype) * box_num * box_dim;

  VLOG(5) << "[mluOpNmsRotated] launch kernel [" << k_dim.x << ", "
          << k_dim.y << ", " << k_dim.z << "].";

  KERNEL_CHECK((mluOpUnionKernelNmsRotatedFloat(
      k_dim, k_type, handle->queue, boxes, box_workspace, scores,
      scores_workspace, output, result_num, box_num, box_dim, p)));

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
