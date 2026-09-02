/*************************************************************************
 * Copyright (C) [2026] by Cambricon, Inc.
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
#include "box_overlap_bev.h"
#include <algorithm>

#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"

static void policyFunc(const mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type, const int num_box1,
                       const int num_box2) {
  // When current MLU arch only support Block type job
  if (mluop::runtime::getJobLimitCapability(handle) == CN_KERNEL_CLASS_BLOCK) {
    *k_type = cnrtFuncTypeBlock;
    k_dim->x = 1;
    k_dim->y = 1;
    k_dim->z = 1;
    VLOG(5) << "Launch Kernel MLUKernelBoxOverlapDev in BLOCK type";
    return;
  }

  // union1 policy func
  *k_type = cnrtFuncTypeUnion1;
  // dimx equals to num of mlu cores in each cluster
  k_dim->x = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  // dimy equals to num of current available clusters
  k_dim->y = mluop::runtime::getClusterLimitCapability(handle);
  k_dim->z = 1;

  // if total_num < 64, use only one mlu core;
  const uint32_t single_core_small_case = 64;

  if (single_core_small_case >= num_box1) {  // only 1 mlu core enough
    *k_type = cnrtFuncTypeBlock;
    k_dim->x = 1;
    k_dim->y = 1;
    VLOG(5) << "Launch Kernel MLUKernelBoxOverlapBev in BLOCK type";
    return;
  }
}

mluOpStatus_t MLUOP_WIN_API mluOpBoxOverlapBev(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t boxes1_desc,
    const void *boxes1, const mluOpTensorDescriptor_t boxes2_desc,
    const void *boxes2, const mluOpTensorDescriptor_t overlaps_desc,
    void *overlaps) {
  // desc null pointer check
  PARAM_CHECK("[mluOpBoxOverlapBev]", handle != NULL);
  PARAM_CHECK("[mluOpBoxOverlapBev]", boxes1_desc != NULL);
  PARAM_CHECK("[mluOpBoxOverlapBev]", boxes2_desc != NULL);
  PARAM_CHECK("[mluOpBoxOverlapBev]", overlaps_desc != NULL);

  // datatype check
  PARAM_CHECK("[mluOpBoxOverlapBev]",
              boxes1_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK_EQ("[mluOpBoxOverlapBev]", boxes1_desc->getDtype(),
                 boxes2_desc->getDtype());
  PARAM_CHECK_EQ("[mluOpBoxOverlapBev]", boxes1_desc->getDtype(),
                 overlaps_desc->getDtype());

  // dims and shape check
  PARAM_CHECK_EQ("[mluOpBoxOverlapBev]", boxes1_desc->getDim(), 2);
  PARAM_CHECK_EQ("[mluOpBoxOverlapBev]", boxes2_desc->getDim(), 2);
  if (boxes1_desc->getDimIndex(0) > MAX_BOX_NUM) {
    LOG(ERROR)
        << "[mluOpBoxOverlapBev]: Check failed: "
        << "the number of boxes must not exceed "
        << MAX_BOX_NUM << "."
        << "But now box1's first dimension is "
        << boxes1_desc->getDimIndex(0) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (boxes1_desc->getDimIndex(boxes1_desc->getDim() - 1) !=
      PCDET_SINGLE_BOX_DIM) {
    LOG(ERROR)
        << "[mluOpBoxOverlapBev]: Check failed: The Boxes' last dimension "
           "should be 7 ."
        << "But now box1's last dimension is "
        << boxes1_desc->getDimIndex(boxes1_desc->getDim() - 1) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (boxes2_desc->getDimIndex(0) > MAX_BOX_NUM) {
    LOG(ERROR)
        << "[mluOpBoxOverlapBev]: Check failed: "
        << "the number of boxes must not exceed "
        << MAX_BOX_NUM << "."
        << "But now box2's first dimension is "
        << boxes2_desc->getDimIndex(0) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (boxes2_desc->getDimIndex(boxes2_desc->getDim() - 1) !=
      PCDET_SINGLE_BOX_DIM) {
    LOG(ERROR)
        << "[mluOpBoxOverlapBev]: Check failed: The Boxes' last dimension "
           "should be 7 ."
        << "But now box2's last dimension is "
        << boxes2_desc->getDimIndex(boxes2_desc->getDim() - 1) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (overlaps_desc->getDimIndex(0) != boxes1_desc->getDimIndex(0)) {
    LOG(ERROR) << "[mluOpBoxOverlapBev] Check failed: "
               << "overlaps_desc->getDimIndex(0) should equal to "
                  "boxes1_desc->getDimIndex(0). But now "
               << "overlaps_desc->getDimIndex(0) is "
               << overlaps_desc->getDimIndex(0)
               << ", boxes1_desc->getDimIndex(0) is "
               << boxes1_desc->getDimIndex(0) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if (overlaps_desc->getDim() != 2) {
    LOG(ERROR) << "[mluOpBoxOverlapBev] Check failed: "
               << "overlaps_desc->getDim() should equal to 2. But now is "
               << overlaps_desc->getDim() << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if (overlaps_desc->getDimIndex(1) != boxes2_desc->getDimIndex(0)) {
    LOG(ERROR)
        << "[mluOpBoxOverlapBev] Check failed: "
        << "overlaps_desc's last dim should equal to boxes2_desc's first dim "
        << boxes2_desc->getDimIndex(0) << ", But now overlaps_desc's last dim is "
        << overlaps_desc->getDimIndex(1) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // 0-element check, after dim and shape check
  if (boxes1_desc->getDimIndex(0) * boxes2_desc->getDimIndex(0) == 0) {
    VLOG(5) << "[mluOpBoxOverlapBev] Skip zero element boxes.";
    return MLUOP_STATUS_SUCCESS;
  }

  // data nullptr should check after 0-element check
  PARAM_CHECK("[mluOpBoxOverlapBev]", boxes1 != NULL);
  PARAM_CHECK("[mluOpBoxOverlapBev]", boxes2 != NULL);
  PARAM_CHECK("[mluOpBoxOverlapBev]", overlaps != NULL);

  // check large tensor
  {
    LARGE_TENSOR_CHECK("[mluOpBoxOverlapBev]", boxes1_desc);
    LARGE_TENSOR_CHECK("[mluOpBoxOverlapBev]", boxes2_desc);
    LARGE_TENSOR_CHECK("[mluOpBoxOverlapBev]", overlaps_desc);
  }

  // stride tensor check
  STRIDE_TENSOR_CHECK("[mluOpBoxOverlapBev]:", boxes1_desc,
                      "Boxes1 tensor must be contiguous.");
  STRIDE_TENSOR_CHECK("[mluOpBoxOverlapBev]:", boxes2_desc,
                      "Boxes2 tensor must be contiguous.");
  STRIDE_TENSOR_CHECK("[mluOpBoxOverlapBev]:", overlaps_desc,
                      "Overlaps tensor must be contiguous.");

  // generate prototxt
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("box_overlap_bev", "BOX_OVERLAP_BEV");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", boxes1, boxes1_desc, 10, 0);
    GEN_CASE_DATA(true, "input", boxes2, boxes2_desc, 10, 0);
    GEN_CASE_DATA(false, "output", overlaps, overlaps_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 3e-3, 3e-3, 0);
  }

  int32_t num_box1 =
      mluOpGetTensorElementNum(boxes1_desc) / PCDET_SINGLE_BOX_DIM;
  int32_t num_box2 =
      mluOpGetTensorElementNum(boxes2_desc) / PCDET_SINGLE_BOX_DIM;

  // Choose the best task dimension.
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, &k_dim, &k_type, num_box1, num_box2);

  VLOG(5) << "[mluOpBoxOverlapBev] launch kernel policyFunc"
          << "[" << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << "].";

  CHECK_RETURN("[mluOpBoxOverlapBev]",
               (KernelBoxOverlapBev(k_dim, k_type, handle->queue,
                                    boxes1_desc->getDtype(), boxes1, boxes2,
                                    overlaps, num_box1, num_box2)));
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
