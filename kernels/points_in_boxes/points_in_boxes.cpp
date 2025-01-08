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
#include "kernels/points_in_boxes/points_in_boxes.h"

#include <string>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"

static mluOpStatus_t pointsInBoxesPreCheck(
    const mluOpTensorDescriptor_t points_desc,
    const mluOpTensorDescriptor_t boxes_desc,
    const mluOpTensorDescriptor_t points_indices_desc) {
  // tensor dim check
  if (points_desc->getDim() != 3 || boxes_desc->getDim() != 3) {
    LOG(ERROR) << "[mluOpPointsInBoxes] The dim size of the points and boxes"
               << " tensor must be 3."
               << " points dim size is " << points_desc->getDim()
               << ", boxes dim size is " << boxes_desc->getDim();
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (points_indices_desc->getDim() != 2) {
    LOG(ERROR) << "[mluOpPointsInBoxes] The dim size of the points_indices"
               << " tensor must be 2."
               << " points_indices dim size is "
               << points_indices_desc->getDim();
    return MLUOP_STATUS_BAD_PARAM;
  }

  // input tensor datatype check
  if (points_desc->getDtype() != MLUOP_DTYPE_FLOAT ||
      boxes_desc->getDtype() != MLUOP_DTYPE_FLOAT) {
    LOG(ERROR) << "[mluOpPointsInBoxes] The data type of the points and boxes"
               << " tensor must be MLUOP_DTYPE_FLOAT.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // output tensor datatype check
  if (points_indices_desc->getDtype() != MLUOP_DTYPE_INT32) {
    LOG(ERROR) << "[mluOpPointsInBoxes] The data type of the points_indices"
               << " tensor must be MLUOP_DTYPE_INT32.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // input tensor layout check
  if (points_desc->getLayout() != MLUOP_LAYOUT_ARRAY ||
      boxes_desc->getLayout() != MLUOP_LAYOUT_ARRAY) {
    LOG(ERROR) << "[mluOpPointsInBoxes] The layout of the points and boxes"
               << " tensor must be MLUOP_LAYOUT_ARRAY.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // output tensor layout check
  if (points_indices_desc->getLayout() != MLUOP_LAYOUT_ARRAY) {
    LOG(ERROR) << "[mluOpPointsInBoxes] The layout of the points_indices"
               << " tensor must be MLUOP_LAYOUT_ARRAY.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // tensor shape check
  if ((points_desc->getDimIndex(0) != boxes_desc->getDimIndex(0)) ||
      (points_desc->getDimIndex(0) != points_indices_desc->getDimIndex(0))) {
    LOG(ERROR) << "[mluOpPointsInBoxes] The batch size of the points, boxes, "
                  "points_indices "
               << "tensor must be same."
               << " points batch size is " << points_desc->getDimIndex(0)
               << ", boxes batch size is " << boxes_desc->getDimIndex(0)
               << ", points_indices batch size is "
               << points_indices_desc->getDimIndex(0);
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (points_desc->getDimIndex(1) != points_indices_desc->getDimIndex(1)) {
    LOG(ERROR) << "[mluOpPointsInBoxes] The points num of the points and "
                  "points_indices"
               << " tensor must be same."
               << " num of the points is " << points_desc->getDimIndex(1)
               << ", num of the points_indices is "
               << points_indices_desc->getDimIndex(1);
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (points_desc->getDimIndex(2) != 3) {
    LOG(ERROR) << "[mluOpPointsInBoxes] The points "
               << "must be 3D Coordinate System ie [x, y, z]. vs "
               << points_desc->getDimIndex(2);
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (boxes_desc->getDimIndex(2) != 7) {
    LOG(ERROR)
        << "[mluOpPointsInBoxes] The boxes "
        << "must be 3D Coordinate System ie [x, y, z, dx, dy, dz, heading]. vs "
        << boxes_desc->getDimIndex(2);
    return MLUOP_STATUS_BAD_PARAM;
  }

  // stride check
  STRIDE_TENSOR_CHECK("[mluOpPointsInBoxes]:", points_desc,
                      "points_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpPointsInBoxes]:", boxes_desc,
                      "boxes_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpPointsInBoxes]:", points_indices_desc,
                      "points_indices_desc must be contiguous");

  const size_t points_element_num = mluOpGetTensorElementNum(points_desc);
  const size_t boxes_element_num = mluOpGetTensorElementNum(boxes_desc);
  TENSOR_NUM_CHECK("[mluOpPointsInBoxes]", points_element_num, LARGE_TENSOR_NUM,
                   "");
  TENSOR_NUM_CHECK("[mluOpPointsInBoxes]", boxes_element_num, LARGE_TENSOR_NUM,
                   "");

  return MLUOP_STATUS_SUCCESS;
}

static bool isPointsInBoxes(const mluOpHandle_t handle, cnrtDim3_t &k_dim,
                            cnrtFunctionType_t &k_type,
                            const mluOpTensorDescriptor_t points_desc,
                            const mluOpTensorDescriptor_t boxes_desc,
                            const mluOpTensorDescriptor_t points_indices_desc,
                            pointsInBoxesTSI &points_in_boxes_info) {
  uint32_t max_nram_size = mluop::runtime::getNramSizeInBytes(handle);

  uint32_t boxes_size =
      boxes_desc->getDimIndex(1) * boxes_desc->getDimIndex(2) * sizeof(float);
  uint32_t nram_low_space = 9 * sizeof(float) + boxes_size;

  if (nram_low_space > max_nram_size) {
    return false;
  }

  uint32_t cluster_num = mluop::runtime::getClusterLimitCapability(handle);
  uint32_t core_dim = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  uint32_t cluster_used =
      PAD_UP(points_desc->getDimIndex(1), core_dim) / core_dim;
  cluster_used = cluster_used > cluster_num ? cluster_num : cluster_used;
  k_type = cnrtFuncTypeBlock;
  k_dim.x = 1;
  k_dim.y = cluster_used * core_dim;
  k_dim.z = 1;

  uint32_t points_deal_num = (max_nram_size - boxes_size) / (9 * sizeof(float));

  points_in_boxes_info.points_batch_offset = points_desc->getDimIndex(1) * 3;
  points_in_boxes_info.boxes_batch_offset = boxes_desc->getDimIndex(1) * 7;
  points_in_boxes_info.idx_batch_offset = points_indices_desc->getDimIndex(1);
  points_in_boxes_info.points_deal_num = points_deal_num;
  points_in_boxes_info.points_deal_offset = points_deal_num * 3;
  points_in_boxes_info.idx_deal_num = points_deal_num;

  return true;
}

mluOpStatus_t MLUOP_WIN_API mluOpPointsInBoxes(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t points_desc,
    const void *points, const mluOpTensorDescriptor_t boxes_desc,
    const void *boxes, const mluOpTensorDescriptor_t points_indices_desc,
    void *points_indices) {
  const std::string API = "[mluOpPointsInBoxes]";
  // check desc
  PARAM_CHECK(API, handle != NULL);
  PARAM_CHECK(API, points_desc != NULL);
  PARAM_CHECK(API, boxes_desc != NULL);
  PARAM_CHECK(API, points_indices_desc != NULL);

  // check dim, shape and dtype
  auto status =
      pointsInBoxesPreCheck(points_desc, boxes_desc, points_indices_desc);
  if (MLUOP_STATUS_SUCCESS != status) {
    return status;
  }

  // check zero element
  if (mluOpGetTensorElementNum(points_desc) == 0 ||
      mluOpGetTensorElementNum(boxes_desc) == 0 ||
      mluOpGetTensorElementNum(points_indices_desc) == 0) {
    VLOG(5) << "[mluOpPointsInBoxes] Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  // check ptr
  PARAM_CHECK(API, points != NULL);
  PARAM_CHECK(API, boxes != NULL);
  PARAM_CHECK(API, points_indices != NULL);

  // limitations check and policFun
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  pointsInBoxesTSI points_in_boxes_info;
  if (!isPointsInBoxes(handle, k_dim, k_type, points_desc, boxes_desc,
                       points_indices_desc, points_in_boxes_info)) {
    LOG(ERROR) << "[mluOpPointsInBoxes] The boxes "
               << "must be not exceed 23404 on MLU370 or 14042 on MLU590. vs "
               << boxes_desc->getDimIndex(1) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  // generate points_in_boxes prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("points_in_boxes", "POINTS_IN_BOXES");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "points", points, points_desc, 100.0, 0.0);
    GEN_CASE_DATA(true, "boxes", boxes, boxes_desc, 100.0, 0.0);
    GEN_CASE_DATA(false, "points_indices", points_indices, points_indices_desc,
                  0, 0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 3e-3, 3e-3, 0);
  }
  // generate points_in_boxes prototxt end!
  VLOG(5) << "[[mluOpPointsInBoxes]] Launch Kernel "
             "KernelPointsInBoxes.";
  CHECK_RETURN("[mluOpPointsInBoxes]",
               KernelPointsInBoxes(
                   k_dim, k_type, handle->queue, points_desc->getDimIndex(0),
                   points_desc->getDimIndex(1), boxes_desc->getDimIndex(1),
                   (float *)points, (float *)boxes, (int *)points_indices,
                   points_in_boxes_info.points_batch_offset,
                   points_in_boxes_info.boxes_batch_offset,
                   points_in_boxes_info.idx_batch_offset,
                   points_in_boxes_info.points_deal_num,
                   points_in_boxes_info.points_deal_offset,
                   points_in_boxes_info.idx_deal_num));

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
