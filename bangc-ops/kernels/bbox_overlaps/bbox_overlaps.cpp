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
#include "mlu_op_kernel.h"
#include "core/logging.h"
#include "core/gen_case.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"

// policyFunc
static void policyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type,
                       const int32_t batch_num_all) {
  uint32_t union_num = mluop::runtime::getClusterLimitCapability(handle);
  uint32_t core_dim = handle->core_num_per_cluster;
  uint32_t core_num = union_num * core_dim;

  // Union1 policyFunc
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = core_dim;
  uint32_t need_core_num = (batch_num_all + core_dim - 1) / core_dim * core_dim;
  if (need_core_num < core_num) {
    k_dim->y = need_core_num / core_dim;
  } else {
    k_dim->y = union_num;
  }
  k_dim->z = 1;
  return;
}

mluOpStatus_t MLUOP_WIN_API mluOpBboxOverlaps(
    mluOpHandle_t handle, const int mode, const bool aligned, const int offset,
    const mluOpTensorDescriptor_t bbox1_desc, const void *bbox1,
    const mluOpTensorDescriptor_t bbox2_desc, const void *bbox2,
    const mluOpTensorDescriptor_t ious_desc, void *ious) {
  PARAM_CHECK("[mluOpBboxOverlaps]", handle != NULL);
  PARAM_CHECK("[mluOpBboxOverlaps]", bbox1_desc != NULL);
  PARAM_CHECK("[mluOpBboxOverlaps]", bbox2_desc != NULL);
  PARAM_CHECK("[mluOpBboxOverlaps]", ious_desc != NULL);

  PARAM_CHECK("[mluOpBboxOverlaps]", bbox1_desc->dtype == MLUOP_DTYPE_FLOAT ||
                                         bbox1_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK("[mluOpBboxOverlaps]", bbox1_desc->dtype == bbox2_desc->dtype);
  PARAM_CHECK("[mluOpBboxOverlaps]", bbox1_desc->dtype == ious_desc->dtype);
  PARAM_CHECK("[mluOpBboxOverlaps]", bbox1_desc->dim == 2);
  PARAM_CHECK("[mluOpBboxOverlaps]", bbox2_desc->dim == 2);

  // param check
  if (mode != 1 && mode != 0) {
    LOG(ERROR) << "[mluOpBboxOverlaps] Check failed: The mode must be 0 or 1, "
                  "but now is "
               << mode << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if (bbox1_desc->dims[bbox1_desc->dim - 1] != 4 && bbox1_desc->dims[0] != 0) {
    LOG(ERROR)
        << "[mluOpBboxOverlaps] Check failed: The Boxes' last dimenstion "
           "should be 4 or "
        << "the first dimension should be 0. But now bbox1's last dimension is "
        << bbox1_desc->dims[bbox1_desc->dim - 1]
        << ", bbox1's first dimension is " << bbox1_desc->dims[0] << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if (bbox2_desc->dims[bbox2_desc->dim - 1] != 4 && bbox2_desc->dims[0] != 0) {
    LOG(ERROR)
        << "[mluOpBboxOverlaps] Check failed: The Boxes' last dimenstion "
           "should  be 4 or "
        << "the first dimension should be 0. But now bbox2's last dimension is "
        << bbox2_desc->dims[bbox2_desc->dim - 1]
        << ", bbox2's first dimension is " << bbox2_desc->dims[0] << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if (offset != 1 && offset != 0) {
    LOG(ERROR) << "[mluOpBboxOverlaps] Check failed: The offset must be 0 or "
                  "1,  but now is "
               << offset << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // param check
  int32_t rows = bbox1_desc->dims[0];
  int32_t cols = bbox2_desc->dims[0];
  int32_t batch_num_all = rows;

  if (ious_desc->dims[0] != rows) {
    LOG(ERROR) << "[mluOpBboxOverlaps] Check failed: Whether it is aligned "
                  "mode or not,"
               << "ious_desc->dims[0] == bbox1_desc->dims[0]. But now "
               << "ious_desc->dims[0] is " << ious_desc->dims[0]
               << ", bbox1_desc->dims[0] is " << rows << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if (aligned) {
    if (rows != cols) {
      LOG(ERROR) << "[mluOpBboxOverlaps] Check failed: If it is aligned mode, "
                 << "bbox1_desc->dims[0] == bbox2_desc->dims[0]. But now "
                 << "bbox1_desc->dims[0] is " << rows
                 << ", bbox2_desc->dims[0] is " << cols << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
    if (rows * cols == 0) {
      if ((ious_desc->dims[0] == rows) &&
          (ious_desc->dims[ious_desc->dim - 1] == 1)) {
        return MLUOP_STATUS_SUCCESS;
      } else {
        LOG(ERROR)
            << "[mluOpBboxOverlaps] Check failed: If it is aligned mode and "
            << "rows * cols = 0, ious_desc's first dim should be 0, "
            << "and ious_desc's last dim should be 1. "
            << "But now ious_desc's first dim is " << ious_desc->dims[0]
            << ", and ious_desc's last dim is "
            << ious_desc->dims[ious_desc->dim - 1] << ".";
        return MLUOP_STATUS_BAD_PARAM;
      }
    } else if ((ious_desc->dims[0] != rows || ious_desc->dim != 1)) {
      LOG(ERROR) << "[mluOpBboxOverlaps] Check failed: If it is aligned mode, "
                 << "ious_desc's first dim should equal to bbox1's first dim, "
                    "ious_desc's dim "
                 << "should be 1. But now ious_desc's first dim is "
                 << ious_desc->dims[0] << ", bbox1's first dim is " << rows
                 << ", and ious_desc's dim is " << ious_desc->dim << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  } else {
    if (ious_desc->dim != 2) {
      LOG(ERROR)
          << "[mluOpBboxOverlaps] Check failed: If it is non-aligned mode, "
          << "ious_desc->dim == 2. But now ious_desc->dim is " << ious_desc->dim
          << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
    if (ious_desc->dims[0] != rows ||
        ious_desc->dims[ious_desc->dim - 1] != cols) {
      LOG(ERROR)
          << "[mluOpBboxOverlaps] Check failed: If it is non-aligned mode, "
          << "ious_desc's first dim should be " << rows << ", ious_desc's last "
          << "dim should be " << cols << "."
          << "But now ious_desc's first dim is " << ious_desc->dims[0]
          << ", and ious_desc's last dim is "
          << ious_desc->dims[ious_desc->dim - 1] << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
    if (rows * cols == 0) {
      return MLUOP_STATUS_SUCCESS;
    }
  }

  PARAM_CHECK("[mluOpBboxOverlaps]", bbox1 != NULL);
  PARAM_CHECK("[mluOpBboxOverlaps]", bbox2 != NULL);
  PARAM_CHECK("[mluOpBboxOverlaps]", ious != NULL);

  // generate mluOpBboxOverlaps prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("bbox_overlaps");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA_REAL(true, "input", bbox1, bbox1_desc);
    GEN_CASE_DATA_REAL(true, "input", bbox2, bbox2_desc);
    GEN_CASE_DATA(false, "output", ious, ious_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "bbox_overlaps", "mode", mode);
    GEN_CASE_OP_PARAM_SINGLE(1, "bbox_overlaps", "aligned", aligned);
    GEN_CASE_OP_PARAM_SINGLE(2, "bbox_overlaps", "offset", offset);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }
  // generate mluOpBboxOverlaps prototxt end!

  mluOpDataType_t k_datatype = bbox1_desc->dtype;
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  policyFunc(handle, &k_dim, &k_type, batch_num_all);

  VLOG(5) << "Launch Kernel MLUUnion1BboxOverlapsKernel";
  VLOG(5) << "  kDim :[ " << k_dim.x << ", " << k_dim.y << ", " << k_dim.z
          << " ]";
  switch (k_datatype) {
    default: { VLOG(5) << "Not Implemented."; }
    case MLUOP_DTYPE_HALF: {
      KERNEL_CHECK(mluOpUnion1BboxOverlapsKernelHalf(
          k_dim, k_type, handle->queue, bbox1, bbox2, ious, rows, cols, mode,
          aligned, offset));
    }; break;
    case MLUOP_DTYPE_FLOAT: {
      KERNEL_CHECK(mluOpUnion1BboxOverlapsKernelFloat(
          k_dim, k_type, handle->queue, bbox1, bbox2, ious, rows, cols, mode,
          aligned, offset));
    }; break;
  }
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
