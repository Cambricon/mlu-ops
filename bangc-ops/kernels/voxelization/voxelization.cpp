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

static void policyFuncDefault(const mluOpHandle_t handle,
                              const size_t num_points, cnrtDim3_t *k_dim,
                              cnrtFunctionType_t *k_type) {
  k_dim->x = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  k_dim->y =
      std::min((num_points + k_dim->x - 1) / k_dim->x,
               (size_t)mluop::runtime::getClusterLimitCapability(handle));
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
}

static void policyFuncCalcPointsPerVoxel(const mluOpHandle_t handle,
                                         const size_t num_points,
                                         cnrtDim3_t *k_dim,
                                         cnrtFunctionType_t *k_type) {
  k_dim->x = 1;
  k_dim->y = 1;
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_BLOCK;
}

mluOpStatus_t voxelizationParamCheck(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t points_desc,
    const mluOpTensorDescriptor_t voxel_size_desc,
    const mluOpTensorDescriptor_t coors_range_desc, const int32_t max_points,
    const int32_t max_voxels, const int32_t NDim, const bool deterministic,
    const mluOpTensorDescriptor_t voxels_desc,
    const mluOpTensorDescriptor_t coors_desc,
    const mluOpTensorDescriptor_t num_points_per_voxel_desc,
    const mluOpTensorDescriptor_t voxel_num_desc, bool is_zero_element) {
  // check arch
  if (handle->arch < MLUOP_MLU370) {
    LOG(ERROR) << "[mluOpVoxelization] The operator only support architecture "
                  "which is greater than or equal to 372.";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  if (deterministic == true) {
    // check tensor shape
    // params points: [num_points, num_features]
    PARAM_CHECK_EQ("[mluOpVoxelization]", points_desc->dim, 2);
    // params voxel_size: [3]
    PARAM_CHECK_EQ("[mluOpVoxelization]", voxel_size_desc->dim, 1);
    PARAM_CHECK_EQ("[mluOpVoxelization]", voxel_size_desc->dims[0], 3);
    // params coors_range: [6]
    PARAM_CHECK_EQ("[mluOpVoxelization]", coors_range_desc->dim, 1);
    PARAM_CHECK_EQ("[mluOpVoxelization]", coors_range_desc->dims[0], 6);
    // params voxels: [max_voxels, max_points, num_features]
    PARAM_CHECK_EQ("[mluOpVoxelization]", voxels_desc->dim, 3);
    PARAM_CHECK_EQ("[mluOpVoxelization]", voxels_desc->dims[0], max_voxels);
    PARAM_CHECK_EQ("[mluOpVoxelization]", voxels_desc->dims[1], max_points);
    PARAM_CHECK_EQ("[mluOpVoxelization]", voxels_desc->dims[2],
                   points_desc->dims[1]);
    // params coors: [max_voxels, 3]
    PARAM_CHECK_EQ("[mluOpVoxelization]", coors_desc->dim, 2);
    PARAM_CHECK_EQ("[mluOpVoxelization]", coors_desc->dims[0], max_voxels);
    PARAM_CHECK_EQ("[mluOpVoxelization]", coors_desc->dims[1], 3);
    // params num_points_per_voxel: [max_voxels]
    PARAM_CHECK_EQ("[mluOpVoxelization]", num_points_per_voxel_desc->dim, 1);
    PARAM_CHECK_EQ("[mluOpVoxelization]", num_points_per_voxel_desc->dims[0],
                   max_voxels);
    // params voxel_num: [1]
    PARAM_CHECK_EQ("[mluOpVoxelization]", voxel_num_desc->dim, 1);
    PARAM_CHECK_EQ("[mluOpVoxelization]", voxel_num_desc->dims[0], 1);

    // check params
    PARAM_CHECK_EQ("[mluOpVoxelization]", NDim, 3);

    // check tensor datatype
    PARAM_CHECK("[mluOpVoxelization]", points_desc->dtype == MLUOP_DTYPE_FLOAT);
    PARAM_CHECK("[mluOpVoxelization]",
                voxel_size_desc->dtype == MLUOP_DTYPE_FLOAT);
    PARAM_CHECK("[mluOpVoxelization]",
                coors_range_desc->dtype == MLUOP_DTYPE_FLOAT);
    PARAM_CHECK("[mluOpVoxelization]", voxels_desc->dtype == MLUOP_DTYPE_FLOAT);
    PARAM_CHECK("[mluOpVoxelization]", coors_desc->dtype == MLUOP_DTYPE_INT32);
    PARAM_CHECK("[mluOpVoxelization]",
                num_points_per_voxel_desc->dtype == MLUOP_DTYPE_INT32);
    PARAM_CHECK("[mluOpVoxelization]",
                voxel_num_desc->dtype == MLUOP_DTYPE_INT32);

    size_t points_element_num = mluOpGetTensorElementNum(points_desc);
    size_t voxel_size_element_num = mluOpGetTensorElementNum(voxel_size_desc);
    size_t coors_range_element_num = mluOpGetTensorElementNum(coors_range_desc);
    size_t voxels_element_num = mluOpGetTensorElementNum(voxels_desc);
    size_t coors_element_num = mluOpGetTensorElementNum(coors_desc);
    size_t num_points_per_voxel_element_num =
        mluOpGetTensorElementNum(num_points_per_voxel_desc);
    size_t voxel_num_element_num = mluOpGetTensorElementNum(voxel_num_desc);

    // check large tensor
    if (points_element_num >= LARGE_TENSOR_NUM ||
        voxel_size_element_num >= LARGE_TENSOR_NUM ||
        coors_range_element_num >= LARGE_TENSOR_NUM ||
        voxels_element_num >= LARGE_TENSOR_NUM ||
        coors_element_num >= LARGE_TENSOR_NUM ||
        num_points_per_voxel_element_num >= LARGE_TENSOR_NUM ||
        voxel_num_element_num >= LARGE_TENSOR_NUM) {
      LOG(ERROR) << "[mluOpVoxelization] Overflow max tensor num."
                 << "Currently, MLU-OPS supports tensor num smaller than 2^31.";
      return MLUOP_STATUS_NOT_SUPPORTED;
    }

    // check element num zero
    is_zero_element = false;
    if (points_element_num == 0 || voxel_size_element_num == 0 ||
        coors_range_element_num == 0) {
      LOG(ERROR)
          << "[mluOpVoxelization] Check failed: Input zero element tensor.";
      return MLUOP_STATUS_BAD_PARAM;
    }

    if (max_points == 0 || max_voxels == 0) {
      is_zero_element = true;
    }
  } else {
    VLOG(5) << "[mluOpVoxelization] Currently, Non-deterministic mode not "
               "supported.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetVoxelizationWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t points_desc,
    const mluOpTensorDescriptor_t voxel_size_desc,
    const mluOpTensorDescriptor_t coors_range_desc, const int32_t max_points,
    const int32_t max_voxels, const int32_t NDim, const bool deterministic,
    const mluOpTensorDescriptor_t voxels_desc,
    const mluOpTensorDescriptor_t coors_desc,
    const mluOpTensorDescriptor_t num_points_per_voxel_desc,
    const mluOpTensorDescriptor_t voxel_num_desc, size_t *size) {
  // handle and desc ptr check null
  PARAM_CHECK("[mluOpGetVoxelizationWorkspaceSize]", handle != NULL);
  PARAM_CHECK("[mluOpGetVoxelizationWorkspaceSize]", points_desc != NULL);
  PARAM_CHECK("[mluOpGetVoxelizationWorkspaceSize]", voxel_size_desc != NULL);
  PARAM_CHECK("[mluOpGetVoxelizationWorkspaceSize]", coors_range_desc != NULL);
  PARAM_CHECK("[mluOpGetVoxelizationWorkspaceSize]", voxels_desc != NULL);
  PARAM_CHECK("[mluOpGetVoxelizationWorkspaceSize]", coors_desc != NULL);
  PARAM_CHECK("[mluOpGetVoxelizationWorkspaceSize]",
              num_points_per_voxel_desc != NULL);
  PARAM_CHECK("[mluOpGetVoxelizationWorkspaceSize]", voxel_num_desc != NULL);
  PARAM_CHECK("[mluOpGetVoxelizationWorkspaceSize]", size != NULL);

  // check params
  bool is_zero_element = false;
  mluOpStatus_t paramcheck_status = voxelizationParamCheck(
      handle, points_desc, voxel_size_desc, coors_range_desc, max_points,
      max_voxels, NDim, deterministic, voxels_desc, coors_desc,
      num_points_per_voxel_desc, voxel_num_desc, is_zero_element);
  if (paramcheck_status != MLUOP_STATUS_SUCCESS) {
    return paramcheck_status;
  }

  if (is_zero_element == true) {
    VLOG(5) << "[mluOpVoxelization] Skip output zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  const size_t num_points = points_desc->dims[0];
  const size_t num_features = points_desc->dims[1];
  const size_t temp_coors_size = 3 * num_points * sizeof(int32_t);
  const size_t point_to_pointidx_size = num_points * sizeof(int32_t);
  const size_t point_to_voxelidx_size = num_points * sizeof(int32_t);
  const size_t coor_to_voxelidx_size = num_points * sizeof(int32_t);
  *size = temp_coors_size + point_to_pointidx_size + point_to_voxelidx_size +
          coor_to_voxelidx_size;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpVoxelization(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t points_desc,
    const void *points, const mluOpTensorDescriptor_t voxel_size_desc,
    const void *voxel_size, const mluOpTensorDescriptor_t coors_range_desc,
    const void *coors_range, const int32_t max_points, const int32_t max_voxels,
    const int32_t NDim, const bool deterministic, void *workspace,
    size_t workspace_size, const mluOpTensorDescriptor_t voxels_desc,
    void *voxels, const mluOpTensorDescriptor_t coors_desc, void *coors,
    const mluOpTensorDescriptor_t num_points_per_voxel_desc,
    void *num_points_per_voxel, const mluOpTensorDescriptor_t voxel_num_desc,
    void *voxel_num) {
  // handle and desc ptr check null
  PARAM_CHECK("[mluOpVoxelization]", handle != NULL);
  PARAM_CHECK("[mluOpVoxelization]", points_desc != NULL);
  PARAM_CHECK("[mluOpVoxelization]", voxel_size_desc != NULL);
  PARAM_CHECK("[mluOpVoxelization]", coors_range_desc != NULL);
  PARAM_CHECK("[mluOpVoxelization]", voxels_desc != NULL);
  PARAM_CHECK("[mluOpVoxelization]", coors_desc != NULL);
  PARAM_CHECK("[mluOpVoxelization]", num_points_per_voxel_desc != NULL);
  PARAM_CHECK("[mluOpVoxelization]", voxel_num_desc != NULL);

  // check params
  bool is_zero_element = false;
  mluOpStatus_t paramcheck_status = voxelizationParamCheck(
      handle, points_desc, voxel_size_desc, coors_range_desc, max_points,
      max_voxels, NDim, deterministic, voxels_desc, coors_desc,
      num_points_per_voxel_desc, voxel_num_desc, is_zero_element);
  if (paramcheck_status != MLUOP_STATUS_SUCCESS) {
    return paramcheck_status;
  }

  // check workspace
  if (workspace_size > 0) {
    PARAM_CHECK("[mluOpVoxelization]", workspace != NULL);
  }

  if (is_zero_element == true) {
    VLOG(5) << "[mluOpVoxelization] Skip output zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("voxelization");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "points", points, points_desc, 10, -10);
    GEN_CASE_DATA(true, "voxel_size", voxel_size, voxel_size_desc, 10, -10);
    GEN_CASE_DATA(true, "coors_range", coors_range, coors_range_desc, 10, -10);
    GEN_CASE_DATA(false, "voxels", voxels, voxels_desc, 0, 0);
    GEN_CASE_DATA(false, "coors", coors, coors_desc, 0, 0);
    GEN_CASE_DATA(false, "num_points_per_voxel", num_points_per_voxel,
                  num_points_per_voxel_desc, 0, 0);
    GEN_CASE_DATA(false, "voxel_num", voxel_num, voxel_num_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "voxelization", "max_points", max_points);
    GEN_CASE_OP_PARAM_SINGLE(1, "voxelization", "max_voxels", max_voxels);
    GEN_CASE_OP_PARAM_SINGLE(2, "voxelization", "NDim", NDim);
    GEN_CASE_OP_PARAM_SINGLE(3, "voxelization", "deterministic", deterministic);
    GEN_CASE_TEST_PARAM_NEW(false, false, true, 0, 0, 0);
  }

  const size_t num_points = points_desc->dims[0];
  const size_t num_features = points_desc->dims[1];

  // temp_coors : [3, num_points]
  void *temp_coors = workspace;
  // point_to_pointidx : [num_points]
  void *point_to_pointidx =
      (char *)temp_coors + num_points * 3 * sizeof(int32_t);
  // point_to_voxelidx : [num_points]
  void *point_to_voxelidx =
      (char *)point_to_pointidx + num_points * sizeof(int32_t);
  // coor_to_voxelidx : [num_points]
  void *coor_to_voxelidx =
      (char *)point_to_voxelidx + num_points * sizeof(int32_t);

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFuncDefault(handle, num_points, &k_dim, &k_type);

  const int32_t voxels_size =
      max_voxels * max_points * num_features * sizeof(float);
  KERNEL_CHECK((mluOpBlockKernelFillZeroByte(k_dim, k_type, handle->queue,
                                             voxels_size, voxels)));

  const int32_t coors_size = max_voxels * 3 * sizeof(int32_t);
  KERNEL_CHECK((mluOpBlockKernelFillZeroByte(k_dim, k_type, handle->queue,
                                             coors_size, coors)));

  const int32_t num_points_per_voxel_size = max_voxels * sizeof(int32_t);
  KERNEL_CHECK((mluOpBlockKernelFillZeroByte(k_dim, k_type, handle->queue,
                                             num_points_per_voxel_size,
                                             num_points_per_voxel)));

  VLOG(5) << "Launch Kernel mluOpUnionKernelDynamicVoxelize.";
  KERNEL_CHECK((mluOpUnionKernelDynamicVoxelize(
      k_dim, k_type, handle->queue, points, voxel_size, coors_range, temp_coors,
      num_points, num_features)));

  VLOG(5) << "Launch Kernel mluOpUnionKernelPoint2Voxel.";
  KERNEL_CHECK((mluOpUnionKernelPoint2Voxel(
      k_dim, k_type, handle->queue, temp_coors, point_to_pointidx,
      point_to_voxelidx, num_points, max_points)));

  cnrtDim3_t k_dim_calc_points_per_voxel;
  cnrtFunctionType_t k_type_calc_points_per_voxel;
  policyFuncCalcPointsPerVoxel(handle, num_points, &k_dim_calc_points_per_voxel,
                               &k_type_calc_points_per_voxel);

  VLOG(5) << "Launch Kernel mluOpUnionKernelCalcPointsPerVoxel.";
  KERNEL_CHECK((mluOpUnionKernelCalcPointsPerVoxel(
      k_dim_calc_points_per_voxel, k_type_calc_points_per_voxel, handle->queue,
      point_to_pointidx, point_to_voxelidx, coor_to_voxelidx,
      num_points_per_voxel, voxel_num, max_voxels, num_points)));

  VLOG(5) << "Launch Kernel mluOpUnionKernelAssignVoxelsCoors.";
  KERNEL_CHECK((mluOpUnionKernelAssignVoxelsCoors(
      k_dim, k_type, handle->queue, points, temp_coors, point_to_voxelidx,
      coor_to_voxelidx, voxels, coors, max_points, num_points, num_features)));

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
