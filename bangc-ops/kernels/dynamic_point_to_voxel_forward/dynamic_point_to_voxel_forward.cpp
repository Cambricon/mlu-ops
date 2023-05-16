/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
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
#include "dynamic_point_to_voxel_forward.h"

#include <string>

#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"

// policy function
static void policyFuncDynamicPointToVoxelForward(const mluOpHandle_t handle,
                                                 cnrtDim3_t *k_dim,
                                                 cnrtFunctionType_t *k_type,
                                                 const int nums) {
  int max_core_num = mluop::runtime::getCoreNumOfJobLimitCapability(handle);
  size_t core_num = handle->core_num_per_cluster;
  if (nums > max_core_num) {
    k_dim->x = max_core_num;
    *k_type = mluop::runtime::getJobLimitCapabilityCnrtFuncType(handle);
  } else {
    if (nums == 1) {
      k_dim->x = 1;
      *k_type = CNRT_FUNC_TYPE_BLOCK;
    } else if (nums <= 4) {
      k_dim->x = core_num * 1;
      *k_type = CNRT_FUNC_TYPE_UNION1;
    } else if (nums <= 8) {
      k_dim->x = core_num * 2;
      *k_type = CNRT_FUNC_TYPE_UNION2;
    } else if (nums <= 16) {
      k_dim->x = core_num * 4;
      *k_type = CNRT_FUNC_TYPE_UNION4;
    } else if (nums <= 32) {
      k_dim->x = core_num * 8;
      *k_type = CNRT_FUNC_TYPE_UNION8;
    } else if (nums <= 64) {
      k_dim->x = core_num * 16;
      *k_type = CNRT_FUNC_TYPE_UNION16;
    }
  }
  k_dim->y = 1;
  k_dim->z = 1;
  return;
}

static mluOpStatus_t DynamicPointToVoxelForwardParamCheck(
    const std::string &api, const mluOpHandle_t handle,
    const mluOpReduceMode_t reduce_type, const void *feats, const void *coors,
    const void *voxel_feats, const void *voxel_coors,
    const void *point2voxel_map, const void *voxel_points_count,
    const void *voxel_num, void *workspace, const size_t workspace_size,
    const mluOpTensorDescriptor_t feats_desc,
    const mluOpTensorDescriptor_t coors_desc,
    const mluOpTensorDescriptor_t voxel_feats_desc,
    const mluOpTensorDescriptor_t voxel_coors_desc,
    const mluOpTensorDescriptor_t point2voxel_map_desc,
    const mluOpTensorDescriptor_t voxel_points_count_desc,
    const mluOpTensorDescriptor_t voxel_num_desc, bool *zero_element) {
  // check descriptor
  PARAM_CHECK(api, handle != NULL);
  // platform check
  if (handle->arch < MLUOP_MLU370) {
    LOG(ERROR) << api << "Only mlu300 and above devices are supported. "
               << "Please check the device version!";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }
  PARAM_CHECK(api, feats_desc != NULL);
  PARAM_CHECK(api, coors_desc != NULL);
  PARAM_CHECK(api, voxel_feats_desc != NULL);
  PARAM_CHECK(api, voxel_coors_desc != NULL);
  PARAM_CHECK(api, point2voxel_map_desc != NULL);
  PARAM_CHECK(api, voxel_points_count_desc != NULL);
  PARAM_CHECK(api, voxel_num_desc != NULL);
  // check shape
  PARAM_CHECK(api, feats_desc->dim == 2);
  PARAM_CHECK(api, coors_desc->dim == 2);
  PARAM_CHECK(api, voxel_feats_desc->dim == 2);
  PARAM_CHECK(api, voxel_coors_desc->dim == 2);
  PARAM_CHECK(api, point2voxel_map_desc->dim == 1);
  PARAM_CHECK(api, voxel_points_count_desc->dim == 1);
  PARAM_CHECK(api, voxel_num_desc->dim == 1);

  // check data type
  PARAM_CHECK_V2(api, (feats_desc->dtype == MLUOP_DTYPE_FLOAT),
                 "Only float are supported in feats tensor, but the data "
                 "type of tensor is "
                     << mluOpGetNameOfDataType(feats_desc->dtype) << ".");
  PARAM_CHECK_V2(api, (coors_desc->dtype == MLUOP_DTYPE_INT32),
                 "Only int32 are supported in coors tensor, but the data "
                 "type of tensor is "
                     << mluOpGetNameOfDataType(coors_desc->dtype) << ".");
  PARAM_CHECK_V2(
      api, (point2voxel_map_desc->dtype == MLUOP_DTYPE_INT32),
      "Only int32 are supported in point2voxel_map tensor, but the data "
      "type of tensor is "
          << mluOpGetNameOfDataType(point2voxel_map_desc->dtype) << ".");

  PARAM_CHECK(api, voxel_feats_desc->dtype == feats_desc->dtype);
  PARAM_CHECK(api, voxel_coors_desc->dtype == coors_desc->dtype);
  PARAM_CHECK(api,
              voxel_points_count_desc->dtype == point2voxel_map_desc->dtype);
  PARAM_CHECK(api, voxel_num_desc->dtype == point2voxel_map_desc->dtype);
  printf("reduce_type:%d\n", reduce_type);
  printf("reduce_type1:%d\n", MLUOP_REDUCE_DMAX);
  printf("reduce_type2:%d\n", MLUOP_REDUCE_DMEAN);
  if (reduce_type != MLUOP_REDUCE_DMAX && reduce_type != MLUOP_REDUCE_DMEAN) {
    LOG(ERROR) << api << "Only support max and mean. "
               << "Please check reduce_type!";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // check dim
  PARAM_CHECK(api, feats_desc->dims[0] == coors_desc->dims[0]);
  PARAM_CHECK(api, feats_desc->dims[0] == point2voxel_map_desc->dims[0]);
  PARAM_CHECK(api, voxel_feats_desc->dims[0] == voxel_coors_desc->dims[0]);
  PARAM_CHECK(api,
              voxel_feats_desc->dims[0] == voxel_points_count_desc->dims[0]);
  PARAM_CHECK(api, voxel_num_desc->dims[0] == 1);
  PARAM_CHECK(api, feats_desc->dims[1] == voxel_feats_desc->dims[1]);
  PARAM_CHECK(api, coors_desc->dims[1] == voxel_coors_desc->dims[1]);
  PARAM_CHECK(api, coors_desc->dims[1] == 3);
  PARAM_CHECK(api, feats_desc->dims[0] >= voxel_feats_desc->dims[0]);

  // check large tensor
  const size_t feats_element_num = mluOpGetTensorElementNum(feats_desc);
  const size_t coors_element_num = mluOpGetTensorElementNum(coors_desc);
  TENSOR_NUM_CHECK(api, feats_element_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(api, coors_element_num, LARGE_TENSOR_NUM, "");

  // check element num zero
  if (feats_element_num == 0 || coors_element_num == 0) {
    *zero_element = true;
    return MLUOP_STATUS_SUCCESS;
  }

  // check workspace ptr
  if (workspace_size > 0) {
    PARAM_CHECK(api, workspace != NULL);
  }
  // input and output ptr check null
  PARAM_CHECK(api, feats != NULL);
  PARAM_CHECK(api, coors != NULL);
  PARAM_CHECK(api, voxel_feats != NULL);
  PARAM_CHECK(api, voxel_coors != NULL);
  PARAM_CHECK(api, point2voxel_map != NULL);
  PARAM_CHECK(api, voxel_points_count != NULL);
  PARAM_CHECK(api, voxel_num != NULL);

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetDynamicPointToVoxelForwardWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t feats_desc,
    const mluOpTensorDescriptor_t coors_desc, size_t *workspace_size) {
  const std::string api = "[mluOpGetDynamicPointToVoxelForwardWorkspaceSize]";
  PARAM_CHECK(api, handle != NULL);
  // platform check
  if (handle->arch < MLUOP_MLU370) {
    LOG(ERROR) << "[mluOpGetDynamicPointToVoxelForwardWorkspaceSize] Only "
                  "mlu300 and above "
                  "devices are supported. "
               << "Please check the device version!";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  PARAM_CHECK(api, feats_desc != NULL);
  PARAM_CHECK(api, coors_desc != NULL);
  PARAM_CHECK(api, workspace_size != NULL);

  mluOpUniqueSort_t unique_mode = MLUOP_SORT_ASCEND;
  mluOpUniqueDescriptor_t unique_desc;
  MLUOP_CHECK(mluOpCreateUniqueDescriptor(&unique_desc));
  MLUOP_CHECK(
      mluOpSetUniqueDescriptor(unique_desc, unique_mode, 0, true, true));
  MLUOP_CHECK(mluOpGetUniqueWorkspaceSize(handle, unique_desc, coors_desc,
                                          workspace_size));
  MLUOP_CHECK(mluOpDestroyUniqueDescriptor(unique_desc));

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpDynamicPointToVoxelForward(
    const mluOpHandle_t handle, const mluOpReduceMode_t reduce_type,
    const mluOpTensorDescriptor_t feats_desc, const void *feats,
    const mluOpTensorDescriptor_t coors_desc, void *coors, void *workspace,
    const size_t workspace_size, const mluOpTensorDescriptor_t voxel_feats_desc,
    void *voxel_feats, const mluOpTensorDescriptor_t voxel_coors_desc,
    void *voxel_coors, const mluOpTensorDescriptor_t point2voxel_map_desc,
    void *point2voxel_map,
    const mluOpTensorDescriptor_t voxel_points_count_desc,
    void *voxel_points_count, const mluOpTensorDescriptor_t voxel_num_desc,
    void *voxel_num) {
  const std::string api = "[mluOpDynamicPointToVoxelForward]";
  // check params
  bool zero_element = false;

  mluOpStatus_t ret = DynamicPointToVoxelForwardParamCheck(
      api, handle, reduce_type, feats, coors, voxel_feats, voxel_coors,
      point2voxel_map, voxel_points_count, voxel_num, workspace, workspace_size,
      feats_desc, coors_desc, voxel_feats_desc, voxel_coors_desc,
      point2voxel_map_desc, voxel_points_count_desc, voxel_num_desc,
      &zero_element);

  if (ret != MLUOP_STATUS_SUCCESS) {
    LOG(ERROR) << api
               << " Error found during element verification, please check.";
    return ret;
  }
  // check zero element
  if (zero_element) {
    VLOG(5) << "[mluOpDynamicPointToVoxelForward] Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }
  // generator
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("dynamic_point_to_voxel_forward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "feats", feats, feats_desc, -100, 100);
    GEN_CASE_DATA_REAL(true, "coors", coors, coors_desc);
    GEN_CASE_DATA(false, "voxel_feats", voxel_feats, voxel_feats_desc, 0, 0);
    GEN_CASE_DATA(false, "voxel_coors", voxel_coors, voxel_coors_desc, 0, 0);
    GEN_CASE_DATA(false, "point2voxel_map", point2voxel_map,
                  point2voxel_map_desc, 0, 0);
    GEN_CASE_DATA(false, "voxel_points_count", voxel_points_count,
                  voxel_points_count_desc, 0, 0);
    GEN_CASE_DATA(false, "voxel_num", voxel_num, voxel_num_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "dynamic_point_to_voxel_forward", "reduce_type",
                             reduce_type);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  const int num_points = feats_desc->dims[0];
  const int num_feats = feats_desc->dims[1];
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFuncDynamicPointToVoxelForward(handle, &k_dim, &k_type, num_points);
  VLOG(5) << api << " Launch [" << k_type << ", " << k_dim.x << ", " << k_dim.y
          << ", " << k_dim.z << "].";
  // 1. mask_fill coors
  VLOG(5) << api << " launch KernelMaskFillCoorsForward start.";
  KERNEL_CHECK((KernelMaskFillCoorsForward(k_dim, k_type, handle->queue,
                                           num_points, coors)));
  VLOG(5) << api << " launch KernelMaskFillCoorsForward end.";

  // 2. unique op
  mluOpUniqueSort_t unique_mode = MLUOP_SORT_ASCEND;
  mluOpUniqueDescriptor_t unique_desc;
  MLUOP_CHECK(mluOpCreateUniqueDescriptor(&unique_desc));
  MLUOP_CHECK(
      mluOpSetUniqueDescriptor(unique_desc, unique_mode, 0, true, true));
  MLUOP_CHECK((mluOpUnique_v2(
      handle, unique_desc, coors_desc, coors, workspace, workspace_size,
      (int *)voxel_num, voxel_coors_desc, voxel_coors, point2voxel_map_desc,
      point2voxel_map, voxel_points_count_desc, voxel_points_count)));
  MLUOP_CHECK(mluOpDestroyUniqueDescriptor(unique_desc));

  // 3. reduce
  // fill -inf or zero
  VLOG(5) << "mluopFill min value start.";
  float inf_value = 0x0;
  if (reduce_type == MLUOP_REDUCE_DMAX) {
    inf_value = -INFINITY;
  }
  const float fill_value = inf_value;
  MLUOP_CHECK(mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST, &fill_value,
                           voxel_feats_desc, voxel_feats));
  VLOG(5) << "mluopFill min value end.";

  VLOG(5) << api << " launch KernelDynamicPointToVoxelForward start.";
  KERNEL_CHECK((KernelDynamicPointToVoxelForward(
      k_dim, k_type, handle->queue, reduce_type, feats, num_points, num_feats,
      voxel_coors, voxel_num, point2voxel_map, voxel_points_count,
      voxel_feats)));
  VLOG(5) << api << " launch KernelDynamicPointToVoxelForward end.";

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
