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
#include "dynamic_point_to_voxel_backward.h"

#include <algorithm>  // std::min
#include <string>
#include <vector>  // std::vector

#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"

mluOpStatus_t MLUOP_WIN_API DynamicPointToVoxelBackwardParamCheck(
    const char *interface_name, const mluOpHandle_t handle,
    const mluOpReduceMode_t reduce_type,
    const mluOpTensorDescriptor_t grad_voxel_feats_desc,
    const void *grad_voxel_feats, const mluOpTensorDescriptor_t feats_desc,
    const void *feats, const mluOpTensorDescriptor_t voxel_feats_desc,
    const void *voxel_feats, const mluOpTensorDescriptor_t point2voxel_map_desc,
    const void *point2voxel_map,
    const mluOpTensorDescriptor_t voxel_points_count_desc,
    const void *voxel_points_count,
    const mluOpTensorDescriptor_t voxel_num_desc, const void *voxel_num,
    void *workspace, const size_t workspace_size,
    const mluOpTensorDescriptor_t grad_feats_desc, void *grad_feats,
    bool &zero_element) {
  // check handle
  PARAM_CHECK(interface_name, handle != NULL);
  // platform check
  if (handle->arch < MLUOP_MLU370) {
    LOG(ERROR) << interface_name
               << "Only mlu300 and above devices are supported. "
               << "Please check the device version!";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }
  // check desc
  PARAM_CHECK(interface_name, grad_voxel_feats_desc != NULL);
  PARAM_CHECK(interface_name, feats_desc != NULL);
  PARAM_CHECK(interface_name, voxel_feats_desc != NULL);
  PARAM_CHECK(interface_name, point2voxel_map_desc != NULL);
  PARAM_CHECK(interface_name, voxel_points_count_desc != NULL);
  PARAM_CHECK(interface_name, voxel_num_desc != NULL);
  PARAM_CHECK(interface_name, grad_feats_desc != NULL);

  // check data type
  PARAM_CHECK(interface_name,
              grad_voxel_feats_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(interface_name, feats_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(interface_name, voxel_feats_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(interface_name, grad_feats_desc->dtype == MLUOP_DTYPE_FLOAT);

  PARAM_CHECK(interface_name, point2voxel_map_desc->dtype == MLUOP_DTYPE_INT32);
  PARAM_CHECK(interface_name,
              voxel_points_count_desc->dtype == MLUOP_DTYPE_INT32);
  PARAM_CHECK(interface_name, voxel_num_desc->dtype == MLUOP_DTYPE_INT32);

  // check shape
  PARAM_CHECK(interface_name, grad_voxel_feats_desc->dim == 2);
  PARAM_CHECK(interface_name, feats_desc->dim == 2);
  PARAM_CHECK(interface_name, voxel_feats_desc->dim == 2);
  PARAM_CHECK(interface_name, point2voxel_map_desc->dim == 1);
  PARAM_CHECK(interface_name, voxel_points_count_desc->dim == 1);
  PARAM_CHECK(interface_name, voxel_num_desc->dim == 1);
  PARAM_CHECK(interface_name, grad_feats_desc->dim == 2);

  PARAM_CHECK(interface_name,
              feats_desc->dims[1] == grad_voxel_feats_desc->dims[1]);
  PARAM_CHECK(interface_name,
              voxel_feats_desc->dims[0] == grad_voxel_feats_desc->dims[0]);
  PARAM_CHECK(interface_name,
              voxel_feats_desc->dims[1] == grad_voxel_feats_desc->dims[1]);
  PARAM_CHECK(interface_name,
              point2voxel_map_desc->dims[0] == feats_desc->dims[0]);
  PARAM_CHECK(interface_name, voxel_points_count_desc->dims[0] ==
                                  grad_voxel_feats_desc->dims[0]);
  PARAM_CHECK(interface_name, voxel_num_desc->dims[0] == 1);
  PARAM_CHECK(interface_name, grad_feats_desc->dims[0] == feats_desc->dims[0]);
  PARAM_CHECK(interface_name,
              grad_feats_desc->dims[1] == grad_voxel_feats_desc->dims[1]);
  PARAM_CHECK(interface_name,
              feats_desc->dims[0] >= grad_voxel_feats_desc->dims[0]);

  // param check
  if (reduce_type != MLUOP_REDUCE_DMAX) {
    LOG(ERROR) << interface_name
               << " only supports max reduce in current version. ";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // large tensor
  const uint64_t grad_voxel_feats_element_num =
      mluOpGetTensorElementNum(grad_voxel_feats_desc);
  const uint64_t feats_element_num = mluOpGetTensorElementNum(feats_desc);
  TENSOR_NUM_CHECK(interface_name, grad_voxel_feats_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(interface_name, feats_element_num, LARGE_TENSOR_NUM, "");

  // 0-element check, after dim and shape check
  if (grad_voxel_feats_element_num == 0 || feats_element_num == 0) {
    VLOG(5) << interface_name << " Skip zero element boxes.";
    zero_element = true;
    return MLUOP_STATUS_SUCCESS;
  }
  PARAM_CHECK(interface_name, grad_voxel_feats != NULL);
  PARAM_CHECK(interface_name, feats != NULL);
  PARAM_CHECK(interface_name, voxel_feats != NULL);
  PARAM_CHECK(interface_name, point2voxel_map != NULL);
  PARAM_CHECK(interface_name, voxel_points_count != NULL);
  PARAM_CHECK(interface_name, voxel_num != NULL);
  PARAM_CHECK(interface_name, grad_feats != NULL);
  if (workspace_size != 0) {
    PARAM_CHECK(interface_name, workspace != NULL);
  }
  return MLUOP_STATUS_SUCCESS;
}

static void policyFunc(const mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type, int N) {
  size_t cluster_num = mluop::runtime::getClusterLimitCapability(handle);
  size_t core_per_cluster =
      mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  // dimx equals to num of ipu cores in each cluster

  // M <= N, we can not get M's value on host, so use N instead.
  size_t cluster_num_need = DIV_UP(N, core_per_cluster);
  size_t eight = 8;
  size_t cluster_num_real = std::min(cluster_num, eight);
  size_t need_cluster = std::min(cluster_num_need, cluster_num_real);

  size_t k_dim_x = 0;
  // kernel1 need sync all task before kernel2, so UNIONX type is need.
  if (need_cluster == 1) {
    *k_type = CNRT_FUNC_TYPE_UNION1;
    k_dim_x = 1 * core_per_cluster;
  } else if (need_cluster == 2) {
    *k_type = CNRT_FUNC_TYPE_UNION2;
    k_dim_x = 2 * core_per_cluster;
  } else if (need_cluster > 2 && need_cluster <= 4) {
    *k_type = CNRT_FUNC_TYPE_UNION4;
    k_dim_x = 4 * core_per_cluster;
  } else if (need_cluster > 4 && need_cluster <= 8) {
    *k_type = CNRT_FUNC_TYPE_UNION8;
    k_dim_x = 8 * core_per_cluster;
  } else {
    LOG(ERROR) << "[mluOpDynamicPointToVoxelBackward]: failed to choose kernel "
                  "to launch";
    return;
  }

  k_dim->x = k_dim_x;
  k_dim->y = 1;
  k_dim->z = 1;
  VLOG(5) << "Launch Kernel MLUKernelDynamicPointToVoxelBackward in UNION"
          << k_dim_x / core_per_cluster << " type";
}

mluOpStatus_t MLUOP_WIN_API mluOpDynamicPointToVoxelBackward(
    const mluOpHandle_t handle, const mluOpReduceMode_t reduce_type,
    const mluOpTensorDescriptor_t grad_voxel_feats_desc,
    const void *grad_voxel_feats, const mluOpTensorDescriptor_t feats_desc,
    const void *feats, const mluOpTensorDescriptor_t voxel_feats_desc,
    const void *voxel_feats, const mluOpTensorDescriptor_t point2voxel_map_desc,
    const void *point2voxel_map,
    const mluOpTensorDescriptor_t voxel_points_count_desc,
    const void *voxel_points_count,
    const mluOpTensorDescriptor_t voxel_num_desc, const void *voxel_num,
    void *workspace, const size_t workspace_size,
    const mluOpTensorDescriptor_t grad_feats_desc, void *grad_feats) {
  const char *interface_name = "[mluOpDynamicPointToVoxelBackward]";
  bool zero_element = false;
  mluOpStatus_t param_check = DynamicPointToVoxelBackwardParamCheck(
      interface_name, handle, reduce_type, grad_voxel_feats_desc,
      grad_voxel_feats, feats_desc, feats, voxel_feats_desc, voxel_feats,
      point2voxel_map_desc, point2voxel_map, voxel_points_count_desc,
      voxel_points_count, voxel_num_desc, voxel_num, workspace, workspace_size,
      grad_feats_desc, grad_feats, zero_element);
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }
  if (zero_element) {
    return MLUOP_STATUS_SUCCESS;
  }

  // generator
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("dynamic_point_to_voxel_backward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA_REAL(true, "grad_voxel_feats", grad_voxel_feats,
                       grad_voxel_feats_desc);
    GEN_CASE_DATA_REAL(true, "feats", feats, feats_desc);
    GEN_CASE_DATA_REAL(true, "voxel_feats", voxel_feats, voxel_feats_desc);
    GEN_CASE_DATA_REAL(true, "point2voxel_map", point2voxel_map,
                       point2voxel_map_desc);
    GEN_CASE_DATA_REAL(true, "voxel_points_count", voxel_points_count,
                       voxel_points_count_desc);
    GEN_CASE_DATA_REAL(true, "voxel_num", voxel_num, voxel_num_desc);
    GEN_CASE_DATA(false, "grad_feats", grad_feats, grad_feats_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "dynamic_point_to_voxel_backward",
                             "reduce_type", reduce_type);
    GEN_CASE_TEST_PARAM_NEW(false, false, true, 0.003, 0.003, 0);
  }

  int N = feats_desc->dims[0];
  int C = feats_desc->dims[1];
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, &k_dim, &k_type, N);
  // MLU500: 1. get scatter indices and scatter
  // MLU300: 1. get scatter indices
  KERNEL_CHECK((KernelDynamicPointToVoxelBackward(
      k_dim, k_type, handle->queue, feats_desc->dtype, reduce_type,
      grad_voxel_feats, feats, voxel_feats, point2voxel_map, voxel_points_count,
      voxel_num, workspace, grad_feats, N, C)));
  if (handle->arch == MLUOP_MLU370) {
    // MLU300: 2. scatter
    mluOpScatterNdMode_t scatter_mode = MLUOP_SCATTERND_ADD;
    mluOpTensorDescriptor_t indices_desc;
    INTERNAL_CHECK(
        interface_name,
        MLUOP_STATUS_SUCCESS == mluOpCreateTensorDescriptor(&indices_desc));
    std::vector<int> indices_dims = {N * C, 1};
    INTERNAL_CHECK(interface_name,
                   MLUOP_STATUS_SUCCESS ==
                       mluOpSetTensorDescriptor(
                           indices_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
                           indices_dims.size(), indices_dims.data()));
    mluOpTensorDescriptor_t updates_desc;
    INTERNAL_CHECK(
        interface_name,
        MLUOP_STATUS_SUCCESS == mluOpCreateTensorDescriptor(&updates_desc));
    std::vector<int> updates_dims = {N * C};
    INTERNAL_CHECK(interface_name,
                   MLUOP_STATUS_SUCCESS ==
                       mluOpSetTensorDescriptor(
                           updates_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                           updates_dims.size(), updates_dims.data()));
    mluOpTensorDescriptor_t output_desc;
    INTERNAL_CHECK(
        interface_name,
        MLUOP_STATUS_SUCCESS == mluOpCreateTensorDescriptor(&output_desc));
    std::vector<int> output_dims = {N * C};
    INTERNAL_CHECK(interface_name,
                   MLUOP_STATUS_SUCCESS ==
                       mluOpSetTensorDescriptor(
                           output_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                           output_dims.size(), output_dims.data()));
    MLUOP_CHECK((mluOpScatterNd_v2(handle, scatter_mode, indices_desc,
                                   workspace, updates_desc, grad_voxel_feats,
                                   NULL, NULL, output_desc, grad_feats)));
    INTERNAL_CHECK(
        interface_name,
        MLUOP_STATUS_SUCCESS == mluOpDestroyTensorDescriptor(indices_desc));
    INTERNAL_CHECK(
        interface_name,
        MLUOP_STATUS_SUCCESS == mluOpDestroyTensorDescriptor(updates_desc));
    INTERNAL_CHECK(
        interface_name,
        MLUOP_STATUS_SUCCESS == mluOpDestroyTensorDescriptor(output_desc));
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetDynamicPointToVoxelBackwardWorkspaceSize(
    const mluOpHandle_t handle, const mluOpReduceMode_t reduce_type,
    const mluOpTensorDescriptor_t feats_desc, size_t *workspace_size) {
  const char *interface_name =
      "[mluOpGetDynamicPointToVoxelBackwardWorkspaceSize]";
  PARAM_CHECK(interface_name, handle != NULL);
  if (handle->arch < MLUOP_MLU370) {
    LOG(ERROR) << interface_name
               << "Only mlu300 and above devices are supported. "
               << "Please check the device version!";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }
  PARAM_CHECK(interface_name, feats_desc != NULL);
  PARAM_CHECK(interface_name, workspace_size != NULL);
  int N = feats_desc->dims[0];
  int C = feats_desc->dims[1];
  *workspace_size = N * C * sizeof(int);
  return MLUOP_STATUS_SUCCESS;
}
