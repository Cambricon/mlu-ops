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
#include <string>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"

// policy function
static mluOpStatus_t kernelPtsIdxOfVoxelsPolicyFunc(
    const mluOpHandle_t handle,
    const mluOpTensorDescriptor_t pts_idx_of_voxels_desc, cnrtDim3_t *k_dim,
    cnrtFunctionType_t *k_type) {
  const int cluster_limit = mluop::runtime::getClusterLimitCapability(handle);
  const int core_limit =
      mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  const int core_num = core_limit * cluster_limit;
  const int boxes_num = pts_idx_of_voxels_desc->dims[0];
  const int task_dim = boxes_num > core_num ? core_num : boxes_num;
  k_dim->x = core_limit;
  k_dim->y = (task_dim / core_limit) > 0 ? (task_dim / core_limit) : 1;
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
  return MLUOP_STATUS_SUCCESS;
}

// policy function
static mluOpStatus_t kernelRoiawarePool3dForwardPolicyFunc(
    const mluOpHandle_t handle,
    const mluOpTensorDescriptor_t pooled_features_desc, cnrtDim3_t *k_dim,
    cnrtFunctionType_t *k_type) {
  const int cluster_limit = mluop::runtime::getClusterLimitCapability(handle);
  const int core_limit =
      mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  const int core_num = core_limit * cluster_limit;
  const int boxes_num = pooled_features_desc->dims[0];
  const int out_x = pooled_features_desc->dims[1];
  const int out_y = pooled_features_desc->dims[2];
  const int out_z = pooled_features_desc->dims[3];
  const int voxeles_num = boxes_num * out_x * out_y * out_z;
  const int task_dim = voxeles_num > core_num ? core_num : voxeles_num;
  k_dim->x = core_limit;
  k_dim->y = (task_dim / core_limit) > 0 ? (task_dim / core_limit) : 1;
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
  return MLUOP_STATUS_SUCCESS;
}

// policy function
static mluOpStatus_t kernelRoiawarePool3dBackwardPolicyFunc(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t grad_out_desc,
    cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  const int cluster_limit = mluop::runtime::getClusterLimitCapability(handle);
  const int core_limit =
      mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  const int core_num = core_limit * cluster_limit;
  const int boxes_num = grad_out_desc->dims[0];
  const int out_x = grad_out_desc->dims[1];
  const int out_y = grad_out_desc->dims[2];
  const int out_z = grad_out_desc->dims[3];
  const int voxels_num = boxes_num * out_x * out_y * out_z;
  const int task_dim = voxels_num > core_num ? core_num : voxels_num;
  k_dim->x = core_limit;
  k_dim->y = (task_dim / core_limit) > 0 ? (task_dim / core_limit) : 1;
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t transposeTensor(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, const int *permute,
    const mluOpTensorDescriptor_t workspace_dst_desc, void *workspace_dst,
    void *transpose_workspace) {
  const std::string API = "[mluOpRoiawarePool3dForward]";
  int input_dim = input_desc->dim;
  mluOpTransposeDescriptor_t trans_desc = NULL;
  size_t transpose_workspace_size = 0;
  PARAM_CHECK(
      API, MLUOP_STATUS_SUCCESS == mluOpCreateTransposeDescriptor(&trans_desc));
  PARAM_CHECK(API, MLUOP_STATUS_SUCCESS == mluOpSetTransposeDescriptor(
                                               trans_desc, input_dim, permute));
  PARAM_CHECK(API, MLUOP_STATUS_SUCCESS == mluOpGetTransposeWorkspaceSize(
                                               handle, input_desc, trans_desc,
                                               &transpose_workspace_size));
  PARAM_CHECK(API, MLUOP_STATUS_SUCCESS ==
                       mluOpTranspose_v2(handle, trans_desc, input_desc, input,
                                         workspace_dst_desc, workspace_dst,
                                         transpose_workspace,
                                         transpose_workspace_size));
  PARAM_CHECK(
      API, MLUOP_STATUS_SUCCESS == mluOpDestroyTransposeDescriptor(trans_desc));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetRoiawarePool3dForwardWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t rois_desc,
    const mluOpTensorDescriptor_t pts_desc,
    const mluOpTensorDescriptor_t pts_feature_desc, size_t *workspace_size) {
  // rois_desc and pts_desc is unused parameter.
  PARAM_CHECK("[mluOpGetRoiawarePool3dForwardWorkspaceSize]",
              handle != nullptr);
  PARAM_CHECK("[mluOpGetRoiawarePool3dForwardWorkspaceSize]",
              pts_feature_desc != nullptr);
  PARAM_CHECK("[mluOpGetRoiawarePool3dForwardWorkspaceSize]",
              workspace_size != nullptr);
  const int pts_num = pts_feature_desc->dims[0];
  const int channels = pts_feature_desc->dims[1];
  int element_num = pts_num * 3 + pts_num * channels;
  element_num += (channels > 3) ? (pts_num * channels) : (pts_num * 3);
  // check zero
  if (element_num == 0) {
    *workspace_size = 0;
    return MLUOP_STATUS_SUCCESS;
  }
  *workspace_size =
      element_num * mluop::getSizeOfDataType(pts_feature_desc->dtype);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpRoiawarePool3dForward(
    mluOpHandle_t handle, const int pool_method, const int boxes_num,
    const int pts_num, const int channels,
    const mluOpTensorDescriptor_t rois_desc, const void *rois,
    const mluOpTensorDescriptor_t pts_desc, const void *pts,
    const mluOpTensorDescriptor_t pts_feature_desc, const void *pts_feature,
    void *workspace, size_t workspace_size, const int max_pts_each_voxel,
    const int out_x, const int out_y, const int out_z,
    const mluOpTensorDescriptor_t argmax_desc, void *argmax,
    const mluOpTensorDescriptor_t pts_idx_of_voxels_desc,
    void *pts_idx_of_voxels, const mluOpTensorDescriptor_t pooled_features_desc,
    void *pooled_features) {
  // rois: (boxes_num, 7) [cx, cy, cz, dx, dy, dz, rz]
  // pts: (pts_num, 3) [x, y, z]
  // pts_feature: (pts_num, channels)
  // argmax: (boxes_num, out_x, out_y, out_z, channels)
  // pts_idx_of_voxels: (boxes_num, out_x, out_y, out_z, max_pts_each_voxel)
  // pooled_features: (boxes_num, out_x, out_y, out_z, channels)

  const std::string API = "[mluOpRoiawarePool3dForward]";
  // check desc
  PARAM_CHECK(API, handle != NULL);
  PARAM_CHECK(API, rois_desc != NULL);
  PARAM_CHECK(API, pts_desc != NULL);
  PARAM_CHECK(API, pts_feature_desc != NULL);
  PARAM_CHECK(API, pts_idx_of_voxels_desc != NULL);
  PARAM_CHECK(API, pooled_features_desc != NULL);

  // check dim
  PARAM_CHECK(API, rois_desc->dim == 2);
  PARAM_CHECK(API, pts_desc->dim == 2);
  PARAM_CHECK(API, pts_feature_desc->dim == 2);
  PARAM_CHECK(API, pts_idx_of_voxels_desc->dim == 5);
  PARAM_CHECK(API, pooled_features_desc->dim == 5);

  // check shape
  PARAM_CHECK(API, rois_desc->dims[0] == boxes_num);
  PARAM_CHECK(API, rois_desc->dims[1] == 7);
  PARAM_CHECK(API, pts_desc->dims[0] == pts_num);
  PARAM_CHECK(API, pts_desc->dims[1] == 3);
  PARAM_CHECK(API, pts_feature_desc->dims[0] == pts_num);
  PARAM_CHECK(API, pts_feature_desc->dims[1] == channels);
  PARAM_CHECK(API, pts_idx_of_voxels_desc->dims[0] == boxes_num);
  PARAM_CHECK(API, pts_idx_of_voxels_desc->dims[1] == out_x);
  PARAM_CHECK(API, pts_idx_of_voxels_desc->dims[2] == out_y);
  PARAM_CHECK(API, pts_idx_of_voxels_desc->dims[3] == out_z);
  PARAM_CHECK(API, pts_idx_of_voxels_desc->dims[4] == max_pts_each_voxel);
  PARAM_CHECK(API, pooled_features_desc->dims[0] == boxes_num);
  PARAM_CHECK(API, pooled_features_desc->dims[1] == out_x);
  PARAM_CHECK(API, pooled_features_desc->dims[2] == out_y);
  PARAM_CHECK(API, pooled_features_desc->dims[3] == out_z);
  PARAM_CHECK(API, pooled_features_desc->dims[4] == channels);

  // check dtype
  PARAM_CHECK(API, rois_desc->dtype == MLUOP_DTYPE_FLOAT ||
                       rois_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK(API, pts_desc->dtype == rois_desc->dtype);
  PARAM_CHECK(API, pts_feature_desc->dtype == rois_desc->dtype);
  PARAM_CHECK(API, pts_idx_of_voxels_desc->dtype == MLUOP_DTYPE_INT32);
  PARAM_CHECK(API, pooled_features_desc->dtype == rois_desc->dtype);

  // check other parms : pool_method
  PARAM_CHECK(API, pool_method == 0 || pool_method == 1);

  // check tensor dim
  PARAM_CHECK(API, boxes_num > 0);
  PARAM_CHECK(API, pts_num > 0);
  PARAM_CHECK(API, channels > 0);
  PARAM_CHECK(API, max_pts_each_voxel > 0);
  PARAM_CHECK(API, out_x > 0);
  PARAM_CHECK(API, out_y > 0);
  PARAM_CHECK(API, out_z > 0);

  const uint64_t tensor_rois_num = mluOpGetTensorElementNum(rois_desc);
  const uint64_t tensor_pts_num = mluOpGetTensorElementNum(pts_desc);
  const uint64_t tensor_pts_feature_num =
      mluOpGetTensorElementNum(pts_feature_desc);
  const uint64_t tensor_pts_idx_of_voxels_num =
      mluOpGetTensorElementNum(pts_idx_of_voxels_desc);
  const uint64_t tensor_pooled_features_num =
      mluOpGetTensorElementNum(pooled_features_desc);

  // check large tensor
  TENSOR_NUM_CHECK(API, tensor_rois_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, tensor_pts_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, tensor_pts_feature_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, tensor_pts_idx_of_voxels_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, tensor_pooled_features_num, LARGE_TENSOR_NUM, "");

  // check zero element
  PARAM_CHECK(API, tensor_rois_num > 0);
  PARAM_CHECK(API, tensor_pts_num > 0);
  PARAM_CHECK(API, tensor_pts_feature_num > 0);
  PARAM_CHECK(API, tensor_pts_idx_of_voxels_num > 0);
  PARAM_CHECK(API, tensor_pooled_features_num > 0);

  // check workspace
  if (workspace_size > 0 && workspace == NULL) {
    LOG(ERROR) << API
               << ": workspace shouldn't be null when workspace_size > 0.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // check ptr
  PARAM_CHECK(API, rois != NULL);
  PARAM_CHECK(API, pts != NULL);
  PARAM_CHECK(API, pts_feature != NULL);
  PARAM_CHECK(API, pts_idx_of_voxels != NULL);
  PARAM_CHECK(API, pooled_features != NULL);

  // pool_method: 0 'max'  1 'avg'
  if (pool_method == 0) {
    PARAM_CHECK(API, argmax_desc != NULL);
    PARAM_CHECK(API, argmax_desc->dim == 5);
    PARAM_CHECK(API, argmax_desc->dims[0] == boxes_num);
    PARAM_CHECK(API, argmax_desc->dims[1] == out_x);
    PARAM_CHECK(API, argmax_desc->dims[2] == out_y);
    PARAM_CHECK(API, argmax_desc->dims[3] == out_z);
    PARAM_CHECK(API, argmax_desc->dims[4] == channels);
    PARAM_CHECK(API, argmax_desc->dtype == MLUOP_DTYPE_INT32);
    const uint64_t tensor_argmax_num = mluOpGetTensorElementNum(argmax_desc);
    TENSOR_NUM_CHECK(API, tensor_argmax_num, LARGE_TENSOR_NUM, "");
    PARAM_CHECK(API, tensor_argmax_num > 0);
    PARAM_CHECK(API, argmax != NULL);
  }

  // check arch
  if (handle->arch < MLUOP_MLU370) {
    LOG(ERROR) << API
               << " The operator does not match the current architecture.";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  // generate mluOpRoiawarePool3dForward prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("roiaware_pool3d_forward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA_REAL(true, "rois", rois, rois_desc);
    GEN_CASE_DATA_REAL(true, "pts", pts, pts_desc);
    GEN_CASE_DATA(true, "pts_feature", pts_feature, pts_feature_desc, -10, 10);
    GEN_CASE_DATA(false, "argmax", argmax, argmax_desc, 0, 0);
    GEN_CASE_DATA(false, "pts_idx_of_voxels", pts_idx_of_voxels,
                  pts_idx_of_voxels_desc, 0, 0);
    GEN_CASE_DATA(false, "pooled_features", pooled_features,
                  pooled_features_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "roiaware_pool3d_forward", "pool_method",
                             pool_method);
    GEN_CASE_OP_PARAM_SINGLE(1, "roiaware_pool3d_forward", "boxes_num",
                             boxes_num);
    GEN_CASE_OP_PARAM_SINGLE(2, "roiaware_pool3d_forward", "pts_num", pts_num);
    GEN_CASE_OP_PARAM_SINGLE(3, "roiaware_pool3d_forward", "channels",
                             channels);
    GEN_CASE_OP_PARAM_SINGLE(4, "roiaware_pool3d_forward", "max_pts_each_voxel",
                             max_pts_each_voxel);
    GEN_CASE_OP_PARAM_SINGLE(5, "roiaware_pool3d_forward", "out_x", out_x);
    GEN_CASE_OP_PARAM_SINGLE(6, "roiaware_pool3d_forward", "out_y", out_y);
    GEN_CASE_OP_PARAM_SINGLE(7, "roiaware_pool3d_forward", "out_z", out_z);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }
  // generate mluOpRoiawarePool3dForward prototxt end!

  mluOpDataType_t data_dtype = pts_desc->dtype;
  uint64_t pts_dtype_size =
      mluOpGetTensorElementNum(pts_desc) * mluop::getSizeOfDataType(data_dtype);
  uint64_t pts_feature_dtype_size = mluOpGetTensorElementNum(pts_feature_desc) *
                                    mluop::getSizeOfDataType(data_dtype);
  void *pts_workspace = workspace;
  void *pts_feature_workspace = (char *)pts_workspace + pts_dtype_size;
  void *transpose_workspace =
      (char *)pts_feature_workspace + pts_feature_dtype_size;

  VLOG(5) << "[mluOpRoiawarePool3dForward] mluOpTranspose pts start.";
  int pts_dim = pts_desc->dim;
  int pts_permute[2] = {1, 0};
  int pts_tmp_dims[2] = {0, 0};
  for (int i = 0; i < pts_dim; ++i) {
    pts_tmp_dims[i] = pts_desc->dims[pts_permute[i]];
  }
  mluOpTensorDescriptor_t pts_desc_tmp = NULL;
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&pts_desc_tmp));
  PARAM_CHECK(API,
              MLUOP_STATUS_SUCCESS ==
                  mluOpSetTensorDescriptor(pts_desc_tmp, MLUOP_LAYOUT_ARRAY,
                                           data_dtype, pts_dim, pts_tmp_dims));
  PARAM_CHECK(
      API, MLUOP_STATUS_SUCCESS ==
               transposeTensor(handle, pts_desc, pts, pts_permute, pts_desc_tmp,
                               pts_workspace, transpose_workspace));
  PARAM_CHECK(
      API, MLUOP_STATUS_SUCCESS == mluOpDestroyTensorDescriptor(pts_desc_tmp));
  VLOG(5) << "[mluOpRoiawarePool3dForward] mluOpTranspose pts end.";

  VLOG(5) << "[mluOpRoiawarePool3dForward] mluOpTranspose pts_feature start.";
  int pts_feature_dim = pts_feature_desc->dim;
  int pts_feature_permute[2] = {1, 0};
  int pts_feature_tmp_dims[2] = {0, 0};
  for (int i = 0; i < pts_feature_dim; ++i) {
    pts_feature_tmp_dims[i] = pts_feature_desc->dims[pts_feature_permute[i]];
  }
  mluOpTensorDescriptor_t pts_feature_desc_tmp = NULL;
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&pts_feature_desc_tmp));
  PARAM_CHECK(API, MLUOP_STATUS_SUCCESS ==
                       mluOpSetTensorDescriptor(
                           pts_feature_desc_tmp, MLUOP_LAYOUT_ARRAY, data_dtype,
                           pts_feature_dim, pts_feature_tmp_dims));
  PARAM_CHECK(API,
              MLUOP_STATUS_SUCCESS ==
                  transposeTensor(handle, pts_feature_desc, pts_feature,
                                  pts_feature_permute, pts_feature_desc_tmp,
                                  pts_feature_workspace, transpose_workspace));
  PARAM_CHECK(API, MLUOP_STATUS_SUCCESS ==
                       mluOpDestroyTensorDescriptor(pts_feature_desc_tmp));
  VLOG(5) << "[mluOpRoiawarePool3dForward] mluOpTranspose pts_feature end.";

  VLOG(5) << "[mluOpRoiawarePool3dForward] mluOpFill_v3 host pointer start.";
  int argmax_initial_value = (pool_method == 0) ? -1 : 0;
  PARAM_CHECK(
      API, MLUOP_STATUS_SUCCESS == mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST,
                                                &argmax_initial_value,
                                                argmax_desc, argmax));
  int pts_idx_initial_value = 0;
  PARAM_CHECK(API, MLUOP_STATUS_SUCCESS ==
                       mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST,
                                    &pts_idx_initial_value,
                                    pts_idx_of_voxels_desc, pts_idx_of_voxels));
  int pooled_features_initial_value = 0;
  PARAM_CHECK(API, MLUOP_STATUS_SUCCESS ==
                       mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST,
                                    &pooled_features_initial_value,
                                    pooled_features_desc, pooled_features));
  VLOG(5) << "[mluOpRoiawarePool3dForward] mluOpFill host pointer end.";

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  mluOpStatus_t status = MLUOP_STATUS_BAD_PARAM;
  status = kernelPtsIdxOfVoxelsPolicyFunc(handle, pts_idx_of_voxels_desc,
                                          &k_dim, &k_type);
  if (MLUOP_STATUS_SUCCESS != status) {
    GEN_CASE_END();
    return status;
  }

  int core_dim = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  VLOG(5) << "[mluOpRoiawarePool3dForward] Launch Kernel "
             "MLUUnionKernelPtsIdxOfVoxels<<< Union"
          << k_type / core_dim << ", " << k_dim.x << ", " << k_dim.y << ", "
          << k_dim.z << " >>>"
          << " core_dim : " << core_dim;
  if (rois_desc->dtype == MLUOP_DTYPE_HALF) {
    VLOG(5) << "[mluOpRoiawarePool3dForward] Launch Kernel "
               "mluOpUnionKernelPtsIdxOfVoxelsHalf().";
    KERNEL_CHECK((mluOpUnionKernelPtsIdxOfVoxelsHalf(
        k_dim, k_type, handle->queue, pool_method, boxes_num, pts_num,
        max_pts_each_voxel, out_x, out_y, out_z, rois, pts_workspace,
        pts_idx_of_voxels)));
  } else {
    VLOG(5) << "[mluOpRoiawarePool3dForward] Launch Kernel "
               "mluOpUnionKernelPtsIdxOfVoxelsFloat().";
    KERNEL_CHECK((mluOpUnionKernelPtsIdxOfVoxelsFloat(
        k_dim, k_type, handle->queue, pool_method, boxes_num, pts_num,
        max_pts_each_voxel, out_x, out_y, out_z, rois, pts_workspace,
        pts_idx_of_voxels)));
  }
  VLOG(5) << "[mluOpRoiawarePool3dForward] Finish kernel "
             "MLUUnionKernelPtsIdxOfVoxels.";

  status = kernelRoiawarePool3dForwardPolicyFunc(handle, pooled_features_desc,
                                                 &k_dim, &k_type);
  if (MLUOP_STATUS_SUCCESS != status) {
    GEN_CASE_END();
    return status;
  }

  VLOG(5) << "[mluOpRoiawarePool3dForward] Launch Kernel "
             "MLUUnionKernelRoiawarePool3dForward<<< Union"
          << k_type / core_dim << ", " << k_dim.x << ", " << k_dim.y << ", "
          << k_dim.z << " >>>"
          << " core_dim : " << core_dim;
  if (pooled_features_desc->dtype == MLUOP_DTYPE_HALF) {
    VLOG(5) << "[mluOpRoiawarePool3dForward] Launch Kernel "
               "mluOpUnionKernelRoiawarePool3dForwardHalf().";
    KERNEL_CHECK((mluOpUnionKernelRoiawarePool3dForwardHalf(
        k_dim, k_type, handle->queue, pool_method, boxes_num, pts_num, channels,
        max_pts_each_voxel, out_x, out_y, out_z, pts_feature_workspace,
        pts_idx_of_voxels, pooled_features, argmax)));
  } else {
    VLOG(5) << "[mluOpRoiawarePool3dForward] Launch Kernel "
               "mluOpUnionKernelRoiawarePool3dForwardFloat().";
    KERNEL_CHECK((mluOpUnionKernelRoiawarePool3dForwardFloat(
        k_dim, k_type, handle->queue, pool_method, boxes_num, pts_num, channels,
        max_pts_each_voxel, out_x, out_y, out_z, pts_feature_workspace,
        pts_idx_of_voxels, pooled_features, argmax)));
  }
  VLOG(5) << "[mluOpRoiawarePool3dForward] Finish kernel "
             "MLUUnionKernelRoiawarePool3dForward.";
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpRoiawarePool3dBackward(
    mluOpHandle_t handle, const int pool_method, const int boxes_num,
    const int out_x, const int out_y, const int out_z, const int channels,
    const int max_pts_each_voxel,
    const mluOpTensorDescriptor_t pts_idx_of_voxels_desc,
    const void *pts_idx_of_voxels, const mluOpTensorDescriptor_t argmax_desc,
    const void *argmax, const mluOpTensorDescriptor_t grad_out_desc,
    const void *grad_out, const mluOpTensorDescriptor_t grad_in_desc,
    void *grad_in) {
  // pts_idx_of_voxels: (boxes_num, out_x, out_y, out_z, max_pts_each_voxel)
  // argmax: (boxes_num, out_x, out_y, out_z, channels)
  // grad_out: (boxes_num, out_x, out_y, out_z, channels)
  // grad_in: (pts_num, channels)

  const std::string API = "[mluOpRoiawarePool3dBackward]";
  // check desc
  PARAM_CHECK(API, handle != NULL);
  PARAM_CHECK(API, pts_idx_of_voxels_desc != NULL);
  PARAM_CHECK(API, argmax_desc != NULL);
  PARAM_CHECK(API, grad_out_desc != NULL);
  PARAM_CHECK(API, grad_in_desc != NULL);

  // check dim
  PARAM_CHECK_EQ(API, pts_idx_of_voxels_desc->dim, 5);
  PARAM_CHECK_EQ(API, argmax_desc->dim, 5);
  PARAM_CHECK_EQ(API, grad_out_desc->dim, 5);
  PARAM_CHECK_EQ(API, grad_in_desc->dim, 2);

  // check shape
  PARAM_CHECK_EQ(API, pts_idx_of_voxels_desc->dims[0], boxes_num);
  PARAM_CHECK_EQ(API, pts_idx_of_voxels_desc->dims[1], out_x);
  PARAM_CHECK_EQ(API, pts_idx_of_voxels_desc->dims[2], out_y);
  PARAM_CHECK_EQ(API, pts_idx_of_voxels_desc->dims[3], out_z);
  PARAM_CHECK_EQ(API, pts_idx_of_voxels_desc->dims[4], max_pts_each_voxel);
  PARAM_CHECK_EQ(API, argmax_desc->dims[0], boxes_num);
  PARAM_CHECK_EQ(API, argmax_desc->dims[1], out_x);
  PARAM_CHECK_EQ(API, argmax_desc->dims[2], out_y);
  PARAM_CHECK_EQ(API, argmax_desc->dims[3], out_z);
  PARAM_CHECK_EQ(API, argmax_desc->dims[4], channels);
  PARAM_CHECK_EQ(API, grad_out_desc->dims[0], boxes_num);
  PARAM_CHECK_EQ(API, grad_out_desc->dims[1], out_x);
  PARAM_CHECK_EQ(API, grad_out_desc->dims[2], out_y);
  PARAM_CHECK_EQ(API, grad_out_desc->dims[3], out_z);
  PARAM_CHECK_EQ(API, grad_out_desc->dims[4], channels);
  // grad_in_desc->dims[0] == pts_num
  PARAM_CHECK_EQ(API, grad_in_desc->dims[1], channels);

  // check dtype
  PARAM_CHECK(API, pts_idx_of_voxels_desc->dtype == MLUOP_DTYPE_INT32);
  PARAM_CHECK(API, argmax_desc->dtype == MLUOP_DTYPE_INT32);
  PARAM_CHECK(API, grad_out_desc->dtype == MLUOP_DTYPE_FLOAT ||
                       grad_out_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK(API, grad_in_desc->dtype == grad_out_desc->dtype);

  // check other parms : pool_method
  PARAM_CHECK(API, pool_method == 0 || pool_method == 1);

  const int pts_num = grad_in_desc->dims[0];
  // check tensor dim
  PARAM_CHECK(API, boxes_num > 0);
  PARAM_CHECK(API, pts_num > 0);
  PARAM_CHECK(API, channels > 0);
  PARAM_CHECK(API, max_pts_each_voxel > 0);
  PARAM_CHECK(API, out_x > 0);
  PARAM_CHECK(API, out_y > 0);
  PARAM_CHECK(API, out_z > 0);

  const uint64_t tensor_pts_idx_of_voxels_num =
      mluOpGetTensorElementNum(pts_idx_of_voxels_desc);
  const uint64_t tensor_argmax_num = mluOpGetTensorElementNum(argmax_desc);
  const uint64_t tensor_grad_out_num = mluOpGetTensorElementNum(grad_out_desc);
  const uint64_t tensor_grad_in_num = mluOpGetTensorElementNum(grad_in_desc);

  // check large tensor
  TENSOR_NUM_CHECK(API, tensor_pts_idx_of_voxels_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, tensor_argmax_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, tensor_grad_out_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, tensor_grad_in_num, LARGE_TENSOR_NUM, "");

  // check zero element
  PARAM_CHECK_GT(API, tensor_pts_idx_of_voxels_num, 0);
  PARAM_CHECK_GT(API, tensor_argmax_num, 0);
  PARAM_CHECK_GT(API, tensor_grad_out_num, 0);
  PARAM_CHECK_GT(API, tensor_grad_in_num, 0);

  // check ptr
  PARAM_CHECK(API, pts_idx_of_voxels != NULL);
  PARAM_CHECK(API, argmax != NULL);
  PARAM_CHECK(API, grad_out != NULL);
  PARAM_CHECK(API, grad_in != NULL);

  // check arch
  if (handle->arch < MLUOP_MLU370) {
    LOG(ERROR) << API
               << " The operator does not match the current architecture.";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  VLOG(5) << "pool_method = " << pool_method
          << ", boxes_num = " << boxes_num
          << ", out_x = " << out_x
          << ", out_y = " << out_y
          << ", out_z = " << out_z
          << ", channels = " << channels
          << ", max_pts_each_voxel = " << max_pts_each_voxel
          << ", points num = " << grad_in_desc->dims[0];

  // generate mluOpRoiawarePool3dBackward prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("roiaware_pool3d_backward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA_REAL(true, "pts_idx_of_voxels", pts_idx_of_voxels,
                       pts_idx_of_voxels_desc);
    GEN_CASE_DATA_REAL(true, "argmax", argmax, argmax_desc);
    GEN_CASE_DATA_REAL(true, "grad_out", grad_out, grad_out_desc);
    GEN_CASE_DATA(false, "grad_in", grad_in, grad_in_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "roiaware_pool3d_backward", "pool_method",
                             pool_method);
    GEN_CASE_OP_PARAM_SINGLE(1, "roiaware_pool3d_backward", "boxes_num",
                             boxes_num);
    GEN_CASE_OP_PARAM_SINGLE(2, "roiaware_pool3d_backward", "out_x", out_x);
    GEN_CASE_OP_PARAM_SINGLE(3, "roiaware_pool3d_backward", "out_y", out_y);
    GEN_CASE_OP_PARAM_SINGLE(4, "roiaware_pool3d_backward", "out_z", out_z);
    GEN_CASE_OP_PARAM_SINGLE(5, "roiaware_pool3d_backward", "channels",
                             channels);
    GEN_CASE_OP_PARAM_SINGLE(6, "roiaware_pool3d_backward",
                             "max_pts_each_voxel", max_pts_each_voxel);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }
  // generate mluOpRoiawarePool3dBackward prototxt end!

  int grad_in_initial_value = 0;
  PARAM_CHECK(
      API, MLUOP_STATUS_SUCCESS == mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST,
                                                &grad_in_initial_value,
                                                grad_in_desc, grad_in));
  VLOG(5)
      << "[mluOpRoiawarePool3dBackward] Initialize output space successfully.";

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  mluOpStatus_t status = MLUOP_STATUS_BAD_PARAM;
  status = kernelRoiawarePool3dBackwardPolicyFunc(handle, grad_out_desc, &k_dim,
                                                  &k_type);
  if (MLUOP_STATUS_SUCCESS != status) {
    GEN_CASE_END();
    return status;
  }

  int core_dim = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  VLOG(5) << "[mluOpRoiawarePool3dBackward] Launch Kernel "
             "MLUUnionKernelRoiawarePool3dBackward<<< Union"
          << k_type / core_dim << ", " << k_dim.x << ", " << k_dim.y << ", "
          << k_dim.z << " >>>"
          << " core_dim : " << core_dim;
  if (grad_out_desc->dtype == MLUOP_DTYPE_HALF) {
    VLOG(5) << "[mluOpRoiawarePool3dBackward] Launch Kernel "
               "mluOpUnionKernelRoiawarePool3dBackwardHalf().";
    KERNEL_CHECK((mluOpUnionKernelRoiawarePool3dBackwardHalf(
        k_dim, k_type, handle->queue, pool_method, boxes_num, out_x, out_y,
        out_z, channels, max_pts_each_voxel, pts_idx_of_voxels, argmax,
        grad_out, grad_in)));
  } else {
    VLOG(5) << "[mluOpRoiawarePool3dBackward] Launch Kernel "
               "mluOpUnionKernelRoiawarePool3dBackwardFloat().";
    KERNEL_CHECK((mluOpUnionKernelRoiawarePool3dBackwardFloat(
        k_dim, k_type, handle->queue, pool_method, boxes_num, out_x, out_y,
        out_z, channels, max_pts_each_voxel, pts_idx_of_voxels, argmax,
        grad_out, grad_in)));
  }
  VLOG(5) << "[mluOpRoiawarePool3dBackward] Finish kernel "
             "MLUUnionKernelRoiawarePool3dBackward.";
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
