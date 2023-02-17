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
#include "core/context.h"
#include "core/logging.h"
#include "core/gen_case.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "core/tool.h"
#include "kernels/kernel.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))

static mluOpStatus_t paramcheck(const int batch_size, const int pts_num,
                         const int boxes_num, const int feature_in_len,
                         const int sampled_pts_num,
                         const mluOpTensorDescriptor_t points_desc,
                         const mluOpTensorDescriptor_t point_features_desc,
                         const mluOpTensorDescriptor_t boxes3d_desc,
                         const mluOpTensorDescriptor_t pooled_features_desc,
                         const mluOpTensorDescriptor_t pooled_empty_flag_desc) {
  // check tensor dim
  // params points: [B, N, 3]
  PARAM_CHECK_EQ("[mluOpRoiPointPool3d]", points_desc->dim, 3);
  PARAM_CHECK_EQ("[mluOpRoiPointPool3d]", points_desc->dims[2], 3);
  // params point_features: [B, N, C]
  PARAM_CHECK_EQ("[mluOpRoiPointPool3d]", point_features_desc->dim, 3);
  // params boxes3d: [B, M, 7]
  PARAM_CHECK_EQ("[mluOpRoiPointPool3d]", boxes3d_desc->dim, 3);
  PARAM_CHECK_EQ("[mluOpRoiPointPool3d]", boxes3d_desc->dims[2], 7);
  // params pooled_features: [B, M, sampled_pts_num, 3+C]
  PARAM_CHECK_EQ("[mluOpRoiPointPool3d]", pooled_features_desc->dim, 4);
  // params pooled_empty_flag: [B, M]
  PARAM_CHECK_EQ("[mluOpRoiPointPool3d]", pooled_empty_flag_desc->dim, 2);

  // check tensor shape
  PARAM_CHECK(
      "[mluOpRoiPointPool3d]",
      (points_desc->dims[0] == pooled_features_desc->dims[0]) &&
          (point_features_desc->dims[0] == pooled_features_desc->dims[0]) &&
          (boxes3d_desc->dims[0] == pooled_features_desc->dims[0]) &&
          (pooled_empty_flag_desc->dims[0] == pooled_features_desc->dims[0]));
  PARAM_CHECK("[mluOpRoiPointPool3d]",
              (pooled_features_desc->dims[1] == boxes3d_desc->dims[1]) &&
                  (pooled_empty_flag_desc->dims[1] == boxes3d_desc->dims[1]));
  PARAM_CHECK("[mluOpRoiPointPool3d]",
              points_desc->dims[1] == point_features_desc->dims[1]);
  PARAM_CHECK("[mluOpRoiPointPool3d]", point_features_desc->dims[2] + 3 ==
                                           pooled_features_desc->dims[3]);

  // check params
  PARAM_CHECK_EQ("[mluOpRoiPointPool3d]", batch_size, points_desc->dims[0]);
  PARAM_CHECK_EQ("[mluOpRoiPointPool3d]", pts_num, points_desc->dims[1]);
  PARAM_CHECK_EQ("[mluOpRoiPointPool3d]", boxes_num, boxes3d_desc->dims[1]);
  PARAM_CHECK_EQ("[mluOpRoiPointPool3d]", feature_in_len,
                 point_features_desc->dims[2]);
  PARAM_CHECK_EQ("[mluOpRoiPointPool3d]", sampled_pts_num,
                 pooled_features_desc->dims[2]);

  // check tensor datatype
  PARAM_CHECK("[mluOpRoiPointPool3d]",
              (points_desc->dtype == MLUOP_DTYPE_FLOAT) ||
                  (points_desc->dtype == MLUOP_DTYPE_HALF));
  PARAM_CHECK("[mluOpRoiPointPool3d]",
              pooled_empty_flag_desc->dtype == MLUOP_DTYPE_INT32);
  // points, point_features, boxes3d_desc, pooled_features datatype must be the
  // same
  PARAM_CHECK("[mluOpRoiPointPool3d]",
              (points_desc->dtype == pooled_features_desc->dtype) &&
                  (point_features_desc->dtype == pooled_features_desc->dtype) &&
                  (boxes3d_desc->dtype == pooled_features_desc->dtype));

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetRoiPointPool3dWorkspaceSize(
    mluOpHandle_t handle, const int batch_size, const int pts_num,
    const int boxes_num, const int feature_in_len, const int sampled_pts_num,
    const mluOpTensorDescriptor_t points_desc,
    const mluOpTensorDescriptor_t point_features_desc,
    const mluOpTensorDescriptor_t boxes3d_desc,
    const mluOpTensorDescriptor_t pooled_features_desc,
    const mluOpTensorDescriptor_t pooled_empty_flag_desc, size_t *size) {
  // handle and desc ptr check null
  PARAM_CHECK("[mluOpRoiPointPool3d]", handle != NULL);
  PARAM_CHECK("[mluOpRoiPointPool3d]", points_desc != NULL);
  PARAM_CHECK("[mluOpRoiPointPool3d]", point_features_desc != NULL);
  PARAM_CHECK("[mluOpRoiPointPool3d]", boxes3d_desc != NULL);
  PARAM_CHECK("[mluOpRoiPointPool3d]", pooled_features_desc != NULL);
  PARAM_CHECK("[mluOpRoiPointPool3d]", pooled_empty_flag_desc != NULL);
  PARAM_CHECK("[mluOpRoiPointPool3d]", size != NULL);

  // check params
  mluOpStatus_t paramcheck_status =
      paramcheck(batch_size, pts_num, boxes_num, feature_in_len,
                 sampled_pts_num, points_desc, point_features_desc,
                 boxes3d_desc, pooled_features_desc, pooled_empty_flag_desc);
  if (paramcheck_status != MLUOP_STATUS_SUCCESS) {
    return paramcheck_status;
  }

  size_t points_element_num = mluOpGetTensorElementNum(points_desc);
  size_t point_features_element_num =
      mluOpGetTensorElementNum(point_features_desc);
  size_t boxes3d_element_num = mluOpGetTensorElementNum(boxes3d_desc);
  size_t pooled_features_element_num =
      mluOpGetTensorElementNum(pooled_features_desc);
  size_t pooled_empty_flag_num =
      mluOpGetTensorElementNum(pooled_empty_flag_desc);

  // check large tensor
  TENSOR_NUM_CHECK("[mluOpRoiPointPool3d]", points_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpRoiPointPool3d]", point_features_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpRoiPointPool3d]", boxes3d_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpRoiPointPool3d]", pooled_features_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpRoiPointPool3d]", pooled_empty_flag_num,
                   LARGE_TENSOR_NUM, "");

  // check element num zero
  if (points_element_num == 0 || point_features_element_num == 0 ||
      boxes3d_element_num == 0 || pooled_features_element_num == 0 ||
      pooled_empty_flag_num == 0) {
    LOG(ERROR) << "[mluOpRoiPointPool3d] Check failed: element num zero.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // workspace for points_xyz : [3, B, N]
  *size = points_element_num * mluop::getSizeOfDataType(points_desc->dtype);
  // workspace for point_features_transpose : [C, B, N]
  *size += point_features_element_num *
           mluop::getSizeOfDataType(point_features_desc->dtype);

  mluOpTransposeDescriptor_t trans_desc;
  size_t transpose_workspace_size0 = 0;
  size_t transpose_workspace_size1 = 0;
  int permute[3] = {2, 0, 1};
  PARAM_CHECK(
      "[mluOpGetRoiPointPool3dWorkspaceSize]",
      MLUOP_STATUS_SUCCESS == mluOpCreateTransposeDescriptor(&trans_desc));
  PARAM_CHECK("[mluOpGetRoiPointPool3dWorkspaceSize]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpSetTransposeDescriptor(trans_desc, (int)3, permute));
  PARAM_CHECK("[mluOpGetRoiPointPool3dWorkspaceSize]",
              MLUOP_STATUS_SUCCESS == mluOpGetTransposeWorkspaceSize(
                                          handle, points_desc, trans_desc,
                                          &transpose_workspace_size0));
  PARAM_CHECK("[mluOpGetRoiPointPool3dWorkspaceSize]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpGetTransposeWorkspaceSize(handle, point_features_desc,
                                                 trans_desc,
                                                 &transpose_workspace_size1));
  *size += MAX(transpose_workspace_size0, transpose_workspace_size1);
  PARAM_CHECK(
      "[mluOpGetRoiPointPool3dWorkspaceSize]",
      MLUOP_STATUS_SUCCESS == mluOpDestroyTransposeDescriptor(trans_desc));

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpRoiPointPool3d(
    mluOpHandle_t handle, const int batch_size, const int pts_num,
    const int boxes_num, const int feature_in_len, const int sampled_pts_num,
    const mluOpTensorDescriptor_t points_desc, const void *points,
    const mluOpTensorDescriptor_t point_features_desc,
    const void *point_features, const mluOpTensorDescriptor_t boxes3d_desc,
    const void *boxes3d, void *workspace, size_t workspace_size,
    const mluOpTensorDescriptor_t pooled_features_desc, void *pooled_features,
    const mluOpTensorDescriptor_t pooled_empty_flag_desc,
    void *pooled_empty_flag) {
  // handle and desc ptr check null
  PARAM_CHECK("[mluOpRoiPointPool3d]", handle != NULL);
  PARAM_CHECK("[mluOpRoiPointPool3d]", points_desc != NULL);
  PARAM_CHECK("[mluOpRoiPointPool3d]", point_features_desc != NULL);
  PARAM_CHECK("[mluOpRoiPointPool3d]", boxes3d_desc != NULL);
  PARAM_CHECK("[mluOpRoiPointPool3d]", pooled_features_desc != NULL);
  PARAM_CHECK("[mluOpRoiPointPool3d]", pooled_empty_flag_desc != NULL);

  // check params
  mluOpStatus_t paramcheck_status =
      paramcheck(batch_size, pts_num, boxes_num, feature_in_len,
                 sampled_pts_num, points_desc, point_features_desc,
                 boxes3d_desc, pooled_features_desc, pooled_empty_flag_desc);
  if (paramcheck_status != MLUOP_STATUS_SUCCESS) {
    return paramcheck_status;
  }

  // check workspace
  if (workspace_size > 0) {
    PARAM_CHECK("[mluOpRoiPointPool3d]", workspace != NULL);
  }

  // check element num zero
  size_t points_element_num = mluOpGetTensorElementNum(points_desc);
  size_t point_features_element_num =
      mluOpGetTensorElementNum(point_features_desc);
  size_t boxes3d_element_num = mluOpGetTensorElementNum(boxes3d_desc);
  size_t pooled_features_element_num =
      mluOpGetTensorElementNum(pooled_features_desc);
  size_t pooled_empty_flag_num =
      mluOpGetTensorElementNum(pooled_empty_flag_desc);

  if (points_element_num == 0 || point_features_element_num == 0 ||
      boxes3d_element_num == 0 || pooled_features_element_num == 0 ||
      pooled_empty_flag_num == 0) {
    LOG(ERROR) << "[mluOpRoiPointPool3d] Check failed: element num zero.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  PARAM_CHECK("[mluOpRoiPointPool3d]", points != NULL);
  PARAM_CHECK("[mluOpRoiPointPool3d]", point_features != NULL);
  PARAM_CHECK("[mluOpRoiPointPool3d]", boxes3d != NULL);
  PARAM_CHECK("[mluOpRoiPointPool3d]", pooled_features != NULL);
  PARAM_CHECK("[mluOpRoiPointPool3d]", pooled_empty_flag != NULL);

  // generate mluOpRoiPointPool3d prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("roipoint_pool3d");
    // set handle dump mlu output
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "points", points, points_desc, 10, -10);
    GEN_CASE_DATA(true, "point_features", point_features, point_features_desc,
                  10, -10);
    GEN_CASE_DATA(true, "boxes3d", boxes3d, boxes3d_desc, 10, -10);
    GEN_CASE_DATA(false, "pooled_features", pooled_features,
                  pooled_features_desc, 0, 0);
    GEN_CASE_DATA(false, "pooled_empty_flag", pooled_empty_flag,
                  pooled_empty_flag_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "roipoint_pool3d", "num_sampled_points",
                  sampled_pts_num);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0, 0, 0);
  }

  // points_xyz : [3, B, N]
  void *points_xyz = workspace;
  // point_features : [B, C, N]
  void *point_features_transpose =
      (char *)workspace +
      points_element_num * mluop::getSizeOfDataType(points_desc->dtype);
  void *transpose_workspace =
      (char *)point_features_transpose +
      point_features_element_num *
          mluop::getSizeOfDataType(point_features_desc->dtype);
  size_t transpose_workspace_size = 0;
  mluOpTensorDescriptor_t output_transpose_desc;
  mluOpTransposeDescriptor_t trans_desc;
  const int dims = 3;
  int points_permute[3] = {2, 0, 1};
  int points_dims[3];
  points_dims[0] = points_desc->dims[2];
  points_dims[1] = points_desc->dims[0];
  points_dims[2] = points_desc->dims[1];

  PARAM_CHECK("[mluOpGetRoiPointPool3d]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpCreateTensorDescriptor(&output_transpose_desc));
  PARAM_CHECK(
      "[mluOpGetRoiPointPool3d]",
      MLUOP_STATUS_SUCCESS ==
          mluOpSetTensorDescriptor(output_transpose_desc, MLUOP_LAYOUT_ARRAY,
                                   points_desc->dtype, dims, points_dims));
  PARAM_CHECK(
      "[mluOpGetRoiPointPool3d]",
      MLUOP_STATUS_SUCCESS == mluOpCreateTransposeDescriptor(&trans_desc));
  PARAM_CHECK("[mluOpGetRoiPointPool3d]",
              MLUOP_STATUS_SUCCESS == mluOpSetTransposeDescriptor(
                                          trans_desc, dims, points_permute));
  PARAM_CHECK("[mluOpGetRoiPointPool3d]",
              MLUOP_STATUS_SUCCESS == mluOpGetTransposeWorkspaceSize(
                                          handle, points_desc, trans_desc,
                                          &transpose_workspace_size));
  // transpose points [B, N ,3] -> [3, B, N]
  PARAM_CHECK(
      "[mluOpGetRoiPointPool3d]",
      MLUOP_STATUS_SUCCESS == mluOpTranspose_v2(handle, trans_desc, points_desc,
                                                points, output_transpose_desc,
                                                points_xyz, transpose_workspace,
                                                transpose_workspace_size));

  int point_features_permute[3] = {0, 2, 1};
  int point_features_dims[3];
  point_features_dims[0] = point_features_desc->dims[0];
  point_features_dims[1] = point_features_desc->dims[2];
  point_features_dims[2] = point_features_desc->dims[1];
  PARAM_CHECK("[mluOpGetRoiPointPool3d]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpSetTensorDescriptor(
                      output_transpose_desc, MLUOP_LAYOUT_ARRAY,
                      point_features_desc->dtype, dims, point_features_dims));
  PARAM_CHECK(
      "[mluOpGetRoiPointPool3d]",
      MLUOP_STATUS_SUCCESS == mluOpSetTransposeDescriptor(
                                  trans_desc, dims, point_features_permute));
  PARAM_CHECK("[mluOpGetRoiPointPool3d]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpGetTransposeWorkspaceSize(handle, point_features_desc,
                                                 trans_desc,
                                                 &transpose_workspace_size));
  // transpose point_features [B, N, C] -> [B, C, N]
  PARAM_CHECK("[mluOpGetRoiPointPool3d]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpTranspose_v2(
                      handle, trans_desc, point_features_desc, point_features,
                      output_transpose_desc, point_features_transpose,
                      transpose_workspace, transpose_workspace_size));
  PARAM_CHECK("[mluOpGetRoiPointPool3d]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpDestroyTensorDescriptor(output_transpose_desc));
  PARAM_CHECK(
      "[mluOpGetRoiPointPool3d]",
      MLUOP_STATUS_SUCCESS == mluOpDestroyTransposeDescriptor(trans_desc));

  // start U1 task, occupy all available clusters
  cnrtDim3_t k_dims;
  k_dims.x = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  k_dims.y = mluop::runtime::getClusterLimitCapability(handle);
  k_dims.z = 1;
  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;

  switch (points_desc->dtype) {
    default: {
      VLOG(5) << "Not implemented";
      break;
    }
    case MLUOP_DTYPE_FLOAT: {
      if (boxes_num <= 10240) {
        VLOG(5) << "Launch Kernel roipointPool3dUnion1<<<Union"
                << k_type / CORE_DIM << ", " << k_dims.x << ", " << k_dims.y
                << ", " << k_dims.z << ">>>";
        KERNEL_CHECK((mluOpUnion1KernelRoipointPool3dFloat(
            k_dims, k_type, handle->queue, batch_size, pts_num, boxes_num,
            feature_in_len, sampled_pts_num, (char *)points_xyz,
            (char *)point_features_transpose, (char *)boxes3d,
            (char *)pooled_features, (char *)pooled_empty_flag)));
      } else {
        VLOG(5) << "Launch Kernel roipointPool3dUnion1LargeBoxesNum<<<Union"
                << k_type / CORE_DIM << ", " << k_dims.x << ", " << k_dims.y
                << ", " << k_dims.z << ">>>";
        KERNEL_CHECK((mluOpUnion1KernelRoipointPool3dLargeBoxesNumFloat(
            k_dims, k_type, handle->queue, batch_size, pts_num, boxes_num,
            feature_in_len, sampled_pts_num, (char *)points_xyz,
            (char *)point_features_transpose, (char *)boxes3d,
            (char *)pooled_features, (char *)pooled_empty_flag)));
      }
    }; break;
    case MLUOP_DTYPE_HALF: {
      if (boxes_num <= 10240) {
        VLOG(5) << "Launch Kernel roipointPool3dUnion1<<<Union"
                << k_type / CORE_DIM << ", " << k_dims.x << ", " << k_dims.y
                << ", " << k_dims.z << ">>>";
        KERNEL_CHECK((mluOpUnion1KernelRoipointPool3dHalf(
            k_dims, k_type, handle->queue, batch_size, pts_num, boxes_num,
            feature_in_len, sampled_pts_num, (char *)points_xyz,
            (char *)point_features_transpose, (char *)boxes3d,
            (char *)pooled_features, (char *)pooled_empty_flag)));
      } else {
        VLOG(5) << "Launch Kernel roipointPool3dUnion1LargeBoxesNum<<<Union"
                << k_type / CORE_DIM << ", " << k_dims.x << ", " << k_dims.y
                << ", " << k_dims.z << ">>>";
        KERNEL_CHECK((mluOpUnion1KernelRoipointPool3dLargeBoxesNumHalf(
            k_dims, k_type, handle->queue, batch_size, pts_num, boxes_num,
            feature_in_len, sampled_pts_num, (char *)points_xyz,
            (char *)point_features_transpose, (char *)boxes3d,
            (char *)pooled_features, (char *)pooled_empty_flag)));
      }
    }; break;
  }

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
