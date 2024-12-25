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

#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"  // mluop::getSizeOfDataType
#include "kernels/kernel.h"
#include "kernels/utils/cnnl_helper.h"

static mluOpStatus_t DynamicPointToVoxelBackwardParamCheck(
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

  // check stride
  STRIDE_TENSOR_CHECK(
      "[mluOpDynamicPointToVoxelBackward]:", grad_voxel_feats_desc,
      "grad_voxel_feats_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpDynamicPointToVoxelBackward]:", feats_desc,
                      "feats_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpDynamicPointToVoxelBackward]:", voxel_feats_desc,
                      "voxel_feats_desc must be contiguous");
  STRIDE_TENSOR_CHECK(
      "[mluOpDynamicPointToVoxelBackward]:", point2voxel_map_desc,
      "point2voxel_map_desc must be contiguous");
  STRIDE_TENSOR_CHECK(
      "[mluOpDynamicPointToVoxelBackward]:", voxel_points_count_desc,
      "voxel_points_count_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpDynamicPointToVoxelBackward]:", voxel_num_desc,
                      "voxel_num_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpDynamicPointToVoxelBackward]:", grad_feats_desc,
                      "grad_feats_desc must be contiguous");

  // check data type
  PARAM_CHECK(interface_name,
              grad_voxel_feats_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(interface_name, feats_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(interface_name,
              voxel_feats_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(interface_name, grad_feats_desc->getDtype() == MLUOP_DTYPE_FLOAT);

  PARAM_CHECK(interface_name,
              point2voxel_map_desc->getDtype() == MLUOP_DTYPE_INT32);
  PARAM_CHECK(interface_name,
              voxel_points_count_desc->getDtype() == MLUOP_DTYPE_INT32);
  PARAM_CHECK(interface_name, voxel_num_desc->getDtype() == MLUOP_DTYPE_INT32);

  // check shape
  PARAM_CHECK(interface_name, grad_voxel_feats_desc->getDim() == 2);
  PARAM_CHECK(interface_name, feats_desc->getDim() == 2);
  PARAM_CHECK(interface_name, voxel_feats_desc->getDim() == 2);
  PARAM_CHECK(interface_name, point2voxel_map_desc->getDim() == 1);
  PARAM_CHECK(interface_name, voxel_points_count_desc->getDim() == 1);
  PARAM_CHECK(interface_name, voxel_num_desc->getDim() == 1);
  PARAM_CHECK(interface_name, grad_feats_desc->getDim() == 2);

  PARAM_CHECK(interface_name, feats_desc->getDimIndex(1) ==
                                  grad_voxel_feats_desc->getDimIndex(1));
  PARAM_CHECK(interface_name, voxel_feats_desc->getDimIndex(0) ==
                                  grad_voxel_feats_desc->getDimIndex(0));
  PARAM_CHECK(interface_name, voxel_feats_desc->getDimIndex(1) ==
                                  grad_voxel_feats_desc->getDimIndex(1));
  PARAM_CHECK(interface_name, point2voxel_map_desc->getDimIndex(0) ==
                                  feats_desc->getDimIndex(0));
  PARAM_CHECK(interface_name, voxel_points_count_desc->getDimIndex(0) ==
                                  grad_voxel_feats_desc->getDimIndex(0));
  PARAM_CHECK(interface_name, voxel_num_desc->getDimIndex(0) == 1);
  PARAM_CHECK(interface_name,
              grad_feats_desc->getDimIndex(0) == feats_desc->getDimIndex(0));
  PARAM_CHECK(interface_name, grad_feats_desc->getDimIndex(1) ==
                                  grad_voxel_feats_desc->getDimIndex(1));
  PARAM_CHECK(interface_name, feats_desc->getDimIndex(0) >=
                                  grad_voxel_feats_desc->getDimIndex(0));

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

  // kernel size check
  const int N = feats_desc->getDimIndex(0);
  const int C = feats_desc->getDimIndex(1);
  const size_t dtype_bytes = mluop::getSizeOfDataType(feats_desc->getDtype());
  const size_t idx_dtype_bytes =
      mluop::getSizeOfDataType(point2voxel_map_desc->getDtype());
  if (N * (idx_dtype_bytes + 1) + C * (2 * dtype_bytes + 3 * idx_dtype_bytes) +
          idx_dtype_bytes >
      handle->nram_size) {
    // float + int
    LOG(ERROR)
        << interface_name
        << " The feats dtype is float, point2voxel_map dtype is int. The feats "
           "shape is ["
        << N << ", " << C << "]"
        << ", should meet constraint : "
           "5*feats_desc->getDimIndex(0)+20*feats_desc->getDimIndex(1)+sizeof("
           "int) <= "
        << handle->nram_size;
    return MLUOP_STATUS_BAD_PARAM;
  }

  // 0-element check, after dim and shape check
  if (mluOpGetTensorElementNum(grad_feats_desc) == 0) {
    zero_element = true;
    return MLUOP_STATUS_SUCCESS;
  }
  if (grad_voxel_feats_element_num != 0) {
    PARAM_CHECK(interface_name, grad_voxel_feats != NULL);
  }
  PARAM_CHECK(interface_name, feats != NULL);
  if (mluOpGetTensorElementNum(voxel_feats_desc) != 0) {
    PARAM_CHECK(interface_name, voxel_feats != NULL);
  }
  PARAM_CHECK(interface_name, point2voxel_map != NULL);
  if (mluOpGetTensorElementNum(voxel_points_count_desc) != 0) {
    PARAM_CHECK(interface_name, voxel_points_count != NULL);
  }
  PARAM_CHECK(interface_name, voxel_num != NULL);
  PARAM_CHECK(interface_name, grad_feats != NULL);
  if (workspace_size != 0) {
    PARAM_CHECK(interface_name, workspace != NULL);
  }
  return MLUOP_STATUS_SUCCESS;
}

static void policyFunc(const mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type, int N) {
  int max_core_num = mluop::runtime::getCoreNumOfJobLimitCapability(handle);
  size_t core_num = handle->core_num_per_cluster;
  if (N > max_core_num) {
    k_dim->x = max_core_num;
    *k_type = mluop::runtime::getJobLimitCapabilityCnrtFuncType(handle);
  } else {
    if (N <= 4) {
      k_dim->x = core_num * 1;
      *k_type = cnrtFuncTypeUnion1;
    } else if (N <= 8) {
      k_dim->x = core_num * 2;
      *k_type = cnrtFuncTypeUnion2;
    } else if (N <= 16) {
      k_dim->x = core_num * 4;
      *k_type = cnrtFuncTypeUnion4;
    } else if (N <= 32) {
      k_dim->x = core_num * 8;
      *k_type = cnrtFuncTypeUnion8;
    } else if (N <= 64) {
      k_dim->x = core_num * 16;
      *k_type = cnrtFuncTypeUnion16;
    } else {
      LOG(ERROR)
          << "[mluOpDynamicPointToVoxelBackward]: failed to choose kernel "
             "to launch";
      return;
    }
  }
  k_dim->y = 1;
  k_dim->z = 1;
  VLOG(5) << "Launch Kernel MLUKernelDynamicPointToVoxelBackward in UNION"
          << *k_type / 4 << " type";
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
    VLOG(5) << interface_name << " Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  // generator
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("dynamic_point_to_voxel_backward",
                   "DYNAMIC_POINT_TO_VOXEL_BACKWARD");
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

  const int N = feats_desc->getDimIndex(0);
  const int C = feats_desc->getDimIndex(1);
  const auto grad_voxel_feats_element_num =
      mluOpGetTensorElementNum(grad_voxel_feats_desc);
  const auto grad_feats_element_num = mluOpGetTensorElementNum(grad_feats_desc);
  VLOG(5) << interface_name << " N = " << N << ", C = " << C
          << ", grad_voxel_feats_element_num=" << grad_voxel_feats_element_num
          << ", grad_feats_element_num=" << grad_feats_element_num;
  // 1. init output
  uint64_t fill_0 = 0x0;
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grad_feats_desc,
                                                 cnnl_output_desc);
    CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &fill_0,
                          cnnl_output_desc, grad_feats));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, &k_dim, &k_type, N);
  if (grad_voxel_feats_element_num != 0) {
    // 2. init workspace
    mluOpTensorDescriptor_t indices_desc;
    CHECK_RETURN(interface_name, mluOpCreateTensorDescriptor(&indices_desc));
    int indices_dims[2] = {(int)grad_voxel_feats_element_num, 1};
    CHECK_RETURN(interface_name,
                 mluOpSetTensorDescriptor(indices_desc, MLUOP_LAYOUT_ARRAY,
                                          MLUOP_DTYPE_INT32, 2, indices_dims));
    {
      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(indices_desc,
                                                   cnnl_output_desc);
      CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST,
                            &grad_feats_element_num, cnnl_output_desc,
                            workspace));
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }
    // 3. get scatter indices
    CHECK_RETURN("[mluOpDynamicPointToVoxelBackward]",
                 KernelDynamicPointToVoxelBackward(
                     k_dim, k_type, handle->queue, feats, voxel_feats,
                     workspace, point2voxel_map, voxel_num, N, C));
    // 4. scatter
    cnnlScatterNdMode_t scatter_mode = CNNL_SCATTERND_ADD;
    mluOpTensorDescriptor_t updates_desc;
    CHECK_RETURN(interface_name, mluOpCreateTensorDescriptor(&updates_desc));
    int updates_dims[1] = {(int)grad_voxel_feats_element_num};
    CHECK_RETURN(interface_name,
                 mluOpSetTensorDescriptor(updates_desc, MLUOP_LAYOUT_ARRAY,
                                          MLUOP_DTYPE_FLOAT, 1, updates_dims));
    mluOpTensorDescriptor_t output_desc;
    CHECK_RETURN(interface_name, mluOpCreateTensorDescriptor(&output_desc));
    int output_dims[1] = {(int)grad_feats_element_num};
    CHECK_RETURN(interface_name,
                 mluOpSetTensorDescriptor(output_desc, MLUOP_LAYOUT_ARRAY,
                                          MLUOP_DTYPE_FLOAT, 1, output_dims));
    {
      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(indices_desc,
                                                   cnnl_indices_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(updates_desc,
                                                   cnnl_updates_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc,
                                                   cnnl_output_desc);

      CALL_CNNL(cnnlScatterNd_v2(cnnl_handle, scatter_mode, cnnl_indices_desc,
                                 workspace, cnnl_updates_desc, grad_voxel_feats,
                                 NULL, NULL, cnnl_output_desc, grad_feats));
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_indices_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_updates_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }
    CHECK_RETURN(interface_name, mluOpDestroyTensorDescriptor(updates_desc));
    CHECK_RETURN(interface_name, mluOpDestroyTensorDescriptor(output_desc));
    CHECK_RETURN(interface_name, mluOpDestroyTensorDescriptor(indices_desc));
  }
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetDynamicPointToVoxelBackwardWorkspaceSize(
    const mluOpHandle_t handle, const mluOpReduceMode_t reduce_type,
    const mluOpTensorDescriptor_t grad_voxel_feats_desc,
    const mluOpTensorDescriptor_t feats_desc,
    const mluOpTensorDescriptor_t voxel_feats_desc,
    const mluOpTensorDescriptor_t point2voxel_map_desc,
    const mluOpTensorDescriptor_t voxel_points_count_desc,
    const mluOpTensorDescriptor_t voxel_num_desc, size_t *workspace_size) {
  const char *interface_name =
      "[mluOpGetDynamicPointToVoxelBackwardWorkspaceSize]";
  PARAM_CHECK(interface_name, handle != NULL);
  if (handle->arch < MLUOP_MLU370) {
    LOG(ERROR) << interface_name
               << "Only mlu300 and above devices are supported. "
               << "Please check the device version!";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  PARAM_CHECK(interface_name, grad_voxel_feats_desc != NULL);
  PARAM_CHECK(interface_name, feats_desc != NULL);
  PARAM_CHECK(interface_name, voxel_feats_desc != NULL);
  PARAM_CHECK(interface_name, point2voxel_map_desc != NULL);
  PARAM_CHECK(interface_name, voxel_points_count_desc != NULL);
  PARAM_CHECK(interface_name, voxel_num_desc != NULL);
  PARAM_CHECK(interface_name, workspace_size != NULL);
  const int N = feats_desc->getDimIndex(0);
  const int C = feats_desc->getDimIndex(1);
  *workspace_size = N * C * sizeof(int);
  return MLUOP_STATUS_SUCCESS;
}
