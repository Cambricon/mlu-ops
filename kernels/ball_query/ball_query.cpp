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
#include "ball_query.h"

#include <string>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"

static inline bool isSupportType(const mluOpDataType_t check_type,
                                 const mluOpDataType_t support_type[],
                                 const int len) {
  for (int i = 0; i < len; ++i) {
    if (check_type == support_type[i]) {
      return true;
    }
  }
  return false;
}

void policyFuncBallQuery(const mluOpHandle_t &handle,
                         const mluOpTensorDescriptor_t &desc, cnrtDim3_t *k_dim,
                         cnrtFunctionType_t *k_type) {
  size_t cluster_num = mluop::runtime::getClusterLimitCapability(handle);
  VLOG(5) << "In current device, cluster_num:" << cluster_num;
  size_t core_in_cluster = handle->core_num_per_cluster;
  VLOG(5) << "In current device, core_in_cluster:" << core_in_cluster;

  size_t total_data_num = desc->getTotalElementNum();

  // On a core, a lot of new_xyz data element can be stored; but only one data
  // element can be processed at a time. So a cluster can only process four data
  // element.
  size_t needed_cluster_num =
      (total_data_num + core_in_cluster - 1) / core_in_cluster;
  *k_type = cnrtFuncTypeUnion1;
  k_dim->x = core_in_cluster;
  k_dim->y =
      needed_cluster_num > cluster_num ? cluster_num : needed_cluster_num;
  k_dim->z = 1;
}

mluOpStatus_t MLUOP_WIN_API mluOpBallQuery(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t new_xyz_desc,
    const void *new_xyz, const mluOpTensorDescriptor_t xyz_desc,
    const void *xyz, const float min_radius, const float max_radius,
    const int nsample, const mluOpTensorDescriptor_t idx_desc, void *idx) {
  VLOG(5) << "go into mluOpBallQuery.";
  mluOpDataType_t support_type[2] = {MLUOP_DTYPE_HALF, MLUOP_DTYPE_FLOAT};
  // check inputs params
  PARAM_CHECK("[mluOpBallQuery]", min_radius >= 0);
  PARAM_CHECK("[mluOpBallQuery]", max_radius >= 0);
  PARAM_CHECK("[mluOpBallQuery]", nsample >= 0);

  // handle and desc ptr check null
  PARAM_CHECK("[mluOpBallQuery]", handle != NULL);
  PARAM_CHECK("[mluOpBallQuery]", new_xyz_desc != NULL);
  PARAM_CHECK("[mluOpBallQuery]", xyz_desc != NULL);
  PARAM_CHECK("[mluOpBallQuery]", idx_desc != NULL);

  // check dims
  PARAM_CHECK("[mluOpBallQuery]", new_xyz_desc->getDim() == 3);
  PARAM_CHECK("[mluOpBallQuery]", xyz_desc->getDim() == 3);
  PARAM_CHECK("[mluOpBallQuery]", idx_desc->getDim() == 3);

  // check dim0
  PARAM_CHECK("[mluOpBallQuery]", new_xyz_desc->getDimIndex(0) == xyz_desc->getDimIndex(0));
  PARAM_CHECK("[mluOpBallQuery]", new_xyz_desc->getDimIndex(0) == idx_desc->getDimIndex(0));

  // check dim1
  PARAM_CHECK("[mluOpBallQuery]", new_xyz_desc->getDimIndex(1) == idx_desc->getDimIndex(1));

  // check dim2
  PARAM_CHECK("[mluOpBallQuery]", new_xyz_desc->getDimIndex(2) == 3);
  PARAM_CHECK("[mluOpBallQuery]", xyz_desc->getDimIndex(2) == 3);
  PARAM_CHECK("[mluOpBallQuery]", idx_desc->getDimIndex(2) == nsample);

  // check stride
  STRIDE_TENSOR_CHECK("[mluOpBallQuery]:", new_xyz_desc,
                      "new_xyz_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpBallQuery]:", xyz_desc,
                      "xyz_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpBallQuery]:", idx_desc,
                      "idx_desc must be contiguous");

  // check dtype
  if (!isSupportType(new_xyz_desc->getDtype(), support_type, 2)) {
    LOG(ERROR) << "[mluOpBallQuery]:Only half and float are supported in input "
                  "new_xyz tensor, but the data type of tensor is "
               << mluOpGetNameOfDataType(new_xyz_desc->getDtype()) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  PARAM_CHECK_EQ("[mluOpBallQuery]", new_xyz_desc->getDtype(), xyz_desc->getDtype());

  if (idx_desc->getDtype() != MLUOP_DTYPE_INT32) {
    LOG(ERROR) << "[mluOpBallQuery]:Only int32 is supportedin output idx, but "
                  "data type of tensor is "
               << mluOpGetNameOfDataType(idx_desc->getDtype()) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // check LargeTensor
  // expression: char *xyz_tmp = (char *)xyz + offset * sizeof(T), T is the data
  // type of xyz. this expression is used to calculate the xyz offset
  // address. for 370 series, offset * sizeof(T) must be less than 2^31,
  // otherwise, it will cause access to gdram out of bounds. according to the
  // above constraints, we can calculate the limit of the number of new tensor
  // elements which is 536895361.
  uint32_t max_input_num = 536895361;
  if (handle->arch > 372) {
    max_input_num = LARGE_TENSOR_NUM;
  }

  if ((mluOpGetTensorElementNum(new_xyz_desc) >= LARGE_TENSOR_NUM) ||
      (mluOpGetTensorElementNum(idx_desc) >= LARGE_TENSOR_NUM)) {
    LOG(ERROR) << "ball_query Overflow max tensor num."
               << " Currently, MLU-OPS supports tensor num smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (mluOpGetTensorElementNum(xyz_desc) >= max_input_num) {
    LOG(ERROR)
        << "ball_query's xyz_tensor element number is bigger max tensor num."
        << " Currently, xyz_tensor supports tensor num smaller than "
        << max_input_num << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // check 0 element
  // for new_xyz, zero elements are not supported
  if (mluOpGetTensorElementNum(new_xyz_desc) == 0) {
    VLOG(5) << "[mluOpBallQuery] new_xyz tensor is a zero element tensor. The "
               "shape of new_xyz tensor is ["
            << new_xyz_desc->getDimIndex(0) << ", " << new_xyz_desc->getDimIndex(1) << ", "
            << new_xyz_desc->getDimIndex(2) << "].";
    return MLUOP_STATUS_BAD_PARAM;
  }
  // the shape of xyz is [b, n, 3]. currently only n equal to 0 is supported
  if (xyz_desc->getDimIndex(1) == 0) {
    return MLUOP_STATUS_SUCCESS;
  }
  // the shape of idx is [b, m, nsample]. currently only nsample equal to 0 is
  // supported
  if (idx_desc->getDimIndex(2) == 0) {
    return MLUOP_STATUS_SUCCESS;
  }

  // check ptr
  PARAM_CHECK("[mluOpBallQuery]", new_xyz != NULL);
  PARAM_CHECK("[mluOpBallQuery]", xyz != NULL);
  PARAM_CHECK("[mluOpBallQuery]", idx != NULL);

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("ball_query", "BALL_QUERY");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA_REAL(true, "input1", new_xyz, new_xyz_desc);
    GEN_CASE_DATA_REAL(true, "input2", xyz, xyz_desc);
    GEN_CASE_DATA(false, "output", idx, idx_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "ball_query", "min_radius", min_radius);
    GEN_CASE_OP_PARAM_SINGLE(0, "ball_query", "max_radius", max_radius);
    GEN_CASE_OP_PARAM_SINGLE(0, "ball_query", "nsample", nsample);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0, 0, 0);
  }
  // choose the best task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFuncBallQuery(handle, new_xyz_desc, &k_dim, &k_type);

  // launch kernel
  uint32_t b = new_xyz_desc->getDimIndex(0);
  uint32_t m = new_xyz_desc->getDimIndex(1);
  uint32_t n = xyz_desc->getDimIndex(1);
  mluOpDataType_t d_type = new_xyz_desc->getDtype();
  VLOG(5) << "[mluOpBallQuery] launch kernel KernelBallQuery[" << k_dim.x
          << ", " << k_dim.y << ", " << k_dim.z << "]";
  CHECK_RETURN(
      "[mluOpBallQuery]",
      KernelBallQuery(k_dim, k_type, handle->queue, d_type, b, n, m, min_radius,
                      max_radius, nsample, new_xyz, xyz, (int32_t *)idx));

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
