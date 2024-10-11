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
#include "diff_iou_rotated_sort_vertices_forward.h"

#include <string>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"

static void policyFunc(const mluOpHandle_t handle, const int bn_num,
                       cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  *k_type = cnrtFuncTypeBlock;
  uint32_t cluster_num = mluop::runtime::getClusterLimitCapability(handle);
  uint32_t core_num_per_cluster =
      mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  uint32_t core_max = cluster_num * core_num_per_cluster;
  uint32_t core_used = core_max > bn_num ? bn_num : core_max;
  k_dim->x = core_used;
  k_dim->y = 1;
  k_dim->z = 1;
}

static mluOpStatus_t diffIouRotatedSortVerticesForwardParamCheck(
    const std::string &op_name, mluOpHandle_t handle,
    const mluOpTensorDescriptor_t vertices_desc, const void *vertices,
    const mluOpTensorDescriptor_t mask_desc, const void *mask,
    const mluOpTensorDescriptor_t num_valid_desc, const void *num_valid,
    const mluOpTensorDescriptor_t idx_desc, void *idx, bool *zero_element) {
  // check descriptor and data
  PARAM_CHECK(op_name, handle != NULL);
  // platform check
  if (handle->arch < MLUOP_MLU370) {
    LOG(ERROR) << op_name << "Only mlu300 and above devices are supported. "
               << "Please check the device version!";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  PARAM_CHECK(op_name, vertices_desc != NULL);
  PARAM_CHECK(op_name, mask_desc != NULL);
  PARAM_CHECK(op_name, num_valid_desc != NULL);
  PARAM_CHECK(op_name, idx_desc != NULL);

  // check shape
  PARAM_CHECK(op_name, vertices_desc->dim == 4);
  PARAM_CHECK(op_name, mask_desc->dim == 3);
  PARAM_CHECK(op_name, num_valid_desc->dim == 2);
  PARAM_CHECK(op_name, idx_desc->dim == 3);

  // check stride
  STRIDE_TENSOR_CHECK(op_name + ":", vertices_desc,
                      "vertices_desc must be contiguous");
  STRIDE_TENSOR_CHECK(op_name + ":", mask_desc, "mask_desc must be contiguous");
  STRIDE_TENSOR_CHECK(op_name + ":", num_valid_desc,
                      "num_valid_desc must be contiguous");
  STRIDE_TENSOR_CHECK(op_name + ":", idx_desc, "idx_desc must be contiguous");

  // check data type
  // check tensor datatype, support float32
  PARAM_CHECK_V2(op_name, (vertices_desc->dtype == MLUOP_DTYPE_FLOAT),
                 "Only float are supported in vertices tensor, but the "
                 "data type of tensor is "
                     << mluOpGetNameOfDataType(vertices_desc->dtype) << ".");

  PARAM_CHECK_V2(op_name, (mask_desc->dtype == MLUOP_DTYPE_BOOL),
                 "Only bool are supported in mask tensor, but the data "
                 "type of tensor is "
                     << mluOpGetNameOfDataType(mask_desc->dtype) << ".");

  PARAM_CHECK_V2(op_name, (num_valid_desc->dtype == MLUOP_DTYPE_INT32),
                 "Only int32 are supported in num_valid tensor, but the data "
                 "type of tensor is "
                     << mluOpGetNameOfDataType(num_valid_desc->dtype) << ".");

  PARAM_CHECK_V2(op_name, (idx_desc->dtype == MLUOP_DTYPE_INT32),
                 "Only int32 are supported in idx tensor, but the data "
                 "type of tensor is "
                     << mluOpGetNameOfDataType(idx_desc->dtype) << ".");

  // check dim
  // int dim_b = vertices_desc->dims[0];
  // int dim_n = vertices_desc->dims[1];
  // int dim_m = vertices_desc->dims[2];
  PARAM_CHECK(op_name, (vertices_desc->dims[0] == mask_desc->dims[0]));
  PARAM_CHECK(op_name, (vertices_desc->dims[0] == num_valid_desc->dims[0]));
  PARAM_CHECK(op_name, (vertices_desc->dims[0] == idx_desc->dims[0]));
  PARAM_CHECK(op_name, (vertices_desc->dims[1] == mask_desc->dims[1]));
  PARAM_CHECK(op_name, (vertices_desc->dims[1] == num_valid_desc->dims[1]));
  PARAM_CHECK(op_name, (vertices_desc->dims[1] == idx_desc->dims[1]));
  PARAM_CHECK(op_name, (vertices_desc->dims[2] == mask_desc->dims[2]));
  PARAM_CHECK_V2(
      op_name, (vertices_desc->dims[2] == 24),
      "vertices and mask tensors dims[2] should be 24, but the input value is "
          << vertices_desc->dims[2] << ".");
  PARAM_CHECK_V2(op_name, (vertices_desc->dims[3] == 2),
                 "vertices tensor dims[3] should be 2, but the input value is "
                     << vertices_desc->dims[3] << ".");
  PARAM_CHECK_V2(op_name, (idx_desc->dims[2] == 9),
                 "idx tensor dims[2] should be 9, but the input value is "
                     << idx_desc->dims[2] << ".");

  const size_t vertices_element_num = mluOpGetTensorElementNum(vertices_desc);
  const size_t mask_element_num = mluOpGetTensorElementNum(mask_desc);
  const size_t num_valid_element_num = mluOpGetTensorElementNum(num_valid_desc);
  const size_t idx_element_num = mluOpGetTensorElementNum(idx_desc);

  // check large tensor
  TENSOR_NUM_CHECK(op_name, vertices_element_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(op_name, mask_element_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(op_name, num_valid_element_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(op_name, idx_element_num, LARGE_TENSOR_NUM, "");

  // check element num zero
  if (vertices_element_num == 0) {
    if (vertices_desc->dims[1] == 0) {
      *zero_element = true;
      return MLUOP_STATUS_SUCCESS;
    } else {
      *zero_element = false;
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  // input and output ptr check null
  PARAM_CHECK(op_name, vertices != NULL);
  PARAM_CHECK(op_name, mask != NULL);
  PARAM_CHECK(op_name, num_valid != NULL);
  PARAM_CHECK(op_name, idx != NULL);

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpDiffIouRotatedSortVerticesForward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t vertices_desc,
    const void *vertices, const mluOpTensorDescriptor_t mask_desc,
    const void *mask, const mluOpTensorDescriptor_t num_valid_desc,
    const void *num_valid, const mluOpTensorDescriptor_t idx_desc, void *idx) {
  // check params
  bool zero_element = false;
  mluOpStatus_t param_check = diffIouRotatedSortVerticesForwardParamCheck(
      "[mluOpDiffIouRotatedSortVerticesForward]", handle, vertices_desc,
      vertices, mask_desc, mask, num_valid_desc, num_valid, idx_desc, idx,
      &zero_element);
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }

  // check zero element
  if (zero_element == true) {
    VLOG(5)
        << "[mluOpDiffIouRotatedSortVerticesForward] Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("diff_iou_rotated_sort_vertices_forward",
                   "DIFF_IOU_ROTATED_SORT_VERTICES_FORWARD");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA_REAL(true, "vertices", vertices, vertices_desc);
    GEN_CASE_DATA_REAL_V2(true, "mask", mask, mask_desc, 1, 0);
    GEN_CASE_DATA_REAL_V2(true, "num_valid", num_valid, num_valid_desc, 8, 0);
    GEN_CASE_DATA(false, "idx", idx, idx_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(false, false, true, 0, 0, 0);
  }

  const int dim_b = vertices_desc->dims[0];
  const int dim_n = vertices_desc->dims[1];
  const int dim_m = vertices_desc->dims[2];
  const int bn_num = dim_b * dim_n;
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, bn_num, &k_dim, &k_type);
  VLOG(5) << "Launch Kernel mluOpDiffIouRotatedSortVerticesForward<<<Union"
          << k_type / CORE_DIM << ", " << k_dim.x << ", " << k_dim.y << ", "
          << k_dim.z << ">>>";
  CHECK_RETURN("[mluOpDiffIouRotatedSortVerticesForward]",
               KernelDiffIouRotatedSortVerticesForward(
                   k_dim, k_type, handle->queue, vertices, mask, num_valid, idx,
                   dim_b, dim_n, dim_m));
  VLOG(5) << "Kernel KernelDiffIouRotatedSortVerticesForward.";

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
