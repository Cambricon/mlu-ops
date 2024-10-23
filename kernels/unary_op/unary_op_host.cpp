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
#include <string>
#include <cmath>
#include "mlu_op.h"
#include "kernels/kernel.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "core/type.h"
#include "core/context.h"
#include "kernels/tensor_stride_process/tensor_stride_process_host.h"
#include "unary_op_host.h"

#define UNIT_SIZE 512
#define PWR 0.5
#define OPTIMAL_BOUNDARY 4096

void unaryOpPolicyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type,
                       mluOpTensorDescriptor_t desc) {
  uint64_t union_number = mluop::runtime::getClusterLimitCapability(handle);
  uint64_t core_in_cluster = handle->core_num_per_cluster;
  uint64_t core_number = union_number * core_in_cluster;
  uint64_t element_num = mluOpGetTensorElementNum(desc);
  uint64_t tensor_size = element_num * mluop::getSizeOfDataType(desc->dtype);
  tensor_size = CEIL_ALIGN(tensor_size, NFU_ALIGN_SIZE);
  uint64_t need_core =
      CEIL_ALIGN(tensor_size / NFU_ALIGN_SIZE, core_in_cluster);
  *k_type = cnrtFuncTypeUnion1;  // default func type
  k_dim->x = core_in_cluster;
  if (need_core < core_number) {
    k_dim->y = need_core / core_in_cluster;
  } else {
    k_dim->y = union_number;
  }
  k_dim->z = 1;
}

// the BLOCK policy
void unaryOpPolicyFuncBlock(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                            cnrtFunctionType_t *k_type,
                            mluOpTensorDescriptor_t desc) {
  uint64_t data_size = desc->total_tensor_size;
  uint32_t core_dim = handle->core_num_per_cluster;
  uint32_t cluster_num = mluop::runtime::getClusterLimitCapability(handle);
  uint32_t core_num = core_dim * cluster_num;
  uint32_t core_used =
      CEIL_ALIGN(data_size, OPTIMAL_BOUNDARY) / OPTIMAL_BOUNDARY;
  core_used = core_used > core_num ? core_num : core_used;
  *k_type = cnrtFuncTypeBlock;
  k_dim->x = 1;
  k_dim->y = core_used;
  k_dim->z = 1;
}

// Divide tasks in host, use with UNARY_OP_KERNEL_3PIPELINE_V2
void unaryOpPolicyFuncBlock_v2(mluOpHandle_t handle,
                               const mluOpTensorDescriptor_t desc,
                               size_t single_core_min_load_size,
                               cnrtDim3_t &k_dim, cnrtFunctionType_t &k_type,
                               size_t &normal_core_elem_num,
                               size_t &tail_core_elem_num) {
  k_type = cnrtFuncTypeBlock;
  if (MLUOP_MLU590 == handle->arch) {
    const size_t llc_pending_size = 512;
    single_core_min_load_size =
        std::max(llc_pending_size, single_core_min_load_size);
  }
  const size_t dtype_size = mluop::getSizeOfDataType(desc->dtype);
  const size_t aligned_num = DIV_UP(single_core_min_load_size, dtype_size);
  const size_t element_num = mluOpGetTensorElementNum(desc);
  const size_t core_number =
      mluop::runtime::getMaxParallelJobNum(handle, k_type);
  const size_t elem_num_per_core = DIV_UP(element_num, core_number);
  normal_core_elem_num = CEIL_ALIGN(elem_num_per_core, aligned_num);
  const size_t rem_num = element_num % normal_core_elem_num;
  size_t task_num = element_num / normal_core_elem_num;
  if (0 < rem_num && task_num < core_number) {
    task_num = task_num + 1;
    tail_core_elem_num = rem_num;
  } else { /*rem_num == 0 or task_num == core_number, tail is normal or more*/
    tail_core_elem_num = normal_core_elem_num + rem_num;
  }

  k_dim.x = task_num;
  k_dim.y = 1;
  k_dim.z = 1;
}

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

mluOpStatus_t unaryOpParamCheck(std::string op_name, const mluOpHandle_t handle,
                                const mluOpTensorDescriptor_t x_desc,
                                const void *x,
                                const mluOpTensorDescriptor_t y_desc,
                                const void *y,
                                const mluOpDataType_t support_type[],
                                const int len, bool &zero_element) {
  // check descriptor
  PARAM_CHECK(op_name, handle != NULL);
  PARAM_CHECK(op_name, x_desc != NULL);
  PARAM_CHECK(op_name, y_desc != NULL);

  // check dim and dtype
  PARAM_CHECK_EQ(op_name, x_desc->dtype, y_desc->dtype);
  PARAM_CHECK_EQ(op_name, x_desc->dim, y_desc->dim);
  // check data type
  if (!isSupportType(x_desc->dtype, support_type, len)) {
    LOG(ERROR) << op_name << ":x_desc's data type is not supported.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  PARAM_CHECK_GT(op_name, x_desc->dim, 0);
  PARAM_CHECK_GT(op_name, y_desc->dim, 0);
  for (int i = 0; i < x_desc->dim; i++) {
    if (x_desc->dims[i] != y_desc->dims[i]) {
      LOG(ERROR) << op_name << ":The shape of x should be equal to y"
                 << ". But now x_desc's shape[" << i << "] is "
                 << x_desc->dims[i] << ", y_desc's shape[" << i << "] is "
                 << y_desc->dims[i] << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  // check 0 element
  if (mluOpGetTensorElementNum(x_desc) == 0) {
    VLOG(5) << op_name << "skip zero element tensor.";
    zero_element = true;
    return MLUOP_STATUS_SUCCESS;
  }
  if (handle->arch < MLUOP_MLU590) {
    uint64_t num_input = mluOpGetTensorElementNum(x_desc);
    TENSOR_NUM_CHECK(op_name, num_input, LARGE_TENSOR_NUM,
                     "input tensor num is too large. ");
  }
  if (needStrideProcess(x_desc, y_desc)) {
    PARAM_CHECK(op_name, x_desc->dim <= MLUOP_DIM_MAX);
    if (handle->arch < MLUOP_MLU590) {
      // num_with_stride affects offset (related with mul op, which cannot
      // exceed 32-bit on MLU300)
      uint64_t num_input_with_stride = shapeStrideCount(x_desc);
      uint64_t num_output_with_stride = shapeStrideCount(y_desc);
      TENSOR_NUM_CHECK(op_name, num_input_with_stride, LARGE_TENSOR_NUM,
                       "input tensor num with stride is too large. ");
      TENSOR_NUM_CHECK(op_name, num_output_with_stride, LARGE_TENSOR_NUM,
                       "output tensor num with stride is too large. ");
    }
  }
  PARAM_CHECK(op_name, x != NULL);
  PARAM_CHECK(op_name, y != NULL);
  return MLUOP_STATUS_SUCCESS;
}

bool needStrideProcess(const mluOpTensorDescriptor_t x_desc,
                       const mluOpTensorDescriptor_t y_desc) {
  if (mluop::strideCaseWithNotConsistentDense(2, x_desc, y_desc)) {
    return true;
  }
  return false;
}
