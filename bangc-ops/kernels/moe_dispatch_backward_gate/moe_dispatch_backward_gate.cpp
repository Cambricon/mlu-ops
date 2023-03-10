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
#include "kernels/kernel.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"

static void policyFunc(const mluOpHandle_t handle, const int samples,
                       cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  int max_core_num = mluop::runtime::getCoreNumOfJobLimitCapability(handle);
  if (samples > max_core_num) {
    // union1 policy func
    *k_type = CNRT_FUNC_TYPE_UNION1;
    // dimx equals to num of mlu cores in each cluster
    k_dim->x = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
    // dimy equals to num of current available clusters
    k_dim->y = mluop::runtime::getClusterLimitCapability(handle);
    k_dim->z = 1;
  } else {
    k_dim->x = max_core_num;
    k_dim->y = 1;
    k_dim->z = 1;
    *k_type = mluop::runtime::getJobLimitCapabilityCnrtFuncType(handle);
  }
}

mluOpStatus_t MLUOP_WIN_API mluOpGetMoeDispatchBackwardGateWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    size_t *workspace_size) {
  PARAM_CHECK("[mluOpMoeDispatchBackwardGate]", handle != NULL);
  // platform check
  if (handle->arch < MLUOP_MLU370) {
    LOG(ERROR) << "[mluOpMoeDispatchBackwardGate] Only mlu300 and above "
                  "devices are supported. "
               << "Please check the device version!";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }
  PARAM_CHECK("[mluOpMoeDispatchBackwardGate]", input_desc != NULL);
  PARAM_CHECK("[mluOpMoeDispatchBackwardGate]", workspace_size != NULL);

  int samples = input_desc->dims[0];
  *workspace_size = 0;
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, samples, &k_dim, &k_type);
  int taskNum = k_dim.x * k_dim.y * k_dim.z;
  if ((samples > 0) && (samples < taskNum)) {
    *workspace_size = taskNum * mluop::getSizeOfDataType(input_desc->dtype);
  }

  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t moeDispatchBackwardGateParamCheck(
    const std::string &op_name, const mluOpHandle_t handle,
    const mluOpTensorDescriptor_t indices_desc, const void *indices,
    const mluOpTensorDescriptor_t locations_desc, const void *locations,
    const mluOpTensorDescriptor_t input_desc, const void *input,
    const mluOpTensorDescriptor_t dispatch_desc, const void *dispatch,
    const int samples, const int capacity, const int hidden,
    const int num_experts, void *workspace, const size_t workspace_size,
    const mluOpTensorDescriptor_t grad_gates_desc, const void *grad_gates,
    bool *zero_element) {
  // check descriptor and data
  PARAM_CHECK(op_name, handle != NULL);
  // platform check
  if (handle->arch < MLUOP_MLU370) {
    LOG(ERROR) << op_name << "Only mlu300 and above devices are supported. "
               << "Please check the device version!";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  PARAM_CHECK(op_name, indices_desc != NULL);
  PARAM_CHECK(op_name, locations_desc != NULL);
  PARAM_CHECK(op_name, input_desc != NULL);
  PARAM_CHECK(op_name, dispatch_desc != NULL);
  PARAM_CHECK(op_name, grad_gates_desc != NULL);

  // check shape
  PARAM_CHECK(op_name, indices_desc->dim == 1);
  PARAM_CHECK(op_name, locations_desc->dim == 1);
  PARAM_CHECK(op_name, input_desc->dim == 2);
  PARAM_CHECK(op_name, dispatch_desc->dim == 2);
  PARAM_CHECK(op_name, grad_gates_desc->dim == 1);

  // check data type
  PARAM_CHECK_V2(op_name, (indices_desc->dtype == MLUOP_DTYPE_INT32),
                 "Only int32 are supported in indices tensor, but the data "
                 "type of tensor is "
                     << mluop::getNameOfDataType(indices_desc->dtype) << ".");
  PARAM_CHECK_V2(op_name, (locations_desc->dtype == MLUOP_DTYPE_INT32),
                 "Only int32 are supported in locations tensor, but the data "
                 "type of tensor is "
                     << mluop::getNameOfDataType(locations_desc->dtype) << ".");

  // check tensor datatype, support float32
  PARAM_CHECK_V2(op_name, (input_desc->dtype == MLUOP_DTYPE_FLOAT),
                 "Only float are supported in input tensor, but the "
                 "data type of tensor is "
                     << mluop::getNameOfDataType(input_desc->dtype) << ".");
  PARAM_CHECK(op_name, input_desc->dtype == dispatch_desc->dtype);
  PARAM_CHECK(op_name, input_desc->dtype == grad_gates_desc->dtype);

  // check dim
  PARAM_CHECK(op_name, samples >= 0);
  PARAM_CHECK(op_name, capacity >= 0);
  PARAM_CHECK(op_name, hidden >= 0);
  PARAM_CHECK(op_name, num_experts >= 0);
  PARAM_CHECK(op_name, (samples == indices_desc->dims[0]));
  PARAM_CHECK(op_name, (samples == locations_desc->dims[0]));
  PARAM_CHECK(op_name, (samples == input_desc->dims[0]));
  PARAM_CHECK(op_name, (samples == grad_gates_desc->dims[0]));
  PARAM_CHECK(op_name, ((num_experts * capacity) == dispatch_desc->dims[0]));
  PARAM_CHECK(op_name, (hidden == input_desc->dims[1]));
  PARAM_CHECK(op_name, (hidden == dispatch_desc->dims[1]));

  const size_t indices_element_num = mluOpGetTensorElementNum(indices_desc);
  const size_t locations_element_num = mluOpGetTensorElementNum(locations_desc);
  const size_t input_element_num = mluOpGetTensorElementNum(input_desc);
  const size_t dispatch_element_num = mluOpGetTensorElementNum(dispatch_desc);
  const size_t grad_gates_element_num =
      mluOpGetTensorElementNum(grad_gates_desc);

  // check large tensor
  TENSOR_NUM_CHECK(op_name, indices_element_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(op_name, locations_element_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(op_name, input_element_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(op_name, dispatch_element_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(op_name, grad_gates_element_num, LARGE_TENSOR_NUM, "");

  // check element num zero
  if (indices_element_num == 0 || locations_element_num == 0 ||
      input_element_num == 0 || dispatch_element_num == 0 ||
      grad_gates_element_num == 0) {
    *zero_element = true;
    return MLUOP_STATUS_SUCCESS;
  }

  // check workspace ptr
  if (workspace_size > 0) {
    PARAM_CHECK(op_name, workspace != NULL);
  }

  // input and output ptr check null
  PARAM_CHECK(op_name, indices != NULL);
  PARAM_CHECK(op_name, locations != NULL);
  PARAM_CHECK(op_name, input != NULL);
  PARAM_CHECK(op_name, dispatch != NULL);
  PARAM_CHECK(op_name, grad_gates != NULL);

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpMoeDispatchBackwardGate(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t indices_desc,
    const void *indices, const mluOpTensorDescriptor_t locations_desc,
    const void *locations, const mluOpTensorDescriptor_t input_desc,
    const void *input, const mluOpTensorDescriptor_t dispatch_desc,
    const void *dispatch, const int samples, const int capacity,
    const int hidden, const int num_experts, void *workspace,
    const size_t workspace_size, const mluOpTensorDescriptor_t grad_gates_desc,
    void *grad_gates) {
  // check params
  bool zero_element = false;
  mluOpStatus_t param_check = moeDispatchBackwardGateParamCheck(
      "[mluOpMoeDispatchBackwardGate]", handle, indices_desc, indices,
      locations_desc, locations, input_desc, input, dispatch_desc, dispatch,
      samples, capacity, hidden, num_experts, workspace, workspace_size,
      grad_gates_desc, grad_gates, &zero_element);
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }

  // check zero element
  if (zero_element == true) {
    VLOG(5) << "[mluOpMoeDispatchBackwardGate] Skip zero element tensor.";
    if (samples > 0) {
      VLOG(5) << "mluopFill start.";
      const size_t fill_value = 0x0;
      MLUOP_CHECK(mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST, &fill_value,
                               grad_gates_desc, grad_gates));
      VLOG(5) << "mluopFill end.";
    }
    return MLUOP_STATUS_SUCCESS;
  }

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("moe_dispatch_backward_gate");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA_REAL(true, "indices", indices, indices_desc);
    GEN_CASE_DATA_REAL(true, "locations", locations, locations_desc);
    GEN_CASE_DATA(true, "input", input, input_desc, 0, 0);
    GEN_CASE_DATA(true, "dispatch", dispatch, dispatch_desc, 0, 0);
    GEN_CASE_DATA(false, "grad_gates", grad_gates, grad_gates_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "moe_dispatch_backward_gate", "samples",
                             samples);
    GEN_CASE_OP_PARAM_SINGLE(1, "moe_dispatch_backward_gate", "capacity",
                             capacity);
    GEN_CASE_OP_PARAM_SINGLE(2, "moe_dispatch_backward_gate", "hidden", hidden);
    GEN_CASE_OP_PARAM_SINGLE(3, "moe_dispatch_backward_gate", "num_experts",
                             num_experts);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, samples, &k_dim, &k_type);
  VLOG(5) << "Launch Kernel mluOpMoeDispatchBackwardGate<<<Union"
          << k_type / CORE_DIM << ", " << k_dim.x << ", " << k_dim.y << ", "
          << k_dim.z << ">>>";
  mluOpDataType_t data_type = input_desc->dtype;
  uint32_t taskNum = k_dim.x * k_dim.y * k_dim.z;
  if (samples <= taskNum) {
    if (data_type == MLUOP_DTYPE_FLOAT) {
      VLOG(5) << "[mluOpMoeDispatchBackwardGate] launch "
                 "mluOpUnionKernelMoeDispatchBwdGate1Float";
      KERNEL_CHECK((mluOpUnionKernelMoeDispatchBwdGate1Float(
          k_dim, k_type, handle->queue, indices, locations, input, dispatch,
          samples, capacity, hidden, num_experts, workspace, grad_gates)));
    } else {
      VLOG(5) << "[mluOpMoeDispatchBackwardGate] launch "
                 "mluOpUnionKernelMoeDispatchBwdGate1Half";
      KERNEL_CHECK((mluOpUnionKernelMoeDispatchBwdGate1Half(
          k_dim, k_type, handle->queue, indices, locations, input, dispatch,
          samples, capacity, hidden, num_experts, workspace, grad_gates)));
    }
  } else {
    if (data_type == MLUOP_DTYPE_FLOAT) {
      VLOG(5) << "[mluOpMoeDispatchBackwardGate] launch "
                 "mluOpUnionKernelMoeDispatchBwdGate2Float";
      KERNEL_CHECK((mluOpUnionKernelMoeDispatchBwdGate2Float(
          k_dim, k_type, handle->queue, indices, locations, input, dispatch,
          samples, capacity, hidden, num_experts, grad_gates)));
    } else {
      VLOG(5) << "[mluOpMoeDispatchBackwardGate] launch "
                 "mluOpUnionKernelMoeDispatchBwdGate2Half";
      KERNEL_CHECK((mluOpUnionKernelMoeDispatchBwdGate2Half(
          k_dim, k_type, handle->queue, indices, locations, input, dispatch,
          samples, capacity, hidden, num_experts, grad_gates)));
    }
  }

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
