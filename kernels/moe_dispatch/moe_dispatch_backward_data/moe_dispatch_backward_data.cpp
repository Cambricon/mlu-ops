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
#include "moe_dispatch_backward_data.h"

#include <string>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/utils/cnnl_helper.h"

// policy function
static void PolicyFunc(const mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type) {
  // union1 policy func
  *k_type = cnrtFuncTypeUnion1;
  // dimx equals to num of MLU Cores in each cluster
  k_dim->x = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  // dimy equals to num of current available clusters
  k_dim->y = mluop::runtime::getClusterLimitCapability(handle);
  k_dim->z = 1;
}

mluOpStatus_t MLUOP_WIN_API mluOpMoeDispatchBackwardData(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t gates_desc,
    const void *gates, const mluOpTensorDescriptor_t indices_desc,
    const void *indices, const mluOpTensorDescriptor_t locations_desc,
    const void *locations, const mluOpTensorDescriptor_t dispatch_desc,
    const void *dispatch, const int samples, const int capacity,
    const int hidden, const int num_experts,
    const mluOpTensorDescriptor_t grad_input_desc, void *grad_input) {
  // gates: (samples)
  // indices: (samples)
  // locations: (samples)
  // dispatch: (num_experts * capacity, hidden)
  // grad_input: (samples, hidden)

  const std::string API = "[mluOpMoeDispatchBackwardData]";
  // check desc
  PARAM_CHECK(API, handle != NULL);
  // check arch
  if (handle->arch < MLUOP_MLU370) {
    LOG(ERROR) << API
               << "The operator does not match the current architecture.";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }
  PARAM_CHECK(API, gates_desc != NULL);
  PARAM_CHECK(API, indices_desc != NULL);
  PARAM_CHECK(API, locations_desc != NULL);
  PARAM_CHECK(API, dispatch_desc != NULL);
  PARAM_CHECK(API, grad_input_desc != NULL);

  // check dim
  PARAM_CHECK_EQ(API, gates_desc->dim, 1);
  PARAM_CHECK_EQ(API, indices_desc->dim, 1);
  PARAM_CHECK_EQ(API, locations_desc->dim, 1);
  PARAM_CHECK_EQ(API, dispatch_desc->dim, 2);
  PARAM_CHECK_EQ(API, grad_input_desc->dim, 2);

  // check shape
  PARAM_CHECK_EQ(API, gates_desc->dims[0], samples);
  PARAM_CHECK_EQ(API, indices_desc->dims[0], samples);
  PARAM_CHECK_EQ(API, locations_desc->dims[0], samples);
  PARAM_CHECK_EQ(API, dispatch_desc->dims[0], (num_experts * capacity));
  PARAM_CHECK_EQ(API, dispatch_desc->dims[1], hidden);
  PARAM_CHECK_EQ(API, grad_input_desc->dims[0], samples);
  PARAM_CHECK_EQ(API, grad_input_desc->dims[1], hidden);

  // check dtype
  PARAM_CHECK_V2(API, (gates_desc->dtype == MLUOP_DTYPE_FLOAT),
                 "Only float are supported in input tensor, but the "
                 "data type of tensor is "
                     << mluOpGetNameOfDataType(gates_desc->dtype) << ".");
  PARAM_CHECK_V2(API, (indices_desc->dtype == MLUOP_DTYPE_INT32),
                 "Only int32 are supported in indices tensor, but the data "
                 "type of tensor is "
                     << mluOpGetNameOfDataType(indices_desc->dtype) << ".");
  PARAM_CHECK_V2(API, (locations_desc->dtype == MLUOP_DTYPE_INT32),
                 "Only int32 are supported in locations tensor, but the data "
                 "type of tensor is "
                     << mluOpGetNameOfDataType(locations_desc->dtype) << ".");
  PARAM_CHECK(API, dispatch_desc->dtype == gates_desc->dtype);
  PARAM_CHECK(API, grad_input_desc->dtype == gates_desc->dtype);

  // check tensor dim
  PARAM_CHECK(API, samples >= 0);
  PARAM_CHECK(API, capacity >= 0);
  PARAM_CHECK(API, hidden >= 0);
  PARAM_CHECK(API, num_experts >= 0);

  // check stride
  STRIDE_TENSOR_CHECK(API + ":", gates_desc, "gates_desc must be contiguous");
  STRIDE_TENSOR_CHECK(API + ":", indices_desc,
                      "indices_desc must be contiguous");
  STRIDE_TENSOR_CHECK(API + ":", locations_desc,
                      "locations_desc must be contiguous");
  STRIDE_TENSOR_CHECK(API + ":", dispatch_desc,
                      "dispatch_desc must be contiguous");
  STRIDE_TENSOR_CHECK(API + ":", grad_input_desc,
                      "grad_input_desc must be contiguous");

  const uint64_t gates_element_num = mluOpGetTensorElementNum(gates_desc);
  const uint64_t indices_element_num = mluOpGetTensorElementNum(indices_desc);
  const uint64_t locations_element_num =
      mluOpGetTensorElementNum(locations_desc);
  const uint64_t dispatch_element_num = mluOpGetTensorElementNum(dispatch_desc);
  const uint64_t grad_input_element_num =
      mluOpGetTensorElementNum(grad_input_desc);

  // check large tensor
  TENSOR_NUM_CHECK(API, gates_element_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, indices_element_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, locations_element_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, dispatch_element_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, grad_input_element_num, LARGE_TENSOR_NUM, "");

  // check zero element
  if (samples == 0 || hidden == 0) {
    VLOG(5) << API << "Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  } else {
    // Initialize output space
    PARAM_CHECK(API, grad_input != NULL);
    const size_t grad_input_initial_value = 0x00;
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grad_input_desc,
                                                 cnnl_output_desc);
    CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST,
                          &grad_input_initial_value, cnnl_output_desc,
                          grad_input));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
    VLOG(5) << API << "Initialize output tensor done.";
  }

  // check zero element
  if (capacity == 0 || num_experts == 0) {
    VLOG(5) << API << "Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  // check ptr
  PARAM_CHECK(API, gates != NULL);
  PARAM_CHECK(API, indices != NULL);
  PARAM_CHECK(API, locations != NULL);
  PARAM_CHECK(API, dispatch != NULL);
  PARAM_CHECK(API, grad_input != NULL);

  VLOG(5) << API << "input data shape: "
          << "samples = " << samples << ", "
          << "capacity = " << capacity << ", "
          << "hidden = " << hidden << ", "
          << "num_experts = " << num_experts;

  // generate prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("moe_dispatch_backward_data", "MOE_DISPATCH_BACKWARD_DATA");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "gates", gates, gates_desc, 100, -100);
    GEN_CASE_DATA_REAL_V2(true, "indices", indices, indices_desc, num_experts,
                          0);
    GEN_CASE_DATA_REAL_V2(true, "locations", locations, locations_desc,
                          capacity, 0);
    GEN_CASE_DATA(true, "dispatch", dispatch, dispatch_desc, 100, -100);
    GEN_CASE_DATA(false, "grad_input", grad_input, grad_input_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "moe_dispatch_backward_data", "samples",
                             samples);
    GEN_CASE_OP_PARAM_SINGLE(1, "moe_dispatch_backward_data", "capacity",
                             capacity);
    GEN_CASE_OP_PARAM_SINGLE(2, "moe_dispatch_backward_data", "hidden", hidden);
    GEN_CASE_OP_PARAM_SINGLE(3, "moe_dispatch_backward_data", "num_experts",
                             num_experts);
    GEN_CASE_TEST_PARAM_NEW(false, false, true, 0, 0, 0.0);
  }
  // generate prototxt end!

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  PolicyFunc(handle, &k_dim, &k_type);

  int core_num_per_cluster =
      mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  VLOG(5) << API << "Launch Kernel <<<Union" << k_type / core_num_per_cluster
          << ", " << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>"
          << "core num per cluster: " << core_num_per_cluster;

  mluOpDataType_t data_type = grad_input_desc->dtype;
  uint32_t taskNum = k_dim.x * k_dim.y * k_dim.z;

  if (samples <= taskNum) {
    VLOG(5) << API << "Launch Kernel KernelMoeDispatchBwdData1().";
    CHECK_RETURN(
        "[mluOpMoeDispatchBackwardData1]",
        KernelMoeDispatchBwdData1(k_dim, k_type, handle->queue, data_type,
                                  gates, indices, locations, dispatch, samples,
                                  capacity, hidden, num_experts, grad_input));
    VLOG(5) << API << "Finish Kernel KernelMoeDispatchBwdData1.";
  } else {
    VLOG(5) << API << "Launch Kernel KernelMoeDispatchBwdData2().";
    CHECK_RETURN(
        "[mluOpMoeDispatchBackwardData2]",
        KernelMoeDispatchBwdData2(k_dim, k_type, handle->queue, data_type,
                                  gates, indices, locations, dispatch, samples,
                                  capacity, hidden, num_experts, grad_input));
    VLOG(5) << API << "Finish Kernel KernelMoeDispatchBwdData2.";
  }

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
