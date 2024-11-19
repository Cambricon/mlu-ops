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
#include "active_rotated_filter.h"

#include <cmath>
#include <string>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"

mluOpStatus_t MLUOP_WIN_API mluOpGetActiveRotatedFilterForwardWorkspaceSize(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    size_t *workspace_size) {
  // handle and desc ptr check null
  const std::string api_name = "[mluOpActiveRotatedFilterForwardWorkspace]";
  PARAM_CHECK(api_name, handle != NULL);
  PARAM_CHECK(api_name, input_desc != NULL);
  PARAM_CHECK(api_name, workspace_size != NULL);

  *workspace_size = input_desc->getTotalTensorSize();
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t activeRotatedFilterForwardParamCheck(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, const mluOpTensorDescriptor_t indices_desc,
    const void *indices, void *workspace, const size_t workspace_size,
    const mluOpTensorDescriptor_t output_desc, void *output,
    const std::string api_name) {
  // handle and desc ptr check null
  PARAM_CHECK(api_name, handle != NULL);
  PARAM_CHECK(api_name, input_desc != NULL);
  PARAM_CHECK(api_name, indices_desc != NULL);
  PARAM_CHECK(api_name, output_desc != NULL);

  // check tensor dim
  PARAM_CHECK(api_name, input_desc->getDim() == 5);
  PARAM_CHECK(api_name, indices_desc->getDim() == 4);
  PARAM_CHECK(api_name, output_desc->getDim() == 4);

  // check dim
  PARAM_CHECK(api_name, input_desc->getDimIndex(2) == indices_desc->getDimIndex(0));
  PARAM_CHECK(api_name, input_desc->getDimIndex(3) == input_desc->getDimIndex(4));
  PARAM_CHECK(api_name, input_desc->getDimIndex(3) == indices_desc->getDimIndex(1));
  PARAM_CHECK(api_name, input_desc->getDimIndex(3) == output_desc->getDimIndex(2));
  PARAM_CHECK(api_name, input_desc->getDimIndex(4) == indices_desc->getDimIndex(2));
  PARAM_CHECK(api_name, input_desc->getDimIndex(4) == output_desc->getDimIndex(3));
  PARAM_CHECK(api_name,
              (input_desc->getDimIndex(2) > 0 && input_desc->getDimIndex(2) <= 128));
  PARAM_CHECK_V2(api_name,
                 int(log(float(input_desc->getDimIndex(2))) / log(2.0f)) ==
                     log(float(input_desc->getDimIndex(2))) / log(2.0f),
                 "input_desc->getDimIndex(2) should be the power of 2.");
  PARAM_CHECK(api_name, (input_desc->getDimIndex(3) == 3 || input_desc->getDimIndex(3) == 1));
  PARAM_CHECK(api_name,
              (indices_desc->getDimIndex(3) == 2 || indices_desc->getDimIndex(3) == 4 ||
               indices_desc->getDimIndex(3) == 8));
  PARAM_CHECK(api_name, (output_desc->getDimIndex(0) ==
                         input_desc->getDimIndex(0) * indices_desc->getDimIndex(3)));
  PARAM_CHECK(api_name, (output_desc->getDimIndex(1) ==
                         input_desc->getDimIndex(1) * input_desc->getDimIndex(2)));

  // check stride
  STRIDE_TENSOR_CHECK(api_name + ":", input_desc,
                      "input_desc must be contiguous");
  STRIDE_TENSOR_CHECK(api_name + ":", indices_desc,
                      "indices_desc must be contiguous");
  STRIDE_TENSOR_CHECK(api_name + ":", output_desc,
                      "output_desc must be contiguous");

  // check tensor datatype, support float16 and float32
  PARAM_CHECK_V2(api_name,
                 (input_desc->getDtype() == MLUOP_DTYPE_HALF) ||
                     (input_desc->getDtype() == MLUOP_DTYPE_FLOAT),
                 "Only half and float are supported in input tensor, but the "
                 "data type of tensor is "
                     << mluOpGetNameOfDataType(input_desc->getDtype()) << ".");
  PARAM_CHECK(api_name, input_desc->getDtype() == output_desc->getDtype());

  PARAM_CHECK_V2(
      api_name, (indices_desc->getDtype() == MLUOP_DTYPE_INT32),
      "Only int32 are supported in indices idx, but the data type of tensor is "
          << mluOpGetNameOfDataType(indices_desc->getDtype()) << ".");

  const size_t input_element_num = mluOpGetTensorElementNum(input_desc);
  const size_t indices_element_num = mluOpGetTensorElementNum(indices_desc);
  const size_t output_element_num = mluOpGetTensorElementNum(output_desc);

  // check large tensor
  TENSOR_NUM_CHECK(api_name, input_element_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(api_name, indices_element_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(api_name, output_element_num, LARGE_TENSOR_NUM, "");

  // check element num zero
  if (input_element_num == 0 || indices_element_num == 0 ||
      output_element_num == 0) {
    VLOG(5) << "[mluOpActiveRotatedFilterForward] Zero element tensor failure.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // check workspace ptr
  if (workspace_size > 0) {
    PARAM_CHECK(api_name, workspace != NULL);
  }

  // input and output ptr check null
  PARAM_CHECK(api_name, input != NULL);
  PARAM_CHECK(api_name, indices != NULL);
  PARAM_CHECK(api_name, output != NULL);

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpActiveRotatedFilterForward(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, const mluOpTensorDescriptor_t indices_desc,
    const void *indices, void *workspace, const size_t workspace_size,
    const mluOpTensorDescriptor_t output_desc, void *output) {
  const std::string api_name = "[mluOpActiveRotatedFilterForward]";
  // params check
  mluOpStatus_t status_paramcheck = activeRotatedFilterForwardParamCheck(
      handle, input_desc, input, indices_desc, indices, workspace,
      workspace_size, output_desc, output, api_name);
  if (status_paramcheck != MLUOP_STATUS_SUCCESS) {
    return status_paramcheck;
  }
  const int output_planes = input_desc->getDimIndex(0);
  const int input_planes = input_desc->getDimIndex(1);
  const int orientations = input_desc->getDimIndex(2);
  const int kH = input_desc->getDimIndex(3);
  const int kW = input_desc->getDimIndex(4);
  const int rotations = indices_desc->getDimIndex(3);

  // generate mluOpActiveRotatedFilterForward prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("active_rotated_filter_forward",
                   "ACTIVE_ROTATED_FILTER_FORWARD");
    // set handle dump mlu output
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", input, input_desc, 100, -100);
    GEN_CASE_DATA_REAL(true, "indices", indices, indices_desc);
    GEN_CASE_DATA(false, "output", output, output_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(false, false, true, 0.003, 0.003, 0);
  }

  mluOpDataType_t input_dtype = input_desc->getDtype();

  // start UX task, occupy all available clusters
  cnrtDim3_t k_dims;
  cnrtFunctionType_t k_type;
  k_dims.x = mluop::runtime::getCoreNumOfJobLimitCapability(handle);
  k_dims.y = 1;
  k_dims.z = 1;

  KernelClass job_type =
      static_cast<KernelClass>(mluop::runtime::getJobLimitCapability(handle));
  k_type = mluop::runtime::castCnKernelClassToCnrtFuncType(job_type);

  VLOG(5) << "Launch Kernel KernelActiveRotatedFilterForward<<<Union"
          << k_type / CORE_DIM << ", " << k_dims.x << ", " << k_dims.y << ", "
          << k_dims.z << ">>>.";
  CHECK_RETURN("[mluOpActiveRotatedFilterForward]",
               KernelActiveRotatedFilterForward(
                   k_dims, k_type, handle->queue, input_dtype, output_planes,
                   input_planes, orientations, kH, kW, rotations, input,
                   indices, workspace, output));

  // generate gen_case prototxt
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
