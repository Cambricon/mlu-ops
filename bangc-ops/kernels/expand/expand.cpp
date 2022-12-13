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

#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"

#define INT64_LARGE_TENSOR_NUM ((uint64_t)(1 << 30))

static mluOpStatus_t policyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                                cnrtFunctionType_t *k_type) {
  k_dim->x = mluop::runtime::getCoreNumOfJobLimitCapability(handle);
  k_dim->y = 1;
  k_dim->z = 1;
  *k_type = mluop::runtime::getJobLimitCapabilityCnrtFuncType(handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpExpand(mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
            const void *input, const mluOpTensorDescriptor_t output_desc,
            void *output) {
  PARAM_CHECK("[mluOpExpand]", handle != NULL);
  PARAM_CHECK("[mluOpExpand]", input_desc != NULL);
  PARAM_CHECK("[mluOpExpand]", output_desc != NULL);
  PARAM_CHECK("[mluOpExpand]", input_desc->dtype != MLUOP_DTYPE_INVALID);
  PARAM_CHECK("[mluOpExpand]", input_desc->dtype == output_desc->dtype);
  PARAM_CHECK("[mluOpExpand]", input_desc->dim <= MLUOP_DIM_MAX);
  PARAM_CHECK("[mluOpExpand]", output_desc->dim <= MLUOP_DIM_MAX);
  size_t input_num = mluOpGetTensorElementNum(input_desc);
  size_t output_num = mluOpGetTensorElementNum(output_desc);
  if (getSizeOfDataType(input_desc->dtype) ==
      getSizeOfDataType(MLUOP_DTYPE_INT64)) {
    auto statement_error = "the data type is int64 or complex, ";
    TENSOR_NUM_CHECK("[mluOpExpand]", input_num, INT64_LARGE_TENSOR_NUM,
                     statement_error);
    TENSOR_NUM_CHECK("[mluOpExpand]", output_num, INT64_LARGE_TENSOR_NUM,
                     statement_error);
  } else {
    TENSOR_NUM_CHECK("[mluOpExpand]", input_num, LARGE_TENSOR_NUM, "");
    TENSOR_NUM_CHECK("[mluOpExpand]", output_num, LARGE_TENSOR_NUM, "");
  }
  if (mluOpGetTensorElementNum(input_desc) == 0) {
    VLOG(5) << "mluOpExpand skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }
  PARAM_CHECK("[mluOpExpand]", input != NULL);
  PARAM_CHECK("[mluOpExpand]", output != NULL);

  uint64_t dims_input[MLUOP_DIM_MAX];
  uint64_t dims_output[MLUOP_DIM_MAX];
  uint64_t redims_input[MLUOP_DIM_MAX + 1];
  uint64_t redims_output[MLUOP_DIM_MAX + 1];
  int32_t count_flag = 0;
  int32_t count_index[MLUOP_DIM_MAX + 1];

  int fix_num = 0;
  size_t input_size = input_num;

  // Reshape dims: A(a, b, c) ---> A(1, 1, 1, 1, 1, a, b, c, 1)
  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    dims_input[i] = 1;
    dims_output[i] = 1;
    redims_input[i] = 1;
    redims_output[i] = 1;
  }
  redims_input[MLUOP_DIM_MAX] = 1;
  redims_output[MLUOP_DIM_MAX] = 1;

  for (int i = 0; i < input_desc->dim; i++) {
    dims_input[MLUOP_DIM_MAX - i - 1] =
        input_desc->dims[input_desc->dim - i - 1];
  }
  for (int i = 0; i < output_desc->dim; i++) {
    dims_output[MLUOP_DIM_MAX - i - 1] =
        output_desc->dims[output_desc->dim - i - 1];
  }
  while (dims_output[MLUOP_DIM_MAX - 1 - fix_num] == 1) {
    fix_num++;
  }

  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    if (dims_output[i] % dims_input[i] != 0) {
      LOG(ERROR) << "[mluOpExpand] In expand dimension, the size of output"
                 << " should be times of the size of input. But now in expand "
                    "dimension"
                 << " the size of input is " << dims_input[i]
                 << ", the size of output is " << dims_output[i] << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  // Reshape: dims(1, A, 1, 1, B, 1) change to redims(A, 1, B)
  for (int i = MLUOP_DIM_MAX - 1, j = fix_num; i - j >= 0; i--) {
    redims_input[i] = dims_input[i - j];
    redims_output[i] = dims_output[i - j];
    while ((i - j) > 0 && dims_input[i - j] == 1 &&
           dims_input[i - j - 1] == 1) {
      redims_output[i] *= dims_output[i - j - 1];
      j++;
    }
  }

  size_t output_size = output_num;
  // Count how many dims need to expand.
  for (int i = 0; i < MLUOP_DIM_MAX + 1; i++) {
    count_index[i] = 0;
    if (redims_input[i] != redims_output[i]) {
      count_flag += 1;
      count_index[count_flag - 1] = i;
    }
  }

  // generate mluOpExpand prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("expand");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", input, input_desc, 10, 0);
    GEN_CASE_DATA(false, "output", output, output_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(false, false, true, 0, 0, 0);
  }
  // generate mluOpExpand prototxt end!

  if (count_flag == 0) {
    auto status_copy =
        mluOpCopy(handle, input_desc, input, output_desc, output);
    if (status_copy != MLUOP_STATUS_SUCCESS) {
      KERNEL_CALL_CHECK("mluOpExpand", "mluOpCopy", status_copy, "");
    }
    GEN_CASE_END();
    return MLUOP_STATUS_SUCCESS;
  }

  // Choose best task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  k_type = CNRT_FUNC_TYPE_UNION1;
  int core_dim = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  int32_t union_number = mluop::runtime::getClusterLimitCapability(handle);
  k_dim.x = core_dim;
  k_dim.y = union_number;
  k_dim.z = 1;
  mluOpDataType_t data_type = input_desc->dtype;

  if (getSizeOfDataType(input_desc->dtype) ==
      getSizeOfDataType(MLUOP_DTYPE_INT64)) {
    input_size *= 2;
    output_size *= 2;
    redims_input[MLUOP_DIM_MAX] = 2;
    redims_output[MLUOP_DIM_MAX] = 2;
    data_type = MLUOP_DTYPE_INT32;
  }

  if (count_flag == 1) {
    uint64_t high_num = 1;
    uint64_t expand_num =
        redims_output[count_index[0]] / redims_input[count_index[0]];
    uint64_t low_num = 1;
    for (int i = 0; i < count_index[0]; i++) {
      high_num *= redims_output[i];
    }
    for (int i = count_index[0] + 1; i < MLUOP_DIM_MAX + 1; i++) {
      low_num *= redims_output[i];
    }
    if (redims_input[count_index[0]] != 1) {
      low_num *= redims_input[count_index[0]];
    }
    VLOG(5) << "Launch Kernel MLUUnion1KernelExpandOneDim<<<Union"
            << k_type / CORE_DIM << ", " << k_dim.x << ", " << k_dim.y << ", "
            << k_dim.z << ">>>";
    KERNEL_CHECK((mluOpUnion1KernelExpandOneDim(
        k_dim, k_type, handle->queue, (void *)input, output, (uint32_t)high_num,
        (uint32_t)expand_num, (uint32_t)low_num,
        mluOpDataTypeBytes(data_type))));
  } else {
    INTERNAL_CHECK("mluOpExpand",
                   MLUOP_STATUS_SUCCESS == policyFunc(handle, &k_dim, &k_type));
    VLOG(5) << "Launch Kernel MLUUnion1KernelExpandTensor<<<Union"
            << k_type / CORE_DIM << ", " << k_dim.x << ", " << k_dim.y << ", "
            << k_dim.z << ">>>";
    KERNEL_CHECK((mluOpUnion1KernelExpandTensor(
        k_dim, k_type, handle->queue, (void *)input, output,
        (uint32_t)dims_input[0], (uint32_t)dims_input[1],
        (uint32_t)dims_input[2], (uint32_t)dims_input[3],
        (uint32_t)dims_input[4], (uint32_t)dims_input[5],
        (uint32_t)dims_input[6], (uint32_t)dims_input[7],
        (uint32_t)dims_output[0], (uint32_t)dims_output[1],
        (uint32_t)dims_output[2], (uint32_t)dims_output[3],
        (uint32_t)dims_output[4], (uint32_t)dims_output[5],
        (uint32_t)dims_output[6], (uint32_t)dims_output[7],
        mluOpDataTypeBytes(input_desc->dtype))));
  }

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
