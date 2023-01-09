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

#include "kernels/kernel.h"
#include "core/tensor.h"
#include "core/type.h"
#include "core/context.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "binary_op_host.h"
#include "mlu_op.h"

void binaryOpPolicyFunc(const mluOpHandle_t &handle,
                        const mluOpTensorDescriptor_t &desc,
                        const int &align_param, cnrtDim3_t *k_dim,
                        cnrtFunctionType_t *k_type) {
  int union_number = mluop::runtime::getClusterLimitCapability(handle);
  int core_dim = handle->core_num_per_cluster;
  int core_number = union_number * core_dim;

  int element_num = mluOpGetTensorElementNum(desc);
  int size =
      CEIL_ALIGN(element_num * mluop::getSizeOfDataType(desc->dtype),
                 align_param);
  int core_used = CEIL_ALIGN(size / align_param, core_dim);
  core_used = core_used > core_number ? core_number : core_used;

  *k_type = CNRT_FUNC_TYPE_UNION1;  // default func type
  k_dim->x = core_dim;
  k_dim->y = core_used / core_dim;
  k_dim->z = 1;
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

mluOpStatus_t binaryOpParamCheck(
    const std::string &op_name, const mluOpHandle_t &handle,
    const mluOpTensorDescriptor_t &input1_desc, const void *input1,
    const mluOpTensorDescriptor_t &input2_desc, const void *input2,
    const mluOpTensorDescriptor_t &output_desc, const void *output,
    const mluOpDataType_t support_type[], const int &len, bool &zero_element) {
  // check descriptor
  PARAM_CHECK(op_name, handle != NULL);
  PARAM_CHECK(op_name, input1_desc != NULL);
  PARAM_CHECK(op_name, input2_desc != NULL);
  PARAM_CHECK(op_name, output_desc != NULL);

  // check dtype equal
  PARAM_CHECK_EQ(op_name, input1_desc->dtype, input2_desc->dtype);
  PARAM_CHECK_EQ(op_name, input1_desc->dtype, output_desc->dtype);

  // check dim less than MLUOP_DIM_MAX
  PARAM_CHECK_LE(op_name, input1_desc->dim, MLUOP_DIM_MAX);
  PARAM_CHECK_LE(op_name, input2_desc->dim, MLUOP_DIM_MAX);
  PARAM_CHECK_LE(op_name, output_desc->dim, MLUOP_DIM_MAX);

  // check dims
  for (int i = 0; i < input1_desc->dim; ++i) {
    if (input1_desc->dims[i] != input2_desc->dims[i]) {
      LOG(ERROR) << op_name << ":Check failed: input1_desc->dims[" << i
                 << "] should be equal to input2_desc->dims[" << i << "].";
      return MLUOP_STATUS_BAD_PARAM;
    }
    if (input1_desc->dims[i] != output_desc->dims[i]) {
      LOG(ERROR) << op_name << ":Check failed: input1_desc->dims[" << i
                 << "] should be equal to output_desc->dims[" << i << "].";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  // check data type support
  if (!isSupportType(input1_desc->dtype, support_type, len)) {
    LOG(ERROR) << op_name << ":input1_desc's data type is not supported.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // check 0 element
  if ((mluOpGetTensorElementNum(input1_desc) == 0) ||
      (mluOpGetTensorElementNum(input2_desc) == 0) ||
      (mluOpGetTensorElementNum(output_desc) == 0)) {
    VLOG(5) << op_name << " skip zero element tensor.";
    zero_element = true;
    return MLUOP_STATUS_SUCCESS;
  }

  // check device pointer
  PARAM_CHECK(op_name, input1 != NULL);
  PARAM_CHECK(op_name, input2 != NULL);
  PARAM_CHECK(op_name, output != NULL);

  return MLUOP_STATUS_SUCCESS;
}
