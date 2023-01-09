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
#include "core/logging.h"
#include "core/tensor.h"
#include "core/type.h"
#include "core/context.h"
#include "core/runtime/device.h"
#include "unary_op_host.h"
#include "mlu_op.h"

void unaryOpPolicyFunc(const mluOpHandle_t &handle,
                       const mluOpTensorDescriptor_t &desc, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type) {
  size_t union_number = mluop::runtime::getClusterLimitCapability(handle);
  size_t core_in_cluster = handle->core_num_per_cluster;
  size_t core_number = union_number * core_in_cluster;
  size_t element_num = mluOpGetTensorElementNum(desc);
  size_t tensor_size = element_num * mluop::getSizeOfDataType(desc->dtype);
  tensor_size = CEIL_ALIGN(tensor_size, NFU_ALIGN_SIZE);
  size_t need_core = CEIL_ALIGN(tensor_size / NFU_ALIGN_SIZE, core_in_cluster);
  *k_type = CNRT_FUNC_TYPE_UNION1;  // default func type
  k_dim->x = core_in_cluster;
  if (need_core < core_number) {
    k_dim->y = need_core / core_in_cluster;
  } else {
    k_dim->y = union_number;
  }
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

mluOpStatus_t unaryOpParamCheck(
    const std::string &op_name, const mluOpHandle_t &handle,
    const mluOpTensorDescriptor_t &x_desc, const void *x,
    const mluOpTensorDescriptor_t &y_desc, const void *y,
    const mluOpDataType_t support_type[], const int &len, bool &zero_element) {
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
  PARAM_CHECK(op_name, x != NULL);
  PARAM_CHECK(op_name, y != NULL);
  return MLUOP_STATUS_SUCCESS;
}
