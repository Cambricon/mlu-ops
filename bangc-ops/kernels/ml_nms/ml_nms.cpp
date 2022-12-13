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
************************************************************************/
#include <stdio.h>
#include <string>
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "mlu_op_kernel.h"
#include "mlu_op.h"
#include "cnrt.h"
#include "cndev.h"

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

mluOpStatus_t MlNmsParamCheck(
  const std::string &op_name, const mluOpHandle_t &handle,
  const mluOpTensorDescriptor_t &x_desc, const void *x,
  const mluOpDataType_t support_type[], const int &len) {
  PARAM_CHECK(op_name, x_desc != NULL);
  PARAM_CHECK(op_name, handle != NULL);

  // check data type
  if (!isSupportType(x_desc->dtype, support_type, len)) {
    LOG(ERROR) << op_name << ":x_desc's data type is not supported.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  PARAM_CHECK(op_name, x != NULL);
  return MLUOP_STATUS_SUCCESS;
}


static void policyFunc(const mluOpHandle_t &handle,
                       const mluOpTensorDescriptor_t desc, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type) {
  size_t dim = mluOpGetTensorElementNum(desc);
  // Union1 policyFunc
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = handle->core_num_per_cluster;
  k_dim->y = mluop::runtime::getClusterLimitCapability(handle);
  k_dim->z = 1;
  // if a case is smaller than 2048 , it just need one cluster can work best.
  size_t small_case_thread = 2048;
  if (dim <= small_case_thread) k_dim->y = 1;
}

mluOpStatus_t MLUOP_WIN_API mluOpMlNms(mluOpHandle_t handle,
    const mluOpTensorDescriptor_t boxes_data_ptr_desc, void* boxes_data_ptr,
    float iou_threshold, void* output_boxes_index) {

    mluOpDataType_t support_type[2] = {MLUOP_DTYPE_HALF, MLUOP_DTYPE_FLOAT};
    mluOpStatus_t param_check = MlNmsParamCheck(
      "[mluOpMlNms]", handle, boxes_data_ptr_desc, boxes_data_ptr,
      support_type, 2);

    if (param_check != MLUOP_STATUS_SUCCESS) {
      return param_check;
    }

    cnrtDim3_t k_dim;
    cnrtFunctionType_t k_type;
    policyFunc(handle, boxes_data_ptr_desc, &k_dim, &k_type);
    int input_boxes_num = boxes_data_ptr_desc->total_element_num / 4;
    void (*mluOpFuncKernel)(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
      cnrtQueue_t queue, mluOpDataType_t data_type, void* boxes_data_ptr,
      float nmsThres, int input_boxes_num, uint8_t* output_boxes_index);

      if (boxes_data_ptr_desc->dtype == MLUOP_DTYPE_HALF) {
          mluOpFuncKernel = mluOpKernelMlNmsHalfFast;
      } else {
          mluOpFuncKernel = mluOpKernelMlNmsFloatFast;
      }

    KERNEL_CHECK(
      (mluOpFuncKernel(k_dim, k_type, handle->queue,
         boxes_data_ptr_desc->dtype, boxes_data_ptr,
         iou_threshold, input_boxes_num, (uint8_t*)output_boxes_index)));
    GEN_CASE_END();

    return MLUOP_STATUS_SUCCESS;
}
