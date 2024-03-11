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
#include "border_align_backward.h"

#include <string>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "core/tool.h"
#include "kernels/kernel.h"
#include "kernels/utils/cnnl_helper.h"

static void policyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type) {
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  k_dim->y = mluop::runtime::getClusterLimitCapability(handle);
  k_dim->z = 1;
}

mluOpStatus_t mluOpBorderAlignBackward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t grad_output_desc,
    const void *grad_output, const mluOpTensorDescriptor_t boxes_desc,
    const void *boxes, const mluOpTensorDescriptor_t argmax_idx_desc,
    const void *argmax_idx, const int32_t pool_size,
    const mluOpTensorDescriptor_t grad_input_desc, void *grad_input) {
  const std::string API = "[mluOpBorderAlignBackward]";
  // params check
  PARAM_CHECK(API, handle != nullptr);
  PARAM_CHECK(API, grad_output_desc != nullptr);
  PARAM_CHECK(API, boxes_desc != nullptr);
  PARAM_CHECK(API, argmax_idx_desc != nullptr);
  PARAM_CHECK(API, grad_input_desc != nullptr);

  PARAM_CHECK(API, grad_output_desc->dim == 4);
  PARAM_CHECK(API, boxes_desc->dim == 3);
  PARAM_CHECK(API, argmax_idx_desc->dim == 4);
  PARAM_CHECK(API, grad_input_desc->dim == 4);

  const int32_t border_num = 4;
  const int32_t coord_num = 4;
  const int32_t origin_n = grad_input_desc->dims[0];
  const int32_t origin_h = grad_input_desc->dims[1];
  const int32_t origin_w = grad_input_desc->dims[2];
  const int32_t origin_c = grad_input_desc->dims[3] / border_num;
  const int32_t origin_k = boxes_desc->dims[1];

  PARAM_CHECK(API, grad_output_desc->dtype == MLUOP_DTYPE_FLOAT ||
                       grad_output_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK(API, argmax_idx_desc->dtype == MLUOP_DTYPE_INT32);
  PARAM_CHECK(API, boxes_desc->dtype == grad_output_desc->dtype);
  PARAM_CHECK(API, grad_input_desc->dtype == grad_output_desc->dtype);

  PARAM_CHECK(API, grad_output_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(API, argmax_idx_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(API, grad_input_desc->layout == MLUOP_LAYOUT_NHWC);

  PARAM_CHECK(API, grad_input_desc->dims[3] % border_num == 0);
  PARAM_CHECK_NE(API, origin_n, 0);
  PARAM_CHECK_NE(API, origin_c, 0);
  PARAM_CHECK_NE(API, origin_h, 0);
  PARAM_CHECK_NE(API, origin_w, 0);
  PARAM_CHECK(API, origin_h * origin_w == origin_k);
  PARAM_CHECK(API, boxes_desc->dim == 3);
  PARAM_CHECK(API, boxes_desc->dims[2] == coord_num);
  PARAM_CHECK_NE(API, origin_k, 0);
  PARAM_CHECK_GT(API, pool_size, 0);

  PARAM_CHECK_EQ(API, grad_output_desc->dims[0], origin_n);
  PARAM_CHECK_EQ(API, grad_output_desc->dims[1], origin_k);
  PARAM_CHECK_EQ(API, grad_output_desc->dims[2], border_num);
  PARAM_CHECK_EQ(API, grad_output_desc->dims[3], origin_c);

  PARAM_CHECK_EQ(API, boxes_desc->dims[0], origin_n);
  PARAM_CHECK_EQ(API, boxes_desc->dims[1], origin_k);
  PARAM_CHECK_EQ(API, boxes_desc->dims[2], coord_num);

  PARAM_CHECK_EQ(API, argmax_idx_desc->dims[0], origin_n);
  PARAM_CHECK_EQ(API, argmax_idx_desc->dims[1], origin_k);
  PARAM_CHECK_EQ(API, argmax_idx_desc->dims[2], border_num);
  PARAM_CHECK_EQ(API, argmax_idx_desc->dims[3], origin_c);

  TENSOR_NUM_CHECK(API, mluOpGetTensorElementNum(grad_output_desc),
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, mluOpGetTensorElementNum(boxes_desc), LARGE_TENSOR_NUM,
                   "");
  TENSOR_NUM_CHECK(API, mluOpGetTensorElementNum(grad_input_desc),
                   LARGE_TENSOR_NUM, "");

  PARAM_CHECK(API, grad_output != nullptr);
  PARAM_CHECK(API, boxes != nullptr);
  PARAM_CHECK(API, argmax_idx != nullptr);
  PARAM_CHECK(API, grad_input != nullptr);

  // generate case prototxt
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("border_align_backward", "BORDER_ALIGN_BACKWARD");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input1", grad_output, grad_output_desc, 100, 0);
    GEN_CASE_DATA_REAL(true, "input2", boxes, boxes_desc);
    GEN_CASE_DATA_REAL(true, "input3", argmax_idx, argmax_idx_desc);
    GEN_CASE_DATA(false, "output1", grad_input, grad_input_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "border_align_backward", "pool_size",
                             pool_size);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, &k_dim, &k_type);

  VLOG(5) << "[mluOpBorderAlignBackward] cnnlFill_v3 start.";
  uint64_t fill_value = 0x0;
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grad_input_desc,
                                                 cnnl_output_desc);
    CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &fill_value,
                          cnnl_output_desc, grad_input));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  VLOG(5) << "[mluOpBorderAlignBackward] cnnlFill_v3 end.";
  mluOpDataType_t input_dtype = grad_output_desc->dtype;

  VLOG(5) << "Launch Kernel KernelBorderAlignBackward<<<Union"
          << k_type / CORE_DIM << ", " << k_dim.x << ", " << k_dim.y << ", "
          << k_dim.z << ">>>";
  CHECK_RETURN(
      API, KernelBorderAlignBackward(
               k_dim, k_type, handle->queue, input_dtype, (void *)grad_output,
               (void *)boxes, (int32_t *)argmax_idx, pool_size, origin_n,
               origin_h, origin_w, origin_c, origin_k, (void *)grad_input));

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
