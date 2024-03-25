/*******************************************************************************
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
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *******************************************************************************/
#include "border_align_forward.h"

#include <string>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"

// policyFunc
static void policyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type) {
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  k_dim->y = mluop::runtime::getClusterLimitCapability(handle);
  k_dim->z = 1;
}

mluOpStatus_t mluOpBorderAlignForward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, const mluOpTensorDescriptor_t boxes_desc,
    const void *boxes, const int32_t pool_size,
    const mluOpTensorDescriptor_t output_desc, void *output,
    const mluOpTensorDescriptor_t argmax_idx_desc, void *argmax_idx) {
  const std::string API = "[mluOpBorderAlignForward]";
  PARAM_CHECK(API, handle != nullptr);
  PARAM_CHECK(API, input_desc != nullptr);
  PARAM_CHECK(API, boxes_desc != nullptr);
  PARAM_CHECK(API, output_desc != nullptr);
  PARAM_CHECK(API, argmax_idx_desc != nullptr);

  PARAM_CHECK(API, input_desc->dim == 4);
  PARAM_CHECK(API, boxes_desc->dim == 3);
  PARAM_CHECK(API, output_desc->dim == 4);
  PARAM_CHECK(API, argmax_idx_desc->dim == 4);

  const int32_t border_num = 4;
  const int32_t coord_num = 4;
  const int32_t origin_n = input_desc->dims[0];
  const int32_t origin_h = input_desc->dims[1];
  const int32_t origin_w = input_desc->dims[2];
  const int32_t origin_c = input_desc->dims[3] / border_num;
  const int32_t origin_k = boxes_desc->dims[1];

  PARAM_CHECK(API, input_desc->dtype == boxes_desc->dtype);
  PARAM_CHECK(API, input_desc->dtype == MLUOP_DTYPE_FLOAT ||
                       input_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK(API, boxes_desc->dtype == MLUOP_DTYPE_FLOAT ||
                       boxes_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK(API, output_desc->dtype == input_desc->dtype);
  PARAM_CHECK(API, argmax_idx_desc->dtype == MLUOP_DTYPE_INT32);

  PARAM_CHECK(API, input_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(API, output_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(API, argmax_idx_desc->layout == MLUOP_LAYOUT_NHWC);

  PARAM_CHECK(API, input_desc->dims[3] % 4 == 0);
  PARAM_CHECK_NE(API, input_desc->dims[0], 0);
  PARAM_CHECK_NE(API, input_desc->dims[3] / 4, 0);
  PARAM_CHECK_NE(API, input_desc->dims[1], 0);
  PARAM_CHECK_NE(API, input_desc->dims[2], 0);
  PARAM_CHECK_NE(API, boxes_desc->dims[1], 0);
  PARAM_CHECK(API, boxes_desc->dim == 3);
  PARAM_CHECK(API, boxes_desc->dims[2] == 4);

  PARAM_CHECK(API, input_desc->dims[0]== boxes_desc->dims[0]);
  PARAM_CHECK(API, input_desc->dims[1] * input_desc->dims[2]
                    == boxes_desc->dims[1]);
  PARAM_CHECK_EQ(API, output_desc->dims[0], input_desc->dims[0]);
  PARAM_CHECK_EQ(API, output_desc->dims[1], boxes_desc->dims[1]);
  PARAM_CHECK_EQ(API, output_desc->dims[2], 4);
  PARAM_CHECK_EQ(API, output_desc->dims[3], input_desc->dims[3] / 4);
  PARAM_CHECK_EQ(API, argmax_idx_desc->dims[0], input_desc->dims[0]);
  PARAM_CHECK_EQ(API, argmax_idx_desc->dims[1], boxes_desc->dims[1]);
  PARAM_CHECK_EQ(API, argmax_idx_desc->dims[2], 4);
  PARAM_CHECK_EQ(API, argmax_idx_desc->dims[3], input_desc->dims[3] / 4);

  const size_t input_num = mluOpGetTensorElementNum(input_desc);
  const size_t boxes_num = mluOpGetTensorElementNum(boxes_desc);
  const size_t output_num = mluOpGetTensorElementNum(output_desc);
  TENSOR_NUM_CHECK(API, input_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, boxes_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK(API, output_num, LARGE_TENSOR_NUM, "");

  PARAM_CHECK(API, input != nullptr);
  PARAM_CHECK(API, boxes != nullptr);
  PARAM_CHECK(API, output != nullptr);
  PARAM_CHECK(API, argmax_idx != nullptr);
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("border_align_forward", "BORDER_ALIGN_FORWARD");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input1", input, input_desc, 100, 0);
    GEN_CASE_DATA_REAL(true, "input2", boxes, boxes_desc);
    GEN_CASE_DATA(false, "output1", output, output_desc, 0, 0);
    GEN_CASE_DATA(false, "output2", argmax_idx, argmax_idx_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "border_align", "pool_size", pool_size);
    GEN_CASE_TEST_PARAM_NEW(false, false, true, 0.003, 0, 0);
  }

  cnrtFunctionType_t k_type;
  cnrtDim3_t k_dim;
  policyFunc(handle, &k_dim, &k_type);

  VLOG(5) << "Launch Kernel KernelBorderAlignForward<<<Union"
          << k_type / CORE_DIM << ", " << k_dim.x << ", " << k_dim.y << ", "
          << k_dim.z << ">>>";
  CHECK_RETURN(API, KernelBorderAlignForward(
                        k_dim, k_type, handle->queue, input_desc->dtype, input,
                        boxes, pool_size, origin_n, origin_h, origin_w,
                        origin_c, origin_k, output, (int32_t *)argmax_idx));
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
