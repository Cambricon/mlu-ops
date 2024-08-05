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
#include "kernels/poly_nms/poly_nms.h"

#include <string>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"

namespace {
static inline int64_t getMaskColNum(int box_num) {
  return (box_num + MASK_T_BITWIDTH - 1) / MASK_T_BITWIDTH;
}

static inline int64_t getMaskMatrixByteSize(int box_num) {
  return box_num * getMaskColNum(box_num) * sizeof(uint32_t);
}
}  // namespace

mluOpStatus_t MLUOP_WIN_API mluOpGetPolyNmsWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t boxes_desc,
    size_t *size) {
  const std::string API = "[mluOpGetPolyNmsWorkspaceSize]";
  // check inputs/outputs
  PARAM_CHECK(API, handle != NULL);
  PARAM_CHECK(API, boxes_desc != NULL);
  PARAM_CHECK(API, size != NULL);

  // check inputs shape
  PARAM_CHECK_EQ(API, boxes_desc->dim, 2);
  PARAM_CHECK_EQ(API, boxes_desc->dims[1], 9);

  int box_num = boxes_desc->dims[0];
  auto mask_sz = getMaskMatrixByteSize(box_num);
  auto sort_info_sz = box_num * sizeof(int);
  auto area_sz = box_num * sizeof(int);
  *size = mask_sz + sort_info_sz + area_sz;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpPolyNms(mluOpHandle_t handle, const mluOpTensorDescriptor_t boxes_desc,
             const void *boxes, const float iou_threshold, void *workspace,
             size_t workspace_size, const mluOpTensorDescriptor_t output_desc,
             void *output, void *output_size) {
  const std::string API = "[mluOpPolyNms]";
  // check inputs/outputs
  PARAM_CHECK(API, handle != NULL);
  PARAM_CHECK(API, boxes_desc != NULL);
  PARAM_CHECK(API, output_desc != NULL);
  PARAM_CHECK(API, output_size != NULL);

  // check inputs/outputs data type
  PARAM_CHECK(API, boxes_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(API, output_desc->dtype == MLUOP_DTYPE_INT32);

  // check inputs layout
  PARAM_CHECK(API, boxes_desc->layout == MLUOP_LAYOUT_ARRAY);

  // check inputs shape
  PARAM_CHECK_EQ(API, boxes_desc->dim, 2);
  PARAM_CHECK_EQ(API, boxes_desc->dims[1], 9);
  // check output shape
  PARAM_CHECK_EQ(API, output_desc->dim, 1);

  PARAM_CHECK(API, boxes_desc->dims[0] == output_desc->dims[0]);

  // check stride
  STRIDE_TENSOR_CHECK("[mluOpPolyNms]:", boxes_desc,
                      "boxes_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpPolyNms]:", output_desc,
                      "output_desc must be contiguous");

  int input_boxes_num = boxes_desc->dims[0];

  if (input_boxes_num == 0) {
    VLOG(5) << API << " skip zero element tensor.";
    CNRT_CHECK(cnrtMemset(output_size, 0, sizeof(int)));
    return MLUOP_STATUS_SUCCESS;
  }

  PARAM_CHECK(API, boxes != NULL);
  PARAM_CHECK(API, output != NULL);

  if (workspace_size > 0) {
    PARAM_CHECK(API, workspace != NULL);
  }

  int box_num = boxes_desc->dims[0];
  int real_width = boxes_desc->strides[0];
  auto mask_col_num = getMaskColNum(box_num);
  if ((10 * box_num + mask_col_num * 2) > (MAX_NRAM_SIZE / sizeof(float))) {
    LOG(ERROR) << API << " Too many input boxes, kernel cannot work."
               << " The number of input boxes shoule be less than 9770,"
               << " current input box num is " << box_num << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // generate prototxt
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("poly_nms", "POLY_NMS");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input1", boxes, boxes_desc, 10, 0);
    GEN_CASE_DATA(false, "output1", output, output_desc, 0, 0);
    GEN_CASE_DATA_UNFOLD(false, "output2", output_size, 1, {1},
                         MLUOP_DTYPE_INT32, MLUOP_LAYOUT_ARRAY, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "poly_nms", "iou_threshold", iou_threshold);
    GEN_CASE_TEST_PARAM_NEW(false, false, true, 3e-3, 3e-3, 0);
  }

  float *dev_area = (float *)workspace;
  int *dev_sort_info = (int *)dev_area + box_num;
  uint32_t *dev_mask = (uint32_t *)dev_sort_info + box_num;
  MLUCalcAreaLaunchConfig area_launch_cfg(handle, box_num);
  KernelPolyNmsCalcArea(area_launch_cfg.dim, area_launch_cfg.kernel_type,
                        handle->queue, (float *)boxes, box_num, real_width,
                        dev_area);

  MLUGenNmsMaskLaunchConfig mask_launch_cfg(handle, box_num);
  KernelPolyNmsGenMask(mask_launch_cfg.dim, mask_launch_cfg.kernel_type,
                       handle->queue, (float *)boxes, box_num, real_width,
                       iou_threshold, dev_area, dev_mask, dev_sort_info);

  MLUGenResultLaunchConfig dim_gen_result;
  KernelPolyNmsGenResult(dim_gen_result.dim, dim_gen_result.kernel_type,
                         handle->queue, box_num, dev_mask, dev_sort_info,
                         (int *)output, (int *)output_size);
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
