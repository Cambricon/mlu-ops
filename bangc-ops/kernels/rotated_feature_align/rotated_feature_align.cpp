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
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"

// policy function
static void policyFunc(const mluOpHandle_t handle,
                       const mluOpTensorDescriptor_t output_desc,
                       cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  const size_t num_rois = output_desc->dims[0];
  const size_t pooled_height = output_desc->dims[1];
  const size_t pooled_width = output_desc->dims[2];
  const size_t num_bin = num_rois * pooled_height * pooled_width;
  size_t core_num = handle->core_num_per_cluster;
  size_t cluster_num = mluop::runtime::getJobLimitCapability(handle) / core_num;
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = core_num;
  size_t use_cluster = (num_bin + core_num - 1) / core_num;
  k_dim->y = use_cluster > cluster_num ? cluster_num : use_cluster;
  k_dim->z = 1;
}

static mluOpStatus_t RotatedFeatureAlignForwardPreCheck(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const mluOpTensorDescriptor_t bboxes_desc,
    const mluOpTensorDescriptor_t output_desc) {
  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]", handle != NULL);
  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]", input_desc != NULL);
  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]", bboxes_desc != NULL);
  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]", output_desc != NULL);

  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]", input_desc->dim == 4);
  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]", bboxes_desc->dim == 4);
  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]", output_desc->dim == 4);

  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]",
              input_desc->dtype == MLUOP_DTYPE_FLOAT ||
                  input_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]",
              input_desc->dtype == bboxes_desc->dtype);
  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]",
              input_desc->dtype == output_desc->dtype);

  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]",
              input_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]",
              output_desc->layout == MLUOP_LAYOUT_NHWC);

  for (int i = 0; i < input_desc->dim; i++) {
    if (input_desc->dims[i] != output_desc->dims[i]) {
      LOG(ERROR)
          << "[mluOpRotatedFeatureAlignForward] Check failed: input_desc->dims["
          << i << "] should be equal to output_desc->dims[" << i << "].";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  for (int i = 0; i < input_desc->dim - 1; i++) {
    if (input_desc->dims[i] != bboxes_desc->dims[i]) {
      LOG(ERROR)
          << "[mluOpRotatedFeatureAlignForward] Check failed: input_desc->dims["
          << i << "] should be equal to bboxes_desc->dims[" << i << "].";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]", bboxes_desc->dims[3] == 5);

  const size_t input_element_num = mluOpGetTensorElementNum(input_desc);
  const size_t output_element_num = mluOpGetTensorElementNum(output_desc);
  const size_t bboxes_element_num = mluOpGetTensorElementNum(bboxes_desc);

  TENSOR_NUM_CHECK("[mluOpRotatedFeatureAlignForward]", input_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpRotatedFeatureAlignForward]", output_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpRotatedFeatureAlignForward]", bboxes_element_num,
                   LARGE_TENSOR_NUM, "");

  if (mluOpGetTensorElementNum(bboxes_desc) == 0 ||
      mluOpGetTensorElementNum(input_desc) == 0 ||
      mluOpGetTensorElementNum(output_desc) == 0) {
    VLOG(5) << "[mluOpRotatedFeatureAlignForward] Zero element tensor failure.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t RotatedFeatureAlignBackwardPreCheck(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t top_output_desc,
    const mluOpTensorDescriptor_t bboxes_desc,
    const mluOpTensorDescriptor_t bottom_input_desc, const int points,
    const float spatial_scale) {
  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]", handle != NULL);
  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]", top_output_desc != NULL);
  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]", bboxes_desc != NULL);
  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]", bottom_input_desc != NULL);

  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]", top_output_desc->dim == 4);
  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]", bboxes_desc->dim == 4);
  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]",
              bottom_input_desc->dim == 4);

  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]",
              top_output_desc->dtype == MLUOP_DTYPE_FLOAT ||
                  top_output_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]",
              top_output_desc->dtype == bboxes_desc->dtype);
  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]",
              top_output_desc->dtype == bottom_input_desc->dtype);

  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]",
              top_output_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]",
              bottom_input_desc->layout == MLUOP_LAYOUT_NHWC);

  for (int i = 0; i < top_output_desc->dim; i++) {
    if (top_output_desc->dims[i] != bottom_input_desc->dims[i]) {
      LOG(ERROR) << "[mluOpRotatedFeatureAlignBackward] Check failed: "
                    "top_output_desc->dims["
                 << i << "] should be equal to bottom_input_desc->dims[" << i
                 << "].";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  for (int i = 0; i < top_output_desc->dim - 1; i++) {
    if (top_output_desc->dims[i] != bboxes_desc->dims[i]) {
      LOG(ERROR) << "[mluOpRotatedFeatureAlignBackward] Check failed: "
                    "top_output_desc->dims["
                 << i << "] should be equal to bboxes_desc->dims[" << i << "].";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]", bboxes_desc->dims[3] == 5);
  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]", points == 1 || points == 5);
  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]", spatial_scale > 0);

  const size_t top_output_element_num =
      mluOpGetTensorElementNum(top_output_desc);
  const size_t bottom_input_element_num =
      mluOpGetTensorElementNum(bottom_input_desc);
  const size_t bboxes_element_num = mluOpGetTensorElementNum(bboxes_desc);

  TENSOR_NUM_CHECK("[mluOpRotatedFeatureAlignBackward]", top_output_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpRotatedFeatureAlignBackward]",
                   bottom_input_element_num, LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpRotatedFeatureAlignBackward]", bboxes_element_num,
                   LARGE_TENSOR_NUM, "");

  if (mluOpGetTensorElementNum(bboxes_desc) == 0 ||
      mluOpGetTensorElementNum(top_output_desc) == 0 ||
      mluOpGetTensorElementNum(bottom_input_desc) == 0) {
    VLOG(5)
        << "[mluOpRotatedFeatureAlignBackward] Zero element tensor failure.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpRotatedFeatureAlignForward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, const mluOpTensorDescriptor_t bboxes_desc,
    const void *bboxes, const float spatial_scale, const int points,
    const mluOpTensorDescriptor_t output_desc, void *output) {
  mluOpStatus_t status = MLUOP_STATUS_BAD_PARAM;
  status = RotatedFeatureAlignForwardPreCheck(handle, input_desc, bboxes_desc,
                                              output_desc);
  if (MLUOP_STATUS_SUCCESS != status) {
    return status;
  }
  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]", points == 1 || points == 5);
  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]", spatial_scale > 0);

  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]", input != NULL);
  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]", bboxes != NULL);
  PARAM_CHECK("[mluOpRotatedFeatureAlignForward]", output != NULL);

  // generate mluOpRotatedFeatureAlignForward prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("rotated_feature_align_forward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", input, input_desc, -10, 10);
    GEN_CASE_DATA_REAL(true, "rois", bboxes, bboxes_desc);
    GEN_CASE_DATA(false, "output", output, output_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "rotated_feature_align_forward",
                             "spatial_scale", spatial_scale);
    GEN_CASE_OP_PARAM_SINGLE(1, "rotated_feature_align_forward", "points",
                             points);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }
  // generate mluOpRotatedFeatureAlignForward prototxt end!

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  policyFunc(handle, output_desc, &k_dim, &k_type);

  const int batches = input_desc->dims[0];
  const int height = input_desc->dims[1];
  const int width = input_desc->dims[2];
  const int channels = input_desc->dims[3];
  const int offset_rois = bboxes_desc->dims[3];
  mluOpDataType_t data_dtype = input_desc->dtype;
  VLOG(5) << "[mluOpRotatedFeatureAlignForward] launch kernel policyFunc["
          << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << "].";
  if (data_dtype == MLUOP_DTYPE_FLOAT) {
    KERNEL_CHECK((mluOpBlockKernelRotatedFeatureAlignForwardFloat(
        k_dim, k_type, handle->queue, input, bboxes, batches, height, width,
        channels, offset_rois, spatial_scale, points, output)));
    VLOG(5) << "Kernel mluOpBlockKernelRotatedFeatureAlignForwardFloat.";
  } else {
    KERNEL_CHECK((mluOpBlockKernelRotatedFeatureAlignForwardHalf(
        k_dim, k_type, handle->queue, input, bboxes, batches, height, width,
        channels, offset_rois, spatial_scale, points, output)));
    VLOG(5) << "Kernel mluOpBlockKernelRotatedFeatureAlignForwardHalf.";
  }
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpRotatedFeatureAlignBackward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t top_output_desc,
    const void *top_output, const mluOpTensorDescriptor_t bboxes_desc,
    const void *bboxes, const float spatial_scale, const int points,
    const mluOpTensorDescriptor_t bottom_input_desc, void *bottom_input) {
  mluOpStatus_t status = MLUOP_STATUS_BAD_PARAM;
  status = RotatedFeatureAlignBackwardPreCheck(handle, top_output_desc,
                                               bboxes_desc, bottom_input_desc,
                                               points, spatial_scale);
  if (MLUOP_STATUS_SUCCESS != status) {
    return status;
  }

  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]", top_output != NULL);
  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]", bboxes != NULL);
  PARAM_CHECK("[mluOpRotatedFeatureAlignBackward]", bottom_input != NULL);

  // generate mluOpRotatedFeatureAlignBackward prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("rotated_feature_align_backward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", top_output, top_output_desc, -100, 100);
    GEN_CASE_DATA_REAL(true, "rois", bboxes, bboxes_desc);
    GEN_CASE_DATA(false, "output", bottom_input, bottom_input_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "rotated_feature_align_backward",
                             "spatial_scale", spatial_scale);
    GEN_CASE_OP_PARAM_SINGLE(1, "rotated_feature_align_backward", "points",
                             points);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }
  // generate mluOpRotatedFeatureAlignBackward prototxt end!

  VLOG(5) << "mluOpFill start.";
  const uint32_t fill_value = 0x00;
  MLUOP_CHECK(mluOpFill(handle, MLUOP_POINTER_MODE_HOST, &fill_value,
                        bottom_input_desc, bottom_input));
  VLOG(5) << "mluOpFill end.";

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, top_output_desc, &k_dim, &k_type);

  const int batches = top_output_desc->dims[0];
  const int height = top_output_desc->dims[1];
  const int width = top_output_desc->dims[2];
  const int channels = top_output_desc->dims[3];
  const int offset_rois = bboxes_desc->dims[3];
  mluOpDataType_t data_dtype = top_output_desc->dtype;
  VLOG(5) << "[mluOpRotatedFeatureAlignBackward] launch kernel policyFunc["
          << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << "].";
  if (data_dtype == MLUOP_DTYPE_FLOAT) {
    KERNEL_CHECK((mluOpBlockKernelRotatedFeatureAlignBackwardFloat(
        k_dim, k_type, handle->queue, top_output, bboxes, batches, height,
        width, channels, offset_rois, spatial_scale, points, bottom_input)));
    VLOG(5) << "Kernel mluOpBlockKernelRotatedFeatureAlignBackwardFloat.";
  } else {
    KERNEL_CHECK((mluOpBlockKernelRotatedFeatureAlignBackwardHalf(
        k_dim, k_type, handle->queue, top_output, bboxes, batches, height,
        width, channels, offset_rois, spatial_scale, points, bottom_input)));
    VLOG(5) << "Kernel mluOpBlockKernelRotatedFeatureAlignBackwardHalf.";
  }
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
