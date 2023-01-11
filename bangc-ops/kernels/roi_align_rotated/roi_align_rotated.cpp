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

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"

static void policyFunc(const mluOpHandle_t handle, const int bin_num,
                       cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  size_t core_num = handle->core_num_per_cluster;
  size_t cluster_num = mluop::runtime::getJobLimitCapability(handle) / core_num;
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = core_num;
  size_t use_cluster = (bin_num + core_num - 1) / core_num;
  k_dim->y = use_cluster > cluster_num ? cluster_num : use_cluster;
  k_dim->z = 1;
}

mluOpStatus_t MLUOP_WIN_API mluOpRoiAlignRotatedForward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t features_desc,
    const void *features, const mluOpTensorDescriptor_t rois_desc,
    const void *rois, const int pooled_height, const int pooled_width,
    const int sample_ratio, const float spatial_scale, const bool aligned,
    const bool clockwise, const mluOpTensorDescriptor_t output_desc,
    void *output) {
  const std::string API = "[mluOpRoiAlignRotatedForward]";

  PARAM_CHECK(API, handle != nullptr);
  PARAM_CHECK(API, features_desc != nullptr);
  PARAM_CHECK(API, rois_desc != nullptr);
  PARAM_CHECK(API, output_desc != nullptr);

  PARAM_CHECK(API, features_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(API, output_desc->layout == MLUOP_LAYOUT_NHWC);

  PARAM_CHECK(API, (features_desc->dtype == output_desc->dtype) &&
                       (output_desc->dtype == rois_desc->dtype));
  PARAM_CHECK(API, features_desc->dtype == MLUOP_DTYPE_FLOAT ||
                       features_desc->dtype == MLUOP_DTYPE_HALF);

  PARAM_CHECK_EQ(API, rois_desc->dim, 2);
  PARAM_CHECK_EQ(API, output_desc->dim, 4);
  PARAM_CHECK_EQ(API, features_desc->dim, 4);
  PARAM_CHECK_EQ(API, rois_desc->dims[1], 6);

  PARAM_CHECK_EQ(API, output_desc->dims[1], pooled_height);
  PARAM_CHECK_EQ(API, output_desc->dims[2], pooled_width);

  PARAM_CHECK_GT(API, features_desc->dims[3], 0);
  PARAM_CHECK_GT(API, rois_desc->dims[0], 0);

  if (output_desc->dims[0] != rois_desc->dims[0]) {
    LOG(ERROR) << API << " rois_desc batch = " << rois_desc->dims[0]
               << ", output_desc batch = " << output_desc->dims[0]
               << ". They should be the same.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (output_desc->dims[3] != features_desc->dims[3]) {
    LOG(ERROR) << API << " features_desc channel = " << features_desc->dims[3]
               << ", output_desc channel = " << output_desc->dims[3]
               << ". They should be the same.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  const int channel = features_desc->dims[3];
  const int width = features_desc->dims[2];
  const int height = features_desc->dims[1];
  const int batch = features_desc->dims[0];
  const int rois_nums = rois_desc->dims[0];
  mluOpDataType_t data_type = features_desc->dtype;

  PARAM_CHECK_GT(API, pooled_height, 0);
  PARAM_CHECK_GT(API, pooled_width, 0);
  PARAM_CHECK_GE(API, spatial_scale, 0);
  PARAM_CHECK_GE(API, sample_ratio, 0);

  if (mluOpGetTensorElementNum(features_desc) == 0) {
    VLOG(5) << "[mluOpRoiAlignRotatedForward]] Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  PARAM_CHECK(API, features != nullptr);
  PARAM_CHECK(API, output != nullptr);
  PARAM_CHECK(API, rois != nullptr);

  VLOG(5) << "pool_height: " << pooled_height << ",pool_width: " << pooled_width
          << ",channel: " << channel << ",roi nums: " << rois_nums << ".";
  VLOG(5) << "batch: " << batch << ",height: " << height << ",width: " << width
          << ".";

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("roi_align_rotated_forward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input1", features, features_desc, 10, 0);
    GEN_CASE_DATA_REAL(true, "input2", rois, rois_desc);
    GEN_CASE_DATA(false, "output1", output, output_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "roi_align_rotated_forward", "pooled_height",
                             pooled_height);
    GEN_CASE_OP_PARAM_SINGLE(1, "roi_align_rotated_forward", "pooled_width",
                             pooled_width);
    GEN_CASE_OP_PARAM_SINGLE(1, "roi_align_rotated_forward", "sample_ratio",
                             sample_ratio);
    GEN_CASE_OP_PARAM_SINGLE(1, "roi_align_rotated_forward", "spatial_scale",
                             spatial_scale);
    GEN_CASE_OP_PARAM_SINGLE(1, "roi_align_rotated_forward", "aligned",
                             aligned);
    GEN_CASE_OP_PARAM_SINGLE(2, "roi_align_rotated_forward", "clockwise",
                             clockwise);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }
  mluOpRoiAlignRotatedParams roiAlignRotatedParams{pooled_height, pooled_width,
                                                   sample_ratio,  spatial_scale,
                                                   aligned,       clockwise};
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, rois_nums * pooled_height * pooled_width, &k_dim, &k_type);
  VLOG(5) << "[mluOpRoiAlignRotatedForward] launch kernel policyFunc["
          << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << "].";
  if (features_desc->dtype == MLUOP_DTYPE_FLOAT) {
    KERNEL_CHECK((mluOpBlockKernelRoiAlignRotatedForwardFloat(
        k_dim, k_type, handle->queue, features, rois, batch, height, width,
        channel, rois_nums, roiAlignRotatedParams, output)));
    VLOG(5) << "Kernel mluOpBlockKernelRoiAlignRotatedForwardFloat.";
  } else {
    KERNEL_CHECK((mluOpBlockKernelRoiAlignRotatedForwardHalf(
        k_dim, k_type, handle->queue, features, rois, batch, height, width,
        channel, rois_nums, roiAlignRotatedParams, output)));
    VLOG(5) << "Kernel mluOpBlockKernelRoiAlignRotatedForwardHalf.";
  }
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpRoiAlignRotatedBackward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t top_grad_desc,
    const void *top_grad, const mluOpTensorDescriptor_t rois_desc,
    const void *rois, const int pooled_height, const int pooled_width,
    const int sample_ratio, const float spatial_scale, const bool aligned,
    const bool clockwise, const mluOpTensorDescriptor_t bottom_grad_desc,
    void *bottom_grad) {
  const std::string API = "[mluOpRoiAlignRotatedBackward]";

  PARAM_CHECK(API, handle != nullptr);
  PARAM_CHECK(API, top_grad_desc != nullptr);
  PARAM_CHECK(API, rois_desc != nullptr);
  PARAM_CHECK(API, bottom_grad_desc != nullptr);

  PARAM_CHECK(API, top_grad_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(API, bottom_grad_desc->layout == MLUOP_LAYOUT_NHWC);

  PARAM_CHECK(API, (top_grad_desc->dtype == bottom_grad_desc->dtype) &&
                       (bottom_grad_desc->dtype == rois_desc->dtype));
  PARAM_CHECK(API, bottom_grad_desc->dtype == MLUOP_DTYPE_FLOAT ||
                       bottom_grad_desc->dtype == MLUOP_DTYPE_HALF);

  PARAM_CHECK_EQ(API, rois_desc->dim, 2);
  PARAM_CHECK_EQ(API, top_grad_desc->dim, 4);
  PARAM_CHECK_EQ(API, bottom_grad_desc->dim, 4);
  PARAM_CHECK_EQ(API, rois_desc->dims[1], 6);

  PARAM_CHECK_EQ(API, top_grad_desc->dims[1], pooled_height);
  PARAM_CHECK_EQ(API, top_grad_desc->dims[2], pooled_width);

  PARAM_CHECK_GT(API, bottom_grad_desc->dims[3], 0);
  PARAM_CHECK_GT(API, rois_desc->dims[0], 0);

  if (top_grad_desc->dims[0] != rois_desc->dims[0]) {
    LOG(ERROR) << API << " rois_desc batch = " << rois_desc->dims[0]
               << ", top_grad_desc batch = " << top_grad_desc->dims[0]
               << ". They should be the same.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (top_grad_desc->dims[3] != bottom_grad_desc->dims[3]) {
    LOG(ERROR) << API
               << " bottom_grad_desc channel = " << bottom_grad_desc->dims[3]
               << ", top_grad_desc channel = " << top_grad_desc->dims[3]
               << ". They should be the same.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  const int channel = bottom_grad_desc->dims[3];
  const int width = bottom_grad_desc->dims[2];
  const int height = bottom_grad_desc->dims[1];
  const int batch = bottom_grad_desc->dims[0];
  const int rois_nums = rois_desc->dims[0];
  mluOpDataType_t data_type = bottom_grad_desc->dtype;

  PARAM_CHECK_GT(API, pooled_height, 0);
  PARAM_CHECK_GT(API, pooled_width, 0);
  PARAM_CHECK_GE(API, spatial_scale, 0);
  PARAM_CHECK_GE(API, sample_ratio, 0);

  if (mluOpGetTensorElementNum(bottom_grad_desc) == 0) {
    VLOG(5) << "[mluOpRoiAlignRotatedBackward]] Skip zero element tensor";
    return MLUOP_STATUS_SUCCESS;
  }

  PARAM_CHECK(API, top_grad != nullptr);
  PARAM_CHECK(API, bottom_grad != nullptr);
  PARAM_CHECK(API, rois != nullptr);

  VLOG(5) << "pool_height: " << pooled_height << ",pool_width: " << pooled_width
          << ",channel: " << channel << ",roi nums: " << rois_nums << ".";
  VLOG(5) << "batch: " << batch << ",height: " << height << ",width: " << width
          << ".";

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("roi_align_rotated_backward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input1", top_grad, top_grad_desc, 10, 0);
    GEN_CASE_DATA_REAL(true, "input2", rois, rois_desc);
    GEN_CASE_DATA(false, "output1", bottom_grad, bottom_grad_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "roi_align_rotated_backward", "pooled_height",
                             pooled_height);
    GEN_CASE_OP_PARAM_SINGLE(1, "roi_align_rotated_backward", "pooled_width",
                             pooled_width);
    GEN_CASE_OP_PARAM_SINGLE(1, "roi_align_rotated_backward", "sample_ratio",
                             sample_ratio);
    GEN_CASE_OP_PARAM_SINGLE(1, "roi_align_rotated_backward", "spatial_scale",
                             spatial_scale);
    GEN_CASE_OP_PARAM_SINGLE(1, "roi_align_rotated_backward", "aligned",
                             aligned);
    GEN_CASE_OP_PARAM_SINGLE(2, "roi_align_rotated_backward", "clockwise",
                             clockwise);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }
  mluOpRoiAlignRotatedParams roiAlignRotatedParams{pooled_height, pooled_width,
                                                   sample_ratio,  spatial_scale,
                                                   aligned,       clockwise};

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, rois_nums * pooled_height * pooled_width, &k_dim, &k_type);
  VLOG(5) << "[mluOpRoiAlignRotatedBackward] launch kernel policyFunc["
          << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << "].";

  VLOG(5) << "mluopFill start.";
  const size_t fill_value = 0x0;
  MLUOP_CHECK(mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST, &fill_value,
                        bottom_grad_desc, bottom_grad));
  VLOG(5) << "mluopFill end.";

  if (top_grad_desc->dtype == MLUOP_DTYPE_FLOAT) {
    KERNEL_CHECK((mluOpBlockKernelRoiAlignRotatedBackwardFloat(
        k_dim, k_type, handle->queue, top_grad, rois, batch, height, width,
        channel, rois_nums, roiAlignRotatedParams, bottom_grad)));
    VLOG(5) << "Kernel mluOpBlockKernelRoiAlignRotatedForwardFloat.";
  } else {
    KERNEL_CHECK((mluOpBlockKernelRoiAlignRotatedBackwardHalf(
        k_dim, k_type, handle->queue, top_grad, rois, batch, height, width,
        channel, rois_nums, roiAlignRotatedParams, bottom_grad)));
    VLOG(5) << "Kernel mluOpBlockKernelRoiAlignRotatedForwardHalf.";
  }
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
