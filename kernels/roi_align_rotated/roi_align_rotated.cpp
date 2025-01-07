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
#include "roi_align_rotated.h"

#include <string>

#include "core/cnnl_helper.h"
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"

static void policyFunc(const mluOpHandle_t handle, const int bin_num,
                       cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  size_t core_num = handle->core_num_per_cluster;
  size_t cluster_num = mluop::runtime::getJobLimitCapability(handle) / core_num;
  *k_type = cnrtFuncTypeUnion1;
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

  PARAM_CHECK(API, features_desc->getLayout() == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(API, output_desc->getLayout() == MLUOP_LAYOUT_NHWC);

  PARAM_CHECK(API, features_desc->getDtype() == output_desc->getDtype());
  PARAM_CHECK(API, output_desc->getDtype() == rois_desc->getDtype());
  PARAM_CHECK(API, features_desc->getDtype() == MLUOP_DTYPE_FLOAT ||
                       features_desc->getDtype() == MLUOP_DTYPE_HALF);

  STRIDE_TENSOR_CHECK("[mluOpRoiAlignRotatedForward]:", features_desc,
                      "features_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpRoiAlignRotatedForward]:", rois_desc,
                      "rois_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpRoiAlignRotatedForward]:", output_desc,
                      "output_desc must be contiguous");

  PARAM_CHECK_EQ(API, rois_desc->getDim(), 2);
  PARAM_CHECK_EQ(API, output_desc->getDim(), 4);
  PARAM_CHECK_EQ(API, features_desc->getDim(), 4);
  PARAM_CHECK_EQ(API, rois_desc->getDimIndex(1), 6);

  PARAM_CHECK_EQ(API, output_desc->getDimIndex(1), pooled_height);
  PARAM_CHECK_EQ(API, output_desc->getDimIndex(2), pooled_width);

  PARAM_CHECK_GT(API, features_desc->getDimIndex(3), 0);
  PARAM_CHECK_GT(API, rois_desc->getDimIndex(0), 0);

  if (output_desc->getDimIndex(0) != rois_desc->getDimIndex(0)) {
    LOG(ERROR) << API << " rois_desc batch = " << rois_desc->getDimIndex(0)
               << ", output_desc batch = " << output_desc->getDimIndex(0)
               << ". They should be the same.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (output_desc->getDimIndex(3) != features_desc->getDimIndex(3)) {
    LOG(ERROR) << API
               << " features_desc channel = " << features_desc->getDimIndex(3)
               << ", output_desc channel = " << output_desc->getDimIndex(3)
               << ". They should be the same.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  const int channel = features_desc->getDimIndex(3);
  const int width = features_desc->getDimIndex(2);
  const int height = features_desc->getDimIndex(1);
  const int batch = features_desc->getDimIndex(0);
  const int rois_nums = rois_desc->getDimIndex(0);
  mluOpDataType_t data_type = features_desc->getDtype();

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
          << ",sample_ratio: " << sample_ratio << ".";

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("roi_align_rotated_forward", "ROI_ALIGN_ROTATED_FORWARD");
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
  mluOpRoiAlignRotatedParams roiAlignRotatedParams{
      aligned,      clockwise,    pooled_height,
      pooled_width, sample_ratio, spatial_scale};

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, rois_nums * pooled_height * pooled_width, &k_dim, &k_type);

  uint32_t sample_ratio_split = 3, channels_split = 1024;
  if (handle->arch >= MLUOP_MLU590 && channel <= channels_split &&
      (sample_ratio >= sample_ratio_split || sample_ratio <= 0)) {
    VLOG(5) << "[mluOpRoiAlignRotatedForwardVector] launch kernel policyFunc["
            << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << "].";
    CHECK_RETURN(API,
                 KernelRoiAlignRotatedForwardVector(
                     k_dim, k_type, handle->queue, features_desc->getDtype(),
                     features, rois, batch, height, width, channel, rois_nums,
                     roiAlignRotatedParams, output));
  } else {
    VLOG(5) << "[mluOpRoiAlignRotatedForward] launch kernel policyFunc["
            << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << "].";
    CHECK_RETURN(API,
                 KernelRoiAlignRotatedForward(
                     k_dim, k_type, handle->queue, features_desc->getDtype(),
                     features, rois, batch, height, width, channel, rois_nums,
                     roiAlignRotatedParams, output));
  }

  VLOG(5) << "Kernel KernelRoiAlignRotatedForward.";
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

  PARAM_CHECK(API, top_grad_desc->getLayout() == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(API, bottom_grad_desc->getLayout() == MLUOP_LAYOUT_NHWC);

  PARAM_CHECK(API,
              (top_grad_desc->getDtype() == bottom_grad_desc->getDtype()) &&
                  (bottom_grad_desc->getDtype() == rois_desc->getDtype()));
  PARAM_CHECK(API, bottom_grad_desc->getDtype() == MLUOP_DTYPE_FLOAT ||
                       bottom_grad_desc->getDtype() == MLUOP_DTYPE_HALF);

  STRIDE_TENSOR_CHECK("[mluOpRoiAlignRotatedBackward]:", top_grad_desc,
                      "top_grad_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpRoiAlignRotatedBackward]:", rois_desc,
                      "rois_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpRoiAlignRotatedBackward]:", bottom_grad_desc,
                      "bottom_grad_desc must be contiguous");

  PARAM_CHECK_EQ(API, rois_desc->getDim(), 2);
  PARAM_CHECK_EQ(API, top_grad_desc->getDim(), 4);
  PARAM_CHECK_EQ(API, bottom_grad_desc->getDim(), 4);
  PARAM_CHECK_EQ(API, rois_desc->getDimIndex(1), 6);

  PARAM_CHECK_EQ(API, top_grad_desc->getDimIndex(1), pooled_height);
  PARAM_CHECK_EQ(API, top_grad_desc->getDimIndex(2), pooled_width);

  PARAM_CHECK_GT(API, bottom_grad_desc->getDimIndex(3), 0);
  PARAM_CHECK_GT(API, rois_desc->getDimIndex(0), 0);

  if (top_grad_desc->getDimIndex(0) != rois_desc->getDimIndex(0)) {
    LOG(ERROR) << API << " rois_desc batch = " << rois_desc->getDimIndex(0)
               << ", top_grad_desc batch = " << top_grad_desc->getDimIndex(0)
               << ". They should be the same.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (top_grad_desc->getDimIndex(3) != bottom_grad_desc->getDimIndex(3)) {
    LOG(ERROR) << API << " bottom_grad_desc channel = "
               << bottom_grad_desc->getDimIndex(3)
               << ", top_grad_desc channel = " << top_grad_desc->getDimIndex(3)
               << ". They should be the same.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  const int channel = bottom_grad_desc->getDimIndex(3);
  const int width = bottom_grad_desc->getDimIndex(2);
  const int height = bottom_grad_desc->getDimIndex(1);
  const int batch = bottom_grad_desc->getDimIndex(0);
  const int rois_nums = rois_desc->getDimIndex(0);
  mluOpDataType_t data_type = bottom_grad_desc->getDtype();

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
    GEN_CASE_START("roi_align_rotated_backward", "ROI_ALIGN_ROTATED_BACKWARD");
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
  mluOpRoiAlignRotatedParams roiAlignRotatedParams{
      aligned,      clockwise,    pooled_height,
      pooled_width, sample_ratio, spatial_scale};

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, rois_nums * pooled_height * pooled_width, &k_dim, &k_type);
  VLOG(5) << "[mluOpRoiAlignRotatedBackward] launch kernel policyFunc["
          << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << "].";

  VLOG(5) << "cnnlFill_v3 start.";
  const size_t fill_value = 0x0;
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(bottom_grad_desc,
                                                 cnnl_output_desc);
    CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &fill_value,
                          cnnl_output_desc, bottom_grad));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  VLOG(5) << "cnnlFill_v3 end.";

  CHECK_RETURN(API, KernelRoiAlignRotatedBackward(
                        k_dim, k_type, handle->queue, top_grad_desc->getDtype(),
                        top_grad, rois, batch, height, width, channel,
                        rois_nums, roiAlignRotatedParams, bottom_grad));
  VLOG(5) << "Kernel KernelRoiAlignRotatedBackward.";
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
