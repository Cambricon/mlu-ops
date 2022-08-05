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

// policy function
static void policyFuncPsRoiPool(const mluOpHandle_t handle, cnrtDim3_t *k_dim,
                                cnrtFunctionType_t *k_type,
                                const int rois_num) {
  size_t union_number = mluop::runtime::getClusterLimitCapability(handle);
  size_t core_in_cluster = handle->core_num_per_cluster;
  uint32_t use_cluster = (rois_num + core_in_cluster - 1) / core_in_cluster;
  *k_type = CNRT_FUNC_TYPE_UNION1;  // default func type
  k_dim->x = core_in_cluster;
  k_dim->y = use_cluster > core_in_cluster ? core_in_cluster : use_cluster;
  k_dim->z = 1;
}

static mluOpStatus_t paramCheck(
    const std::string &api, const mluOpHandle_t handle, const int pooled_height,
    const int pooled_width, const float spatial_scale, const int group_size,
    const int output_dim, const void *input, const void *rois,
    const void *output, const void *mapping_channel,
    const mluOpTensorDescriptor_t input_desc,
    const mluOpTensorDescriptor_t rois_desc,
    const mluOpTensorDescriptor_t output_desc,
    const mluOpTensorDescriptor_t mapping_channel_desc) {
  PARAM_CHECK(api, handle != NULL);
  PARAM_CHECK(api, input_desc != NULL);
  PARAM_CHECK(api, rois_desc != NULL);
  PARAM_CHECK(api, output_desc != NULL);
  PARAM_CHECK(api, mapping_channel_desc != NULL);
  PARAM_CHECK(api, input_desc->dim == 4);
  PARAM_CHECK(api, rois_desc->dim == 2);
  PARAM_CHECK(api, output_desc->dim == 4);
  PARAM_CHECK(api, mapping_channel_desc->dim == 4);
  // check the input and output datatype
  PARAM_CHECK(api, input_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(api, rois_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(api, output_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(api, mapping_channel_desc->dtype == MLUOP_DTYPE_INT32);
  // check layout
  PARAM_CHECK(api, input_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(api, output_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(api, mapping_channel_desc->layout == MLUOP_LAYOUT_NHWC);
  // param check
  PARAM_CHECK(api, pooled_height == output_desc->dims[1]);
  PARAM_CHECK(api, pooled_width == output_desc->dims[2]);
  PARAM_CHECK(api, output_dim == output_desc->dims[3]);
  // group_size == pooled_height
  PARAM_CHECK(api, group_size == output_desc->dims[1]);
  // pooled_height == pooled_width
  PARAM_CHECK(api, output_desc->dims[1] == output_desc->dims[2]);
  // group_size >= 1.
  PARAM_CHECK(api, group_size >= 1);
  // output_dim >= 1.
  PARAM_CHECK(api, output_desc->dims[3] >= 1);
  // spatial_scale > 0
  PARAM_CHECK(api, spatial_scale > 0);
  // rois_offset = 5.
  PARAM_CHECK(api, rois_desc->dims[1] == 5);
  // roi_num check
  PARAM_CHECK(api, output_desc->dims[0] == rois_desc->dims[0]);
  // channels == pooled_height * pooled_width * output_dim
  PARAM_CHECK(api, input_desc->dims[3] == output_desc->dims[1] *
                                              output_desc->dims[2] *
                                              output_desc->dims[3]);
  for (int i = 0; i < output_desc->dim; ++i) {
    if (output_desc->dims[i] != mapping_channel_desc->dims[i]) {
      LOG(ERROR) << api << " Check failed: output_desc->dims[" << i
                 << "] should be equal to mapping_channel_desc->dims[" << i
                 << "].";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  const size_t max_input_num = 2147483648;  // 2^31 2G num
  if ((mluOpGetTensorElementNum(output_desc) >= max_input_num) ||
      (mluOpGetTensorElementNum(input_desc) >= max_input_num) ||
      (mluOpGetTensorElementNum(rois_desc) >= max_input_num)) {
    LOG(ERROR) << api << " Overflow max tensor num."
               << " Currently, MLU-OPS supports tensor num smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (mluOpGetTensorElementNum(input_desc) == 0) {
    VLOG(5) << api << " input skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }
  if (mluOpGetTensorElementNum(rois_desc) == 0) {
    LOG(ERROR) << api << " Roi_data can not be zero element tensor.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  PARAM_CHECK(api, input != NULL);
  PARAM_CHECK(api, rois != NULL);
  PARAM_CHECK(api, output != NULL);
  PARAM_CHECK(api, mapping_channel != NULL);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpPsRoiPoolForward(
    mluOpHandle_t handle, const int pooled_height, const int pooled_width,
    const float spatial_scale, const int group_size, const int output_dim,
    const mluOpTensorDescriptor_t input_desc, const void *input,
    const mluOpTensorDescriptor_t rois_desc, const void *rois,
    const mluOpTensorDescriptor_t output_desc, void *output,
    const mluOpTensorDescriptor_t mapping_channel_desc, void *mapping_channel) {
  const std::string api = "[mluOpPsRoiPoolForward]";
  mluOpStatus_t ret =
      paramCheck(api, handle, pooled_height, pooled_width, spatial_scale,
                 group_size, output_dim, input, rois, output, mapping_channel,
                 input_desc, rois_desc, output_desc, mapping_channel_desc);
  if (ret != MLUOP_STATUS_SUCCESS) {
    LOG(ERROR) << api
               << " Error found during element verification, please check.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  const int batch_size = input_desc->dims[0];
  const int height = input_desc->dims[1];
  const int width = input_desc->dims[2];
  const int channels = input_desc->dims[3];
  const int rois_sum = output_desc->dims[0];
  const int rois_offset = rois_desc->dims[1];

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("psroipool_forward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", input, input_desc, 1, 0);
    GEN_CASE_DATA_REAL(true, "rois", rois, rois_desc);
    GEN_CASE_DATA(false, "output", output, output_desc, 0, 0);
    GEN_CASE_DATA(false, "mapping_channel", mapping_channel,
                  mapping_channel_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "psroipool_forward", "output_dim", output_dim);
    GEN_CASE_OP_PARAM_SINGLE(1, "psroipool_forward", "pooled_height",
                             pooled_height);
    GEN_CASE_OP_PARAM_SINGLE(1, "psroipool_forward", "pooled_width",
                             pooled_width);
    GEN_CASE_OP_PARAM_SINGLE(1, "psroipool_forward", "spatial_scale",
                             spatial_scale);
    GEN_CASE_OP_PARAM_SINGLE(2, "psroipool_forward", "group_size", group_size);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFuncPsRoiPool(handle, &k_dim, &k_type, rois_sum);
  VLOG(5) << api << " Launch [" << k_type << ", " << k_dim.x << ", " << k_dim.y
          << ", " << k_dim.z << "].";
  KERNEL_CHECK((mluOpBlockKernelPsRoiPoolForwardFloat(
      k_dim, k_type, handle->queue, input, rois, output, mapping_channel,
      batch_size, height, width, channels, pooled_height, pooled_width,
      output_dim, group_size, rois_sum, rois_offset, spatial_scale)));
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
