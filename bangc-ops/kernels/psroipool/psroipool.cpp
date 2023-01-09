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
                                cnrtFunctionType_t *k_type, const int nums) {
  size_t union_number = mluop::runtime::getClusterLimitCapability(handle);
  size_t core_in_cluster = handle->core_num_per_cluster;
  *k_type = CNRT_FUNC_TYPE_UNION1;  // default func type
  k_dim->x = core_in_cluster;
  uint32_t use_cluster = (nums + core_in_cluster - 1) / core_in_cluster;
  k_dim->y = use_cluster > union_number ? union_number : use_cluster;
  k_dim->z = 1;
}

static mluOpStatus_t psRoiPoolForwardParamCheck(
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
  PARAM_CHECK(api, group_size == output_desc->dims[1]);
  PARAM_CHECK(api, output_desc->dims[1] == output_desc->dims[2]);
  PARAM_CHECK(api, group_size >= 1);
  PARAM_CHECK(api, output_desc->dims[3] >= 1);
  PARAM_CHECK(api, spatial_scale > 0);
  PARAM_CHECK(api, rois_desc->dims[1] == 5);
  // roi_num check
  PARAM_CHECK(api, output_desc->dims[0] == rois_desc->dims[0]);
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
  if ((mluOpGetTensorElementNum(output_desc) *
           mluop::getSizeOfDataType(output_desc->dtype) >=
       LARGE_TENSOR_SIZE) ||
      (mluOpGetTensorElementNum(input_desc) *
           mluop::getSizeOfDataType(input_desc->dtype) >=
       LARGE_TENSOR_SIZE) ||
      (mluOpGetTensorElementNum(rois_desc) *
           mluop::getSizeOfDataType(rois_desc->dtype) >=
       LARGE_TENSOR_SIZE) ||
      (mluOpGetTensorElementNum(mapping_channel_desc) *
           mluop::getSizeOfDataType(mapping_channel_desc->dtype) >=
       LARGE_TENSOR_SIZE)) {
    LOG(ERROR) << api << " Overflow max tensor size."
               << " Currently, MLU-OPS supports tensor size smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  const size_t max_input_num = 2147483648;  // 2^31 2G num
  if ((mluOpGetTensorElementNum(output_desc) >= max_input_num) ||
      (mluOpGetTensorElementNum(input_desc) >= max_input_num) ||
      (mluOpGetTensorElementNum(rois_desc) >= max_input_num) ||
      (mluOpGetTensorElementNum(mapping_channel_desc) >= max_input_num)) {
    LOG(ERROR) << api << " Overflow max tensor num."
               << " Currently, MLU-OPS supports tensor num smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (mluOpGetTensorElementNum(input_desc) == 0) {
    VLOG(5) << api << " Input skip zero element tensor.";
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

static mluOpStatus_t psRoiPoolBackwardParamCheck(
    const std::string &api, const mluOpHandle_t handle, const int pooled_height,
    const int pooled_width, const float spatial_scale, const int output_dim,
    const void *top_grad, const void *rois, const void *mapping_channel,
    const void *bottom_grad, const mluOpTensorDescriptor_t top_grad_desc,
    const mluOpTensorDescriptor_t rois_desc,
    const mluOpTensorDescriptor_t mapping_channel_desc,
    const mluOpTensorDescriptor_t bottom_grad_desc) {
  PARAM_CHECK(api, handle != NULL);
  PARAM_CHECK(api, top_grad_desc != NULL);
  PARAM_CHECK(api, rois_desc != NULL);
  PARAM_CHECK(api, mapping_channel_desc != NULL);
  PARAM_CHECK(api, bottom_grad_desc != NULL);
  PARAM_CHECK(api, top_grad_desc->dim == 4);
  PARAM_CHECK(api, rois_desc->dim == 2);
  PARAM_CHECK(api, mapping_channel_desc->dim == 4);
  PARAM_CHECK(api, bottom_grad_desc->dim == 4);
  // check the input and output datatype
  PARAM_CHECK(api, top_grad_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(api, rois_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(api, mapping_channel_desc->dtype == MLUOP_DTYPE_INT32);
  PARAM_CHECK(api, bottom_grad_desc->dtype == MLUOP_DTYPE_FLOAT);
  // check layout
  PARAM_CHECK(api, top_grad_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(api, mapping_channel_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(api, bottom_grad_desc->layout == MLUOP_LAYOUT_NHWC);
  // param check
  PARAM_CHECK(api, pooled_height == top_grad_desc->dims[1]);
  PARAM_CHECK(api, pooled_width == top_grad_desc->dims[2]);
  PARAM_CHECK(api, output_dim == top_grad_desc->dims[3]);
  PARAM_CHECK(api, top_grad_desc->dims[1] == top_grad_desc->dims[2]);
  PARAM_CHECK(api, top_grad_desc->dims[3] >= 1);
  PARAM_CHECK(api, spatial_scale > 0);
  PARAM_CHECK(api, rois_desc->dims[1] == 5);
  // roi_num check
  PARAM_CHECK(api, top_grad_desc->dims[0] == rois_desc->dims[0]);
  PARAM_CHECK(api, bottom_grad_desc->dims[3] ==
                       output_dim * pooled_width * pooled_height);
  for (int i = 0; i < top_grad_desc->dim; ++i) {
    if (top_grad_desc->dims[i] != mapping_channel_desc->dims[i]) {
      LOG(ERROR) << api << " Check failed: top_grad_desc->dims[" << i
                 << "] should be equal to mapping_channel_desc->dims[" << i
                 << "].";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  if ((mluOpGetTensorElementNum(top_grad_desc) *
           mluop::getSizeOfDataType(top_grad_desc->dtype) >=
       LARGE_TENSOR_SIZE) ||
      (mluOpGetTensorElementNum(bottom_grad_desc) *
           mluop::getSizeOfDataType(bottom_grad_desc->dtype) >=
       LARGE_TENSOR_SIZE) ||
      (mluOpGetTensorElementNum(rois_desc) *
           mluop::getSizeOfDataType(rois_desc->dtype) >=
       LARGE_TENSOR_SIZE) ||
      (mluOpGetTensorElementNum(mapping_channel_desc) *
           mluop::getSizeOfDataType(mapping_channel_desc->dtype) >=
       LARGE_TENSOR_SIZE)) {
    LOG(ERROR) << api << " Overflow max tensor size."
               << " Currently, MLU-OPS supports tensor size smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

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
  mluOpStatus_t ret = psRoiPoolForwardParamCheck(
      api, handle, pooled_height, pooled_width, spatial_scale, group_size,
      output_dim, input, rois, output, mapping_channel, input_desc, rois_desc,
      output_desc, mapping_channel_desc);
  if (ret != MLUOP_STATUS_SUCCESS) {
    LOG(ERROR) << api
               << " Error found during element verification, please check.";
    return ret;
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

mluOpStatus_t MLUOP_WIN_API mluOpPsRoiPoolBackward(
    mluOpHandle_t handle, const int pooled_height, const int pooled_width,
    const float spatial_scale, const int output_dim,
    const mluOpTensorDescriptor_t top_grad_desc, const void *top_grad,
    const mluOpTensorDescriptor_t rois_desc, const void *rois,
    const mluOpTensorDescriptor_t mapping_channel_desc,
    const void *mapping_channel, const mluOpTensorDescriptor_t bottom_grad_desc,
    void *bottom_grad) {
  const std::string api = "[mluOpPsRoiPoolBackward]";
  mluOpStatus_t ret = psRoiPoolBackwardParamCheck(
      api, handle, pooled_height, pooled_width, spatial_scale, output_dim,
      top_grad, rois, mapping_channel, bottom_grad, top_grad_desc, rois_desc,
      mapping_channel_desc, bottom_grad_desc);
  if (ret != MLUOP_STATUS_SUCCESS) {
    LOG(ERROR) << api
               << " Error found during element verification, please check.";
    return ret;
  }

  if (mluOpGetTensorElementNum(rois_desc) == 0) {
    LOG(ERROR) << api << " Roi_data can not be zero element tensor.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (mluOpGetTensorElementNum(top_grad_desc) == 0 ||
      mluOpGetTensorElementNum(mapping_channel_desc) == 0 ||
      mluOpGetTensorElementNum(bottom_grad_desc) == 0) {
    VLOG(5) << api << " Input skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  PARAM_CHECK(api, top_grad != NULL);
  PARAM_CHECK(api, rois != NULL);
  PARAM_CHECK(api, bottom_grad != NULL);
  PARAM_CHECK(api, mapping_channel != NULL);

  const int batch_size = bottom_grad_desc->dims[0];
  const int height = bottom_grad_desc->dims[1];
  const int width = bottom_grad_desc->dims[2];
  const int channels = bottom_grad_desc->dims[3];
  const int rois_sum = rois_desc->dims[0];
  const int rois_offset = rois_desc->dims[1];

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("psroipool_backward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", top_grad, top_grad_desc, 1, 0);
    GEN_CASE_DATA_REAL(true, "mapping_channel", mapping_channel,
                       mapping_channel_desc);
    GEN_CASE_DATA_REAL(true, "rois", rois, rois_desc);
    GEN_CASE_DATA(false, "output", bottom_grad, bottom_grad_desc, 0, 0);

    GEN_CASE_OP_PARAM_SINGLE(0, "psroipool_backward", "output_dim", output_dim);
    GEN_CASE_OP_PARAM_SINGLE(1, "psroipool_backward", "pooled_height",
                             pooled_height);
    GEN_CASE_OP_PARAM_SINGLE(1, "psroipool_backward", "pooled_width",
                             pooled_width);
    GEN_CASE_OP_PARAM_SINGLE(2, "psroipool_backward", "spatial_scale",
                             spatial_scale);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  const int nums = rois_sum * pooled_height * pooled_width;
  policyFuncPsRoiPool(handle, &k_dim, &k_type, nums);
  VLOG(5) << api << " Launch [" << k_type << ", " << k_dim.x << ", " << k_dim.y
          << ", " << k_dim.z << "].";

  // gdram set zero
  int gdramset_size = channels * width * height * batch_size * sizeof(float);
  KERNEL_CHECK((mluOpBlockKernelFillZeroByte(k_dim, k_type, handle->queue,
                                             gdramset_size, bottom_grad)));
  VLOG(5) << "Kernel mluOpBlockKernelFillZero.";

  KERNEL_CHECK((mluOpBlockKernelPsRoiPoolBackwardFloat(
      k_dim, k_type, handle->queue, top_grad, mapping_channel, rois,
      bottom_grad, batch_size, height, width, channels, pooled_height,
      pooled_width, output_dim, rois_sum, rois_offset, spatial_scale)));
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
