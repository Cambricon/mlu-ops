/*******************************************************************************
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
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *******************************************************************************/
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"

// policy function
void policyFunc(const mluOpHandle_t handle,
                const mluOpTensorDescriptor_t output_desc, cnrtDim3_t *k_dim,
                cnrtFunctionType_t *k_type) {
  const size_t cluster_limit =
      mluop::runtime::getClusterLimitCapability(handle);
  const size_t core_limit =
      mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  const size_t num_rois = output_desc->dims[0];
  const size_t pooled_height = output_desc->dims[1];
  const size_t pooled_width = output_desc->dims[2];
  const size_t num_bins =
      CEIL_ALIGN(num_rois * pooled_height * pooled_width, core_limit);
  k_dim->x = core_limit;
  k_dim->y = (num_bins / core_limit) > cluster_limit ? cluster_limit
                                                     : (num_bins / core_limit);
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
}

static mluOpStatus_t DeformRoiPoolForwardPreCheck(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const mluOpTensorDescriptor_t rois_desc,
    const mluOpTensorDescriptor_t offset_desc,
    const mluOpTensorDescriptor_t output_desc, const int pooled_height,
    const int pooled_width) {
  PARAM_CHECK("[mluOpDeformRoiPoolForward]",
              input_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]",
              output_desc->layout == MLUOP_LAYOUT_NHWC);

  PARAM_CHECK("[mluOpDeformRoiPoolForward]",
              input_desc->dtype == MLUOP_DTYPE_FLOAT ||
                  input_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]",
              input_desc->dtype == rois_desc->dtype);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]",
              input_desc->dtype == output_desc->dtype);

  PARAM_CHECK("[mluOpDeformRoiPoolForward]", rois_desc->dim == 2);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]", rois_desc->dims[1] == 5);

  PARAM_CHECK("[mluOpDeformRoiPoolForward]", pooled_height > 0);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]", pooled_width > 0);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]",
              output_desc->dims[1] == pooled_height);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]",
              output_desc->dims[2] == pooled_width);

  if (offset_desc != NULL) {
    PARAM_CHECK("[mluOpDeformRoiPoolForward]",
                offset_desc->dtype == input_desc->dtype);
    PARAM_CHECK("[mluOpDeformRoiPoolForward]", offset_desc->dim == 4);
    PARAM_CHECK("[mluOpDeformRoiPoolForward]",
                offset_desc->dims[0] == rois_desc->dims[0]);
    PARAM_CHECK("[mluOpDeformRoiPoolForward]", offset_desc->dims[1] == 2);
    PARAM_CHECK("[mluOpDeformRoiPoolForward]",
                offset_desc->dims[2] == pooled_height);
    PARAM_CHECK("[mluOpDeformRoiPoolForward]",
                offset_desc->dims[3] == pooled_width);
    const size_t offset_element_num = mluOpGetTensorElementNum(offset_desc);
    TENSOR_NUM_CHECK("[mluOpDeformRoiPoolForward]", offset_element_num,
                     LARGE_TENSOR_NUM, "");
  }
  if (rois_desc->dims[0] != output_desc->dims[0]) {
    LOG(ERROR) << "[mluOpDeformRoiPoolForward] rois number = "
               << rois_desc->dims[0]
               << ", output batch = " << output_desc->dims[0]
               << ", they should be equal.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (input_desc->dims[3] != output_desc->dims[3]) {
    LOG(ERROR) << "[mluOpDeformRoiPoolForward] input channel = "
               << input_desc->dims[3]
               << ", output channel = " << output_desc->dims[3]
               << ", they should be equal.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  const size_t input_element_num = mluOpGetTensorElementNum(input_desc);
  const size_t rois_element_num = mluOpGetTensorElementNum(rois_desc);
  const size_t output_element_num = mluOpGetTensorElementNum(output_desc);
  TENSOR_NUM_CHECK("[mluOpDeformRoiPoolForward]", input_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpDeformRoiPoolForward]", rois_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpDeformRoiPoolForward]", output_element_num,
                   LARGE_TENSOR_NUM, "");
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t DeformRoiPoolBackwardPreCheck(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t grad_output_desc,
    const mluOpTensorDescriptor_t input_desc,
    const mluOpTensorDescriptor_t rois_desc, const void *offset,
    const mluOpTensorDescriptor_t offset_desc,
    const mluOpTensorDescriptor_t grad_input_desc, const void *grad_offset,
    const mluOpTensorDescriptor_t grad_offset_desc, const int pooled_height,
    const int pooled_width) {
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              grad_output_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              input_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              grad_input_desc->layout == MLUOP_LAYOUT_NHWC);

  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              input_desc->dtype == MLUOP_DTYPE_FLOAT ||
                  input_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              input_desc->dtype == grad_output_desc->dtype);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              input_desc->dtype == rois_desc->dtype);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              input_desc->dtype == grad_input_desc->dtype);

  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", rois_desc->dim == 2);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", rois_desc->dims[1] == 5);

  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", pooled_height > 0);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", pooled_width > 0);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              grad_output_desc->dims[1] == pooled_height);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              grad_output_desc->dims[2] == pooled_width);

  for (int i = 0; i < input_desc->dim; ++i) {
    if (input_desc->dims[i] != grad_input_desc->dims[i]) {
      LOG(ERROR) << "[mluOpDeformRoiPoolBackward] input's shape is ["
                 << input_desc->dims[0] << " " << input_desc->dims[1] << " "
                 << input_desc->dims[2] << " " << input_desc->dims[3]
                 << "], grad_input's shape is [" << grad_input_desc->dims[0]
                 << " " << grad_input_desc->dims[1] << " "
                 << grad_input_desc->dims[2] << " " << grad_input_desc->dims[3]
                 << "]. They should be the same.";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  if (offset_desc == NULL && offset != NULL) {
    LOG(ERROR) << "[mluOpDeformRoiPoolBackward] offset_desc is NULL, but "
                  "offset is not NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (offset_desc != NULL && offset == NULL) {
    LOG(ERROR) << "[mluOpDeformRoiPoolBackward] offset_desc is not NULL, but "
                  "offset is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (grad_offset_desc == NULL && grad_offset != NULL) {
    LOG(ERROR) << "[mluOpDeformRoiPoolBackward] grad_offset_desc is NULL, but "
                  "grad_offset is not NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (grad_offset_desc != NULL && grad_offset == NULL) {
    LOG(ERROR) << "[mluOpDeformRoiPoolBackward] grad_offset_desc is not NULL, "
                  "but grad_offset is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (offset != NULL && grad_offset == NULL) {
    LOG(ERROR) << "[mluOpDeformRoiPoolBackward] offset is not NULL, but "
                  "grad_offset is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  } else if (offset == NULL && grad_offset != NULL) {
    LOG(ERROR) << "[mluOpDeformRoiPoolBackward] offset is NULL, but "
                  "grad_offset is not NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (offset_desc != NULL) {
    if (grad_offset_desc == NULL) {
      LOG(ERROR) << "[mluOpDeformRoiPoolBackward] offset_desc is not NULL, but "
                    "grad_offset_desc is NULL.";
      return MLUOP_STATUS_BAD_PARAM;
    }
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
                offset_desc->dtype == input_desc->dtype);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]", offset_desc->dim == 4);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
                offset_desc->dims[0] == rois_desc->dims[0]);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]", offset_desc->dims[1] == 2);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
                offset_desc->dims[2] == pooled_height);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
                offset_desc->dims[3] == pooled_width);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
                grad_offset_desc->dtype == input_desc->dtype);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]", grad_offset_desc->dim == 4);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
                grad_offset_desc->dims[0] == rois_desc->dims[0]);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]", grad_offset_desc->dims[1] == 2);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
                grad_offset_desc->dims[2] == pooled_height);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
                grad_offset_desc->dims[3] == pooled_width);
    const size_t offset_element_num = mluOpGetTensorElementNum(offset_desc);
    const size_t grad_offset_element_num =
        mluOpGetTensorElementNum(grad_offset_desc);
    TENSOR_NUM_CHECK("[mluOpDeformRoiPoolBackward]", offset_element_num,
                     LARGE_TENSOR_NUM, "");
    TENSOR_NUM_CHECK("[mluOpDeformRoiPoolBackward]", grad_offset_element_num,
                     LARGE_TENSOR_NUM, "");
  } else {
    if (grad_offset_desc != NULL) {
      LOG(ERROR) << "[mluOpDeformRoiPoolBackward] offset_desc is NULL, but "
                    "grad_offset_desc is not NULL.";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  if (rois_desc->dims[0] != grad_output_desc->dims[0]) {
    LOG(ERROR) << "[mluOpDeformRoiPoolBackward] rois number = "
               << rois_desc->dims[0]
               << ", grad_output batch = " << grad_output_desc->dims[0]
               << ", they should be equal.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (input_desc->dims[3] != grad_output_desc->dims[3]) {
    LOG(ERROR) << "[mluOpDeformRoiPoolBackward] input channel = "
               << input_desc->dims[3]
               << ", output channel = " << grad_output_desc->dims[3]
               << ", they should be equal.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  const size_t grad_output_element_num =
      mluOpGetTensorElementNum(grad_output_desc);
  const size_t input_element_num = mluOpGetTensorElementNum(input_desc);
  const size_t rois_element_num = mluOpGetTensorElementNum(rois_desc);
  const size_t grad_input_element_num =
      mluOpGetTensorElementNum(grad_input_desc);
  TENSOR_NUM_CHECK("[mluOpDeformRoiPoolBackward]", grad_output_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpDeformRoiPoolBackward]", input_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpDeformRoiPoolBackward]", rois_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpDeformRoiPoolBackward]", grad_input_element_num,
                   LARGE_TENSOR_NUM, "");

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpDeformRoiPoolForward(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, const mluOpTensorDescriptor_t rois_desc,
    const void *rois, const mluOpTensorDescriptor_t offset_desc,
    const void *offset, const int pooled_height, const int pooled_width,
    const float spatial_scale, const int sampling_ratio, const float gamma,
    const mluOpTensorDescriptor_t output_desc, void *output) {
  PARAM_CHECK("[mluOpDeformRoiPoolForward]", handle != NULL);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]", input_desc != NULL);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]", rois_desc != NULL);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]", output_desc != NULL);

  mluOpStatus_t status = MLUOP_STATUS_BAD_PARAM;
  status =
      DeformRoiPoolForwardPreCheck(handle, input_desc, rois_desc, offset_desc,
                                   output_desc, pooled_height, pooled_width);
  if (MLUOP_STATUS_SUCCESS != status) {
    return status;
  }
  if (offset_desc == NULL && offset != NULL) {
    LOG(ERROR) << "[mluOpDeformRoiPoolForward] offset_desc is NULL, but offset "
                  "is not NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (offset_desc != NULL && offset == NULL) {
    LOG(ERROR) << "[mluOpDeformRoiPoolForward] offset_desc is not NULL, but "
                  "offset is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (input_desc->dims[0] == 0 || mluOpGetTensorElementNum(rois_desc) == 0 ||
      mluOpGetTensorElementNum(output_desc) == 0) {
    LOG(ERROR) << "[mluOpDeformRoiPoolForward] Zero element tensor failure";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (mluOpGetTensorElementNum(input_desc) == 0 ||
      mluOpGetTensorElementNum(output_desc) == 0) {
    VLOG(5) << "[mluOpDeformRoiPoolForward] Skip zero element tensor";
    return MLUOP_STATUS_SUCCESS;
  }
  PARAM_CHECK("[mluOpDeformRoiPoolForward]", input != NULL);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]", rois != NULL);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]", output != NULL);

  // generate mluOpDeformRoiPoolForward prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("deform_roi_pool_forward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", input, input_desc, -10, 10);
    GEN_CASE_DATA_REAL(true, "rois", rois, rois_desc);
    if (offset != NULL) {
      GEN_CASE_DATA_REAL(true, "offset", offset, offset_desc);
    }
    GEN_CASE_DATA(false, "output", output, output_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "deform_roi_pool_forward", "spatial_scale",
                             spatial_scale);
    GEN_CASE_OP_PARAM_SINGLE(1, "deform_roi_pool_forward", "sampling_ratio",
                             sampling_ratio);
    GEN_CASE_OP_PARAM_SINGLE(1, "deform_roi_pool_forward", "gamma", gamma);
    GEN_CASE_OP_PARAM_SINGLE(1, "deform_roi_pool_forward", "pooled_height",
                             pooled_height);
    GEN_CASE_OP_PARAM_SINGLE(2, "deform_roi_pool_forward", "pooled_width",
                             pooled_width);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }
  // generate mluOpDeformRoiPoolForward prototxt end!

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  policyFunc(handle, output_desc, &k_dim, &k_type);

  const int batches = input_desc->dims[0];
  const int height = input_desc->dims[1];
  const int width = input_desc->dims[2];
  const int channels = input_desc->dims[3];
  const int num_rois = output_desc->dims[0];
  mluOpDataType_t data_dtype = input_desc->dtype;
  VLOG(5) << "[mluOpDeformRoiPoolForward] Launch kernel policyFunc[" << k_dim.x
          << ", " << k_dim.y << ", " << k_dim.z << "].";
  switch (data_dtype) {
    case MLUOP_DTYPE_HALF: {
      KERNEL_CHECK((MLUUnion1DeformRoiPoolForwardHalf(
          k_dim, k_type, handle->queue, input, rois, offset, output, batches,
          channels, height, width, num_rois, pooled_height, pooled_width,
          spatial_scale, sampling_ratio, gamma)));
    }; break;
    case MLUOP_DTYPE_FLOAT: {
      KERNEL_CHECK((MLUUnion1DeformRoiPoolForwardFloat(
          k_dim, k_type, handle->queue, input, rois, offset, output, batches,
          channels, height, width, num_rois, pooled_height, pooled_width,
          spatial_scale, sampling_ratio, gamma)));
    }; break;
    default: {
      VLOG(5) << "Input Date Type Not Supported. Only support half and float !";
    }
  }
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpDeformRoiPoolBackward(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t grad_output_desc,
    const void *grad_output, const mluOpTensorDescriptor_t input_desc,
    const void *input, const mluOpTensorDescriptor_t rois_desc,
    const void *rois, const mluOpTensorDescriptor_t offset_desc,
    const void *offset, const int pooled_height, const int pooled_width,
    const float spatial_scale, const int sampling_ratio, const float gamma,
    const mluOpTensorDescriptor_t grad_input_desc, void *grad_input,
    const mluOpTensorDescriptor_t grad_offset_desc, void *grad_offset) {
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", handle != NULL);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", grad_output_desc != NULL);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", input_desc != NULL);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", rois_desc != NULL);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", grad_input_desc != NULL);

  mluOpStatus_t status = MLUOP_STATUS_BAD_PARAM;
  status = DeformRoiPoolBackwardPreCheck(
      handle, grad_output_desc, input_desc, rois_desc, offset, offset_desc,
      grad_input_desc, grad_offset, grad_offset_desc, pooled_height,
      pooled_width);
  if (MLUOP_STATUS_SUCCESS != status) {
    return status;
  }

  if (mluOpGetTensorElementNum(grad_output_desc) == 0 ||
      input_desc->dims[0] == 0 || mluOpGetTensorElementNum(rois_desc) == 0) {
    LOG(ERROR) << "[mluOpDeformRoiPoolBackward] Zero element tensor failure";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (mluOpGetTensorElementNum(input_desc) == 0) {
    VLOG(5) << "[mluOpDeformRoiPoolBackward] Skip zero element tensor";
    return MLUOP_STATUS_SUCCESS;
  }

  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", grad_output != NULL);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", input != NULL);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", rois != NULL);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", grad_input != NULL);

  // generate mluOpDeformRoiPoolBackward prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("deform_roi_pool_backward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "grad_output", grad_output, grad_output_desc, -10, 10);
    GEN_CASE_DATA(true, "input", input, input_desc, -10, 10);
    GEN_CASE_DATA_REAL(true, "rois", rois, rois_desc);
    if (offset != NULL) {
      GEN_CASE_DATA_REAL(true, "offset", offset, offset_desc);
    }
    GEN_CASE_DATA(false, "grad_input", grad_input, grad_input_desc, 0, 0);
    if (grad_offset != NULL) {
      GEN_CASE_DATA(false, "grad_offset", grad_offset, grad_offset_desc, 0, 0);
    }
    GEN_CASE_OP_PARAM_SINGLE(0, "deform_roi_pool_backward", "spatial_scale",
                             spatial_scale);
    GEN_CASE_OP_PARAM_SINGLE(1, "deform_roi_pool_backward", "sampling_ratio",
                             sampling_ratio);
    GEN_CASE_OP_PARAM_SINGLE(1, "deform_roi_pool_backward", "gamma", gamma);
    GEN_CASE_OP_PARAM_SINGLE(1, "deform_roi_pool_backward", "pooled_height",
                             pooled_height);
    GEN_CASE_OP_PARAM_SINGLE(2, "deform_roi_pool_backward", "pooled_width",
                             pooled_width);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }
  // generate mluOpDeformRoiPoolBackward prototxt end!
  VLOG(5) << "[mluOpDeformRoiPoolBackward] mluOpFill start.";
  const uint32_t fill_value = 0x00;
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST, &fill_value,
                               grad_input_desc, grad_input));
  if (offset != NULL) {
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
                MLUOP_STATUS_SUCCESS ==
                    mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST, &fill_value,
                                 grad_offset_desc, grad_offset));
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  policyFunc(handle, grad_output_desc, &k_dim, &k_type);

  const int batches = input_desc->dims[0];
  const int height = input_desc->dims[1];
  const int width = input_desc->dims[2];
  const int channels = input_desc->dims[3];
  const int num_rois = rois_desc->dims[0];
  mluOpDataType_t data_dtype = input_desc->dtype;
  VLOG(5) << "[mluOpDeformRoiPoolBackward] Launch kernel policyFunc[" << k_dim.x
          << ", " << k_dim.y << ", " << k_dim.z << "].";
  switch (data_dtype) {
    case MLUOP_DTYPE_HALF: {
      KERNEL_CHECK((MLUUnion1DeformRoiPoolBackwardHalf(
          k_dim, k_type, handle->queue, grad_output, input, rois, offset,
          grad_input, grad_offset, batches, channels, height, width, num_rois,
          pooled_height, pooled_width, spatial_scale, sampling_ratio, gamma)));
    }; break;
    case MLUOP_DTYPE_FLOAT: {
      KERNEL_CHECK((MLUUnion1DeformRoiPoolBackwardFloat(
          k_dim, k_type, handle->queue, grad_output, input, rois, offset,
          grad_input, grad_offset, batches, channels, height, width, num_rois,
          pooled_height, pooled_width, spatial_scale, sampling_ratio, gamma)));
    }; break;
    default: {
      VLOG(5) << "Input Date Type Not Supported. Only support half and float !";
    }
  }
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
