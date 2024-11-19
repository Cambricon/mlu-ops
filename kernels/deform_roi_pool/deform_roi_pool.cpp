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
#include "deform_roi_pool.h"

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"
#include "kernels/utils/cnnl_helper.h"

// policy function
void policyFunc(const mluOpHandle_t handle,
                const mluOpTensorDescriptor_t output_desc, cnrtDim3_t *k_dim,
                cnrtFunctionType_t *k_type) {
  const size_t cluster_limit =
      mluop::runtime::getClusterLimitCapability(handle);
  const size_t core_limit =
      mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  const size_t num_rois = output_desc->getDimIndex(0);
  const size_t pooled_height = output_desc->getDimIndex(1);
  const size_t pooled_width = output_desc->getDimIndex(2);
  const size_t num_bins =
      CEIL_ALIGN(num_rois * pooled_height * pooled_width, core_limit);
  k_dim->x = core_limit;
  k_dim->y = (num_bins / core_limit) > cluster_limit ? cluster_limit
                                                     : (num_bins / core_limit);
  k_dim->z = 1;
  *k_type = cnrtFuncTypeUnion1;
}

static mluOpStatus_t DeformRoiPoolForwardPreCheck(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const mluOpTensorDescriptor_t rois_desc,
    const mluOpTensorDescriptor_t offset_desc,
    const mluOpTensorDescriptor_t output_desc, const int pooled_height,
    const int pooled_width) {
  PARAM_CHECK("[mluOpDeformRoiPoolForward]",
              input_desc->getLayout() == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]",
              output_desc->getLayout() == MLUOP_LAYOUT_NHWC);

  STRIDE_TENSOR_CHECK("[mluOpDeformRoiPoolForward]:", input_desc,
                      "input_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpDeformRoiPoolForward]:", rois_desc,
                      "rois_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpDeformRoiPoolForward]:", offset_desc,
                      "offset_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpDeformRoiPoolForward]:", output_desc,
                      "output_desc must be contiguous");

  PARAM_CHECK("[mluOpDeformRoiPoolForward]",
              input_desc->getDtype() == MLUOP_DTYPE_FLOAT ||
                  input_desc->getDtype() == MLUOP_DTYPE_HALF);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]",
              input_desc->getDtype() == rois_desc->getDtype());
  PARAM_CHECK("[mluOpDeformRoiPoolForward]",
              input_desc->getDtype() == output_desc->getDtype());

  PARAM_CHECK("[mluOpDeformRoiPoolForward]", rois_desc->getDim() == 2);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]", rois_desc->getDimIndex(1) == 5);

  PARAM_CHECK("[mluOpDeformRoiPoolForward]", pooled_height > 0);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]", pooled_width > 0);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]",
              output_desc->getDimIndex(1) == pooled_height);
  PARAM_CHECK("[mluOpDeformRoiPoolForward]",
              output_desc->getDimIndex(2) == pooled_width);

  if (offset_desc != NULL) {
    PARAM_CHECK("[mluOpDeformRoiPoolForward]",
                offset_desc->getDtype() == input_desc->getDtype());
    PARAM_CHECK("[mluOpDeformRoiPoolForward]", offset_desc->getDim() == 4);
    PARAM_CHECK("[mluOpDeformRoiPoolForward]",
                offset_desc->getDimIndex(0) == rois_desc->getDimIndex(0));
    PARAM_CHECK("[mluOpDeformRoiPoolForward]", offset_desc->getDimIndex(1) == 2);
    PARAM_CHECK("[mluOpDeformRoiPoolForward]",
                offset_desc->getDimIndex(2) == pooled_height);
    PARAM_CHECK("[mluOpDeformRoiPoolForward]",
                offset_desc->getDimIndex(3) == pooled_width);
    const size_t offset_element_num = mluOpGetTensorElementNum(offset_desc);
    TENSOR_NUM_CHECK("[mluOpDeformRoiPoolForward]", offset_element_num,
                     LARGE_TENSOR_NUM, "");
  }
  if (rois_desc->getDimIndex(0) != output_desc->getDimIndex(0)) {
    LOG(ERROR) << "[mluOpDeformRoiPoolForward] rois number = "
               << rois_desc->getDimIndex(0)
               << ", output batch = " << output_desc->getDimIndex(0)
               << ", they should be equal.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (input_desc->getDimIndex(3) != output_desc->getDimIndex(3)) {
    LOG(ERROR) << "[mluOpDeformRoiPoolForward] input channel = "
               << input_desc->getDimIndex(3)
               << ", output channel = " << output_desc->getDimIndex(3)
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
              grad_output_desc->getLayout() == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              input_desc->getLayout() == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              grad_input_desc->getLayout() == MLUOP_LAYOUT_NHWC);

  STRIDE_TENSOR_CHECK("[mluOpDeformRoiPoolBackward]:", input_desc,
                      "input_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpDeformRoiPoolBackward]:", grad_output_desc,
                      "grad_output_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpDeformRoiPoolBackward]:", rois_desc,
                      "rois_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpDeformRoiPoolBackward]:", offset_desc,
                      "offset_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpDeformRoiPoolBackward]:", grad_input_desc,
                      "grad_input_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpDeformRoiPoolBackward]:", grad_offset_desc,
                      "grad_offset_desc must be contiguous");

  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              input_desc->getDtype() == MLUOP_DTYPE_FLOAT ||
                  input_desc->getDtype() == MLUOP_DTYPE_HALF);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              input_desc->getDtype() == grad_output_desc->getDtype());
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              input_desc->getDtype() == rois_desc->getDtype());
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              input_desc->getDtype() == grad_input_desc->getDtype());

  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", rois_desc->getDim() == 2);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", rois_desc->getDimIndex(1) == 5);

  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", pooled_height > 0);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]", pooled_width > 0);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              grad_output_desc->getDimIndex(1) == pooled_height);
  PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
              grad_output_desc->getDimIndex(2) == pooled_width);

  for (int i = 0; i < input_desc->getDim(); ++i) {
    if (input_desc->getDimIndex(i) != grad_input_desc->getDimIndex(i)) {
      LOG(ERROR) << "[mluOpDeformRoiPoolBackward] input's shape is ["
                 << input_desc->getDimIndex(0) << " " << input_desc->getDimIndex(1) << " "
                 << input_desc->getDimIndex(2) << " " << input_desc->getDimIndex(3)
                 << "], grad_input's shape is [" << grad_input_desc->getDimIndex(0)
                 << " " << grad_input_desc->getDimIndex(1) << " "
                 << grad_input_desc->getDimIndex(2) << " " << grad_input_desc->getDimIndex(3)
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
                offset_desc->getDtype() == input_desc->getDtype());
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]", offset_desc->getDim() == 4);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
                offset_desc->getDimIndex(0) == rois_desc->getDimIndex(0));
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]", offset_desc->getDimIndex(1) == 2);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
                offset_desc->getDimIndex(2) == pooled_height);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
                offset_desc->getDimIndex(3) == pooled_width);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
                grad_offset_desc->getDtype() == input_desc->getDtype());
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]", grad_offset_desc->getDim() == 4);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
                grad_offset_desc->getDimIndex(0) == rois_desc->getDimIndex(0));
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]", grad_offset_desc->getDimIndex(1) == 2);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
                grad_offset_desc->getDimIndex(2) == pooled_height);
    PARAM_CHECK("[mluOpDeformRoiPoolBackward]",
                grad_offset_desc->getDimIndex(3) == pooled_width);
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
  if (rois_desc->getDimIndex(0) != grad_output_desc->getDimIndex(0)) {
    LOG(ERROR) << "[mluOpDeformRoiPoolBackward] rois number = "
               << rois_desc->getDimIndex(0)
               << ", grad_output batch = " << grad_output_desc->getDimIndex(0)
               << ", they should be equal.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (input_desc->getDimIndex(3) != grad_output_desc->getDimIndex(3)) {
    LOG(ERROR) << "[mluOpDeformRoiPoolBackward] input channel = "
               << input_desc->getDimIndex(3)
               << ", output channel = " << grad_output_desc->getDimIndex(3)
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
  if (input_desc->getDimIndex(0) == 0 || mluOpGetTensorElementNum(rois_desc) == 0 ||
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
    GEN_CASE_START("deform_roi_pool_forward", "DEFORM_ROI_POOL_FORWARD");
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

  const int batches = input_desc->getDimIndex(0);
  const int height = input_desc->getDimIndex(1);
  const int width = input_desc->getDimIndex(2);
  const int channels = input_desc->getDimIndex(3);
  const int num_rois = output_desc->getDimIndex(0);
  mluOpDataType_t data_dtype = input_desc->getDtype();
  VLOG(5) << "[mluOpDeformRoiPoolForward] Launch kernel policyFunc[" << k_dim.x
          << ", " << k_dim.y << ", " << k_dim.z << "].";
  CHECK_RETURN(
      "[mluOpDeformRoiPoolForward]",
      KernelDeformRoiPoolForward(
          k_dim, k_type, handle->queue, data_dtype, input, rois, offset, output,
          batches, channels, height, width, num_rois, pooled_height,
          pooled_width, spatial_scale, sampling_ratio, gamma));

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
      input_desc->getDimIndex(0) == 0 || mluOpGetTensorElementNum(rois_desc) == 0) {
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
    GEN_CASE_START("deform_roi_pool_backward", "DEFORM_ROI_POOL_BACKWARD");
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
  VLOG(5) << "[mluOpDeformRoiPoolBackward] cnnlFill_v3 start.";
  const uint32_t fill_value = 0x00;
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grad_input_desc,
                                                 cnnl_output_desc);
    CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &fill_value,
                          cnnl_output_desc, grad_input));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  if (offset != NULL) {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grad_offset_desc,
                                                 cnnl_output_desc);
    CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &fill_value,
                          cnnl_output_desc, grad_offset));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  policyFunc(handle, grad_output_desc, &k_dim, &k_type);

  const int batches = input_desc->getDimIndex(0);
  const int height = input_desc->getDimIndex(1);
  const int width = input_desc->getDimIndex(2);
  const int channels = input_desc->getDimIndex(3);
  const int num_rois = rois_desc->getDimIndex(0);
  mluOpDataType_t data_dtype = input_desc->getDtype();
  VLOG(5) << "[mluOpDeformRoiPoolBackward] Launch kernel policyFunc[" << k_dim.x
          << ", " << k_dim.y << ", " << k_dim.z << "].";
  CHECK_RETURN("[mluOpDeformRoiPoolBackward]",
               KernelDeformRoiPoolBackward(
                   k_dim, k_type, handle->queue, data_dtype, grad_output, input,
                   rois, offset, grad_input, grad_offset, batches, channels,
                   height, width, num_rois, pooled_height, pooled_width,
                   spatial_scale, sampling_ratio, gamma));

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
