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
#include "roi_crop.h"

#include <string>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/utils/cnnl_helper.h"

static void policyFunc(const mluOpHandle_t handle, int bin_num,
                       cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  uint32_t cluster_num = mluop::runtime::getClusterLimitCapability(handle);
  uint32_t core_in_cluster = handle->core_num_per_cluster;
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = core_in_cluster;
  uint32_t use_cluster = (bin_num + core_in_cluster - 1) / core_in_cluster;
  k_dim->y = use_cluster > cluster_num ? cluster_num : use_cluster;
  k_dim->z = 1;
}

/* user param check
 * step1:check desc and data ptr is not nullptr_t
 * step2:check shape and data type
 * step3:check zero element
 * */
mluOpStatus_t RoiCropForwardParamCheck(
    const std::string &op_name, const mluOpHandle_t handle,
    const mluOpTensorDescriptor_t input_desc, const void *input,
    const mluOpTensorDescriptor_t grid_desc, const void *grid,
    const mluOpTensorDescriptor_t output_desc, const void *output) {
  // check descriptor and data
  PARAM_CHECK(op_name, handle != NULL);
  PARAM_CHECK(op_name, input_desc != NULL);
  PARAM_CHECK(op_name, grid_desc != NULL);
  PARAM_CHECK(op_name, output_desc != NULL);
  // check data type and dim
  PARAM_CHECK(op_name, input_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(op_name, input_desc->dim == 4);
  PARAM_CHECK(op_name, grid_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(op_name, grid_desc->dim == 4);
  PARAM_CHECK(op_name, output_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(op_name, output_desc->dim == 4);
  // check shape and layout
  PARAM_CHECK(op_name, input_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(op_name, output_desc->layout == MLUOP_LAYOUT_NHWC);
  // check stride
  STRIDE_TENSOR_CHECK("[mluOpRoiCropForward]:", input_desc,
                      "input_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpRoiCropForward]:", grid_desc,
                      "grid_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpRoiCropForward]:", output_desc,
                      "output_desc must be contiguous");

  for (int i = 0; i < output_desc->dim - 1; ++i) {
    if (output_desc->dims[i] != grid_desc->dims[i]) {
      LOG(ERROR) << op_name << " Check failed: output_desc->dims[" << i
                 << "] should be equal to grid_desc->dims[" << i << "].";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  if (output_desc->dims[3] != input_desc->dims[3]) {
    LOG(ERROR) << op_name
               << " Check failed: output_desc->dims[3] should be "
                  "equal to input_desc->dims[3].";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (grid_desc->dims[3] != 2) {
    LOG(ERROR) << op_name
               << " Check failed: grid_desc->dims[3] should be equal to 2.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  const size_t max_input_num = 2147483648;  // 2^31, 2G num
  if ((mluOpGetTensorElementNum(input_desc) >= max_input_num) ||
      (mluOpGetTensorElementNum(grid_desc) >= max_input_num) ||
      (mluOpGetTensorElementNum(output_desc) >= max_input_num)) {
    LOG(ERROR) << op_name << " Overflow max tensor num."
               << " Currently, MLU-OPS supports tensor num smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  // check zero element
  if ((mluOpGetTensorElementNum(input_desc) == 0) ||
      (mluOpGetTensorElementNum(grid_desc) == 0) ||
      (mluOpGetTensorElementNum(output_desc) == 0)) {
    LOG(ERROR) << op_name << " Zero element tensor failure.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  PARAM_CHECK(op_name, input != NULL);
  PARAM_CHECK(op_name, grid != NULL);
  PARAM_CHECK(op_name, output != NULL);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t RoiCropBackwardParamCheck(
    const std::string &op_name, const mluOpHandle_t handle,
    const mluOpTensorDescriptor_t grad_output_desc, const void *grad_output,
    const mluOpTensorDescriptor_t grid_desc, const void *grid,
    const mluOpTensorDescriptor_t grad_input_desc, const void *grad_input) {
  // check descriptor and data
  PARAM_CHECK(op_name, handle != NULL);
  PARAM_CHECK(op_name, grad_output_desc != NULL);
  PARAM_CHECK(op_name, grid_desc != NULL);
  PARAM_CHECK(op_name, grad_input_desc != NULL);
  // check data type
  PARAM_CHECK(op_name, grad_output_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(op_name, grad_output_desc->dim == 4);
  PARAM_CHECK(op_name, grid_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(op_name, grid_desc->dim == 4);
  PARAM_CHECK(op_name, grad_input_desc->dtype == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(op_name, grad_input_desc->dim == 4);
  // check shape and layout
  PARAM_CHECK(op_name, grad_output_desc->layout == MLUOP_LAYOUT_NHWC);
  PARAM_CHECK(op_name, grad_input_desc->layout == MLUOP_LAYOUT_NHWC);
  // check stride
  STRIDE_TENSOR_CHECK("[mluOpRoiCropBackward]:", grad_output_desc,
                      "grad_output_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpRoiCropBackward]:", grid_desc,
                      "grid_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpRoiCropBackward]:", grad_input_desc,
                      "grad_input_desc must be contiguous");

  for (int i = 0; i < grad_output_desc->dim - 1; ++i) {
    if (grad_output_desc->dims[i] != grid_desc->dims[i]) {
      LOG(ERROR) << op_name << " Check failed: grad_output_desc->dims[" << i
                 << "] should be equal to grid_desc->dims[" << i << "].";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  if (grad_output_desc->dims[3] != grad_input_desc->dims[3]) {
    LOG(ERROR) << op_name
               << " Check failed: grad_output_desc->dims[3] should be "
                  "equal to grad_input_desc->dims[3].";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (grid_desc->dims[3] != 2) {
    LOG(ERROR) << op_name
               << " Check failed: grid_desc->dims[3] should be equal to 2.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  const size_t max_input_num = 2147483648;  // 2^31 2G num
  if ((mluOpGetTensorElementNum(grad_output_desc) >= max_input_num) ||
      (mluOpGetTensorElementNum(grid_desc) >= max_input_num) ||
      (mluOpGetTensorElementNum(grad_input_desc) >= max_input_num)) {
    LOG(ERROR) << op_name << " Overflow max tensor num."
               << " Currently, MLU-OPS supports tensor num smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  // check zero element
  if ((mluOpGetTensorElementNum(grad_input_desc) == 0) ||
      (mluOpGetTensorElementNum(grid_desc) == 0) ||
      (mluOpGetTensorElementNum(grad_output_desc) == 0)) {
    LOG(ERROR) << op_name << " Zero element tensor failure.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  PARAM_CHECK(op_name, grad_output != NULL);
  PARAM_CHECK(op_name, grid != NULL);
  PARAM_CHECK(op_name, grad_input != NULL);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpRoiCropForward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, const mluOpTensorDescriptor_t grid_desc,
    const void *grid, const mluOpTensorDescriptor_t output_desc, void *output) {
  // check params
  mluOpStatus_t param_check =
      RoiCropForwardParamCheck("[mluOpRoiCropForward]", handle, input_desc,
                               input, grid_desc, grid, output_desc, output);
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }

  uint32_t batch = input_desc->dims[0];
  uint32_t height = input_desc->dims[1];
  uint32_t width = input_desc->dims[2];
  uint32_t channels = input_desc->dims[3];
  uint32_t grid_n = grid_desc->dims[0];
  uint32_t output_h = output_desc->dims[1];
  uint32_t output_w = output_desc->dims[2];
  uint32_t bin_num = grid_n * output_h * output_w;

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("roi_crop_forward", "ROI_CROP_FORWARD");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", input, input_desc, -10, 10);
    GEN_CASE_DATA(true, "grid", grid, grid_desc, -1, 1);
    GEN_CASE_DATA(false, "output", output, output_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  policyFunc(handle, bin_num, &k_dim, &k_type);
  VLOG(5) << "[mluOpRoiCropForward] launch kernel policyFunc[" << k_dim.x
          << ", " << k_dim.y << ", " << k_dim.z << "].";

  CHECK_RETURN("[mluOpRoiCropForward]",
               KernelRoiCropForward(k_dim, k_type, handle->queue, input, grid,
                                    batch, height, width, channels, grid_n,
                                    output_h, output_w, output));
  VLOG(5) << "Kernel KernelRoiCropForward.";
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpRoiCropBackward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t grad_output_desc,
    const void *grad_output, const mluOpTensorDescriptor_t grid_desc,
    const void *grid, const mluOpTensorDescriptor_t grad_input_desc,
    void *grad_input) {
  // check params
  mluOpStatus_t param_check = RoiCropBackwardParamCheck(
      "[mluOpRoiCropBackward]", handle, grad_output_desc, grad_output,
      grid_desc, grid, grad_input_desc, grad_input);
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }

  uint32_t batch = grad_input_desc->dims[0];
  uint32_t height = grad_input_desc->dims[1];
  uint32_t width = grad_input_desc->dims[2];
  uint32_t channels = grad_input_desc->dims[3];
  uint32_t grid_n = grid_desc->dims[0];
  uint32_t output_h = grad_output_desc->dims[1];
  uint32_t output_w = grad_output_desc->dims[2];
  uint32_t bin_num = grid_n * output_h * output_w;

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("roi_crop_backward", "ROI_CROP_BACKWARD");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "grad_output", grad_output, grad_output_desc, -10, 10);
    GEN_CASE_DATA(true, "grid", grid, grid_desc, -1, 1);
    GEN_CASE_DATA(false, "grad_input", grad_input, grad_input_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  policyFunc(handle, bin_num, &k_dim, &k_type);
  VLOG(5) << "[mluOpRoiCropBackward] launch kernel policyFunc[" << k_dim.x
          << ", " << k_dim.y << ", " << k_dim.z << "].";
  // gdram set zero
  float fill_value = 0;
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(grad_input_desc,
                                                 cnnl_output_desc);
    CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &fill_value,
                          cnnl_output_desc, grad_input));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }

  CHECK_RETURN("[mluOpRoiCropBackward]",
               KernelRoiCropBackward(k_dim, k_type, handle->queue, grad_output,
                                     grid, batch, height, width, channels,
                                     grid_n, output_h, output_w, grad_input));
  VLOG(5) << "kernel KernelRoiCropBackward.";
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
