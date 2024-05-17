/*************************************************************************
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
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "kernels/tin_shift/tin_shift.h"

#include <string>

#include "core/gen_case.h"
#include "core/runtime/device.h"
#include "core/type.h"

// policy function
static void policyFunc(const mluOpHandle_t handle,
                       const mluOpTensorDescriptor_t input_desc,
                       cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type,
                       int *channel_per_core, int *max_number_hw_per_core,
                       int *max_length_per_core) {
  const int32_t cluster_limit =
      mluop::runtime::getClusterLimitCapability(handle);
  const int32_t core_limit =
      mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  const int core_num = core_limit * cluster_limit;
  const size_t batch_size = input_desc->dims[0];
  const size_t time_size = input_desc->dims[1];
  const size_t channel_size = input_desc->dims[2];
  const size_t hw_size = input_desc->dims[3];
  const size_t size_per_channel =
      time_size * hw_size * mluop::getSizeOfDataType(input_desc->dtype);
  *channel_per_core = handle->nram_size / size_per_channel;
  int task_dim = 0;
  if (*channel_per_core == 0) {
    const size_t size_per_hw =
        hw_size * mluop::getSizeOfDataType(input_desc->dtype);
    *max_number_hw_per_core = handle->nram_size / size_per_hw;
    if (*max_number_hw_per_core <= 0) {
      *max_length_per_core =
          handle->nram_size / mluop::getSizeOfDataType(input_desc->dtype);
    }
    int tmp_max_number_hw_per_core =
        *max_number_hw_per_core > 0 ? *max_number_hw_per_core : 1;
    const int loop_time =
        (time_size / (tmp_max_number_hw_per_core)) +
        ((time_size % (tmp_max_number_hw_per_core)) > 0 ? 1 : 0);
    task_dim = batch_size * channel_size * loop_time < core_num
                   ? batch_size * channel_size * loop_time
                   : core_num;
  } else {
    task_dim = batch_size * channel_size < core_num ? batch_size * channel_size
                                                    : core_num;
  }

  k_dim->x = core_limit;
  k_dim->y = (task_dim / core_limit) > 0 ? (task_dim / core_limit) : 1;
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
}

static mluOpStatus_t TinShiftPreCheck(
    const std::string direction, const mluOpTensorDescriptor_t input_desc,
    const mluOpTensorDescriptor_t shifts_desc,
    const mluOpTensorDescriptor_t output_desc) {
  int input_dims = input_desc->dim;
  int output_dims = output_desc->dim;
  if (input_dims != 4) {
    LOG(ERROR) << "[mluOpTinShift " + direction +
                      "] The input dims should be 4. "
               << "But now the input dims is " << input_dims << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (shifts_desc->dim != 2) {
    LOG(ERROR) << "[mluOpTinShift " + direction +
                      "] The shifts dims should be 2. "
               << "But now the shifts dims is " << shifts_desc->dim << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (input_dims != output_dims) {
    LOG(ERROR) << "[mluOpTinShift " + direction + "] "
               << "The input dims and the output dims should be the same. "
               << "But now the input dims is " << input_dims
               << ", and the output dims is " << output_dims << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  const int channel_size = input_desc->dims[2];
  const int group_size = shifts_desc->dims[1];
  if (channel_size == 0) {
    LOG(ERROR) << "[mluOpTinShift " + direction + "] "
               << "The channel size should not be zero.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (group_size == 0) {
    LOG(ERROR) << "[mluOpTinShift " + direction + "] "
               << "The group size should not be zero.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (channel_size % group_size != 0) {
    LOG(ERROR) << "[mluOpTinShift " + direction + "] "
               << "The channel size should be multiple of group size, "
               << "But now channel size is " << channel_size
               << " and group size is " << group_size << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  for (int i = 0; i < input_dims; i++) {
    if (input_desc->dims[i] != output_desc->dims[i]) {
      LOG(ERROR) << "[mluOpTinShift " + direction + "] The size of input dims["
                 << i << "] is " << input_desc->dims[i]
                 << ", and the size of output dims[" << i << "] is "
                 << output_desc->dims[i] << ". They should be the same.";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  if (input_desc->dims[0] != shifts_desc->dims[0]) {
    LOG(ERROR) << "[mluOpTinShift " + direction + "] "
               << "The input batch size should be the same as shifts's,"
               << " input batch size is " << input_desc->dims[0]
               << " and shifts batch size is " << shifts_desc->dims[0] << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (shifts_desc->dtype != MLUOP_DTYPE_INT32) {
    LOG(ERROR)
        << "[mluOpTinShift " + direction + "] "
        << "The data type of the shift tensor should be MLUOP_DTYPE_INT32. "
        << "Current data type is "
        << mluOpGetNameOfDataType(shifts_desc->dtype);
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (input_desc->dtype != MLUOP_DTYPE_HALF &&
      input_desc->dtype != MLUOP_DTYPE_FLOAT) {
    LOG(ERROR) << "[mluOpTinShift " + direction + "] "
               << "The data type of the input tensor should be half or float.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (input_desc->dtype != output_desc->dtype) {
    LOG(ERROR) << "[mluOpTinShift " + direction + "] "
               << "The data type of the input tensor and output tensor should "
                  "be the same.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  const int input_batches_size = input_desc->dims[0];
  const int input_hw_size = input_desc->dims[3];
  if (input_batches_size == 0) {
    LOG(ERROR) << "[mluOpTinShift " + direction + "] "
               << "The input batch size should not be zero.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (input_hw_size == 0) {
    LOG(ERROR) << "[mluOpTinShift " + direction + "] "
               << "The size of input hw size should not be zero.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  const uint64_t input_element_num = mluOpGetTensorElementNum(input_desc);
  const uint64_t shifts_element_num = mluOpGetTensorElementNum(shifts_desc);
  const uint64_t output_element_num = mluOpGetTensorElementNum(output_desc);
  TENSOR_NUM_CHECK("[mluOpTinShift " + direction + "] ", input_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpTinShift " + direction + "] ", shifts_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpTinShift " + direction + "] ", output_element_num,
                   LARGE_TENSOR_NUM, "");
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpTinShiftForward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, const mluOpTensorDescriptor_t shifts_desc,
    const void *shifts, const mluOpTensorDescriptor_t output_desc,
    void *output) {
  PARAM_CHECK("[mluOpTinShift forward]", handle != NULL);
  PARAM_CHECK("[mluOpTinShift forward]", input_desc != NULL);
  PARAM_CHECK("[mluOpTinShift forward]", shifts_desc != NULL);
  PARAM_CHECK("[mluOpTinShift forward]", output_desc != NULL);
  mluOpStatus_t status =
      TinShiftPreCheck("forward", input_desc, shifts_desc, output_desc);
  if (MLUOP_STATUS_SUCCESS != status) {
    return status;
  }
  if (input_desc->dims[1] == 0) {
    return MLUOP_STATUS_SUCCESS;
  }
  PARAM_CHECK("[mluOpTinShift forward]", input != NULL);
  PARAM_CHECK("[mluOpTinShift forward]", shifts != NULL);
  PARAM_CHECK("[mluOpTinShift forward]", output != NULL);

  // generate mluOpTinShiftForward prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("tin_shift_forward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", input, input_desc, -10, 10);
    GEN_CASE_DATA_REAL(true, "shifts", shifts, shifts_desc);
    GEN_CASE_DATA(false, "output", output, output_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(false, false, true, 0, 0, 0);
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  int channel_per_core = 0;
  int max_number_hw_per_core = 0;
  int max_length_per_core = 0;
  policyFunc(handle, input_desc, &k_dim, &k_type, &channel_per_core,
             &max_number_hw_per_core, &max_length_per_core);

  const int batch_size = input_desc->dims[0];
  const int time_size = input_desc->dims[1];
  const int channel_size = input_desc->dims[2];
  const int hw_size = input_desc->dims[3];
  const int group_size = shifts_desc->dims[1];
  int group_channel = channel_size / group_size;
  mluOpDataType_t data_dtype = input_desc->dtype;
  VLOG(5) << "batch_size: " << batch_size << " time_size: " << time_size
          << " channel_size: " << channel_size << " hw_size: " << hw_size
          << " group_channel: " << group_channel
          << " channel_per_core: " << channel_per_core
          << " max_number_hw_per_core: " << max_number_hw_per_core;

  CHECK_RETURN(
      "[mluOpTinShift forward]",
      KernelTinShift(k_dim, k_type, handle->queue, data_dtype, input, shifts,
                     batch_size, time_size, channel_size, hw_size, group_size,
                     group_channel, channel_per_core, max_number_hw_per_core,
                     max_length_per_core, output));

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpTinShiftBackward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t grad_output_desc,
    const void *grad_output, const mluOpTensorDescriptor_t shifts_desc,
    const void *shifts, const mluOpTensorDescriptor_t grad_input_desc,
    void *grad_input) {
  PARAM_CHECK("[mluOpTinShift backward]", handle != NULL);
  PARAM_CHECK("[mluOpTinShift backward]", grad_output_desc != NULL);
  PARAM_CHECK("[mluOpTinShift backward]", shifts_desc != NULL);
  PARAM_CHECK("[mluOpTinShift backward]", grad_input_desc != NULL);
  mluOpStatus_t status = MLUOP_STATUS_BAD_PARAM;
  status = TinShiftPreCheck("backward", grad_output_desc, shifts_desc,
                            grad_input_desc);
  if (MLUOP_STATUS_SUCCESS != status) {
    return status;
  }
  if (grad_output_desc->dims[1] == 0) {
    return MLUOP_STATUS_SUCCESS;
  }
  PARAM_CHECK("[mluOpTinShift backward]", grad_output != NULL);
  PARAM_CHECK("[mluOpTinShift backward]", shifts != NULL);
  PARAM_CHECK("[mluOpTinShift backward]", grad_input != NULL);

  // generate mluOpTinShiftBackward prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("tin_shift_backward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "grad_output", grad_output, grad_output_desc, -10, 10);
    GEN_CASE_DATA_REAL(true, "shifts", shifts, shifts_desc);
    GEN_CASE_DATA(false, "grad_input", grad_input, grad_input_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(false, false, true, 0, 0, 0);
  }
  // generate mluOpTinShiftBackward prototxt end!

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  int channel_per_core = 0;
  int max_number_hw_per_core = 0;
  int max_length_per_core = 0;
  policyFunc(handle, grad_output_desc, &k_dim, &k_type, &channel_per_core,
             &max_number_hw_per_core, &max_length_per_core);

  const int batch_size = grad_output_desc->dims[0];
  const int time_size = grad_output_desc->dims[1];
  const int channel_size = grad_output_desc->dims[2];
  const int hw_size = grad_output_desc->dims[3];
  const int group_size = shifts_desc->dims[1];
  int group_channel = channel_size / group_size;
  mluOpDataType_t data_dtype = grad_output_desc->dtype;
  VLOG(5) << "batch_size: " << batch_size << " time_size: " << time_size
          << " channel_size: " << channel_size << " hw_size: " << hw_size
          << " group_channel: " << group_channel
          << " channel_per_core: " << channel_per_core
          << " max_number_hw_per_core: " << max_number_hw_per_core;
  CHECK_RETURN(
      "[mluOpTinShift backward]",
      KernelTinShift(k_dim, k_type, handle->queue, data_dtype, grad_output,
                     shifts, batch_size, time_size, channel_size, hw_size,
                     group_size, group_channel, channel_per_core,
                     max_number_hw_per_core, max_length_per_core, grad_input));
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
