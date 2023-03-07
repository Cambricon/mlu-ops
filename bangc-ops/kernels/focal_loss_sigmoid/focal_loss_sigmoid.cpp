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
#include "kernels/kernel.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"

static void policyFunc(const mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type) {
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = handle->core_num_per_cluster;
  k_dim->y = mluop::runtime::getClusterLimitCapability(handle);
  k_dim->z = 1;
}

/*
 * FocalLossSigmoidForwrad
 */
static mluOpStatus_t checkFocalLossSigmoidForwardValidation(
    const mluOpTensorDescriptor_t input_desc,
    const mluOpTensorDescriptor_t target_desc,
    const mluOpTensorDescriptor_t weight_desc,
    const mluOpTensorDescriptor_t output_desc) {
  const std::string interface_name = "[mluOpFocalLossSigmoidForward] ";
  const mluOpDataType_t input_dtype = input_desc->dtype;
  const mluOpDataType_t target_dtype = target_desc->dtype;
  const mluOpDataType_t output_dtype = output_desc->dtype;

  // check shape
  if (input_desc->dim != 2) {
    LOG(ERROR) << interface_name << "Dimension num of input should be 2. "
               << "But now input_desc->dim is " << input_desc->dim << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (target_desc->dim != 1) {
    LOG(ERROR) << interface_name << "Dimension num of target should be 1. "
               << "But now target_desc->dim is " << target_desc->dim << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (target_desc->dims[0] != input_desc->dims[0]) {
    LOG(ERROR) << interface_name << "Element num of target should be "
               << input_desc->dims[0] << ", But now target_desc->dims[0] is "
               << target_desc->dims[0] << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (output_desc->dim != 2) {
    LOG(ERROR) << interface_name << "Dimension num of output should be 2. "
               << "But now output_desc->dim is " << output_desc->dim << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (output_desc->dims[0] != input_desc->dims[0] ||
      output_desc->dims[1] != input_desc->dims[1]) {
    LOG(ERROR) << interface_name << "Shape of output and input must be euqal. "
               << "But now output.shape is [" << output_desc->dims[0] << ", "
               << output_desc->dims[1] << "], "
               << "and input.shape is [" << input_desc->dims[0] << ", "
               << input_desc->dims[1] << "]. ";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // check dtype
  if (input_dtype != MLUOP_DTYPE_FLOAT && input_dtype != MLUOP_DTYPE_HALF) {
    LOG(ERROR) << interface_name << "Types of input should be HALF or FLOAT. "
               << "But now input_dtype is "
               << mluop::getNameOfDataType(input_dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (target_dtype != MLUOP_DTYPE_INT32) {
    LOG(ERROR) << interface_name << "Type of target should be INT32. "
               << "But now target_dtype is "
               << mluop::getNameOfDataType(input_dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (output_dtype != input_dtype) {
    LOG(ERROR) << interface_name
               << "Both types of input and output should be equal. "
               << "But now input_dtype is "
               << mluop::getNameOfDataType(input_dtype) << ", "
               << "output_dtype is " << mluop::getNameOfDataType(output_dtype)
               << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // check weight
  if (weight_desc != NULL && mluOpGetTensorElementNum(weight_desc) != 0) {
    if (weight_desc->dtype != input_dtype) {
      LOG(ERROR) << interface_name
                 << "Both types of weight and output should be equal. "
                 << "But now input_dtype is "
                 << mluop::getNameOfDataType(input_dtype) << ", "
                 << "weight_dtype is "
                 << mluop::getNameOfDataType(weight_desc->dtype) << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
    if (weight_desc->dim != 1) {
      LOG(ERROR) << interface_name << "Dimension num of weight should be 1. "
                 << "But now weight_desc->dim is " << weight_desc->dim << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
    if (weight_desc->dims[0] != input_desc->dims[1]) {
      LOG(ERROR) << interface_name << "Element num of weight should be "
                 << input_desc->dims[1] << ", But now weight_desc->dims[0] is "
                 << weight_desc->dims[0] << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  } else {
    VLOG(5) << interface_name << "weight is null.";
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpFocalLossSigmoidForward(
    mluOpHandle_t handle, const mluOpComputationPreference_t prefer,
    const mluOpLossReduction_t reduction,
    const mluOpTensorDescriptor_t input_desc, const void *input,
    const mluOpTensorDescriptor_t target_desc, const void *target,
    const mluOpTensorDescriptor_t weight_desc, const void *weight,
    const float alpha, const float gamma,
    const mluOpTensorDescriptor_t output_desc, void *output) {
  const std::string interface_name = "[mluOpFocalLossSigmoidForward] ";
  PARAM_CHECK("[mluOpFocalLossSigmoidForward]", handle != NULL);
  PARAM_CHECK("[mluOpFocalLossSigmoidForward]", input_desc != NULL);
  PARAM_CHECK("[mluOpFocalLossSigmoidForward]", target_desc != NULL);
  PARAM_CHECK("[mluOpFocalLossSigmoidForward]", output_desc != NULL);

  // params check
  if (prefer !=
      mluOpComputationPreference_t::MLUOP_COMPUTATION_HIGH_PRECISION) {
    LOG(ERROR) << interface_name << "only support HIGH_PRECISION currently.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (reduction != mluOpLossReduction_t::MLUOP_LOSS_REDUCTION_NONE) {
    LOG(ERROR) << interface_name << "only support REDUCTION_NONE currently.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (gamma < 0) {
    LOG(ERROR) << interface_name
               << "gamma should be greater than or equal to 0."
               << "But now gamma is " << gamma << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (MLUOP_STATUS_SUCCESS !=
      checkFocalLossSigmoidForwardValidation(input_desc, target_desc,
                                             weight_desc, output_desc)) {
    return MLUOP_STATUS_BAD_PARAM;
  }

  if (mluOpGetTensorElementNum(input_desc) == 0 ||
      mluOpGetTensorElementNum(target_desc) == 0 ||
      mluOpGetTensorElementNum(output_desc) == 0) {
    VLOG(5) << interface_name << "Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }
  if (weight_desc != NULL && mluOpGetTensorElementNum(weight_desc) != 0) {
    PARAM_CHECK("[mluOpFocalLossSigmoidForward]", weight != NULL);
  }
  PARAM_CHECK("[mluOpFocalLossSigmoidForward]", input != NULL);
  PARAM_CHECK("[mluOpFocalLossSigmoidForward]", target != NULL);
  PARAM_CHECK("[mluOpFocalLossSigmoidForward]", output != NULL);

  // generate case prototxt.
  const uint64_t N = input_desc->dims[0];
  const uint64_t C = input_desc->dims[1];
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("focal_loss_sigmoid_forward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA_REAL(true, "input", input, input_desc);
    GEN_CASE_DATA_REAL(true, "target", target, target_desc);
    if (weight != NULL) {
      GEN_CASE_DATA_REAL(true, "weight", weight, weight_desc);
    }
    GEN_CASE_DATA(false, "output", output, output_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "focal_loss_sigmoid_forward", "prefer", 1);
    GEN_CASE_OP_PARAM_SINGLE(1, "focal_loss_sigmoid_forward", "reduction", 0);
    GEN_CASE_OP_PARAM_SINGLE(1, "focal_loss_sigmoid_forward", "alpha", alpha);
    GEN_CASE_OP_PARAM_SINGLE(2, "focal_loss_sigmoid_forward", "gamma", gamma);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  // calculate task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, &k_dim, &k_type);

  // Launch Kernel
  const uint64_t core_dim = handle->core_num_per_cluster;
  VLOG(5) << "Launch Kernel MLUKernelFocalLossSigmoidForward<<<Union"
          << k_type / core_dim << ", " << k_dim.x << ", " << k_dim.y << ", "
          << k_dim.z << ">>>";
  if (input_desc->dtype == MLUOP_DTYPE_HALF) {
    KERNEL_CHECK((mluOpBlockKernelFocalLossSigmoidForwardHalf(
        k_dim, k_type, handle->queue, input, target, weight, N, C, alpha, gamma,
        output)));
  } else {
    KERNEL_CHECK((mluOpBlockKernelFocalLossSigmoidForwardFloat(
        k_dim, k_type, handle->queue, input, target, weight, N, C, alpha, gamma,
        output)));
  }

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}

/*
 * FocalLossSigmoidBackwrad
 */
static void getDealNAndThresholdC(const mluOpHandle_t handle,
                                  const int compute_data_bytes,
                                  const int target_data_bytes,
                                  const int total_c, int *deal_n_ptr,
                                  int *threshold_c_ptr, const bool has_weight,
                                  const bool is_half) {
  /* NRAM partition:
   *
   * |-----------------ping pong--------------------|
   * |input | pt | alpha_t | temp | output | target | flt_min | gamma | weight|
   *
   * split_pipeline_num is 5: including input, pt, alpha_t, temp, output.
   */
  const int nram_split_num = 5;
  const int nram_split_pingpong = 2;
  const int max_nram_size = handle->nram_size;
  int compute_align_size = NFU_ALIGN_SIZE;
  if (is_half) {
    compute_align_size += NFU_ALIGN_SIZE;
  }
  const int compute_align_num = compute_align_size / compute_data_bytes;
  // reservered_align_size: including input(ping pong), pt(ping pong),
  //                        alpha_t(ping pong), temp(ping pong),
  //                        output(ping pong), target(ping pong),
  //                        flt_min and gamma.
  const int reservered_align_size =
      ((nram_split_num + 1) * nram_split_pingpong + 2) * compute_align_size;
  int nram_pingpong_size = max_nram_size - reservered_align_size;

  int compute_c = total_c;
  int threshold_c = 0;
  if (has_weight) {
    // reserved space for weight to align
    nram_pingpong_size -= NFU_ALIGN_SIZE;

    // threshold_c * nram_split_pingpong * compute_data_bytes * nram_split_num +
    //     nram_split_pingpong * target_data_bytes +
    //     threshold_c * compute_data_bytes <= nram_pingpong_size
    threshold_c =
        (nram_pingpong_size - nram_split_pingpong * target_data_bytes) /
        (compute_data_bytes * (nram_split_num * nram_split_pingpong + 1));
    threshold_c = PAD_DOWN(threshold_c, compute_align_num);
    int weight_space = PAD_UP(total_c * compute_data_bytes, NFU_ALIGN_SIZE);

    // reserved space for weight
    nram_pingpong_size -= weight_space;
    compute_c = PAD_UP(total_c, compute_align_num);
  } else {
    // threshold_c * nram_split_pingpong * compute_data_bytes * nram_split_num +
    //     nram_split_pingpong * target_data_bytes <= nram_pingpong_size
    threshold_c =
        (nram_pingpong_size / nram_split_pingpong - target_data_bytes) /
        (nram_split_num * compute_data_bytes);
  }
  // deal_n * compute_c * nram_split_pingpong * compute_data_bytes *
  //     nram_split_num + deal_n * nram_split_pingpong * target_data_bytes <=
  //     nram_pingpong_size
  *deal_n_ptr =
      nram_pingpong_size /
      ((nram_split_num * compute_c * compute_data_bytes + target_data_bytes) *
       nram_split_pingpong);
  *threshold_c_ptr = threshold_c;
}

static mluOpStatus_t checkParams(const mluOpTensorDescriptor_t input_desc,
                                 const mluOpTensorDescriptor_t target_desc,
                                 const mluOpTensorDescriptor_t weight_desc,
                                 const mluOpTensorDescriptor_t output_desc) {
  const std::string interface_name = "[mluOpFocalLossSigmoidBackward]: ";

  // check shape
  PARAM_CHECK(interface_name, input_desc->dim == output_desc->dim);
  if (input_desc->dim != 2) {
    LOG(ERROR) << interface_name << "input_desc->dim shoule be 2"
               << "but now input_desc->dim is " << input_desc->dim << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (target_desc->dim != 1) {
    LOG(ERROR) << interface_name << "target_desc->dim shoule be 1"
               << "but now target_desc->dim is " << target_desc->dim << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  for (int i = 0; i < input_desc->dim; ++i) {
    if (input_desc->dims[i] != output_desc->dims[i]) {
      LOG(ERROR) << interface_name << "input_desc->dims[" << i
                 << "] should be equal to "
                 << "output_desc->dims[" << i << "]. But now "
                 << "input_desc->dims[" << i << "] is " << input_desc->dims[i]
                 << ", "
                 << "output_desc->dims[" << i << "] is " << output_desc->dims[i]
                 << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  if (input_desc->dims[0] != target_desc->dims[0]) {
    LOG(ERROR) << interface_name << "input_desc->dims[0] should be equal to "
               << "target_desc->dim[0]. But now "
               << "input_desc->dims[0] is " << input_desc->dims[0] << ", "
               << "target_desc->dims[0] is " << target_desc->dims[0] << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // check data type
  auto input_dtype = input_desc->dtype;
  auto target_dtype = target_desc->dtype;
  PARAM_CHECK(interface_name, input_desc->dtype == output_desc->dtype);
  if (input_dtype != MLUOP_DTYPE_FLOAT && input_dtype != MLUOP_DTYPE_HALF) {
    LOG(ERROR) << interface_name << "Types of input should be HALF or FLOAT. "
               << "But now input_dtype is "
               << mluop::getNameOfDataType(input_dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (target_desc->dtype != MLUOP_DTYPE_INT32) {
    LOG(ERROR) << interface_name << "The data type of target should be int32, "
               << "but now target dtype is "
               << mluop::getNameOfDataType(target_dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // check weight
  if (weight_desc != NULL && mluOpGetTensorElementNum(weight_desc) != 0) {
    if (weight_desc->dim != 1) {
      LOG(ERROR) << interface_name << "weight_desc->dim shoule be 1"
                 << "but now weight_desc->dim is " << weight_desc->dim << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
    if (input_desc->dims[1] != weight_desc->dims[0]) {
      LOG(ERROR) << interface_name << "input_desc->dims[1] should be equal to "
                 << "weight_desc->dims[0]. But now "
                 << "input_desc->dims[1] is " << input_desc->dims[1] << ", "
                 << "weight_desc->dims[0] is " << weight_desc->dims[0] << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
    if (weight_desc->dtype != input_dtype) {
      LOG(ERROR) << interface_name
                 << "Both types of weight and output should be equal. "
                 << "But now input_dtype is "
                 << mluop::getNameOfDataType(input_dtype) << ", "
                 << "weight_dtype is "
                 << mluop::getNameOfDataType(weight_desc->dtype) << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  } else {
    VLOG(5) << interface_name << "weight is NULL.";
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpFocalLossSigmoidBackward(
    mluOpHandle_t handle, const mluOpComputationPreference_t prefer,
    const mluOpLossReduction_t reduction,
    const mluOpTensorDescriptor_t input_desc, const void *input,
    const mluOpTensorDescriptor_t target_desc, const void *target,
    const mluOpTensorDescriptor_t weight_desc, const void *weight,
    const float alpha, const float gamma,
    const mluOpTensorDescriptor_t output_desc, void *output) {
  const std::string interface_name = "[mluOpFocalLossSigmoidBackward]: ";
  // params check
  PARAM_CHECK(interface_name, handle != NULL);
  PARAM_CHECK(interface_name, input_desc != NULL);
  PARAM_CHECK(interface_name, target_desc != NULL);
  PARAM_CHECK(interface_name, output_desc != NULL);

  if (checkParams(input_desc, target_desc, weight_desc, output_desc) !=
      MLUOP_STATUS_SUCCESS) {
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (prefer != mluOpComputationPreference_t::MLUOP_COMPUTATION_FAST) {
    LOG(ERROR) << interface_name << "only surpport COMPUTATION_FAST currently.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (reduction != mluOpLossReduction_t::MLUOP_LOSS_REDUCTION_NONE) {
    LOG(ERROR) << interface_name << "only surpport REDUCTION_NONE currently.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  if (gamma < 0) {
    LOG(ERROR) << interface_name
               << "gamma should be greater than or equal to 0."
               << "But now gamma is " << gamma << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  bool has_weight = false;
  if (weight_desc != NULL && mluOpGetTensorElementNum(weight_desc) != 0) {
    has_weight = true;
    PARAM_CHECK(interface_name, weight != NULL);
  }

  int deal_n = 0;
  int compute_data_bytes = sizeof(float);
  int target_data_bytes = mluOpDataTypeBytes(target_desc->dtype);
  int threshold_c = 0;
  int dim_n = input_desc->dims[0];
  int dim_c = input_desc->dims[1];

  bool is_half = input_desc->dtype == MLUOP_DTYPE_HALF;
  // calculate deal_n and threshold_c
  getDealNAndThresholdC(handle, compute_data_bytes, target_data_bytes, dim_c,
                        &deal_n, &threshold_c, has_weight, is_half);

  VLOG(5) << interface_name << "threshold_c: " << threshold_c;
  // check C
  if (dim_c > threshold_c) {
    LOG(ERROR) << interface_name
               << " input_desc->dims[1] should be in the range of "
               << "[0, " << threshold_c << "]. "
               << "but now input_desc->dims[1] is " << dim_c;
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  size_t input_size = mluOpGetTensorElementNum(input_desc);
  size_t target_size = mluOpGetTensorElementNum(target_desc);
  size_t output_size = mluOpGetTensorElementNum(output_desc);
  if (input_size == 0 || target_size == 0 || output_size == 0) {
    VLOG(5) << interface_name << "skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }
  PARAM_CHECK(interface_name, input != NULL);
  PARAM_CHECK(interface_name, target != NULL);
  PARAM_CHECK(interface_name, output != NULL);

  // generate focal_loss_sigmoid_backward prototxt
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("focal_loss_sigmoid_backward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", input, input_desc, 20, -20);
    if (weight != NULL) {
      GEN_CASE_DATA_REAL(true, "target", target, target_desc);
      GEN_CASE_DATA(true, "weight", weight, weight_desc, 1, 0);
    } else {
      GEN_CASE_DATA(true, "target", target, target_desc, dim_c, 0);
    }
    GEN_CASE_DATA(false, "output", output, output_desc, 20, -20);
    GEN_CASE_OP_PARAM_SINGLE(0, "focal_loss_sigmoid_backward", "prefer",
                             prefer);
    GEN_CASE_OP_PARAM_SINGLE(1, "focal_loss_sigmoid_backward", "reduction",
                             reduction);
    GEN_CASE_OP_PARAM_SINGLE(1, "focal_loss_sigmoid_backward", "alpha", alpha);
    GEN_CASE_OP_PARAM_SINGLE(2, "focal_loss_sigmoid_backward", "gamma", gamma);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  const int dwidth = mluOpDataTypeBytes(input_desc->dtype);
  policyFunc(handle, &k_dim, &k_type);

  VLOG(5) << "Launch Kernel MLUBlockFocalLossSigmoidBackward<<<Union"
          << k_type / CORE_DIM << ", " << k_dim.x << ", " << k_dim.y << ", "
          << k_dim.z << ">>>";
  if (dwidth == 2) {
    KERNEL_CHECK((mluOpBlockKernelFocalLossSigmoidBackwardHalf(
        k_dim, k_type, handle->queue, input, target, weight, gamma, alpha,
        dim_n, deal_n, dim_c, output)));
  } else {
    KERNEL_CHECK((mluOpBlockKernelFocalLossSigmoidBackwardFloat(
        k_dim, k_type, handle->queue, input, target, weight, gamma, alpha,
        dim_n, deal_n, dim_c, output)));
  }
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
