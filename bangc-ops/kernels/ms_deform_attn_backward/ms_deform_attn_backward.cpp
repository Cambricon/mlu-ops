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
#include "ms_deform_attn_backward.h"

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "core/tool.h"
#include "kernels/kernel.h"

char API[] = "[mluOpMsDeformAttnBackward]";

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

/*!
 * @brief Describes the kernel policy of ms_deform_attn_backward.
 */
typedef enum {
  MLUOP_MS_DEFORM_ATTN_BACKWARD_DEFAULT = 0,
  /*!< Returns the default policy. */
  MLUOP_MS_DEFORM_ATTN_BACKWARD_SMALL_CHANNEL = 1,
  /*!< Returns the small channel policy. */
} mluOpDeformAttnBackwardKernelPolicy_t;

static void policyFunc(mluOpHandle_t handle, const int32_t batch,
                       const int32_t num_query, const int32_t num_heads,
                       const int32_t num_levels, cnrtFunctionType_t *k_type,
                       cnrtDim3_t *k_dim,
                       mluOpDeformAttnBackwardKernelPolicy_t kernelPolicy) {
  size_t cluster_limit = mluop::runtime::getClusterLimitCapability(handle);
  size_t core_limit = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  k_dim->x = core_limit;
  int32_t total_num = batch * num_query * num_heads * num_levels;
  if (kernelPolicy == MLUOP_MS_DEFORM_ATTN_BACKWARD_SMALL_CHANNEL) {
    total_num = batch * num_query;
  }
  size_t total_num_align = CEIL_ALIGN(total_num, core_limit);
  k_dim->y = (total_num_align / core_limit) > cluster_limit
                 ? cluster_limit
                 : (total_num_align / core_limit);
  k_dim->z = 1;
  *k_type = CNRT_FUNC_TYPE_UNION1;
}

mluOpDeformAttnBackwardKernelPolicy_t msDeformAttnBackwardPolicyFunc(
    const int channels, const int num_levels, const int num_points,
    const int num_heads) {
  const int num_hlp = num_heads * num_levels * num_points;
  int num_per_time_theory = (MAX_NRAM_SIZE - num_levels * sizeof(float) -
                             3 * num_levels * sizeof(int32_t)) /
                            sizeof(float) / (8 * PAD_UP(channels, 32) + 28) /
                            PAD_UP((num_hlp), 32);

  if (num_per_time_theory >= 1) {
    return MLUOP_MS_DEFORM_ATTN_BACKWARD_SMALL_CHANNEL;
  }
  return MLUOP_MS_DEFORM_ATTN_BACKWARD_DEFAULT;
}

/* check user entrance param in mluOpMsDeformAttnBackward */
static mluOpStatus_t msDeformAttnBackwardParamCheck(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t value_desc,
    const void *value, const mluOpTensorDescriptor_t spatial_shapes_desc,
    const void *spatial_shapes,
    const mluOpTensorDescriptor_t level_start_index_desc,
    const void *level_start_index,
    const mluOpTensorDescriptor_t sampling_loc_desc, const void *sampling_loc,
    const mluOpTensorDescriptor_t attn_weight_desc, const void *attn_weight,
    const mluOpTensorDescriptor_t grad_output_desc, const void *grad_output,
    const int32_t im2col_step, const mluOpTensorDescriptor_t grad_value_desc,
    void *grad_value, const mluOpTensorDescriptor_t grad_sampling_loc_desc,
    void *grad_sampling_loc,
    const mluOpTensorDescriptor_t grad_attn_weight_desc, void *grad_attn_weight,
    bool *calc_grad_flag) {
  // check desc
  PARAM_CHECK(API, handle != NULL);
  PARAM_CHECK(API, value_desc != NULL);
  PARAM_CHECK(API, spatial_shapes_desc != NULL);
  PARAM_CHECK(API, level_start_index_desc != NULL);
  PARAM_CHECK(API, sampling_loc_desc != NULL);
  PARAM_CHECK(API, attn_weight_desc != NULL);
  PARAM_CHECK(API, grad_output_desc != NULL);
  PARAM_CHECK(API, grad_value_desc != NULL);
  PARAM_CHECK(API, grad_sampling_loc_desc != NULL);
  PARAM_CHECK(API, grad_attn_weight_desc != NULL);

  // check dim
  PARAM_CHECK(API, value_desc->dim == 4);
  PARAM_CHECK(API, spatial_shapes_desc->dim == 2);
  PARAM_CHECK(API, level_start_index_desc->dim == 1);
  PARAM_CHECK(API, sampling_loc_desc->dim == 6);
  PARAM_CHECK(API, attn_weight_desc->dim == 5);
  PARAM_CHECK(API, grad_output_desc->dim == 4);
  PARAM_CHECK(API, grad_value_desc->dim == 4);
  PARAM_CHECK(API, grad_sampling_loc_desc->dim == 6);
  PARAM_CHECK(API, grad_attn_weight_desc->dim == 5);

  // check datatype
  PARAM_CHECK(API, (value_desc->dtype == MLUOP_DTYPE_FLOAT &&
                    spatial_shapes_desc->dtype == MLUOP_DTYPE_INT32 &&
                    level_start_index_desc->dtype == MLUOP_DTYPE_INT32 &&
                    sampling_loc_desc->dtype == MLUOP_DTYPE_FLOAT &&
                    attn_weight_desc->dtype == MLUOP_DTYPE_FLOAT &&
                    grad_output_desc->dtype == MLUOP_DTYPE_FLOAT &&
                    grad_value_desc->dtype == MLUOP_DTYPE_FLOAT &&
                    grad_sampling_loc_desc->dtype == MLUOP_DTYPE_FLOAT &&
                    grad_attn_weight_desc->dtype == MLUOP_DTYPE_FLOAT));

  const int32_t num_key = value_desc->dims[1];
  const int32_t channels = value_desc->dims[3];
  const int32_t batch = attn_weight_desc->dims[0];
  const int32_t num_query = attn_weight_desc->dims[1];
  const int32_t num_heads = attn_weight_desc->dims[2];
  const int32_t num_levels = attn_weight_desc->dims[3];
  const int32_t num_points = attn_weight_desc->dims[4];
  // check input param
  const int32_t im2col_step_ = MIN(batch, im2col_step);
  PARAM_CHECK(API, im2col_step_ > 0);
  PARAM_CHECK(API, batch % im2col_step_ == 0);

  // check all the input relationship
  for (int32_t i = 0; i < value_desc->dim; ++i) {
    if (value_desc->dims[i] != grad_value_desc->dims[i]) {
      LOG(ERROR) << "[mluOpMsDeformAttnBackward] The shape of value should be "
                    "the same as grad_value."
                 << " But now value_desc->dims[" << i << "] is "
                 << value_desc->dims[i] << ", and grad_value_desc->dims[" << i
                 << "] is " << grad_value_desc->dims[i] << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  for (int32_t i = 0; i < sampling_loc_desc->dim; ++i) {
    if (sampling_loc_desc->dims[i] != grad_sampling_loc_desc->dims[i]) {
      LOG(ERROR) << "[mluOpMsDeformAttnBackward] The shape of "
                    "sampling_loc_desc should be the "
                    "same as grad_sampling_loc_desc."
                 << " But now sampling_loc_desc->dims[" << i << "] is "
                 << sampling_loc_desc->dims[i]
                 << ", and grad_sampling_loc_desc->dims[" << i << "] is "
                 << grad_sampling_loc_desc->dims[i] << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  for (int32_t i = 0; i < attn_weight_desc->dim; ++i) {
    if (attn_weight_desc->dims[i] != grad_attn_weight_desc->dims[i]) {
      LOG(ERROR) << "[mluOpMsDeformAttnBackward] The shape of "
                    "attn_weight_desc should be the "
                    "same as grad_attn_weight_desc."
                 << " But now attn_weight_desc->dims[" << i << "] is "
                 << attn_weight_desc->dims[i]
                 << ", and grad_attn_weight_desc->dims[" << i << "] is "
                 << grad_attn_weight_desc->dims[i] << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  PARAM_CHECK_EQ(API, value_desc->dims[0], batch);
  PARAM_CHECK_EQ(API, value_desc->dims[2], num_heads);

  PARAM_CHECK_EQ(API, spatial_shapes_desc->dims[0], num_levels);
  PARAM_CHECK_EQ(API, spatial_shapes_desc->dims[1], 2);

  PARAM_CHECK_EQ(API, level_start_index_desc->dims[0], num_levels);

  PARAM_CHECK_EQ(API, sampling_loc_desc->dims[0], batch);
  PARAM_CHECK_EQ(API, sampling_loc_desc->dims[1], num_query);
  PARAM_CHECK_EQ(API, sampling_loc_desc->dims[2], num_heads);
  PARAM_CHECK_EQ(API, sampling_loc_desc->dims[3], num_levels);
  PARAM_CHECK_EQ(API, sampling_loc_desc->dims[4], num_points);
  PARAM_CHECK_EQ(API, sampling_loc_desc->dims[5], 2);

  PARAM_CHECK_EQ(API, grad_output_desc->dims[0], batch);
  PARAM_CHECK_EQ(API, grad_output_desc->dims[1], num_query);
  PARAM_CHECK_EQ(API, grad_output_desc->dims[2], num_heads);
  PARAM_CHECK_EQ(API, grad_output_desc->dims[3], channels);

  TENSOR_NUM_CHECK(API, mluOpGetTensorElementNum(value_desc), LARGE_TENSOR_NUM,
                   "");
  TENSOR_NUM_CHECK(API, mluOpGetTensorElementNum(sampling_loc_desc),
                   LARGE_TENSOR_NUM, "");

  // check zero
  if (batch * channels * num_key * num_heads * num_query == 0) {
    LOG(ERROR) << "[mluOpMsDeformAttnBackward] The batch, channels, num_key, "
                  "num_heads or "
                  "num_query of the input is zero.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (num_levels == 0 || num_points == 0) {
    *calc_grad_flag = true;
    return MLUOP_STATUS_SUCCESS;
  }

  PARAM_CHECK(API, value != NULL);
  PARAM_CHECK(API, spatial_shapes != NULL);
  PARAM_CHECK(API, level_start_index != NULL);
  PARAM_CHECK(API, sampling_loc != NULL);
  PARAM_CHECK(API, attn_weight != NULL);
  PARAM_CHECK(API, grad_output != NULL);
  PARAM_CHECK(API, grad_value != NULL);
  PARAM_CHECK(API, grad_sampling_loc != NULL);
  PARAM_CHECK(API, grad_attn_weight != NULL);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpMsDeformAttnBackward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t value_desc,
    const void *value, const mluOpTensorDescriptor_t spatial_shapes_desc,
    const void *spatial_shapes,
    const mluOpTensorDescriptor_t level_start_index_desc,
    const void *level_start_index,
    const mluOpTensorDescriptor_t sampling_loc_desc, const void *sampling_loc,
    const mluOpTensorDescriptor_t attn_weight_desc, const void *attn_weight,
    const mluOpTensorDescriptor_t grad_output_desc, const void *grad_output,
    const int32_t im2col_step, const mluOpTensorDescriptor_t grad_value_desc,
    void *grad_value, const mluOpTensorDescriptor_t grad_sampling_loc_desc,
    void *grad_sampling_loc,
    const mluOpTensorDescriptor_t grad_attn_weight_desc,
    void *grad_attn_weight) {
  // entrance param check
  bool calc_grad_flag = false;
  mluOpStatus_t param_check_status = msDeformAttnBackwardParamCheck(
      handle, value_desc, value, spatial_shapes_desc, spatial_shapes,
      level_start_index_desc, level_start_index, sampling_loc_desc,
      sampling_loc, attn_weight_desc, attn_weight, grad_output_desc,
      grad_output, im2col_step, grad_value_desc, grad_value,
      grad_sampling_loc_desc, grad_sampling_loc, grad_attn_weight_desc,
      grad_attn_weight, &calc_grad_flag);
  if (MLUOP_STATUS_SUCCESS != param_check_status) {
    return param_check_status;
  }
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("ms_deform_attn_backward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA_REAL(true, "value", value, value_desc);
    GEN_CASE_DATA_REAL(true, "spatial_shapes", spatial_shapes,
                       spatial_shapes_desc);
    GEN_CASE_DATA_REAL(true, "level_start_index", level_start_index,
                       level_start_index_desc);
    GEN_CASE_DATA_REAL(true, "sampling_loc", sampling_loc, sampling_loc_desc);
    GEN_CASE_DATA_REAL(true, "attn_weight", attn_weight, attn_weight_desc);
    GEN_CASE_DATA_REAL(true, "grad_output", grad_output, grad_output_desc);
    GEN_CASE_DATA(false, "grad_value", grad_value, grad_value_desc, 0, 0);
    GEN_CASE_DATA(false, "grad_sampling_loc", grad_sampling_loc,
                  grad_sampling_loc_desc, 0, 0);
    GEN_CASE_DATA(false, "grad_attn_weight", grad_attn_weight,
                  grad_attn_weight_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "ms_deform_attn_backward", "im2col_step",
                             im2col_step);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  if (calc_grad_flag) {
    uint64_t fill_value = 0x0;
    PARAM_CHECK(API,
                MLUOP_STATUS_SUCCESS ==
                    mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST, &fill_value,
                                 grad_value_desc, grad_value));
    GEN_CASE_END();
    return MLUOP_STATUS_SUCCESS;
  }
  VLOG(5) << "[mluOpMsDeformAttnBackward] mluOpFill_v3 start.";
  uint64_t fill_value = 0x0;

  PARAM_CHECK(API, MLUOP_STATUS_SUCCESS ==
                       mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST,
                                    &fill_value, grad_value_desc, grad_value));
  PARAM_CHECK(API,
              MLUOP_STATUS_SUCCESS ==
                  mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST, &fill_value,
                               grad_sampling_loc_desc, grad_sampling_loc));
  PARAM_CHECK(API,
              MLUOP_STATUS_SUCCESS ==
                  mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST, &fill_value,
                               grad_attn_weight_desc, grad_attn_weight));

  VLOG(5) << "[mluOpMsDeformAttnBackward] mluOpFill_v3 end.";

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  const int32_t spatial_size = value_desc->dims[1];
  const int32_t batch = attn_weight_desc->dims[0];
  const int32_t channels = value_desc->dims[3];
  const int32_t num_query = attn_weight_desc->dims[1];
  const int32_t num_heads = attn_weight_desc->dims[2];
  const int32_t num_levels = attn_weight_desc->dims[3];
  const int32_t num_points = attn_weight_desc->dims[4];
  // generate mluOpMsDeformAttnBackward prototxt start!

  mluOpDeformAttnBackwardKernelPolicy_t kernelPolicy =
      msDeformAttnBackwardPolicyFunc(channels, num_levels, num_points,
                                     num_heads);

  policyFunc(handle, batch, num_query, num_heads, num_levels, &k_type, &k_dim,
             kernelPolicy);
  VLOG(5) << "Launch Kernel MLUUnion1KernelMsDeformAttnBackward<<<Union"
          << k_type / CORE_DIM << ", " << k_dim.x << ", " << k_dim.y << ", "
          << k_dim.z << ">>>";
  switch (kernelPolicy) {
    case MLUOP_MS_DEFORM_ATTN_BACKWARD_DEFAULT: {
      KERNEL_CHECK((KernelMsDeformAttnBackwardDefault(
          k_dim, k_type, handle->queue, (float *)value,
          (int32_t *)spatial_shapes, (int32_t *)level_start_index,
          (float *)sampling_loc, (float *)attn_weight, (float *)grad_output,
          batch, spatial_size, num_heads, channels, num_levels, num_query,
          num_points, (float *)grad_value, (float *)grad_sampling_loc,
          (float *)grad_attn_weight)));
    } break;
    case MLUOP_MS_DEFORM_ATTN_BACKWARD_SMALL_CHANNEL: {
      KERNEL_CHECK((KernelMsDeformAttnBackwardSmallChannels(
          k_dim, k_type, handle->queue, (float *)value,
          (int32_t *)spatial_shapes, (int32_t *)level_start_index,
          (float *)sampling_loc, (float *)attn_weight, (float *)grad_output,
          batch, spatial_size, num_heads, channels, num_levels, num_query,
          num_points, (float *)grad_value, (float *)grad_sampling_loc,
          (float *)grad_attn_weight)));
    }
    default: {
      VLOG(5) << "Not Implemented.";
    }
  }

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
