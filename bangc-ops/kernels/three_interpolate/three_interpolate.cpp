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

#define WEIGHT_N_LIMIT_SIZE 6
#define INDEX_N_LIMIT_SIZE 6
#define OUTPUT_NC_LIMIT_SIZE 2
#define INPUT_MC_LIMIT_SIZE 2
#define INDEX_TYPE_CONVERT_N_LIMIT_SIZE 3
#define REPEATS_TIME_GAP 1
#define REPEATS_TIME_MOD 0
#define INPUT_NC_LIMIT_SIZE 2
#define OUTPUT_MC_LIMIT_SIZE 2

static void PolicyFuncThreeInterpolate(
    const mluOpHandle_t &handle, const mluOpTensorDescriptor_t &desc,
    const int b, const int c, const int m, const int n, const bool is_forward,
    cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type, int *c_limit_size,
    int *m_limit_size, int *n_limit_size) {
  size_t cluster_num = mluop::runtime::getClusterLimitCapability(handle);
  size_t core_in_cluster = handle->core_num_per_cluster;
  size_t cores_in_device = cluster_num * core_in_cluster;
  int input_size = sizeof(float);
  if (desc->dtype == MLUOP_DTYPE_HALF) {
    input_size /= 2;
  }
  int align_base_128 = NFU_ALIGN_SIZE / input_size;
  int max_nram_size = handle->nram_size;
  // according to the kernel nram space usage, simply init the c_limit, m_limit
  // and n_limit
  int m_limit =
      (max_nram_size -
       (OUTPUT_NC_LIMIT_SIZE * align_base_128 * align_base_128 +
        align_base_128 * WEIGHT_N_LIMIT_SIZE) *
           input_size +
       align_base_128 * INDEX_N_LIMIT_SIZE * sizeof(int32_t) +
       align_base_128 * INDEX_TYPE_CONVERT_N_LIMIT_SIZE * sizeof(float)) /
      (align_base_128 * INPUT_MC_LIMIT_SIZE);
  int c_limit = align_base_128;
  int n_limit = align_base_128;
  int c_aligned = CEIL_ALIGN(c, align_base_128);
  int m_aligned = CEIL_ALIGN(m, align_base_128);
  int n_aligned = CEIL_ALIGN(n, align_base_128);
  int limit_size = c_limit * m_limit * n_limit;
  // if nram space is big enough, the limit_size may overflow, reset limit_size
  if (limit_size <= 0) {
    limit_size = handle->nram_size;
  }
  // find the best size of c_limit, m_limit and n_limit
  while (true) {
    int m_max_use_size = limit_size / (align_base_128 * align_base_128);
    int c_max_use_size = m_max_use_size;
    int n_max_use_size = m_max_use_size;
    bool c_is_second_priority = true;
    // m_limit has the first priority
    if (m_aligned <= m_max_use_size) {
      m_limit = m_aligned;
    } else if (m_aligned > m_max_use_size) {
      m_limit = FLOOR_ALIGN(m_max_use_size, align_base_128);
    }
    if (is_forward) {
      if (m_aligned * c_aligned <= (limit_size / align_base_128) ||
          c_aligned > m_aligned) {
        // c_limit has the second priority in this case
        c_is_second_priority = true;
      } else {
        // n_limit has the second priority in this case
        c_is_second_priority = false;
        n_max_use_size = limit_size / (m_limit * align_base_128);
      }
    }
    if (c_is_second_priority) {
      c_max_use_size = limit_size / (m_limit * align_base_128);
      if (c_aligned <= c_max_use_size) {
        c_limit = c_aligned;
      } else if (c_aligned > c_max_use_size) {
        c_limit = FLOOR_ALIGN(c_max_use_size, align_base_128);
      }
      // n_limit has the third priority in this case
      n_max_use_size = limit_size / (c_limit * m_limit);
    }
    // get the n_limit according to n_max_use_size
    if (n_aligned <= n_max_use_size) {
      n_limit = n_aligned;
    } else if (n_aligned > n_max_use_size) {
      n_limit = FLOOR_ALIGN(n_max_use_size, align_base_128);
    }
    // get the best n_limit to use the most clusters
    int best_n_aligned_limit = CEIL_ALIGN(n, n_limit);
    int best_n_limit = n_limit;
    // get the best m_limit to use the most clusters
    int best_m_aligned_limit = CEIL_ALIGN(m, m_limit);
    int best_m_limit = m_limit;
    // try to find the best n_limit or m_limit that can make full use of the
    // device's cores
    while (true) {
      int n_aligned_limit = CEIL_ALIGN(n, n_limit);
      int m_aligned_limit = CEIL_ALIGN(m, m_limit);
      // record the best n_limit
      int best_repeats = b * best_n_aligned_limit / best_n_limit;
      int current_repeats = b * n_aligned_limit / n_limit;
      if (!is_forward) {
        best_repeats = b * best_m_aligned_limit / best_m_limit;
        current_repeats = b * m_aligned_limit / m_limit;
      }
      int best_repeats_mod = best_repeats % cores_in_device;
      int best_repeats_div = best_repeats / cores_in_device;
      int current_repeats_mod = current_repeats % cores_in_device;
      int current_repeats_div = current_repeats / cores_in_device;
      bool update_best = false;
      if ((best_repeats_div - current_repeats_div) > REPEATS_TIME_GAP) {
        // minimize the repeats time
        update_best = true;
      } else if (((best_repeats_div - current_repeats_div) ==
                  REPEATS_TIME_GAP) &&
                 (best_repeats_mod ||
                  (best_repeats_mod == current_repeats_mod &&
                   current_repeats_mod == REPEATS_TIME_MOD))) {
        // when current repeats time is only one number less than the best
        // check the repeats mod to make use of the most cores
        update_best = true;
      } else if (best_repeats_div == current_repeats_div) {
        // when repeats time is the same, make use of the most cores
        if ((best_repeats_mod && current_repeats_mod &&
             (best_repeats_mod < current_repeats_mod)) ||
            (best_repeats_mod && current_repeats_mod == REPEATS_TIME_MOD)) {
          update_best = true;
        }
      } else if (((current_repeats_div - best_repeats_div) ==
                  REPEATS_TIME_GAP) &&
                 best_repeats_mod && current_repeats_mod == REPEATS_TIME_MOD) {
        // when current repeats time is only one number more than the best
        // check the repeats mod to make use of the most cores
        update_best = true;
      }
      if (is_forward) {
        // update the best_n_limit for forward
        if (update_best) {
          best_n_limit = n_limit;
          best_n_aligned_limit = n_aligned_limit;
        }
        n_limit = n_limit - align_base_128;
        if (n_limit < align_base_128) {
          // stop when n_limit reach to the minimum
          break;
        }
        n_limit = FLOOR_ALIGN(n_limit, align_base_128);
      } else {
        // update the best_m_limit for backward
        if (update_best) {
          best_m_limit = m_limit;
          best_m_aligned_limit = m_aligned_limit;
        }
        m_limit = m_limit - align_base_128;
        if (m_limit < align_base_128) {
          // stop when m_limit reach to the minimum
          break;
        }
        m_limit = FLOOR_ALIGN(m_limit, align_base_128);
      }
    }
    n_limit = best_n_limit;
    m_limit = best_m_limit;
    if (!c_is_second_priority) {
      // c_limit has the third priority for forward in this case
      c_max_use_size = limit_size / (m_limit * n_limit);
    }
    if (!is_forward) {
      // c_limit has the second priority for backward
      c_max_use_size = limit_size / (m_limit * align_base_128);
    }
    if (c_aligned <= c_max_use_size) {
      c_limit = c_aligned;
    } else if (c_aligned > c_max_use_size) {
      c_limit = FLOOR_ALIGN(c_max_use_size, align_base_128);
    }
    if (!is_forward) {
      // n_limit has the third priority
      n_max_use_size = limit_size / (m_limit * c_limit);
      if (n_aligned <= n_max_use_size) {
        n_limit = n_aligned;
      } else if (n_aligned > n_max_use_size) {
        n_limit = FLOOR_ALIGN(n_max_use_size, align_base_128);
      }
    }
    // the kernel's total nram space use formula
    int total_nram_size =
        (std::max(c_limit * m_limit, c_limit * n_limit) + m_limit * c_limit +
         OUTPUT_NC_LIMIT_SIZE * n_limit * c_limit +
         n_limit * WEIGHT_N_LIMIT_SIZE) *
            input_size +
        n_limit * INDEX_N_LIMIT_SIZE * sizeof(int32_t) +
        n_limit * INDEX_TYPE_CONVERT_N_LIMIT_SIZE * sizeof(float);
    if (!is_forward) {
      total_nram_size += c_limit * input_size;
    }
    if (total_nram_size <= max_nram_size) {
      // according to the current c_limit, m_limit and n_limit, the
      // total_nram_size meets the max nram size restrictions
      break;
    } else {
      limit_size =
          limit_size - align_base_128 * align_base_128 * align_base_128;
    }
  }
  *c_limit_size = c_limit;
  *m_limit_size = m_limit;
  *n_limit_size = n_limit;

  int n_aligned_limit = CEIL_ALIGN(n, n_limit);
  n_limit = n_limit > n_aligned_limit ? n_aligned_limit : n_limit;
  int m_aligned_limit = CEIL_ALIGN(m, m_limit);
  m_limit = m_limit > m_aligned_limit ? m_aligned_limit : m_limit;

  uint32_t use_cluster =
      (b * n_aligned_limit / n_limit + core_in_cluster - 1) / core_in_cluster;

  if (!is_forward) {
    use_cluster =
        (b * m_aligned_limit / m_limit + core_in_cluster - 1) / core_in_cluster;
  }
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = core_in_cluster;
  k_dim->y = use_cluster > cluster_num ? cluster_num : use_cluster;
  k_dim->z = 1;
}

mluOpStatus_t ThreeInterpolateForwardParamCheck(
    const std::string &op_name, const mluOpHandle_t handle,
    const mluOpTensorDescriptor_t features_desc, const void *features,
    const mluOpTensorDescriptor_t indices_desc, const void *indices,
    const mluOpTensorDescriptor_t weights_desc, const void *weights,
    const mluOpTensorDescriptor_t output_desc, const void *output,
    bool &zero_element) {
  // check handle and descriptor
  PARAM_CHECK(op_name, handle != NULL);
  PARAM_CHECK(op_name, features_desc != NULL);
  PARAM_CHECK(op_name, indices_desc != NULL);
  PARAM_CHECK(op_name, weights_desc != NULL);
  PARAM_CHECK(op_name, output_desc != NULL);
  // check dim
  PARAM_CHECK(op_name, features_desc->dim == 3);
  PARAM_CHECK(op_name, indices_desc->dim == 3);
  PARAM_CHECK(op_name, weights_desc->dim == 3);
  PARAM_CHECK(op_name, output_desc->dim == 3);
  // check layout
  PARAM_CHECK(op_name, features_desc->layout == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(op_name, indices_desc->layout == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(op_name, weights_desc->layout == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(op_name, output_desc->layout == MLUOP_LAYOUT_ARRAY);
  // check data type
  PARAM_CHECK(op_name, features_desc->dtype == weights_desc->dtype);
  PARAM_CHECK(op_name, indices_desc->dtype == MLUOP_DTYPE_INT32);
  PARAM_CHECK(op_name, weights_desc->dtype == output_desc->dtype);
  PARAM_CHECK(op_name, (output_desc->dtype == MLUOP_DTYPE_HALF ||
                        output_desc->dtype == MLUOP_DTYPE_FLOAT));
  // check shape
  if (features_desc->dims[0] != indices_desc->dims[0]) {
    LOG(ERROR) << op_name << "Check failed: features_desc->dims[0] should be "
                             "equal to indices_desc->dims[0].";
    return MLUOP_STATUS_BAD_PARAM;
  }
  for (int i = 0; i < indices_desc->dim; ++i) {
    if (indices_desc->dims[i] != weights_desc->dims[i]) {
      LOG(ERROR) << op_name << " Check failed: indices_desc->dims[" << i
                 << "] should be equal to weightss_desc->dims[" << i << "].";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  if (weights_desc->dims[2] != 3) {
    LOG(ERROR) << op_name
               << " Check failed: weights_desc->dims[2] should be equal to 3.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  for (int i = 0; i < output_desc->dim - 1; ++i) {
    if (output_desc->dims[i] != features_desc->dims[i]) {
      LOG(ERROR) << op_name << " Check failed: output_desc->dims[" << i
                 << "] should be equal to features_desc->dims[" << i << "].";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  if (output_desc->dims[2] != indices_desc->dims[1]) {
    LOG(ERROR) << op_name << " Check failed: output_desc->dims[2] should be "
                             "equal to indices_desc->dims[1].";
    return MLUOP_STATUS_BAD_PARAM;
  }
  // check large tensor
  if ((mluOpGetTensorElementNum(features_desc) >= LARGE_TENSOR_NUM) ||
      (mluOpGetTensorElementNum(indices_desc) >= LARGE_TENSOR_NUM) ||
      (mluOpGetTensorElementNum(weights_desc) >= LARGE_TENSOR_NUM) ||
      (mluOpGetTensorElementNum(output_desc) >= LARGE_TENSOR_NUM)) {
    LOG(ERROR) << op_name << " Overflow max tensor num."
               << " Currently, MLU-OPS supports tensor num smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  // check zero element
  if ((mluOpGetTensorElementNum(indices_desc) == 0) ||
      (mluOpGetTensorElementNum(weights_desc) == 0) ||
      (mluOpGetTensorElementNum(output_desc) == 0)) {
    LOG(ERROR) << op_name << " Zero element tensor failure.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  // the shape of features is [B, C, M]. currently only M equal to 0 is
  // supported
  if (features_desc->dims[2] == 0) {
    VLOG(5) << op_name << " Skip zero element tensor.";
    zero_element = true;
    return MLUOP_STATUS_SUCCESS;
  }

  PARAM_CHECK(op_name, features != NULL);
  PARAM_CHECK(op_name, indices != NULL);
  PARAM_CHECK(op_name, weights != NULL);
  PARAM_CHECK(op_name, output != NULL);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t ThreeInterpolateBackwardParamCheck(
    const std::string &op_name, const mluOpHandle_t handle,
    const mluOpTensorDescriptor_t grad_output_desc, const void *grad_output,
    const mluOpTensorDescriptor_t indices_desc, const void *indices,
    const mluOpTensorDescriptor_t weights_desc, const void *weights,
    const mluOpTensorDescriptor_t grad_features_desc,
    const void *grad_features, bool &zero_element) {
  // check handle and descriptor
  PARAM_CHECK(op_name, handle != NULL);
  PARAM_CHECK(op_name, grad_output_desc != NULL);
  PARAM_CHECK(op_name, indices_desc != NULL);
  PARAM_CHECK(op_name, weights_desc != NULL);
  PARAM_CHECK(op_name, grad_features_desc != NULL);
  // check dim
  PARAM_CHECK(op_name, grad_output_desc->dim == 3);
  PARAM_CHECK(op_name, indices_desc->dim == 3);
  PARAM_CHECK(op_name, weights_desc->dim == 3);
  PARAM_CHECK(op_name, grad_features_desc->dim == 3);
  // check layout
  PARAM_CHECK(op_name, grad_output_desc->layout == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(op_name, indices_desc->layout == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(op_name, weights_desc->layout == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK(op_name, grad_features_desc->layout == MLUOP_LAYOUT_ARRAY);
  // check data type
  PARAM_CHECK(op_name, grad_output_desc->dtype == weights_desc->dtype);
  PARAM_CHECK(op_name, indices_desc->dtype == MLUOP_DTYPE_INT32);
  PARAM_CHECK(op_name, weights_desc->dtype == grad_features_desc->dtype);
  PARAM_CHECK(op_name, (grad_output_desc->dtype == MLUOP_DTYPE_HALF ||
                        grad_output_desc->dtype == MLUOP_DTYPE_FLOAT));
  // check shape
  if (grad_output_desc->dims[0] != indices_desc->dims[0]) {
    LOG(ERROR) << op_name
               << " Check failed: grad_output_desc->dims[0] should be "
                  "equal to indices_desc->dims[0].";
    return MLUOP_STATUS_BAD_PARAM;
  }
  for (int i = 0; i < indices_desc->dim; ++i) {
    if (indices_desc->dims[i] != weights_desc->dims[i]) {
      LOG(ERROR) << op_name << " Check failed: indices_desc->dims[" << i
                 << "] should be equal to weightss_desc->dims[" << i << "].";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  if (weights_desc->dims[2] != 3) {
    LOG(ERROR) << op_name
               << " Check failed: weights_desc->dims[2] should be equal to 3.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  for (int i = 0; i < grad_output_desc->dim - 1; ++i) {
    if (grad_output_desc->dims[i] != grad_features_desc->dims[i]) {
      LOG(ERROR) << op_name << " Check failed: grad_output_desc->dims[" << i
                 << "] should be equal to grad_features_desc->dims[" << i
                 << "].";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  if (grad_output_desc->dims[2] != indices_desc->dims[1]) {
    LOG(ERROR) << op_name
               << " Check failed: grad_output_desc->dims[2] should be "
                  "equal to indices_desc->dims[1].";
    return MLUOP_STATUS_BAD_PARAM;
  }
  // check large tensor
  if ((mluOpGetTensorElementNum(grad_output_desc) >= LARGE_TENSOR_NUM) ||
      (mluOpGetTensorElementNum(indices_desc) >= LARGE_TENSOR_NUM) ||
      (mluOpGetTensorElementNum(weights_desc) >= LARGE_TENSOR_NUM) ||
      (mluOpGetTensorElementNum(grad_features_desc) >= LARGE_TENSOR_NUM)) {
    LOG(ERROR) << op_name << " Overflow max tensor num."
               << " Currently, MLU-OPS supports tensor num smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  // check zero element
  if ((mluOpGetTensorElementNum(grad_output_desc) == 0) ||
      (mluOpGetTensorElementNum(indices_desc) == 0) ||
      (mluOpGetTensorElementNum(weights_desc) == 0)) {
    LOG(ERROR) << op_name << " Zero element tensor failure.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  // the shape of grad_features is [B, C, M]. currently only M equal to 0 is
  // supported
  if (grad_features_desc->dims[2] == 0) {
    VLOG(5) << op_name << " Skip zero element tensor.";
    zero_element = true;
    return MLUOP_STATUS_SUCCESS;
  }

  PARAM_CHECK(op_name, grad_output != NULL);
  PARAM_CHECK(op_name, indices != NULL);
  PARAM_CHECK(op_name, weights != NULL);
  PARAM_CHECK(op_name, grad_features != NULL);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpThreeInterpolateForward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t features_desc,
    const void *features, const mluOpTensorDescriptor_t indices_desc,
    const void *indices, const mluOpTensorDescriptor_t weights_desc,
    const void *weights, const mluOpTensorDescriptor_t output_desc,
    void *output) {
  bool zero_element = false;
  mluOpStatus_t param_check = ThreeInterpolateForwardParamCheck(
      "[mluOpThreeInterpolateForward]", handle, features_desc, features,
      indices_desc, indices, weights_desc, weights, output_desc, output,
      zero_element);
  if (zero_element == true) {
    return MLUOP_STATUS_SUCCESS;
  }
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }
  int b = features_desc->dims[0];
  int c = features_desc->dims[1];
  int m = features_desc->dims[2];
  int n = output_desc->dims[2];

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("three_interpolate_forward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "features", features, features_desc, 0, 100);
    GEN_CASE_DATA(true, "indices", indices, indices_desc, 0, m - 1);
    GEN_CASE_DATA(true, "weights", weights, weights_desc, 0, 1);
    GEN_CASE_DATA(false, "output", output, output_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  int input_size = sizeof(float);
  if (features_desc->dtype == MLUOP_DTYPE_HALF) {
    input_size /= 2;
  }
  int c_limit_size = NFU_ALIGN_SIZE / input_size;
  int m_limit_size = c_limit_size;
  int n_limit_size = c_limit_size;
  PolicyFuncThreeInterpolate(handle, features_desc, b, c, m, n, true, &k_dim,
                             &k_type, &c_limit_size, &m_limit_size,
                             &n_limit_size);
  VLOG(5) << "[mluOpThreeInterpolateForward] launch kernel policyFunc["
          << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << "]";
  if (features_desc->dtype == MLUOP_DTYPE_HALF) {
    VLOG(5) << "Kernel mluOpUnionKernelThreeInterpolateForwardHalf";
    KERNEL_CHECK((mluOpUnionKernelThreeInterpolateForwardHalf(
        k_dim, k_type, handle->queue, features, indices, weights, b, c, m, n,
        c_limit_size, m_limit_size, n_limit_size, output)));
  } else {
    VLOG(5) << "Kernel mluOpUnionKernelThreeInterpolateForwardFloat";
    KERNEL_CHECK((mluOpUnionKernelThreeInterpolateForwardFloat(
        k_dim, k_type, handle->queue, features, indices, weights, b, c, m, n,
        c_limit_size, m_limit_size, n_limit_size, output)));
  }
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpThreeInterpolateBackward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t grad_output_desc,
    const void *grad_output, const mluOpTensorDescriptor_t indices_desc,
    const void *indices, const mluOpTensorDescriptor_t weights_desc,
    const void *weights, const mluOpTensorDescriptor_t grad_features_desc,
    void *grad_features) {
  bool zero_element = false;
  mluOpStatus_t param_check = ThreeInterpolateBackwardParamCheck(
      "[mluOpThreeInterpolateBackward]", handle, grad_output_desc, grad_output,
      indices_desc, indices, weights_desc, weights, grad_features_desc,
      grad_features, zero_element);
  if (zero_element == true) {
    return MLUOP_STATUS_SUCCESS;
  }
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }
  int b = grad_output_desc->dims[0];
  int c = grad_output_desc->dims[1];
  int n = grad_output_desc->dims[2];
  int m = grad_features_desc->dims[2];

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("three_interpolate_backward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "grad_output", grad_output, grad_output_desc, 0, 100);
    GEN_CASE_DATA(true, "indices", indices, indices_desc, 0, m - 1);
    GEN_CASE_DATA(true, "weights", weights, weights_desc, 0, 1);
    GEN_CASE_DATA(false, "grad_features", grad_features, grad_features_desc, 0,
                  0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  int input_size = sizeof(float);
  if (grad_output_desc->dtype == MLUOP_DTYPE_HALF) {
    input_size /= 2;
  }
  int c_limit_size = NFU_ALIGN_SIZE / input_size;
  int m_limit_size = c_limit_size;
  int n_limit_size = c_limit_size;
  PolicyFuncThreeInterpolate(handle, grad_output_desc, b, c, m, n, false,
                             &k_dim, &k_type, &c_limit_size, &m_limit_size,
                             &n_limit_size);
  VLOG(5) << "[mluOpThreeInterpolateBackward] launch kernel policyFunc["
          << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << "]";
  if (grad_output_desc->dtype == MLUOP_DTYPE_HALF) {
    VLOG(5) << "Kernel mluOpUnionKernelThreeInterpolateBackwardHalf";
    KERNEL_CHECK((mluOpUnionKernelThreeInterpolateBackwardHalf(
        k_dim, k_type, handle->queue, grad_output, indices, weights, b, c, m, n,
        c_limit_size, m_limit_size, n_limit_size, grad_features)));
  } else {
    VLOG(5) << "Kernel mluOpUnionKernelThreeInterpolateBackwardFloat";
    KERNEL_CHECK((mluOpUnionKernelThreeInterpolateBackwardFloat(
        k_dim, k_type, handle->queue, grad_output, indices, weights, b, c, m, n,
        c_limit_size, m_limit_size, n_limit_size, grad_features)));
  }
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
