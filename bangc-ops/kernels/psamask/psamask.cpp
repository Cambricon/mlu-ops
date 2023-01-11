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
#include <algorithm>
#include <string>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/tensor.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "kernels/kernel.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"

#define COMPUTE_COUNT_ALIGN 64

inline void getNHWC(const mluOpTensorDescriptor_t desc, int *n, int *h, int *w,
                    int *c) {
  *n = desc->dims[0];
  *h = desc->dims[1];
  *w = desc->dims[2];
  *c = desc->dims[3];
}

static void policyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim_ptr,
                cnrtFunctionType_t *f_type_ptr, PartitionSeg *partition_ptr,
                int n, int h_feature) {
  unsigned int core_dim = handle->core_num_per_cluster;
  unsigned int cluster_num = mluop::runtime::getClusterLimitCapability(handle);
  unsigned int use_cluster_num = (uint32_t)cluster_num;
  unsigned int use_core_num = (uint32_t)core_dim;
  VLOG(5) << "core_dim:" << core_dim;
  VLOG(5) << "cluster_num:" << cluster_num;
  if (n >= cluster_num || n >= h_feature) {
    VLOG(5) << "cluster partition n";
    partition_ptr->cluster_partition = PARTITION_N;
    partition_ptr->n_per_cluster = (n + cluster_num - 1) / cluster_num;
    partition_ptr->h_per_cluster = h_feature;
    use_cluster_num =
        (n + partition_ptr->n_per_cluster - 1) / partition_ptr->n_per_cluster;
  } else {
    VLOG(5) << "cluster partition h";
    partition_ptr->cluster_partition = PARTITION_H;
    partition_ptr->h_per_cluster = (h_feature + cluster_num - 1) / cluster_num;
    partition_ptr->n_per_cluster = n;
    use_cluster_num = (h_feature + partition_ptr->h_per_cluster - 1) /
                      partition_ptr->h_per_cluster;
  }

  if (partition_ptr->n_per_cluster >= core_dim ||
      partition_ptr->n_per_cluster >= partition_ptr->h_per_cluster) {
    VLOG(5) << "core partition n";
    partition_ptr->core_partition = PARTITION_N;
    partition_ptr->n_per_core =
        (partition_ptr->n_per_cluster + core_dim - 1) / core_dim;
    partition_ptr->h_per_core = partition_ptr->h_per_cluster;
    use_core_num =
        (partition_ptr->n_per_cluster + partition_ptr->n_per_core - 1) /
        partition_ptr->n_per_core;
  } else {
    VLOG(5) << "core partition h";
    partition_ptr->core_partition = PARTITION_H;
    partition_ptr->h_per_core =
        (partition_ptr->h_per_cluster + core_dim - 1) / core_dim;
    partition_ptr->n_per_core = partition_ptr->n_per_cluster;
    use_core_num =
        (partition_ptr->h_per_cluster + partition_ptr->h_per_core - 1) /
        partition_ptr->h_per_core;
  }
  VLOG(5) << "n_per_core:" << partition_ptr->n_per_core
          << ",h_per_core:" << partition_ptr->h_per_core
          << ",use_cluster_num:" << use_cluster_num;
  *k_dim_ptr = {core_dim, use_cluster_num, 1};
}

mluOpStatus_t findLimit(mluOpHandle_t handle, const int shape_core_n,
                        const int shape_core_h, const int shape_core_w,
                        const int shape_core_ci, const int shape_core_co,
                        const int input_bytes, int *limit_n_seg_ptr,
                        int *limit_h_seg_ptr, int *limit_w_seg_ptr,
                        const int psa_type, const std::string api) {
  bool need_temp = (bool)psa_type;
  int limit_n_seg = shape_core_n;
  int limit_h_seg = shape_core_h;
  int limit_w_seg = shape_core_w;

  int max_nram_size = handle->nram_size;
  int align_base_128 = NFU_ALIGN_SIZE / input_bytes;
  int align_base_64 = COMPUTE_COUNT_ALIGN / input_bytes;
  int align_co = CEIL_ALIGN(shape_core_co, align_base_64);
  int align_w = CEIL_ALIGN(shape_core_w, align_base_64);
  int align_hw = CEIL_ALIGN(shape_core_h * shape_core_w, align_base_64);
  int max_num = max_nram_size / input_bytes;
  VLOG(5) << "shape_core_n:" << shape_core_n;
  VLOG(5) << "shape_core_h:" << shape_core_h;
  VLOG(5) << "shape_core_w:" << shape_core_w;

  int n_limit =
      max_num /
      (CEIL_ALIGN(shape_core_h * shape_core_w * shape_core_ci, align_base_128) +
       align_hw * align_co * (1 + need_temp));
  if (n_limit > 0) {
    n_limit = std::min(n_limit, shape_core_n);
    limit_n_seg = n_limit;
  } else {
    int h_limit =
        max_num / (CEIL_ALIGN(shape_core_w * shape_core_ci, align_base_128) +
                   align_w * align_co * (1 + need_temp));
    if (h_limit > 0) {
      h_limit = std::min(h_limit, shape_core_h);
      limit_h_seg = h_limit;
      limit_n_seg = 1;
    } else {
      int w_limit =
          max_num / (CEIL_ALIGN(shape_core_ci, align_base_128) +
                     CEIL_ALIGN(align_co, align_base_128) * (1 + need_temp));
      if (w_limit > 0 && w_limit >= (COMPUTE_COUNT_ALIGN / input_bytes)) {
        w_limit = std::min(w_limit, shape_core_w);
        w_limit = w_limit / (COMPUTE_COUNT_ALIGN / input_bytes) *
                  (COMPUTE_COUNT_ALIGN / input_bytes);
        limit_w_seg = w_limit;
        limit_h_seg = 1;
        limit_n_seg = 1;
      } else {
        LOG(ERROR) << api
                   << " Check failed: Psamask mode: the size of input channel "
                      "is too large.";
        return MLUOP_STATUS_NOT_SUPPORTED;
      }
    }
  }
  VLOG(5) << "limit_n_seg:" << limit_n_seg;
  VLOG(5) << "limit_h_seg:" << limit_h_seg;
  VLOG(5) << "limit_w_seg:" << limit_w_seg;
  *limit_n_seg_ptr = limit_n_seg;
  *limit_h_seg_ptr = limit_h_seg;
  *limit_w_seg_ptr = limit_w_seg;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t checkParams(const mluOpTensorDescriptor_t input_desc,
                          const mluOpTensorDescriptor_t output_desc,
                          const int h_mask, const int w_mask,
                          const std::string api) {
  PARAM_CHECK(api, input_desc->dim == 4);
  PARAM_CHECK(api, output_desc->dim == 4);
  int x_n, x_h, x_w, x_c;
  int y_n, y_h, y_w, y_c;
  getNHWC(input_desc, &x_n, &x_h, &x_w, &x_c);
  getNHWC(output_desc, &y_n, &y_h, &y_w, &y_c);
  if (input_desc->layout != MLUOP_LAYOUT_NHWC ||
      output_desc->layout != MLUOP_LAYOUT_NHWC) {
    LOG(ERROR) << api
               << " Check failed: Only support MLUOP_LAYOUT_NHWC input and "
                  "output, but now input is "
               << mluop::getNameOfTensorLayout(input_desc->layout)
               << ", and output is "
               << mluop::getNameOfTensorLayout(output_desc->layout) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (input_desc->dtype != output_desc->dtype ||
      input_desc->dtype != MLUOP_DTYPE_FLOAT) {
    LOG(ERROR)
        << api
        << " Check failed: The data type of input and output should be float.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (x_n != y_n || x_h != y_h || x_w != y_w) {
    LOG(ERROR) << api
               << " Check failed: The size of input and output should be the "
                  "same, except channel.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (h_mask * w_mask != x_c) {
    LOG(ERROR) << api
               << " Check failed: The size of input channel is wrong, it needs "
                  "to be the "
                  "same as the h_mask * w_mask.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (y_h * y_w != y_c) {
    LOG(ERROR) << api
               << " Check failed: The size of output channel is wrong, it "
                  "needs to be the "
                  "same as the h_feature * w_feature.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpPsamaskForward(mluOpHandle_t handle, const int psa_type,
                                  const mluOpTensorDescriptor_t x_desc,
                                  const void *x, const int h_mask,
                                  const int w_mask,
                                  const mluOpTensorDescriptor_t y_desc,
                                  void *y) {
  const std::string api = "[mluOpPsamaskForward]";
  PARAM_CHECK(api, handle != nullptr);
  PARAM_CHECK(api, y_desc != nullptr);
  PARAM_CHECK(api, x_desc != nullptr);
  if (checkParams(x_desc, y_desc, h_mask, w_mask, api) !=
      MLUOP_STATUS_SUCCESS) {
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (!x_desc->total_element_num) {
    return MLUOP_STATUS_SUCCESS;
  }
  PARAM_CHECK(api, x != nullptr);
  PARAM_CHECK(api, y != nullptr);

  auto n = x_desc->dims[0];
  auto h_feature = x_desc->dims[1];
  auto w_feature = x_desc->dims[2];
  auto x_c = x_desc->dims[3];
  auto y_c = y_desc->dims[3];
  auto x_data_type = x_desc->dtype;
  auto half_h_mask = (h_mask - 1) >> 1;
  auto half_w_mask = (w_mask - 1) >> 1;

  // generate mluOpPsamaskForward prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("psamask_forward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "x", x, x_desc, 10, -10);
    GEN_CASE_DATA(false, "y", y, y_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "psamask_forward", "h_mask", h_mask);
    GEN_CASE_OP_PARAM_SINGLE(1, "psamask_forward", "w_mask", w_mask);
    GEN_CASE_OP_PARAM_SINGLE(2, "psamask_forward", "psa_type", psa_type);
    GEN_CASE_TEST_PARAM_NEW(false, false, true, 0.003, 0.003, 0);
  }
  // generate mluOpPsamaskForward prototxt end!

  mluOpStatus_t ret = MLUOP_STATUS_SUCCESS;
  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
  cnrtDim3_t k_dim;
  PartitionSeg partition_info;
  policyFunc(handle, &k_dim, &k_type, &partition_info, n, h_feature);
  int n_limit_seg, h_limit_seg, w_limit_seg;
  ret = findLimit(handle, partition_info.n_per_core, partition_info.h_per_core,
                  w_feature, x_c, y_c, mluop::getSizeOfDataType(x_data_type),
                  &n_limit_seg, &h_limit_seg, &w_limit_seg, psa_type, api);
  if (ret != MLUOP_STATUS_SUCCESS) {
    GEN_CASE_END();
    return ret;
  }

  KERNEL_CHECK((mluOpUnion1KernelPsamaskForwardFloat(
      k_dim, k_type, handle->queue, static_cast<const float *>(x),
      static_cast<float *>(y), (psamaskType_t)psa_type,
      partition_info.core_partition, partition_info.cluster_partition, n,
      h_feature, w_feature, h_mask, w_mask, x_c, y_c, half_h_mask, half_w_mask,
      partition_info.n_per_core, partition_info.h_per_core,
      partition_info.n_per_cluster, partition_info.h_per_cluster, n_limit_seg,
      h_limit_seg, w_limit_seg)));
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpPsamaskBackward(mluOpHandle_t handle, const int psa_type,
                                   const mluOpTensorDescriptor_t dy_desc,
                                   const void *dy, const int h_mask,
                                   const int w_mask,
                                   const mluOpTensorDescriptor_t dx_desc,
                                   void *dx) {
  const std::string api = "[mluOpPsamaskBackward]";
  PARAM_CHECK(api, handle != nullptr);
  PARAM_CHECK(api, dy_desc != nullptr);
  PARAM_CHECK(api, dx_desc != nullptr);
  if (checkParams(dx_desc, dy_desc, h_mask, w_mask, api) !=
      MLUOP_STATUS_SUCCESS) {
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (!dy_desc->total_element_num) {
    return MLUOP_STATUS_SUCCESS;
  }
  PARAM_CHECK(api, dy != nullptr);
  PARAM_CHECK(api, dx != nullptr);

  auto n = dy_desc->dims[0];
  auto h_feature = dy_desc->dims[1];
  auto w_feature = dy_desc->dims[2];
  auto dy_c = dy_desc->dims[3];
  auto dx_c = dx_desc->dims[3];
  auto dy_type = dy_desc->dtype;
  auto half_h_mask = (h_mask - 1) >> 1;
  auto half_w_mask = (w_mask - 1) >> 1;

  // generate mluOpPsamaskBackward prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("psamask_backward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "dy", dy, dy_desc, 10, -10);
    GEN_CASE_DATA(false, "dx", dx, dx_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "psamask_backward", "h_mask", h_mask);
    GEN_CASE_OP_PARAM_SINGLE(1, "psamask_backward", "w_mask", w_mask);
    GEN_CASE_OP_PARAM_SINGLE(2, "psamask_backward", "psa_type", psa_type);
    GEN_CASE_TEST_PARAM_NEW(false, false, true, 0.003, 0.003, 0);
  }
  // generate mluOpPsamaskBackward prototxt end!

  mluOpStatus_t ret = MLUOP_STATUS_SUCCESS;
  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
  cnrtDim3_t k_dim;
  PartitionSeg partition_info;
  policyFunc(handle, &k_dim, &k_type, &partition_info, n, h_feature);
  int n_limit_seg, h_limit_seg, w_limit_seg;
  ret = findLimit(handle, partition_info.n_per_core, partition_info.h_per_core,
                  w_feature, dx_c, dy_c, mluop::getSizeOfDataType(dy_type),
                  &n_limit_seg, &h_limit_seg, &w_limit_seg, psa_type, api);
  if (ret != MLUOP_STATUS_SUCCESS) {
    GEN_CASE_END();
    return ret;
  }
  KERNEL_CHECK((mluOpUnion1KernelPsamaskBackwardFloat(
      k_dim, k_type, handle->queue, static_cast<const float *>(dy),
      static_cast<float *>(dx), (psamaskType_t)psa_type,
      partition_info.core_partition, partition_info.cluster_partition, n,
      h_feature, w_feature, h_mask, w_mask, dx_c, dy_c, half_h_mask,
      half_w_mask, partition_info.n_per_core, partition_info.h_per_core,
      partition_info.n_per_cluster, partition_info.h_per_cluster, n_limit_seg,
      h_limit_seg, w_limit_seg)));
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
