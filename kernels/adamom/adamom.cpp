/*************************************************************************
 * Copyright (C) [2025] by Cambricon, Inc.
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
#include "core/logging.h"
#include "core/gen_case.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "core/tool.h"
#include "kernels/adamom/adamom.h"

static void policyFunc(const mluOpHandle_t &handle,
                       cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type) {
  uint32_t core_dim = handle->core_num_per_cluster;
  uint32_t cluster_num = mluop::runtime::getClusterLimitCapability(handle);

  *k_type = cnrtFuncTypeUnion1;
  k_dim->x = core_dim;
  k_dim->y = cluster_num;
  k_dim->z = 1;
  VLOG(5) << "Adamom policyFunc:(x, y, z) =" << k_dim->x << ", " << k_dim->y << ", "
          << k_dim->z;
}


mluOpStatus_t MLUOP_WIN_API
mluOpAdamom(mluOpHandle_t handle,
            mluOpTensorDescriptor_t grads_desc,  // input
            const void *grads,
            mluOpTensorDescriptor_t ms_desc,   // input output
            void *ms,
            mluOpTensorDescriptor_t vs_desc,   // input output
            void *vs,
            mluOpTensorDescriptor_t v_bias_corrections_desc,  // input output
            void *v_bias_corrections,
            mluOpTensorDescriptor_t weights_desc,   // input output
            void *weights,
            const void *nan_inf_found,
            const void *lr,
            const void *beta1,
            const void *beta2,
            const void *weight_decay,
            const void *epsilon) {
  if (int(handle->arch) < 500) {
    LOG(ERROR) << "[mluOpAdamom] does not supported under 500 series.";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }
  PARAM_CHECK("[mluOpAdamom]", handle != NULL);
  PARAM_CHECK("[mluOpAdamom]", grads_desc != NULL);
  PARAM_CHECK("[mluOpAdamom]", ms_desc != NULL);
  PARAM_CHECK("[mluOpAdamom]", vs_desc != NULL);
  PARAM_CHECK("[mluOpAdamom]", v_bias_corrections_desc != NULL);
  PARAM_CHECK("[mluOpAdamom]", weights_desc != NULL);
  PARAM_CHECK("[mluOpAdamom]", grads != NULL);
  PARAM_CHECK("[mluOpAdamom]", ms != NULL);
  PARAM_CHECK("[mluOpAdamom]", vs != NULL);
  PARAM_CHECK("[mluOpAdamom]", v_bias_corrections != NULL);
  PARAM_CHECK("[mluOpAdamom]", weights != NULL);
  PARAM_CHECK("[mluOpAdamom]", nan_inf_found != NULL);
  PARAM_CHECK("[mluOpAdamom]", lr != NULL);
  PARAM_CHECK("[mluOpAdamom]", beta1 != NULL);
  PARAM_CHECK("[mluOpAdamom]", beta2 != NULL);
  PARAM_CHECK("[mluOpAdamom]", weight_decay != NULL);
  PARAM_CHECK("[mluOpAdamom]", epsilon != NULL);

  size_t grads_nums = mluOpGetTensorElementNum(grads_desc);
  size_t ms_nums = mluOpGetTensorElementNum(ms_desc);
  size_t vs_nums = mluOpGetTensorElementNum(vs_desc);
  size_t v_bias_corrections_nums = mluOpGetTensorElementNum(v_bias_corrections_desc);
  size_t weights_nums = mluOpGetTensorElementNum(weights_desc);

  // check data num
  if (grads_nums != ms_nums || grads_nums != vs_nums ||
      grads_nums != v_bias_corrections_nums ||
      grads_nums != weights_nums) {
    LOG(ERROR) << "[mluOpAdamom] the size of grads, ms, vs, v_bias_corrections"
               << " and weights should be the same. But now the size of grads is " << grads_nums
               << ", the size of ms is " << ms_nums
               << ", the size of v_bias_corrections is " << v_bias_corrections_nums
               << ", the size of weights is " << weights_nums << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // check data type
  PARAM_CHECK("[mluOpAdamom]", grads_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK("[mluOpAdamom]", ms_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK("[mluOpAdamom]", vs_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK("[mluOpAdamom]", v_bias_corrections_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK("[mluOpAdamom]", weights_desc->getDtype() == MLUOP_DTYPE_FLOAT);

  // check data shape
  for (int i = 0; i < grads_desc->getDim(); i++) {
    if (grads_desc->getDims()[i] != ms_desc->getDims()[i] || grads_desc->getDims()[i] != vs_desc->getDims()[i] ||
        grads_desc->getDims()[i] != v_bias_corrections_desc->getDims()[i] ||
        grads_desc->getDims()[i] != weights_desc->getDims()[i]) {
      LOG(ERROR) << "[mluOpAdamom]: The shape of grads, ms, vs, v_bias_corrections"
                  << "and weights should be the same. But now grads_desc's shape[" << i << "] is " << grads_desc->getDims()[i]
                  << ", ms_desc's shape[" << i << "] is " << ms_desc->getDims()[i]
                  << ", vs_desc's shape[" << i << "] is " << vs_desc->getDims()[i]
                  << ", v_bias_corrections_desc's shape[" << i << "] is " << v_bias_corrections_desc->getDims()[i]
                  << ", weights_desc's shape[" << i << "] is " << weights_desc->getDims()[i] << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  // stride tensor check
  STRIDE_TENSOR_CHECK("[mluOpAdamom]:", grads_desc, "grads tensor must be continguous.");
  STRIDE_TENSOR_CHECK("[mluOpAdamom]:", ms_desc, "ms tensor must be continguous.");
  STRIDE_TENSOR_CHECK("[mluOpAdamom]:", vs_desc, "vs tensor must be continguous.");
  STRIDE_TENSOR_CHECK("[mluOpAdamom]:", v_bias_corrections_desc, "v_bias_corrections tensor must be continguous.");
  STRIDE_TENSOR_CHECK("[mluOpAdamom]:", weights_desc, "weights tensor must be continguous.");

  // generate adam prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("adamom", "ADAMOM");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "grads", grads, grads_desc, 1, 0);
    GEN_CASE_DATA(true, "ms", ms, ms_desc, 1, 0);
    GEN_CASE_DATA(true, "vs", vs, vs_desc, 1, 0);
    GEN_CASE_DATA(true, "v_bias_corrections", v_bias_corrections, v_bias_corrections_desc, 1, 0);
    GEN_CASE_DATA(true, "weights", weights, weights_desc, 1, 0);
    GEN_CASE_DATA(false, "ms", ms, ms_desc, 0, 0);
    GEN_CASE_DATA(false, "vs", vs, vs_desc, 0, 0);
    GEN_CASE_DATA(false, "v_bias_corrections", v_bias_corrections, v_bias_corrections_desc, 0, 0);
    GEN_CASE_DATA(false, "weights", weights, weights_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "adamom", "lr", lr, grads_desc->getDtype());
    GEN_CASE_OP_PARAM_SINGLE(1, "adamom", "beta1", beta1, grads_desc->getDtype());
    GEN_CASE_OP_PARAM_SINGLE(2, "adamom", "beta2", beta2, grads_desc->getDtype());
    GEN_CASE_OP_PARAM_SINGLE(3, "adamom", "weight_decay", weight_decay, grads_desc->getDtype());
    GEN_CASE_OP_PARAM_SINGLE(4, "adamom", "epsilon", epsilon, grads_desc->getDtype());
    GEN_CASE_OP_PARAM_SINGLE(5, "adamom", "nan_inf_found", nan_inf_found, MLUOP_DTYPE_BOOL);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }
  // generate adam prototxt end!

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, &k_dim, &k_type);
  mluOpDataType_t k_data_type = grads_desc->getDtype();

  size_t element_num = mluOpGetTensorElementNum(grads_desc);
  switch (k_type) {
    default: {
      LOG(ERROR) << "[mluOpAdamom] Failed to choose kernel to launch";
      return MLUOP_STATUS_ARCH_MISMATCH;
    }
    case cnrtFuncTypeUnion1: {
      VLOG(5) << "Launch Kernel KernelAdamom<<<Union" << k_type / CORE_DIM << ", "
              << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>";
      CHECK_RETURN("[mluOpAdamom]", KernelAdamom(k_dim, k_type, handle->queue,
                    grads_desc->getDtype(), (void *)grads, (void *)ms, (void *)vs,
                    (void *)v_bias_corrections, (void *)weights, (void *)nan_inf_found, (void *)lr,
                    (void *)beta1, (void *)beta2, (void *)weight_decay, (void *)epsilon, element_num));
    }
  }

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
