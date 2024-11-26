/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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
#include "logspace.h"
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/unary_op/unary_op_host.h"

static void LogspacePolicyFunc(const mluOpHandle_t &handle, const int64_t steps,
                               cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  *k_type = CNRT_FUNC_TYPE_BLOCK;
  uint32_t cluster_num =
      mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  uint32_t core_in_cluster = handle->core_num_per_cluster;
  uint32_t core_max = cluster_num * core_in_cluster;
  uint32_t core_used = core_max > steps ? steps : core_max;
  k_dim->x = core_used;
  k_dim->y = 1;
  k_dim->z = 1;
}

static inline bool isSupportType(const mluOpDataType_t check_type,
                                 const mluOpDataType_t support_type[],
                                 const int len) {
  for (int i = 0; i < len; ++i) {
    if (check_type == support_type[i]) {
      return true;
    }
  }
  return false;
}

mluOpStatus_t LogspaceParamCheck(const mluOpHandle_t &handle, const float start,
                                 const float end, const int64_t steps,
                                 const float base,
                                 const mluOpTensorDescriptor_t &res_desc,
                                 const void *res) {
  PARAM_CHECK("[mluOpLogspace]", handle != nullptr);
  PARAM_CHECK("[mluOpLogspace]", res_desc != nullptr);
  PARAM_CHECK("[mluOpLogspace]", steps >= 0);
  size_t element_num = mluOpGetTensorElementNum(res_desc);
  PARAM_CHECK("[mluOpLogspace]", steps <= element_num);
  mluOpDataType_t support_type[3] = {MLUOP_DTYPE_FLOAT, MLUOP_DTYPE_HALF,
                                     MLUOP_DTYPE_INT32};
  if (!isSupportType(res_desc->getDtype(), support_type, 3)) {
    LOG(ERROR) << "[mluOpLogspace]"
               << ":res_desc's data type is not supported.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpLogspace(mluOpHandle_t handle, const float start, const float end,
              const int64_t steps, const float base,
              const mluOpTensorDescriptor_t res_desc, void *res) {
  // param check
  mluOpStatus_t param_check =
      LogspaceParamCheck(handle, start, end, steps, base, res_desc, res);
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }

  if (steps == 0) {
    return MLUOP_STATUS_SUCCESS;
  }

  // generate prototxt
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("logspace", "LOGSPACE");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", nullptr, nullptr, 0, 0);
    GEN_CASE_DATA(false, "res", res, res_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "logspace", "start", start);
    GEN_CASE_OP_PARAM_SINGLE(1, "logspace", "end", end);
    GEN_CASE_OP_PARAM_SINGLE(2, "logspace", "steps", steps);
    GEN_CASE_OP_PARAM_SINGLE(3, "logspace", "base", base);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  // policy select
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  LogspacePolicyFunc(handle, steps, &k_dim, &k_type);
  VLOG(5) << "[mluOpLogspace] launch kernel policyFUnc[" << k_dim.x << ", "
          << k_dim.y << ", " << k_dim.z << "]";

  VLOG(5) << "kernel KernelLogspace.";
  CHECK_RETURN("[mluOpLogspace] ",
               KernelLogspace(k_dim, k_type, handle->queue, res_desc->getDtype(),
                              start, end, steps, base, res));
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
