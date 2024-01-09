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
#include "log.h"

#include <algorithm>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/unary_op/unary_op_host.h"

mluOpStatus_t MLUOP_WIN_API
mluOpLog(mluOpHandle_t handle, const mluOpComputationPreference_t prefer,
         const mluOpLogBase_t base, const mluOpTensorDescriptor_t x_desc,
         const void *x, const mluOpTensorDescriptor_t y_desc, void *y) {
  mluOpDataType_t support_type[2] = {MLUOP_DTYPE_HALF, MLUOP_DTYPE_FLOAT};
  bool zero_element = false;
  mluOpStatus_t param_check =
      unaryOpParamCheck("[mluOpLog]", handle, x_desc, x, y_desc, y,
                        support_type, 2, zero_element);
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }
  if (zero_element == true) {
    return MLUOP_STATUS_SUCCESS;
  }

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("log");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "x", x, x_desc, 10, 0);
    GEN_CASE_DATA(false, "y", y, y_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  cnrtFunctionType_t k_type;
  cnrtDim3_t k_dim;
  unaryOpPolicyFunc(handle, x_desc, &k_dim, &k_type);
  VLOG(5) << "[mluOp] Launch [" << k_type << ", " << k_dim.x << ", " << k_dim.y
          << ", " << k_dim.z << "]";

  float coef = 1.0;
  if (base == mluOpLogBase_t::MLUOP_LOG_E) {
    coef = 1.0;
  } else if (base == mluOpLogBase_t::MLUOP_LOG_2) {
    // log2(x) = loge(x) * log2(e)
    coef = log2(exp(1));
  } else if (base == mluOpLogBase_t::MLUOP_LOG_10) {
    // log10(x) = loge(x) * log10(e)
    coef = log10(exp(1));
  }

  int element_num = mluOpGetTensorElementNum(x_desc);
  if (handle->arch == MLUOP_MLU270) {
    VLOG(5) << "kernel Kernel5StagePipelineLog.";
    CHECK_RETURN("[mluOpLog] ", (Kernel5StagePipelineLog(
                                    k_dim, k_type, handle->queue, x_desc->dtype,
                                    prefer, x, y, element_num, coef)));
  } else {
    VLOG(5) << "kernel Kernel3StagePipelineLog.";
    CHECK_RETURN("[mluOpLog] ", (Kernel3StagePipelineLog(
                                    k_dim, k_type, handle->queue, x_desc->dtype,
                                    prefer, x, y, element_num, coef)));
  }
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
