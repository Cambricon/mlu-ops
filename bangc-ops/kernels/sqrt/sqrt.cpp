/*************************************************************************
 * Copyright (C) 2021 by Cambricon, Inc. All rights reserved.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/binary_op/binary_op_host.h"
#include "kernels/unary_op/unary_op_host.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"

mluOpStatus_t MLUOP_WIN_API mluOpSqrt(mluOpHandle_t handle,
                                      const mluOpComputationPreference_t prefer,
                                      const mluOpTensorDescriptor_t x_desc,
                                      const void *x,
                                      const mluOpTensorDescriptor_t y_desc,
                                      void *y) {
  mluOpDataType_t support_type[2] = {MLUOP_DTYPE_HALF, MLUOP_DTYPE_FLOAT};
  bool zero_element = false;
  mluOpStatus_t param_check =
      unaryOpParamCheck("[mluOpSqrt]", handle, x_desc, x, y_desc, y,
                        support_type, 2, zero_element);
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }
  if (zero_element == true) {
    return MLUOP_STATUS_SUCCESS;
  }

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("sqrt");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "x", x, x_desc, 100, 0.1);
    GEN_CASE_DATA(true, "y", y, y_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  // Choose the best task dimension.
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  unaryOpPolicyFunc(handle, x_desc, &k_dim, &k_type);
  VLOG(5) << "[mluOpSqrt] launch kernel policyFUnc[" << k_dim.x << ", "
          << k_dim.y << ", " << k_dim.z << "]";

  int element_num = mluOpGetTensorElementNum(x_desc);
  void (*mluOpBlockKernelUnary)(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                                cnrtQueue_t queue, const void *x, void *y,
                                int element_num);
  mluOpBlockKernelUnary = nullptr;
  if (handle->arch == MLUOP_MLU270) {
    if (x_desc->dtype == MLUOP_DTYPE_FLOAT) {
      VLOG(5) << "kernel mluOpBlockKernel5StagePipelineSqrtFloatFast";
      mluOpBlockKernelUnary = mluOpBlockKernel5StagePipelineSqrtFloatFast;
    } else {
      if (prefer == MLUOP_COMPUTATION_FAST) {
        VLOG(5) << "kernel mluOpBlockKernel5StagePipelineSqrtHalfFast";
        mluOpBlockKernelUnary = mluOpBlockKernel5StagePipelineSqrtHalfFast;
      } else {
        VLOG(5) << "kernel mluOpBlockKernel5StagePipelineSqrtHalfHighAcc";
        mluOpBlockKernelUnary = mluOpBlockKernel5StagePipelineSqrtHalfHighAcc;
      }
    }
  } else {
    if (x_desc->dtype == MLUOP_DTYPE_FLOAT) {
      VLOG(5) << "kernel mluOpBlockKernel3StagePipelineSqrtFloatFast";
      mluOpBlockKernelUnary = mluOpBlockKernel3StagePipelineSqrtFloatFast;
    } else {
      if (prefer == MLUOP_COMPUTATION_FAST) {
        VLOG(5) << "kernel mluOpBlockKernel3StagePipelineSqrtHalfFast";
        mluOpBlockKernelUnary = mluOpBlockKernel3StagePipelineSqrtHalfFast;
      } else {
        VLOG(5) << "kernel mluOpBlockKernel3StagePipelineSqrtHalfHighAcc";
        mluOpBlockKernelUnary = mluOpBlockKernel3StagePipelineSqrtHalfHighAcc;
      }
    }
  }
  KERNEL_CHECK(
      (mluOpBlockKernelUnary(k_dim, k_type, handle->queue, x, y, element_num)));
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSqrtBackward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t y_desc, const void *y,
    const mluOpTensorDescriptor_t dy_desc, const void *diff_y,
    const mluOpTensorDescriptor_t dx_desc, void *diff_x) {
  mluOpDataType_t support_type[2] = {MLUOP_DTYPE_HALF, MLUOP_DTYPE_FLOAT};
  int number_of_supported_types = 2;
  bool zero_element = false;
  mluOpStatus_t param_check = binaryOpParamCheck(
      "[mluOpSqrtBackward]", handle, y_desc, y, dy_desc, diff_y, dx_desc,
      diff_x, support_type, number_of_supported_types, zero_element);
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }
  if (zero_element == true) {
    return MLUOP_STATUS_SUCCESS;
  }

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("sqrt_backward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "y", y, y_desc, 10, 0.1);
    GEN_CASE_DATA(true, "diff_y", diff_y, dy_desc, -10, 10);
    GEN_CASE_DATA(false, "diff_x", diff_x, dx_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  binaryOpPolicyFunc(handle, y_desc, handle->nram_size, &k_dim, &k_type);

  int num_elem = mluOpGetTensorElementNum(y_desc);
  void (*mluOpBlockKernelBinary)(
      cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
      const void *y, const void *diff_y, void *diff_x, int num_elem);
  mluOpBlockKernelBinary = nullptr;
  if (y_desc->dtype == MLUOP_DTYPE_HALF) {
    VLOG(5) << "Kernel mluOpBlockKernel3StagePipelineSqrtBackwardHalfHighAcc";
    mluOpBlockKernelBinary =
        mluOpBlockKernel3StagePipelineSqrtBackwardHalfHighAcc;
  } else {
    VLOG(5) << "Kernel mluOpBlockKernel3StagePipelineSqrtBackwardFloatFast";
    mluOpBlockKernelBinary =
        mluOpBlockKernel3StagePipelineSqrtBackwardFloatFast;
  }
  KERNEL_CHECK((mluOpBlockKernelBinary(k_dim, k_type, handle->queue, y, diff_y,
                                       diff_x, num_elem)));
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
