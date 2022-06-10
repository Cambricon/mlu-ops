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
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
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

  // Choose the best task dimension.
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  unaryOpPolicyFunc(handle, x_desc, &k_dim, &k_type);
  VLOG(5) << "[mluOpSqrt] launch kernel policyFUnc[" << k_dim.x << ", "
          << k_dim.y << ", " << k_dim.z << "]";

  int32_t element_num = mluOpGetTensorElementNum(x_desc);
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
  return MLUOP_STATUS_SUCCESS;
}
