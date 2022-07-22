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
#include "mlu_op.h"
#include "mlu_op_kernel.h"

// threshold of bytes to be processed by each core
// according to the actual measurement results
#define THRESHOLD_SIZE (3 * 1024)

mluOpStatus_t MLUOP_WIN_API
mluOpDiv(mluOpHandle_t handle, const mluOpComputationPreference_t prefer,
         const mluOpTensorDescriptor_t x_desc, const void *x,
         const mluOpTensorDescriptor_t y_desc, const void *y,
         const mluOpTensorDescriptor_t z_desc, void *z) {
  mluOpDataType_t support_type[2] = {MLUOP_DTYPE_HALF, MLUOP_DTYPE_FLOAT};
  int number_of_supported_types = 2;
  bool zero_element = false;
  mluOpStatus_t param_check =
      binaryOpParamCheck("mluOpDiv", handle, x_desc, x, y_desc, y, z_desc, z,
                         support_type, number_of_supported_types, zero_element);
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }
  if (zero_element == true) {
    return MLUOP_STATUS_SUCCESS;
  }

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("div");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "x", x, x_desc, 10, 0);
    GEN_CASE_DATA(true, "y", y, y_desc, 10, 0);
    GEN_CASE_DATA(false, "z", z, z_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  binaryOpPolicyFunc(handle, x_desc, THRESHOLD_SIZE, &k_dim, &k_type);

  int element_num = mluOpGetTensorElementNum(x_desc);
  void (*mluOpBlockKernelBinary)(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                                 cnrtQueue_t queue, const void *x,
                                 const void *y, void *z, int element_num);
  mluOpBlockKernelBinary = nullptr;
  if (x_desc->dtype == MLUOP_DTYPE_HALF) {
    if (prefer == MLUOP_COMPUTATION_HIGH_PRECISION) {
      VLOG(5) << "kernel mluOpKernel3StagePipelineDivHalfHighAcc";
      mluOpBlockKernelBinary = mluOpBlockKernel3StagePipelineDivHalfHighAcc;
    } else {
      VLOG(5) << "kernel mluOpKernel3StagePipelineDivHalfFast";
      mluOpBlockKernelBinary = mluOpBlockKernel3StagePipelineDivHalfFast;
    }
  } else {
    VLOG(5) << "kernel mluOpKernel3StagePipelineDivFloatFast";
    mluOpBlockKernelBinary = mluOpBlockKernel3StagePipelineDivFloatFast;
  }
  KERNEL_CHECK((mluOpBlockKernelBinary(k_dim, k_type, handle->queue, x, y, z,
                                       element_num)));
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
