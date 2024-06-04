#include "lgamma.h"

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/unary_op/unary_op_host.h"


mluOpStatus_t MLUOP_WIN_API mluOpLgamma(mluOpHandle_t handle,
                                      const mluOpTensorDescriptor_t x_desc,
                                      const void *x,
                                      const mluOpTensorDescriptor_t y_desc,
                                      void *y) {
  // param check
  mluOpDataType_t support_type[2] = {MLUOP_DTYPE_HALF, MLUOP_DTYPE_FLOAT};
  bool zero_element = false;
  mluOpStatus_t param_check =
      unaryOpParamCheck("[mluOpLgamma]", handle, x_desc, x, y_desc, y,
                        support_type, 2, zero_element);
  if (param_check != MLUOP_STATUS_SUCCESS) {
    return param_check;
  }
  if (zero_element == true) {
    return MLUOP_STATUS_SUCCESS;
  }

#if 0
// FIXME
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("sqrt");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "x", x, x_desc, 100, 0.1);
    GEN_CASE_DATA(true, "y", y, y_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }
#endif

  // policy select
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  unaryOpPolicyFunc(handle, x_desc, &k_dim, &k_type);
  VLOG(5) << "[mluOpLgamma] launch kernel policyFUnc[" << k_dim.x << ", "
          << k_dim.y << ", " << k_dim.z << "]";

  int element_num = mluOpGetTensorElementNum(x_desc);
  if (handle->arch != MLUOP_MLU370) {
    LOG(ERROR) << "[mluOpLgamma] now only support <MLU370>\n";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  VLOG(5) << "kernel KernelLgamma.";
  CHECK_RETURN("[mluOpLgamma] ",
               KernelLgamma(k_dim, k_type, handle->queue,
                                       x_desc->dtype, x, y, element_num));
  // GEN_CASE_END();

  return MLUOP_STATUS_SUCCESS;
}    