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
#include "kernels/adam_w/adam_w.h"

#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/type.h"
#include "core/tool.h"

mluOpStatus_t MLUOP_WIN_API
mluOpCreateAdamWDescriptor(mluOpAdamWDescriptor_t *adamw_desc) {
  PARAM_CHECK("mluOpCreateAdamWDescriptor", adamw_desc != nullptr);
  mluOpAdamWStruct *ts = new mluOpAdamWStruct();
  if (ts == nullptr) {
    LOG(ERROR) << "mluOpCreateAdamWDescriptor: alloc failed.";
    return MLUOP_STATUS_ALLOC_FAILED;
  }
  *adamw_desc = ts;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetAdamWDescAttr(
    mluOpAdamWDescriptor_t adamw_desc, mluOpAdamWDescAttribute_t attr,
    const void *buf, const size_t size_in_bytes) {
  switch (attr) {
    case MLUOP_ADAMW_WEIGHT_DECAY: {
      if (size_in_bytes == sizeof(float) && buf != nullptr) {
        adamw_desc->weight_decay = *((float *)buf);
      } else {
        return MLUOP_STATUS_BAD_PARAM;
      }
    }; break;
    case MLUOP_ADAMW_GRAD_SCALE: {
      if (size_in_bytes == sizeof(float) && buf != nullptr) {
        adamw_desc->grad_scale = *((float *)buf);
      } else {
        return MLUOP_STATUS_BAD_PARAM;
      }
    }; break;
    case MLUOP_ADAMW_USE_NESTEROV: {
      if (size_in_bytes == sizeof(bool) && buf != nullptr) {
        adamw_desc->use_nesterov = *((bool *)buf);
      } else {
        return MLUOP_STATUS_BAD_PARAM;
      }
    }; break;
    default: {
      LOG(ERROR) << "[mluOpSetAdamWDescAttr] failed, attr is not supported.";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpDestroyAdamWDescriptor(mluOpAdamWDescriptor_t desc) {
  if (desc == nullptr) {
    LOG(ERROR) << "mluOpDestroyAdamWDescriptor: passing nullptr to this API.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  delete desc;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpAdamW(mluOpHandle_t handle, const mluOpAdamWDescriptor_t adamw_desc,
           const mluOpTensorDescriptor_t param_desc, void *param,
           const mluOpTensorDescriptor_t paramh_desc, void *param_h,
           const mluOpTensorDescriptor_t momentum_desc, void *momentum,
           const mluOpTensorDescriptor_t velocity_desc, void *velocity,
           const mluOpTensorDescriptor_t grad_desc, void *grad, const float lr,
           const float beta1, const float beta2, const float bias1,
           const float bias2, const float epsilon) {
  PARAM_CHECK("[mluOpAdamW]", handle != nullptr);
  PARAM_CHECK("[mluOpAdamW]", param_desc != nullptr || paramh_desc != nullptr);
  PARAM_CHECK("[mluOpAdamW]", momentum_desc != nullptr);
  PARAM_CHECK("[mluOpAdamW]", velocity_desc != nullptr);
  PARAM_CHECK("[mluOpAdamW]", grad_desc != nullptr);
  PARAM_CHECK("[mluOpAdamW]", param_desc->dtype == MLUOP_DTYPE_FLOAT)
  PARAM_CHECK("[mluOpAdamW]", paramh_desc->dtype == MLUOP_DTYPE_BFLOAT16)
  PARAM_CHECK("[mluOpAdamW]", momentum_desc->dtype == MLUOP_DTYPE_FLOAT)
  PARAM_CHECK("[mluOpAdamW]", velocity_desc->dtype == MLUOP_DTYPE_FLOAT)
  PARAM_CHECK("[mluOpAdamW]", grad_desc->dtype == MLUOP_DTYPE_BFLOAT16)

  PARAM_CHECK_LE("[mluOpAdamW]", beta1, 1.0)
  PARAM_CHECK_GE("[mluOpAdamW]", beta1, 0.0)
  PARAM_CHECK_LE("[mluOpAdamW]", beta2, 1.0)
  PARAM_CHECK_GE("[mluOpAdamW]", beta2, 0.0)
  PARAM_CHECK("[mluOpAdamW]", epsilon > 0)

  size_t param_dims = 0;
  size_t paramh_dims = 0;
  size_t momentum_dims = 0;
  size_t velocity_dims = 0;
  size_t grad_dims = 0;
  size_t size = 0;

  mluOpTensorLayout_t momentum_layout;
  mluOpDataType_t momentum_dtype;
  int momentum_dims_num = 0;
  int momentum_dims_shape[8] = {0};
  mluOpGetTensorDescriptor(momentum_desc, &momentum_layout, &momentum_dtype,
                           &momentum_dims_num, momentum_dims_shape);

  momentum_dims = mluOpGetTensorElementNum(momentum_desc);
  velocity_dims = mluOpGetTensorElementNum(velocity_desc);
  grad_dims = mluOpGetTensorElementNum(grad_desc);
  size = momentum_dims * sizeof(momentum_dtype);

  PARAM_CHECK("[mluOpAdamW]", velocity_dims > 0);
  PARAM_CHECK("[mluOpAdamW]", momentum_dims > 0);

  {
    LARGE_TENSOR_CHECK("[mluOpAdamW]", param_desc);
    LARGE_TENSOR_CHECK("[mluOpAdamW]", paramh_desc);
    LARGE_TENSOR_CHECK("[mluOpAdamW]", momentum_desc);
    LARGE_TENSOR_CHECK("[mluOpAdamW]", velocity_desc);
    LARGE_TENSOR_CHECK("[mluOpAdamW]", grad_desc);
  }

  if (param != nullptr && param_h != nullptr) {
    param_dims = mluOpGetTensorElementNum(param_desc);
    paramh_dims = mluOpGetTensorElementNum(paramh_desc);
    if (param_dims != paramh_dims || param_dims != momentum_dims ||
        param_dims != velocity_dims || param_dims != grad_dims) {
      LOG(ERROR)
          << "[mluOpAdamW] the size of param, param_h, momentum, velocity"
          << " and grad should be the same. But now the size of param is "
          << param_dims << ", the size of param_h is " << paramh_dims
          << ", the size of momentum is " << momentum_dims
          << ", the size of velocity is " << velocity_dims
          << ", the size of grad is " << grad_dims << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  } else if (param != nullptr) {
    param_dims = mluOpGetTensorElementNum(param_desc);
    if (param_dims != momentum_dims || param_dims != velocity_dims ||
        param_dims != grad_dims) {
      LOG(ERROR)
          << "[mluOpAdamW] the size of param, momentum, velocity"
          << " and grad should be the same. But now the size of param is "
          << param_dims << ", the size of momentum is " << momentum_dims
          << ", the size of velocity is " << velocity_dims
          << ", the size of grad is " << grad_dims << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  } else {
    paramh_dims = mluOpGetTensorElementNum(paramh_desc);
    if (paramh_dims != momentum_dims || paramh_dims != velocity_dims ||
        paramh_dims != grad_dims) {
      LOG(ERROR)
          << "[mluOpAdamW] the size of param_h, momentum, velocity"
          << " and grad should be the same. But now the size of param_h is "
          << paramh_dims << ", the size of momentum is " << momentum_dims
          << ", the size of velocity is " << velocity_dims
          << ", the size of grad is " << grad_dims << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  PARAM_CHECK("[mluOpAdamW]", momentum != nullptr);
  PARAM_CHECK("[mluOpAdamW]", velocity != nullptr);
  PARAM_CHECK("[mluOpAdamW]", grad != nullptr);

  // stride check
  if (param_desc != nullptr) {
    STRIDE_TENSOR_CHECK("[mluOpAdamW]:", param_desc,
                        "param_desc must be contiguous");
  }
  if (paramh_desc != nullptr) {
    STRIDE_TENSOR_CHECK("[mluOpAdamW]:", paramh_desc,
                        "paramh_desc must be contiguous");
  }
  STRIDE_TENSOR_CHECK("[mluOpAdamW]:", momentum_desc,
                      "momentum_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpAdamW]:", velocity_desc,
                      "velocity_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpAdamW]:", grad_desc,
                      "grad_desc must be contiguous");

  // generate adam prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("adamw", "ADAMW");
    GEN_CASE_HANDLE(handle);
    if (param != nullptr && param_h != nullptr) {
      GEN_CASE_DATA(true, "param", param, param_desc, 1, 0);
      GEN_CASE_DATA(true, "param_h", param_h, paramh_desc, 1, 0);
      GEN_CASE_DATA(false, "param", param, param_desc, 1, 0);
      GEN_CASE_DATA(false, "param_h", param_h, paramh_desc, 1, 0);
    } else if (param != nullptr) {
      GEN_CASE_DATA(true, "param", param, param_desc, 1, 0);
      GEN_CASE_DATA(false, "param", param, param_desc, 1, 0);
    } else {
      GEN_CASE_DATA(true, "param_h", param_h, paramh_desc, 1, 0);
      GEN_CASE_DATA(false, "param_h", param_h, paramh_desc, 1, 0);
    }

    GEN_CASE_DATA(true, "momentum", momentum, momentum_desc, 1, 1);
    GEN_CASE_DATA(true, "velocity", velocity, velocity_desc, 1, 1);
    GEN_CASE_DATA(true, "grad", grad, grad_desc, 1, 1);

    GEN_CASE_DATA(false, "momentum", momentum, momentum_desc, 0, 0);
    GEN_CASE_DATA(false, "velocity", velocity, velocity_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "adamw", "lr", lr, MLUOP_DTYPE_FLOAT);
    GEN_CASE_OP_PARAM_SINGLE(1, "adamw", "beta1", beta1, MLUOP_DTYPE_FLOAT);
    GEN_CASE_OP_PARAM_SINGLE(1, "adamw", "beta2", beta2, MLUOP_DTYPE_FLOAT);
    GEN_CASE_OP_PARAM_SINGLE(1, "adamw", "bias1", bias1, MLUOP_DTYPE_FLOAT);
    GEN_CASE_OP_PARAM_SINGLE(1, "adamw", "bias2", bias2, MLUOP_DTYPE_FLOAT);
    GEN_CASE_OP_PARAM_SINGLE(2, "adamw", "epsilon", epsilon, MLUOP_DTYPE_FLOAT);
    GEN_CASE_OP_PARAM_SINGLE(2, "adamw", "weight_decay",
                             adamw_desc->weight_decay, MLUOP_DTYPE_FLOAT);
    GEN_CASE_OP_PARAM_SINGLE(2, "adamw", "grad_scale", adamw_desc->grad_scale,
                             MLUOP_DTYPE_FLOAT);
    GEN_CASE_OP_PARAM_SINGLE(2, "adamw", "use_nesterov",
                             adamw_desc->use_nesterov, MLUOP_DTYPE_BOOL);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }
  // generate adamw prototxt end!

  mluOpTensorLayout_t grad_layout;
  mluOpDataType_t grad_dtype;
  int grad_dims_num = 0;
  int grad_dims_shape[8] = {0};

  mluOpGetTensorDescriptor(grad_desc, &grad_layout, &grad_dtype, &grad_dims_num,
                           grad_dims_shape);
  mluOpDataType_t k_data_type = grad_dtype;
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim.x = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  k_dim.y = mluop::runtime::getClusterLimitCapability(handle);
  k_dim.z = 1;

  // adamw common data shape in DNN, 512,512*512,512*2048,2048,2048*512;
  // small case such like 2048、512 should use 1 cluster.
  size_t small_case_thread = 2048;
  if (param_dims <= small_case_thread) k_dim.y = 1;
  switch (k_type) {
    default: {
      LOG(ERROR) << "[mluOpAdamW] Failed to choose kernel to launch";
      GEN_CASE_END();
      return MLUOP_STATUS_ARCH_MISMATCH;
    }
    case CNRT_FUNC_TYPE_UNION1: {
      VLOG(5) << "Launch Kernel KernelApplyAdamW<<<Union" << k_type / CORE_DIM
              << ", " << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>";
      CHECK_RETURN(
          "[mluOpAdamW]",
          KernelApplyAdamW(k_dim, k_type, handle->queue, (void *)param,
                           (void *)param_h, (void *)grad, (void *)momentum,
                           (void *)velocity, lr, beta1, beta2, bias1, bias2,
                           epsilon, adamw_desc->weight_decay,
                           adamw_desc->grad_scale, adamw_desc->use_nesterov,
                           size, k_data_type));
    }
  }
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
