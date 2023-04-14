/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
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
#include "mutual_information_backward.h"

#include <algorithm>
#include <string>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"

#define API_NAME "[mluOpMutualInformationBackward]"

mluOpStatus_t MLUOP_WIN_API mluOpGetMutualInformationBackwardWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t px_desc,
    const mluOpTensorDescriptor_t py_desc,
    const mluOpTensorDescriptor_t opt_boundary_desc,
    const mluOpTensorDescriptor_t p_desc,
    const mluOpTensorDescriptor_t ans_grad_desc,
    const bool overwrite_ans_grad, size_t *workspace_size) {
  // Workspace is not required in the current implementation.
  // This interface will be used when the scale limitation is removed.
  *workspace_size = 0;
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t checkTensorDim(
    const mluOpTensorDescriptor_t px_desc,
    const mluOpTensorDescriptor_t py_desc,
    const mluOpTensorDescriptor_t opt_boundary_desc,
    const mluOpTensorDescriptor_t p_desc,
    const mluOpTensorDescriptor_t ans_grad_desc,
    const mluOpTensorDescriptor_t px_grad_desc,
    const mluOpTensorDescriptor_t py_grad_desc) {
  if (3 != px_desc->dim) {
    LOG(ERROR) << API_NAME << " The dim of px must be 3. "
               << "But now the dim of px is " << px_desc->dim << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (3 != py_desc->dim) {
    LOG(ERROR) << API_NAME << " The dim of py must be 3. "
               << "But now the dim of py is " << py_desc->dim << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (nullptr != opt_boundary_desc && 2 != opt_boundary_desc->dim) {
    LOG(ERROR) << API_NAME
               << " The dim of opt_boundary must be 2 when opt_boundary is "
               << "not NULL. But now the dim of opt_boundary is "
               << opt_boundary_desc->dim << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (3 != p_desc->dim) {
    LOG(ERROR) << API_NAME << " The dim of p must be 3. "
               << "But now the dim of p is " << p_desc->dim << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (1 != ans_grad_desc->dim) {
    LOG(ERROR) << API_NAME << " The dim of ans_grad must be 1. "
               << "But now the dim of ans_grad is "
               << ans_grad_desc->dim << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (3 != px_grad_desc->dim) {
    LOG(ERROR) << API_NAME << " The dim of px_grad must be 3. "
               << "But now the dim of px_grad is " << px_grad_desc->dim << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (3 != py_grad_desc->dim) {
    LOG(ERROR) << API_NAME << " The dim of py_grad must be 3. "
               << "But now the dim of py_grad is " << py_grad_desc->dim << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t checkTensorShape(
    const mluOpTensorDescriptor_t px_desc,
    const mluOpTensorDescriptor_t py_desc,
    const mluOpTensorDescriptor_t opt_boundary_desc,
    const mluOpTensorDescriptor_t p_desc,
    const mluOpTensorDescriptor_t ans_grad_desc,
    const mluOpTensorDescriptor_t px_grad_desc,
    const mluOpTensorDescriptor_t py_grad_desc) {
  const int B = px_desc->dims[0];
  const int S = px_desc->dims[1];
  const int T = py_desc->dims[2];
  if (B != py_desc->dims[0] || B != p_desc->dims[0] ||
      B != ans_grad_desc->dims[0] || B != px_grad_desc->dims[0] ||
      B != py_grad_desc->dims[0]) {
    LOG(ERROR) << API_NAME
               << " px.shape[0], py.shape[0], p.shape[0], ans_grad.shape[0], "
               << "px_grad.shape[0] and py_grad.shape[0] must be same. But now "
               << "px.shape[0] is " << px_desc->dims[0]
               << ", py.shape[0] is " << py_desc->dims[0]
               << ", p.shape[0] is " << p_desc->dims[0]
               << ", ans_grad.shape[0] is " << ans_grad_desc->dims[0]
               << ", px_grad.shape[0] is " << px_grad_desc->dims[0]
               << ", py_grad.shape[0] is " << py_grad_desc->dims[0] << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // Currently only supports !modified, so the shape of px must be [B, S, T+1]
  if (T + 1 != px_desc->dims[2]) {
    LOG(ERROR) << API_NAME << " Currently only supports the case that "
               << "px.shape[2] must be equal to py.shape[2] + 1. But now "
               << "px.shape[2] is " << px_desc->dims[2]
               << ", py.shape[2] is " << py_desc->dims[2] << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // the shape of py must be [B, S+1, T]
  if (S + 1 != py_desc->dims[1]) {
    LOG(ERROR) << API_NAME
               << " py.shape[1] must be equal to px.shape[1] + 1. "
               << "But now px.shape[1] is " << px_desc->dims[1]
               << ", py.shape[1] is " << py_desc->dims[1] << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // the shape of opt_boundary must be [B, 4]
  if (nullptr != opt_boundary_desc && (B != opt_boundary_desc->dims[0]
      || 4 != opt_boundary_desc->dims[1])) {
    LOG(ERROR) << API_NAME << " When opt_boundary is not NULL, "
               << "opt_boundary.shape[0] and px.shape[0] must be same, and "
               << "opt_boundary.shape[1] must be 4. But now "
               << "px.shape[0] is " << px_desc->dims[0]
               << ", opt_boundary.shape[0] is " << opt_boundary_desc->dims[0]
               << ", opt_boundary.shape[1] is " << opt_boundary_desc->dims[1]
               << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // the shape of p must be [B, S+1, T+1]
  if (S + 1 != p_desc->dims[1] || T + 1 != p_desc->dims[2]) {
    LOG(ERROR) << API_NAME
               << " p.shape[1] and py.shape[1] must be same, and "
               << "p.shape[2] must be equal to py.shape[2] + 1. "
               << "But now p.shape[1] is " << p_desc->dims[1]
               << ", py.shape[1] is " << py_desc->dims[1]
               << ", p.shape[2] is " << p_desc->dims[2]
               << ", py.shape[2] is " << py_desc->dims[2] << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // the shape of px and px_grad must be same: [B, S, T+1]
  for (int i = 1; i < px_grad_desc->dim; ++i) {
    if (px_grad_desc->dims[i] != px_desc->dims[i]) {
      LOG(ERROR) << API_NAME
                 << " The shape of px and px_grad must be same. But now "
                 << "px.shape[" << i << "] is " << px_desc->dims[i]
                 << ", px_grad.shape[" << i << "] is " << px_grad_desc->dims[i]
                 << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  // the shape of py and py_grad must be same: [B, S+1, T]
  for (int i = 1; i < py_grad_desc->dim; ++i) {
    if (py_grad_desc->dims[i] != py_desc->dims[i]) {
      LOG(ERROR) << API_NAME
                 << " The shape of py and py_grad must be same. But now "
                 << "py.shape[" << i << "] is " << py_desc->dims[i]
                 << ", py_grad.shape[" << i << "] is " << py_grad_desc->dims[i]
                 << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t checkTensorDatatype(
    const mluOpTensorDescriptor_t px_desc,
    const mluOpTensorDescriptor_t py_desc,
    const mluOpTensorDescriptor_t opt_boundary_desc,
    const mluOpTensorDescriptor_t p_desc,
    const mluOpTensorDescriptor_t ans_grad_desc,
    const mluOpTensorDescriptor_t px_grad_desc,
    const mluOpTensorDescriptor_t py_grad_desc) {
  if (MLUOP_DTYPE_FLOAT != px_desc->dtype) {
    LOG(ERROR) << API_NAME
               << "The data type of px currently only support float. But now "
               << "the data type of px is "
               << mluOpGetNameOfDataType(px_desc->dtype) << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (MLUOP_DTYPE_FLOAT != py_desc->dtype) {
    LOG(ERROR) << API_NAME
               << "The data type of py currently only support float. But now "
               << "the data type of py is "
               << mluOpGetNameOfDataType(py_desc->dtype) << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (nullptr != opt_boundary_desc &&
      MLUOP_DTYPE_INT64 != opt_boundary_desc->dtype) {
    LOG(ERROR) << API_NAME
               << "The data type of opt_boundary currently only support int64."
               << " But now the data type of opt_boundary is "
               << mluOpGetNameOfDataType(opt_boundary_desc->dtype) << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (MLUOP_DTYPE_FLOAT != p_desc->dtype) {
    LOG(ERROR) << API_NAME
               << "The data type of p currently only support float. But now "
               << "the data type of p is "
               << mluOpGetNameOfDataType(p_desc->dtype) << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (MLUOP_DTYPE_FLOAT != ans_grad_desc->dtype) {
    LOG(ERROR) << API_NAME
               << "The data type of ans_grad currently only support float. "
               << "But now the data type of ans_grad is "
               << mluOpGetNameOfDataType(ans_grad_desc->dtype) << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (MLUOP_DTYPE_FLOAT != px_grad_desc->dtype) {
    LOG(ERROR) << API_NAME
               << "The data type of px_grad currently only support float. "
               << "But now the data type of px_grad is "
               << mluOpGetNameOfDataType(px_grad_desc->dtype) << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (MLUOP_DTYPE_FLOAT != py_grad_desc->dtype) {
    LOG(ERROR) << API_NAME
               << "The data type of py_grad currently only support float. "
               << "But now the data type of py_grad is "
               << mluOpGetNameOfDataType(py_grad_desc->dtype) << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t checkTensorScaleLimit(
    mluOpHandle_t handle,
    const mluOpTensorDescriptor_t px_desc,
    const mluOpTensorDescriptor_t py_desc,
    const mluOpTensorDescriptor_t opt_boundary_desc,
    const mluOpTensorDescriptor_t p_desc) {
  // check large tensor
  // the shape of px and px_grad are the same,
  // the shape of py and py_grad are the same,
  // so there is no need to check the tensor num of px_grad and py_grad
  if (mluOpGetTensorElementNum(px_desc) >= LARGE_TENSOR_NUM ||
      mluOpGetTensorElementNum(py_desc) >= LARGE_TENSOR_NUM ||
      (nullptr != opt_boundary_desc &&
      mluOpGetTensorElementNum(opt_boundary_desc) >= LARGE_TENSOR_NUM) ||
      mluOpGetTensorElementNum(p_desc) >= LARGE_TENSOR_NUM) {
    LOG(ERROR) << API_NAME << " Overflow max tensor num."
               << " Current operator supports tensor num smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  const int B = px_desc->dims[0];
  const int S = px_desc->dims[1];
  const int T = py_desc->dims[2];
  // check scale limit for compute term1 and term2
  int currnet_size = T * (S + 1) + (T + 1) * S + 5 * (T + 1);
  if (currnet_size > handle->nram_size / sizeof(float)) {
    LOG(ERROR) << API_NAME << " The num of px.shape[1] * px.shape[2] + "
               << "py.shape[1] * py.shape[2] + 5 * (py.shape[2] + 1) shoule be "
               << "less than " << handle->nram_size / sizeof(float)
               << ". But now it is " << currnet_size << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // check scale limit for compute p_grad
  currnet_size = T * (S + 1) + (T + 1) * S + (T + 1) * (S + 1) +
                 3 * std::min(S, T) + 4;
  if (currnet_size > handle->nram_size / sizeof(float)) {
    LOG(ERROR) << API_NAME << " The num of px.shape[1] * px.shape[2] + "
               << "py.shape[1] * py.shape[2] + p.shape[1] * p.shape[2] "
               << "+ 3 * min(px.shape[1], py.shape[2]) + 4 shoule be less than "
               << handle->nram_size / sizeof(float)
               << ". But now it is " << currnet_size << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t checkTensorPtr(
    const void *px, const void *py, const void *p, const void *ans_grad,
    const mluOpTensorDescriptor_t opt_boundary_desc, const void *opt_boundary,
    const void *px_grad, const void *py_grad, const int S, const int T,
    bool &has_boundary) {
  if (S > 0) {
    PARAM_CHECK(API_NAME, px != nullptr);
    PARAM_CHECK(API_NAME, px_grad != nullptr);
  } else {
    VLOG(5) << API_NAME << " px.shape[1] is zero.";
  }

  if (T > 0) {
    PARAM_CHECK(API_NAME, py != nullptr);
    PARAM_CHECK(API_NAME, py_grad != nullptr);
  } else {
    VLOG(5) << API_NAME << " py.shape[2] is zero.";
  }

  PARAM_CHECK(API_NAME, p != nullptr);
  PARAM_CHECK(API_NAME, ans_grad != nullptr);

  if (nullptr != opt_boundary_desc && nullptr != opt_boundary) {
    has_boundary = true;
    VLOG(5) << API_NAME << " opt_boundary is not NULL.";

  } else  if (nullptr == opt_boundary_desc && nullptr == opt_boundary) {
    has_boundary = false;
    VLOG(5) << API_NAME << " opt_boundary is NULL.";
  } else {
    LOG(ERROR) << API_NAME
               << " opt_boundary_desc and opt_boundary must both be NULL, "
               << "or both not be NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t mutualInformationBackwardParamCheck(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t px_desc, const void *px,
    const mluOpTensorDescriptor_t py_desc, const void *py,
    const mluOpTensorDescriptor_t opt_boundary_desc, const void *opt_boundary,
    const mluOpTensorDescriptor_t p_desc, const void *p,
    const mluOpTensorDescriptor_t ans_grad_desc, void *ans_grad,
    void *workspace, const size_t workspace_size,
    const mluOpTensorDescriptor_t px_grad_desc, void *px_grad,
    const mluOpTensorDescriptor_t py_grad_desc, void *py_grad,
    bool &has_boundary, bool &zero_element) {
  // 1. check handle and tensor_desc
  PARAM_CHECK(API_NAME, handle != nullptr);
  PARAM_CHECK(API_NAME, px_desc != nullptr);
  PARAM_CHECK(API_NAME, py_desc != nullptr);
  PARAM_CHECK(API_NAME, p_desc != nullptr);
  PARAM_CHECK(API_NAME, ans_grad_desc != nullptr);
  PARAM_CHECK(API_NAME, px_grad_desc != nullptr);
  PARAM_CHECK(API_NAME, py_grad_desc != nullptr);

  // since the layout of all tensor is ARRAY, so skip check tensor layout

  // 2. check mlu platform
  if (handle->arch < 372) {
    LOG(ERROR) << API_NAME << " Only mlu300 and above devices are supported."
               << " Please check the device version!";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  // 3. check tensor dim
  mluOpStatus_t check_status = checkTensorDim(px_desc, py_desc,
                                              opt_boundary_desc,
                                              p_desc, ans_grad_desc,
                                              px_grad_desc, py_grad_desc);
  if (MLUOP_STATUS_SUCCESS != check_status) {
    return check_status;
  }

  // 4. check tensor shape
  check_status = checkTensorShape(px_desc, py_desc, opt_boundary_desc,
                                  p_desc, ans_grad_desc, px_grad_desc,
                                  py_grad_desc);
  if (MLUOP_STATUS_SUCCESS != check_status) {
    return check_status;
  }

  // 5. check tensor dtype
  check_status = checkTensorDatatype(px_desc, py_desc, opt_boundary_desc,
                                     p_desc, ans_grad_desc, px_grad_desc,
                                     py_grad_desc);
  if (MLUOP_STATUS_SUCCESS != check_status) {
    return check_status;
  }

  // 6. check scale limit
  check_status = checkTensorScaleLimit(handle, px_desc, py_desc,
                                       opt_boundary_desc, p_desc);
  if (MLUOP_STATUS_SUCCESS != check_status) {
    return check_status;
  }

  const int B = px_desc->dims[0];
  const int S = px_desc->dims[1];
  const int T = py_desc->dims[2];

  // 7. check zero element.
  if (0 == B || (0 == S && 0 == T)) {
    zero_element = true;
    VLOG(5) << API_NAME << " Skip zero element tensor when px.shape[0] is zero "
                        << "or px.shape[1] and py.shape[2] are both zero.";
    return MLUOP_STATUS_SUCCESS;
  }

  // 8 check workspace
  if (workspace_size > 0) {
    PARAM_CHECK(API_NAME, workspace != nullptr);
  }

  // 9. check tensor ptr
  check_status = checkTensorPtr(px, py, p, ans_grad, opt_boundary_desc,
                                opt_boundary, px_grad, py_grad, S, T,
                                has_boundary);
  if (MLUOP_STATUS_SUCCESS != check_status) {
    return check_status;
  }

  return MLUOP_STATUS_SUCCESS;
}

static void mutualInformationBackwardGencase(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t px_desc, const void *px,
    const mluOpTensorDescriptor_t py_desc, const void *py,
    const mluOpTensorDescriptor_t opt_boundary_desc, const void *opt_boundary,
    const mluOpTensorDescriptor_t p_desc, const void *p,
    const mluOpTensorDescriptor_t ans_grad_desc, void *ans_grad,
    const bool overwrite_ans_grad,
    const mluOpTensorDescriptor_t px_grad_desc, void *px_grad,
    const mluOpTensorDescriptor_t py_grad_desc, void *py_grad) {
  GEN_CASE_START("mutual_information_backward");
  GEN_CASE_HANDLE(handle);

  GEN_CASE_DATA(true, "px", px, px_desc, -1, 1);
  GEN_CASE_DATA(true, "py", py, py_desc, -1, 1);
  if (nullptr != opt_boundary) {
    GEN_CASE_DATA_REAL(true, "opt_boundary", opt_boundary, opt_boundary_desc);
  }
  GEN_CASE_DATA(true, "p", p, p_desc, -1, 1);
  GEN_CASE_DATA(true, "ans_grad", ans_grad, ans_grad_desc, -1, 1);
  GEN_CASE_DATA(false, "ans_grad", ans_grad, ans_grad_desc, -1, 1);
  GEN_CASE_DATA(false, "px_grad", px_grad, px_grad_desc, -1, 1);
  GEN_CASE_DATA(false, "py_grad", py_grad, py_grad_desc, -1, 1);

  GEN_CASE_OP_PARAM_SINGLE(0, "mutual_information_backward",
                           "overwrite_ans_grad", overwrite_ans_grad);
  GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
}

static void policyFunc(const mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type, int batch_size) {
  int core_num =  mluop::runtime::getClusterLimitCapability(handle) *
                  mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  *k_type = CNRT_FUNC_TYPE_BLOCK;
  k_dim->x = 1;
  k_dim->y = batch_size < core_num ? batch_size : core_num;
  k_dim->z = 1;
}

static mluOpStatus_t launchMutualInformationBackwardKernel(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t px_desc, const void *px,
    const mluOpTensorDescriptor_t py_desc, const void *py,
    const bool has_boundary, const void *opt_boundary, const void *p,
    const bool overwrite_ans_grad, void *ans_grad, void *px_grad,
    void *py_grad) {
  const int B = px_desc->dims[0];
  const int S = px_desc->dims[1];
  const int T = py_desc->dims[2];

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, &k_dim, &k_type, B);
  VLOG(5) << "Launch Kernel MutualInformationBackward<<<Block "
          << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>";
  KERNEL_CHECK(kernelMutualInformationBackward(
      k_dim, k_type, handle->queue, B, S, T, px, py, has_boundary, opt_boundary,
      p, overwrite_ans_grad, ans_grad, px_grad, py_grad));

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpMutualInformationBackward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t px_desc, const void *px,
    const mluOpTensorDescriptor_t py_desc, const void *py,
    const mluOpTensorDescriptor_t opt_boundary_desc, const void *opt_boundary,
    const mluOpTensorDescriptor_t p_desc, const void *p,
    const mluOpTensorDescriptor_t ans_grad_desc, void *ans_grad,
    const bool overwrite_ans_grad, void *workspace, const size_t workspace_size,
    const mluOpTensorDescriptor_t px_grad_desc, void *px_grad,
    const mluOpTensorDescriptor_t py_grad_desc, void *py_grad) {
  // 1. paramcheck
  bool has_boundary = false;
  bool zero_element = false;
  mluOpStatus_t check_status = mutualInformationBackwardParamCheck(
      handle, px_desc, px, py_desc, py, opt_boundary_desc, opt_boundary,
      p_desc, p, ans_grad_desc, ans_grad, workspace, workspace_size,
      px_grad_desc, px_grad, py_grad_desc, py_grad, has_boundary, zero_element);

  if (MLUOP_STATUS_SUCCESS != check_status || zero_element) {
    return check_status;
  }

  // 2. generate case
  if (MLUOP_GEN_CASE_ON_NEW) {
    mutualInformationBackwardGencase(handle, px_desc, px, py_desc, py,
                                     opt_boundary_desc, opt_boundary,
                                     p_desc, p, ans_grad_desc, ans_grad,
                                     overwrite_ans_grad, px_grad_desc, px_grad,
                                     py_grad_desc, py_grad);
  }

  // 3. launch kernel
  mluOpStatus_t return_status = launchMutualInformationBackwardKernel(
      handle, px_desc, px, py_desc, py, has_boundary, opt_boundary, p,
      overwrite_ans_grad, ans_grad, px_grad, py_grad);

  GEN_CASE_END();
  return return_status;
}
