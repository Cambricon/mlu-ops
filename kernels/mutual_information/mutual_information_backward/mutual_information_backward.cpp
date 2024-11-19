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
#include "kernels/utils/cnnl_helper.h"

#define API_NAME "[mluOpMutualInformationBackward]"

mluOpStatus_t MLUOP_WIN_API mluOpGetMutualInformationBackwardWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t px_desc,
    const mluOpTensorDescriptor_t py_desc,
    const mluOpTensorDescriptor_t opt_boundary_desc,
    const mluOpTensorDescriptor_t p_desc,
    const mluOpTensorDescriptor_t ans_grad_desc, const bool overwrite_ans_grad,
    size_t *workspace_size) {
  PARAM_CHECK(API_NAME, handle != nullptr);
  PARAM_CHECK(API_NAME, px_desc != nullptr);
  PARAM_CHECK(API_NAME, py_desc != nullptr);
  PARAM_CHECK(API_NAME, p_desc != nullptr);
  PARAM_CHECK(API_NAME, ans_grad_desc != nullptr);
  PARAM_CHECK(API_NAME, workspace_size != nullptr);
  // Use for p_grad size, only support float data type now
  *workspace_size = mluOpGetTensorElementNum(p_desc) * sizeof(float);
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
  if (3 != px_desc->getDim()) {
    LOG(ERROR) << API_NAME << " The dim of px must be 3. "
               << "But now the dim of px is " << px_desc->getDim() << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (3 != py_desc->getDim()) {
    LOG(ERROR) << API_NAME << " The dim of py must be 3. "
               << "But now the dim of py is " << py_desc->getDim() << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (nullptr != opt_boundary_desc && 2 != opt_boundary_desc->getDim()) {
    LOG(ERROR) << API_NAME
               << " The dim of opt_boundary must be 2 when opt_boundary is "
               << "not NULL. But now the dim of opt_boundary is "
               << opt_boundary_desc->getDim() << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (3 != p_desc->getDim()) {
    LOG(ERROR) << API_NAME << " The dim of p must be 3. "
               << "But now the dim of p is " << p_desc->getDim() << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (1 != ans_grad_desc->getDim()) {
    LOG(ERROR) << API_NAME << " The dim of ans_grad must be 1. "
               << "But now the dim of ans_grad is " << ans_grad_desc->getDim()
               << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (3 != px_grad_desc->getDim()) {
    LOG(ERROR) << API_NAME << " The dim of px_grad must be 3. "
               << "But now the dim of px_grad is " << px_grad_desc->getDim() << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (3 != py_grad_desc->getDim()) {
    LOG(ERROR) << API_NAME << " The dim of py_grad must be 3. "
               << "But now the dim of py_grad is " << py_grad_desc->getDim() << ".";
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
  const int B = px_desc->getDimIndex(0);
  const int S = px_desc->getDimIndex(1);
  const int T = py_desc->getDimIndex(2);
  if (B != py_desc->getDimIndex(0) || B != p_desc->getDimIndex(0) ||
      B != ans_grad_desc->getDimIndex(0) || B != px_grad_desc->getDimIndex(0) ||
      B != py_grad_desc->getDimIndex(0)) {
    LOG(ERROR) << API_NAME
               << " px.shape[0], py.shape[0], p.shape[0], ans_grad.shape[0], "
               << "px_grad.shape[0] and py_grad.shape[0] must be same. But now "
               << "px.shape[0] is " << px_desc->getDimIndex(0) << ", py.shape[0] is "
               << py_desc->getDimIndex(0) << ", p.shape[0] is " << p_desc->getDimIndex(0)
               << ", ans_grad.shape[0] is " << ans_grad_desc->getDimIndex(0)
               << ", px_grad.shape[0] is " << px_grad_desc->getDimIndex(0)
               << ", py_grad.shape[0] is " << py_grad_desc->getDimIndex(0) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // Currently only supports !modified, so the shape of px must be [B, S, T+1]
  if (T + 1 != px_desc->getDimIndex(2)) {
    LOG(ERROR) << API_NAME << " Currently only supports the case that "
               << "px.shape[2] must be equal to py.shape[2] + 1. But now "
               << "px.shape[2] is " << px_desc->getDimIndex(2) << ", py.shape[2] is "
               << py_desc->getDimIndex(2) << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // The shape of py must be [B, S+1, T]
  if (S + 1 != py_desc->getDimIndex(1)) {
    LOG(ERROR) << API_NAME << " py.shape[1] must be equal to px.shape[1] + 1. "
               << "But now px.shape[1] is " << px_desc->getDimIndex(1)
               << ", py.shape[1] is " << py_desc->getDimIndex(1) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // The shape of opt_boundary must be [B, 4]
  if (nullptr != opt_boundary_desc &&
      (B != opt_boundary_desc->getDimIndex(0) || 4 != opt_boundary_desc->getDimIndex(1))) {
    LOG(ERROR) << API_NAME << " When opt_boundary is not NULL, "
               << "opt_boundary.shape[0] and px.shape[0] must be same, and "
               << "opt_boundary.shape[1] must be 4. But now "
               << "px.shape[0] is " << px_desc->getDimIndex(0)
               << ", opt_boundary.shape[0] is " << opt_boundary_desc->getDimIndex(0)
               << ", opt_boundary.shape[1] is " << opt_boundary_desc->getDimIndex(1)
               << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // The shape of p must be [B, S+1, T+1]
  if (S + 1 != p_desc->getDimIndex(1) || T + 1 != p_desc->getDimIndex(2)) {
    LOG(ERROR) << API_NAME << " p.shape[1] and py.shape[1] must be same, and "
               << "p.shape[2] must be equal to py.shape[2] + 1. "
               << "But now p.shape[1] is " << p_desc->getDimIndex(1)
               << ", py.shape[1] is " << py_desc->getDimIndex(1) << ", p.shape[2] is "
               << p_desc->getDimIndex(2) << ", py.shape[2] is " << py_desc->getDimIndex(2)
               << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // The shape of px and px_grad must be same: [B, S, T+1]
  for (int i = 1; i < px_grad_desc->getDim(); ++i) {
    if (px_grad_desc->getDimIndex(i) != px_desc->getDimIndex(i)) {
      LOG(ERROR) << API_NAME
                 << " The shape of px and px_grad must be same. But now "
                 << "px.shape[" << i << "] is " << px_desc->getDimIndex(i)
                 << ", px_grad.shape[" << i << "] is " << px_grad_desc->getDimIndex(i)
                 << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  // The shape of py and py_grad must be same: [B, S+1, T]
  for (int i = 1; i < py_grad_desc->getDim(); ++i) {
    if (py_grad_desc->getDimIndex(i) != py_desc->getDimIndex(i)) {
      LOG(ERROR) << API_NAME
                 << " The shape of py and py_grad must be same. But now "
                 << "py.shape[" << i << "] is " << py_desc->getDimIndex(i)
                 << ", py_grad.shape[" << i << "] is " << py_grad_desc->getDimIndex(i)
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
  if (MLUOP_DTYPE_FLOAT != px_desc->getDtype()) {
    LOG(ERROR) << API_NAME
               << "The data type of px currently only support float. But now "
               << "the data type of px is "
               << mluOpGetNameOfDataType(px_desc->getDtype()) << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (MLUOP_DTYPE_FLOAT != py_desc->getDtype()) {
    LOG(ERROR) << API_NAME
               << "The data type of py currently only support float. But now "
               << "the data type of py is "
               << mluOpGetNameOfDataType(py_desc->getDtype()) << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (nullptr != opt_boundary_desc &&
      MLUOP_DTYPE_INT64 != opt_boundary_desc->getDtype()) {
    LOG(ERROR) << API_NAME
               << "The data type of opt_boundary currently only support int64."
               << " But now the data type of opt_boundary is "
               << mluOpGetNameOfDataType(opt_boundary_desc->getDtype()) << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (MLUOP_DTYPE_FLOAT != p_desc->getDtype()) {
    LOG(ERROR) << API_NAME
               << "The data type of p currently only support float. But now "
               << "the data type of p is "
               << mluOpGetNameOfDataType(p_desc->getDtype()) << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (MLUOP_DTYPE_FLOAT != ans_grad_desc->getDtype()) {
    LOG(ERROR) << API_NAME
               << "The data type of ans_grad currently only support float. "
               << "But now the data type of ans_grad is "
               << mluOpGetNameOfDataType(ans_grad_desc->getDtype()) << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (MLUOP_DTYPE_FLOAT != px_grad_desc->getDtype()) {
    LOG(ERROR) << API_NAME
               << "The data type of px_grad currently only support float. "
               << "But now the data type of px_grad is "
               << mluOpGetNameOfDataType(px_grad_desc->getDtype()) << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (MLUOP_DTYPE_FLOAT != py_grad_desc->getDtype()) {
    LOG(ERROR) << API_NAME
               << "The data type of py_grad currently only support float. "
               << "But now the data type of py_grad is "
               << mluOpGetNameOfDataType(py_grad_desc->getDtype()) << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t checkTensorScaleLimit(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t px_desc,
    const mluOpTensorDescriptor_t py_desc,
    const mluOpTensorDescriptor_t opt_boundary_desc,
    const mluOpTensorDescriptor_t p_desc) {
  // Check large tensor
  // The shape of px and px_grad are the same,
  // The shape of py and py_grad are the same,
  // So there is no need to check the tensor num of px_grad and py_grad
  if (mluOpGetTensorElementNum(px_desc) >= LARGE_TENSOR_NUM ||
      mluOpGetTensorElementNum(py_desc) >= LARGE_TENSOR_NUM ||
      (nullptr != opt_boundary_desc &&
       mluOpGetTensorElementNum(opt_boundary_desc) >= LARGE_TENSOR_NUM) ||
      mluOpGetTensorElementNum(p_desc) >= LARGE_TENSOR_NUM) {
    LOG(ERROR) << API_NAME << " Overflow max tensor num."
               << " Current operator supports tensor num smaller than 2^31.";
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

  } else if (nullptr == opt_boundary_desc && nullptr == opt_boundary) {
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

  // Since the layout of all tensors are ARRAY, so skip check tensor layout

  // 2. check mlu platform
  if (handle->arch < 372) {
    LOG(ERROR) << API_NAME << " Only mlu300 and above devices are supported."
               << " Please check the device version!";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  // 3. check tensor dim
  mluOpStatus_t check_status =
      checkTensorDim(px_desc, py_desc, opt_boundary_desc, p_desc, ans_grad_desc,
                     px_grad_desc, py_grad_desc);
  if (MLUOP_STATUS_SUCCESS != check_status) {
    return check_status;
  }

  // 4. check tensor shape
  check_status = checkTensorShape(px_desc, py_desc, opt_boundary_desc, p_desc,
                                  ans_grad_desc, px_grad_desc, py_grad_desc);
  if (MLUOP_STATUS_SUCCESS != check_status) {
    return check_status;
  }

  // 5. check tensor stride
  STRIDE_TENSOR_CHECK("[mluOpMutualInformationBackward]:", px_desc,
                      "px_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpMutualInformationBackward]:", py_desc,
                      "py_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpMutualInformationBackward]:", opt_boundary_desc,
                      "opt_boundary_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpMutualInformationBackward]:", p_desc,
                      "p_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpMutualInformationBackward]:", ans_grad_desc,
                      "ans_grad_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpMutualInformationBackward]:", px_grad_desc,
                      "px_grad_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpMutualInformationBackward]:", py_grad_desc,
                      "py_grad_desc must be contiguous");

  // 6. check tensor dtype
  check_status =
      checkTensorDatatype(px_desc, py_desc, opt_boundary_desc, p_desc,
                          ans_grad_desc, px_grad_desc, py_grad_desc);
  if (MLUOP_STATUS_SUCCESS != check_status) {
    return check_status;
  }

  // 7. check scale limit, for large tensor
  check_status = checkTensorScaleLimit(handle, px_desc, py_desc,
                                       opt_boundary_desc, p_desc);
  if (MLUOP_STATUS_SUCCESS != check_status) {
    return check_status;
  }

  const int B = px_desc->getDimIndex(0);
  const int S = px_desc->getDimIndex(1);
  const int T = py_desc->getDimIndex(2);

  // 8. check zero element.
  if (0 == B || (0 == S && 0 == T)) {
    zero_element = true;
    VLOG(5) << API_NAME << " Skip zero element tensor when px.shape[0] is zero "
            << "or px.shape[1] and py.shape[2] are both zero.";
    return MLUOP_STATUS_SUCCESS;
  }

  // 9 check workspace
  if (workspace_size > 0) {
    PARAM_CHECK(API_NAME, workspace != nullptr);
  }

  // 10. check tensor ptr
  check_status =
      checkTensorPtr(px, py, p, ans_grad, opt_boundary_desc, opt_boundary,
                     px_grad, py_grad, S, T, has_boundary);
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
    const bool overwrite_ans_grad, const mluOpTensorDescriptor_t px_grad_desc,
    void *px_grad, const mluOpTensorDescriptor_t py_grad_desc, void *py_grad) {
  GEN_CASE_START("mutual_information_backward", "MUTUAL_INFORMATION_BACKWARD");
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

static void policyFunc3Pipeline(const mluOpHandle_t handle, cnrtDim3_t *k_dim,
                                cnrtFunctionType_t *k_type, int batch_size) {
  int core_num = mluop::runtime::getClusterLimitCapability(handle) *
                 mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  *k_type = cnrtFuncTypeBlock;
  k_dim->x = 1;
  k_dim->y = batch_size < core_num ? batch_size : core_num;
  k_dim->z = 1;
}

static mluOpStatus_t launchMutualInformationBackward3PipelineKernel(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t px_desc, const void *px,
    const mluOpTensorDescriptor_t py_desc, const void *py,
    const bool has_boundary, const void *opt_boundary, const void *p,
    const bool overwrite_ans_grad, void *ans_grad, void *px_grad,
    void *py_grad) {
  const int B = px_desc->getDimIndex(0);
  const int S = px_desc->getDimIndex(1);
  const int T = py_desc->getDimIndex(2);

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc3Pipeline(handle, &k_dim, &k_type, B);
  VLOG(5) << "Launch Kernel 3PipelineMutualInformationBackward<<<Block "
          << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>";
  CHECK_RETURN(
      "[MutualInformationBackward]",
      kernel3PipelineMutualInformationBackward(
          k_dim, k_type, handle->queue, B, S, T, px, py, has_boundary,
          opt_boundary, p, overwrite_ans_grad, ans_grad, px_grad, py_grad));

  return MLUOP_STATUS_SUCCESS;
}

// Calculate computing diagonal number of partition mode for default kernel
static void calComputingDiags(const int S, const int T,
                              int64_t *computing_diag_num, int *s_block_size,
                              int *t_block_size, int *s_repeat, int *t_repeat,
                              int *s_remainder, int *t_remainder,
                              const int mode) {
  // If has remainder part, rearrange block size to balance work load
  s_repeat[mode] = S / s_block_size[mode];
  s_remainder[mode] = S % s_block_size[mode];
  if (s_remainder[mode] > 0) {
    s_block_size[mode] = S / (s_repeat[mode] + 1);
    s_repeat[mode] = S / s_block_size[mode];
    s_remainder[mode] = S % s_block_size[mode];
  }

  t_repeat[mode] = T / t_block_size[mode];
  t_remainder[mode] = T % t_block_size[mode];
  if (t_remainder[mode] > 0) {
    t_block_size[mode] = T / (t_repeat[mode] + 1);
    t_repeat[mode] = T / t_block_size[mode];
    t_remainder[mode] = T % t_block_size[mode];
  }

  // Accumulate all block's computing diagonal numbers
  computing_diag_num[mode] = s_repeat[mode] * t_repeat[mode] *
                             (s_block_size[mode] + t_block_size[mode] - 1);
  if (s_remainder[mode] > 0) {
    computing_diag_num[mode] +=
        t_repeat[mode] * (t_block_size[mode] + s_remainder[mode] - 1);
  }

  if (t_remainder[mode] > 0) {
    computing_diag_num[mode] +=
        s_repeat[mode] * (s_block_size[mode] + t_remainder[mode] - 1);
  }

  if (s_remainder[mode] > 0 && t_remainder[mode] > 0) {
    computing_diag_num[mode] += s_remainder[mode] + t_remainder[mode] - 1;
  }
}

static void assignPartitionParams(const int *s_block_size,
                                  const int *t_block_size, const int *s_repeat,
                                  const int *t_repeat, const int *s_remainder,
                                  const int *t_remainder,
                                  int &final_s_block_size,
                                  int &final_t_block_size, int &final_s_repeat,
                                  int &final_t_repeat, int &final_s_remainder,
                                  int &final_t_remainder, const int mode) {
  final_s_block_size = s_block_size[mode];
  final_t_block_size = t_block_size[mode];
  final_s_repeat = s_repeat[mode];
  final_t_repeat = t_repeat[mode];
  final_s_remainder = s_remainder[mode];
  final_t_remainder = t_remainder[mode];
}

static void calDefaultPartition(const int S, const int T, const int N_size,
                                const int nram_size, int &job_diag_num,
                                int &final_s_block_size,
                                int &final_t_block_size, int &final_s_repeat,
                                int &final_t_repeat, int &final_s_remainder,
                                int &final_t_remainder) {
  // Compute each partition's job diagonal number,
  // and choose the partition method with the least job diagonal number:
  // 1) all S and T, no partition, launch once in one batch;
  // 2) S < max_N_size, compare with (S, t) and (S/2, t);
  // 3) T < max_N_size, compare with (s, T) and (s, T/2);
  // 4) both S and T > max_N_size, compare with (N, N), (S, t), (s, T), if
  // exist;
  if (S <= N_size && T <= N_size) {
    // once can compute all SxT onchip
    job_diag_num = 1;
    final_s_block_size = S;
    final_t_block_size = T;
    final_s_repeat = 1;
    final_t_repeat = 1;
    final_s_remainder = 0;
    final_t_remainder = 0;
    return;
  } else {
    // Sum of each partition's number of computing diagonals
    // at most 3 arrays of candidate partition mode
    int mode;
    int64_t computing_diag_num[3] = {0};
    int s_block_size[3] = {0};
    int t_block_size[3] = {0};
    int s_repeat[3] = {0};
    int t_repeat[3] = {0};
    int s_remainder[3] = {0};
    int t_remainder[3] = {0};

    if (S <= N_size && T > N_size) {
      // compare with (S, t) and (S/2, t)
      // 1) deal_s = S; min(s, t) = s;
      mode = 0;
      s_block_size[0] = S;
      t_block_size[0] = (nram_size / sizeof(float) - 8 * s_block_size[0]) /
                        (4 * s_block_size[0] + 2);
      calComputingDiags(S, T, computing_diag_num, s_block_size, t_block_size,
                        s_repeat, t_repeat, s_remainder, t_remainder, mode);
      // 2) deal_s = S/2; min(s, t) = s;
      mode = 1;
      s_block_size[1] = std::max(S / 2, 1);  // at least 1 number in s_block
      t_block_size[1] = (nram_size / sizeof(float) - 8 * s_block_size[1]) /
                        (4 * s_block_size[1] + 2);
      calComputingDiags(S, T, computing_diag_num, s_block_size, t_block_size,
                        s_repeat, t_repeat, s_remainder, t_remainder, mode);

      if (computing_diag_num[0] <= computing_diag_num[1]) {
        assignPartitionParams(
            s_block_size, t_block_size, s_repeat, t_repeat, s_remainder,
            t_remainder, final_s_block_size, final_t_block_size, final_s_repeat,
            final_t_repeat, final_s_remainder, final_t_remainder, 0);
      } else {
        assignPartitionParams(
            s_block_size, t_block_size, s_repeat, t_repeat, s_remainder,
            t_remainder, final_s_block_size, final_t_block_size, final_s_repeat,
            final_t_repeat, final_s_remainder, final_t_remainder, 1);
      }
    } else if (S > N_size && T <= N_size) {
      // compare with (s, T) and (s, T/2)
      // 1) deal_t = T; min(s, t) = t;
      mode = 0;
      t_block_size[0] = T;
      s_block_size[0] = (nram_size / sizeof(float) - 8 * t_block_size[0]) /
                        (4 * t_block_size[0] + 2);
      calComputingDiags(S, T, computing_diag_num, s_block_size, t_block_size,
                        s_repeat, t_repeat, s_remainder, t_remainder, mode);
      // 2) deal_t = T/2; min(s, t) = t;
      mode = 1;
      t_block_size[1] = std::max(T / 2, 1);  // at least 1 number in t_block
      s_block_size[1] = (nram_size / sizeof(float) - 8 * t_block_size[1]) /
                        (4 * t_block_size[1] + 2);
      calComputingDiags(S, T, computing_diag_num, s_block_size, t_block_size,
                        s_repeat, t_repeat, s_remainder, t_remainder, mode);

      if (computing_diag_num[0] <= computing_diag_num[1]) {
        assignPartitionParams(
            s_block_size, t_block_size, s_repeat, t_repeat, s_remainder,
            t_remainder, final_s_block_size, final_t_block_size, final_s_repeat,
            final_t_repeat, final_s_remainder, final_t_remainder, 0);
      } else {
        assignPartitionParams(
            s_block_size, t_block_size, s_repeat, t_repeat, s_remainder,
            t_remainder, final_s_block_size, final_t_block_size, final_s_repeat,
            final_t_repeat, final_s_remainder, final_t_remainder, 1);
      }
    } else {  // S > N_size, T > N_size, choose between (N,N), (S,t), (s,T)
      // 1) deal_s = deal_t = N_size; min(s,t) = s = t;
      mode = 0;
      s_block_size[0] = N_size;
      t_block_size[0] = N_size;
      calComputingDiags(S, T, computing_diag_num, s_block_size, t_block_size,
                        s_repeat, t_repeat, s_remainder, t_remainder, mode);
      // 2) deal_s = S, deal_t = t; min(s,t) = t;
      mode = 1;
      s_block_size[1] = N_size;
      t_block_size[1] = (nram_size / sizeof(float) - 2 * s_block_size[1]) /
                        (4 * s_block_size[1] + 8);
      if (t_block_size[1] > 0) {
        calComputingDiags(S, T, computing_diag_num, s_block_size, t_block_size,
                          s_repeat, t_repeat, s_remainder, t_remainder, mode);
      } else {
        computing_diag_num[1] = -1;  // not support on this partition
      }
      // 3) deal_t = T, deal_s = s; min(s,t) = s;
      mode = 2;
      t_block_size[2] = T;
      s_block_size[2] = (nram_size / sizeof(float) - 2 * t_block_size[2]) /
                        (4 * t_block_size[2] + 8);
      if (s_block_size[2] > 0) {
        calComputingDiags(S, T, computing_diag_num, s_block_size, t_block_size,
                          s_repeat, t_repeat, s_remainder, t_remainder, mode);
      } else {
        computing_diag_num[2] = -1;  // not support on this partition
      }

      if (computing_diag_num[0] > 0 &&      // mode 0 is valid
          ((computing_diag_num[1] <= 0) ||  // mode 1 is invalid or
           computing_diag_num[0] <=
               computing_diag_num[1])) {  // mode 0 is better than mode 1
        if (computing_diag_num[2] > 0 &&  // mode 2 is valid and
            computing_diag_num[2] <
                computing_diag_num[0]) {  // mode 2 is better than mode 0
          // choose mode 2
          assignPartitionParams(s_block_size, t_block_size, s_repeat, t_repeat,
                                s_remainder, t_remainder, final_s_block_size,
                                final_t_block_size, final_s_repeat,
                                final_t_repeat, final_s_remainder,
                                final_t_remainder, 2);
        } else {
          // choose mode 0
          assignPartitionParams(s_block_size, t_block_size, s_repeat, t_repeat,
                                s_remainder, t_remainder, final_s_block_size,
                                final_t_block_size, final_s_repeat,
                                final_t_repeat, final_s_remainder,
                                final_t_remainder, 0);
        }
      } else {  // mode 1 is valid and mode 1 is better than mode 0
        if (computing_diag_num[2] > 0 &&  // mode 2 is valid
            computing_diag_num[2] <
                computing_diag_num[1]) {  // mode 2 is better than mode 1
          // choose mode 2
          assignPartitionParams(s_block_size, t_block_size, s_repeat, t_repeat,
                                s_remainder, t_remainder, final_s_block_size,
                                final_t_block_size, final_s_repeat,
                                final_t_repeat, final_s_remainder,
                                final_t_remainder, 2);
        } else {
          // choose mode 1
          assignPartitionParams(s_block_size, t_block_size, s_repeat, t_repeat,
                                s_remainder, t_remainder, final_s_block_size,
                                final_t_block_size, final_s_repeat,
                                final_t_repeat, final_s_remainder,
                                final_t_remainder, 1);
        }
      }
    }
    // total job diagonal number in parallel
    job_diag_num = final_s_repeat + (int)(final_s_remainder > 0) +
                   final_t_repeat + (int)(final_t_remainder > 0) - 1;
  }
}

static mluOpStatus_t launchMutualInformationBackwardDefaultKernel(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t px_desc, const void *px,
    const mluOpTensorDescriptor_t py_desc, const void *py,
    const bool has_boundary, const void *opt_boundary, const void *p,
    const bool overwrite_ans_grad, void *ans_grad, void *px_grad, void *py_grad,
    void *p_grad) {
  // At first, use Fill Op to set px_grad, py_grad to all 0
  VLOG(5) << API_NAME << " cnnlFill_v3 start.";
  uint64_t fill_value = 0x0;
  if (mluOpGetTensorElementNum(px_desc) > 0) {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(px_desc, cnnl_output_desc);
    CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &fill_value,
                          cnnl_output_desc, px_grad));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  if (mluOpGetTensorElementNum(py_desc) > 0) {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(py_desc, cnnl_output_desc);
    CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &fill_value,
                          cnnl_output_desc, py_grad));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  VLOG(5) << API_NAME << " cnnlFill_v3 end.";

  // When S and T is too large, launch default kernel with partition of S and T
  // 1. Compute current arch max N size, according to NRAM size and device RAM
  // 2. Use max_N_size to calculate different partition mode computing diagonal
  //    numbers and choose the partition mode, which has the least computing
  //    diagonal number
  // 3. Launch default kernels by diagonal in parallel, with check of MaxDimX

  const int B = px_desc->getDimIndex(0);
  const int S = px_desc->getDimIndex(1);
  const int T = py_desc->getDimIndex(2);

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  // 1. According to on-chip RAM size, calculate current arch partition block
  // size by square, Use max_N_size to partition on S and T dimension RAM space:
  //   2*S*T + 2*(S+1)*(T+1) + 2*min(S,T) + 4*min(S,T)+1
  int max_N_size = (int)(std::sqrt(handle->nram_size / sizeof(float) / 4)) - 2;
  // Use max square size N, partition on T and S dimension, launch by diagonal:
  // -|------T--------|
  // :| N1| N2| N3| N4|
  // :|---|---|---|---|
  // S| N2| N3| N4| N5|
  // :|---|---|---|---|
  // :| N3| N4| N5| N6|
  // -|---------------|

  VLOG(5) << "Current arch Max square N size is " << max_N_size;

  int job_diag_num;  // number of default kernel launch steps by diagonal
  int s_block_size, t_block_size, s_repeat, t_repeat, s_remainder, t_remainder;

  // 2. Choose the partition mode, which has the least computing diagonal number
  // NOTE: p_grad has dimension (S+1, T+1), in function directly use (S, T)
  // instead
  calDefaultPartition(S + 1, T + 1, max_N_size, handle->nram_size, job_diag_num,
                      s_block_size, t_block_size, s_repeat, t_repeat,
                      s_remainder, t_remainder);
  int s_block_num = s_repeat + (int)(s_remainder > 0);
  int t_block_num = t_repeat + (int)(t_remainder > 0);
  int max_s_t_block_num = std::max(s_block_num, t_block_num);
  int min_s_t_block_num = std::min(s_block_num, t_block_num);

  k_type = cnrtFuncTypeBlock;
  k_dim.y = 1;
  k_dim.z = 1;
  // Get current arch support max dim_x value
  int task_dim_x_limit;
  cnDeviceGetAttribute(&task_dim_x_limit,
                       CN_DEVICE_ATTRIBUTE_MAX_BLOCK_TASK_DIM_X,
                       handle->device);
  VLOG(5) << "Current arch MAX_BLOCK_TASK_DIM_X is " << task_dim_x_limit;

  // 3. Traverse step_i from 0 to (job_diag_num - 1)
  for (int step_i = 0; step_i < job_diag_num; step_i++) {
    int job_num_on_step = B * (step_i < max_s_t_block_num
                                   ? std::min(step_i + 1, min_s_t_block_num)
                                   : s_block_num + t_block_num - step_i - 1);
    k_dim.x = job_num_on_step;
    // Make sure not exceed max dim x limit
    if (k_dim.x > task_dim_x_limit) {
      int task_dim_change = (k_dim.x + task_dim_x_limit - 1) / task_dim_x_limit;
      k_dim.x = (k_dim.x + task_dim_x_limit - 1) / task_dim_change;
      k_dim.y = k_dim.y * task_dim_change;
    }

    VLOG(5) << "Launch Kernel DefaultMutualInformationBackward<<< step "
            << step_i << " of Batch Block: " << k_dim.x << ", " << k_dim.y
            << ", " << k_dim.z << ">>>";
    CHECK_RETURN("[MutualInformationBackward]",
                 kernelDefaultMutualInformationBackward(
                     k_dim, k_type, handle->queue, B, S, T, step_i,
                     job_num_on_step, s_block_num, t_block_num, s_block_size,
                     t_block_size, px, py, has_boundary, opt_boundary, p,
                     overwrite_ans_grad, ans_grad, px_grad, py_grad, p_grad));
  }
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
  // 1. Paramcheck
  bool has_boundary = false;
  bool zero_element = false;
  mluOpStatus_t check_status = mutualInformationBackwardParamCheck(
      handle, px_desc, px, py_desc, py, opt_boundary_desc, opt_boundary, p_desc,
      p, ans_grad_desc, ans_grad, workspace, workspace_size, px_grad_desc,
      px_grad, py_grad_desc, py_grad, has_boundary, zero_element);

  if (MLUOP_STATUS_SUCCESS != check_status || zero_element) {
    return check_status;
  }

  // 2. Generate case
  if (MLUOP_GEN_CASE_ON_NEW) {
    mutualInformationBackwardGencase(
        handle, px_desc, px, py_desc, py, opt_boundary_desc, opt_boundary,
        p_desc, p, ans_grad_desc, ans_grad, overwrite_ans_grad, px_grad_desc,
        px_grad, py_grad_desc, py_grad);
  }

  // Choose to launch 3pipeline or default kernel
  const int B = px_desc->getDimIndex(0);
  const int S = px_desc->getDimIndex(1);
  const int T = py_desc->getDimIndex(2);

  bool is_launch_3pipeline = true;
  // check 3pipeline scale limit for computing term1 and term2
  int current_size = T * (S + 1) + (T + 1) * S + 5 * (T + 1);
  if (current_size > handle->nram_size / sizeof(float)) {
    is_launch_3pipeline = false;
  }

  // check 3pipeline scale limit for computing p_grad
  current_size =
      T * (S + 1) + (T + 1) * S + (T + 1) * (S + 1) + 3 * std::min(S, T) + 4;
  if (current_size > handle->nram_size / sizeof(float)) {
    is_launch_3pipeline = false;
  }

  // 3. launch kernel
  mluOpStatus_t return_status;
  if (is_launch_3pipeline) {
    // launch 3pipeline kernel when satisfy scale limit
    return_status = launchMutualInformationBackward3PipelineKernel(
        handle, px_desc, px, py_desc, py, has_boundary, opt_boundary, p,
        overwrite_ans_grad, ans_grad, px_grad, py_grad);
  } else {
    // launch default kernel, workspace is for p_grad
    return_status = launchMutualInformationBackwardDefaultKernel(
        handle, px_desc, px, py_desc, py, has_boundary, opt_boundary, p,
        overwrite_ans_grad, ans_grad, px_grad, py_grad, workspace);
  }

  GEN_CASE_END();
  return return_status;
}
