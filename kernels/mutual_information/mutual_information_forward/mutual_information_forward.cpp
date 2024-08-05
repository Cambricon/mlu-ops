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
#include "mutual_information_forward.h"

#include <algorithm>
#include <string>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"

#define API_NAME "[mluOpMutualInformationForward]"

mluOpStatus_t MLUOP_WIN_API mluOpGetMutualInformationForwardWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t px_desc,
    const mluOpTensorDescriptor_t py_desc,
    const mluOpTensorDescriptor_t opt_boundary_desc,
    const mluOpTensorDescriptor_t p_desc,
    const mluOpTensorDescriptor_t ans_desc, size_t *workspace_size) {
  PARAM_CHECK(API_NAME, handle != nullptr);
  PARAM_CHECK(API_NAME, px_desc != nullptr);
  PARAM_CHECK(API_NAME, py_desc != nullptr);
  PARAM_CHECK(API_NAME, p_desc != nullptr);
  PARAM_CHECK(API_NAME, ans_desc != nullptr);
  PARAM_CHECK(API_NAME, workspace_size != nullptr);
  // Workspace is not required in the current implementation.
  *workspace_size = 0;
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t checkTensorDim(
    const mluOpTensorDescriptor_t px_desc,
    const mluOpTensorDescriptor_t py_desc,
    const mluOpTensorDescriptor_t opt_boundary_desc,
    const mluOpTensorDescriptor_t p_desc,
    const mluOpTensorDescriptor_t ans_desc) {
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
  if (1 != ans_desc->dim) {
    LOG(ERROR) << API_NAME << " The dim of ans must be 1. "
               << "But now the dim of ans is " << ans_desc->dim << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t checkTensorShape(
    const mluOpTensorDescriptor_t px_desc,
    const mluOpTensorDescriptor_t py_desc,
    const mluOpTensorDescriptor_t opt_boundary_desc,
    const mluOpTensorDescriptor_t p_desc,
    const mluOpTensorDescriptor_t ans_desc) {
  const int B = px_desc->dims[0];
  const int S = px_desc->dims[1];
  const int T = py_desc->dims[2];
  if (B != py_desc->dims[0] || B != p_desc->dims[0] || B != ans_desc->dims[0]) {
    LOG(ERROR) << API_NAME
               << " px.shape[0], py.shape[0], p.shape[0], ans.shape[0], "
               << "must be same. But now "
               << "px.shape[0] is " << px_desc->dims[0] << ", py.shape[0] is "
               << py_desc->dims[0] << ", p.shape[0] is " << p_desc->dims[0]
               << ", ans.shape[0] is " << ans_desc->dims[0] << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // Currently only supports !modified, so the shape of px must be [B, S, T+1]
  if (T + 1 != px_desc->dims[2]) {
    LOG(ERROR) << API_NAME << " Currently only supports the case that "
               << "px.shape[2] must be equal to py.shape[2] + 1. But now "
               << "px.shape[2] is " << px_desc->dims[2] << ", py.shape[2] is "
               << py_desc->dims[2] << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // The shape of py must be [B, S+1, T]
  if (S + 1 != py_desc->dims[1]) {
    LOG(ERROR) << API_NAME << " py.shape[1] must be equal to px.shape[1] + 1. "
               << "But now px.shape[1] is " << px_desc->dims[1]
               << ", py.shape[1] is " << py_desc->dims[1] << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // The shape of opt_boundary must be [B, 4]
  if (nullptr != opt_boundary_desc &&
      (B != opt_boundary_desc->dims[0] || 4 != opt_boundary_desc->dims[1])) {
    LOG(ERROR) << API_NAME << " When opt_boundary is not NULL, "
               << "opt_boundary.shape[0] and px.shape[0] must be same, and "
               << "opt_boundary.shape[1] must be 4. But now "
               << "px.shape[0] is " << px_desc->dims[0]
               << ", opt_boundary.shape[0] is " << opt_boundary_desc->dims[0]
               << ", opt_boundary.shape[1] is " << opt_boundary_desc->dims[1]
               << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // The shape of p must be [B, S+1, T+1]
  if (S + 1 != p_desc->dims[1] || T + 1 != p_desc->dims[2]) {
    LOG(ERROR) << API_NAME << " p.shape[1] and py.shape[1] must be same, and "
               << "p.shape[2] must be equal to py.shape[2] + 1. "
               << "But now p.shape[1] is " << p_desc->dims[1]
               << ", py.shape[1] is " << py_desc->dims[1] << ", p.shape[2] is "
               << p_desc->dims[2] << ", py.shape[2] is " << py_desc->dims[2]
               << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t checkTensorDatatype(
    const mluOpTensorDescriptor_t px_desc,
    const mluOpTensorDescriptor_t py_desc,
    const mluOpTensorDescriptor_t opt_boundary_desc,
    const mluOpTensorDescriptor_t p_desc,
    const mluOpTensorDescriptor_t ans_desc) {
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
  if (MLUOP_DTYPE_FLOAT != ans_desc->dtype) {
    LOG(ERROR) << API_NAME
               << "The data type of ans currently only support float. "
               << "But now the data type of ans is "
               << mluOpGetNameOfDataType(ans_desc->dtype) << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t checkTensorScaleLimit(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t px_desc,
    const mluOpTensorDescriptor_t py_desc,
    const mluOpTensorDescriptor_t opt_boundary_desc,
    const mluOpTensorDescriptor_t p_desc) {
  // check large tensor
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
    const void *px, const void *py, const void *p, const void *ans,
    const mluOpTensorDescriptor_t opt_boundary_desc, const void *opt_boundary,
    const int S, const int T, bool &has_boundary) {
  if (S > 0) {
    PARAM_CHECK(API_NAME, px != nullptr);
  } else {
    VLOG(5) << API_NAME << " px.shape[1] is zero.";
  }

  if (T > 0) {
    PARAM_CHECK(API_NAME, py != nullptr);
  } else {
    VLOG(5) << API_NAME << " py.shape[2] is zero.";
  }

  PARAM_CHECK(API_NAME, p != nullptr);
  PARAM_CHECK(API_NAME, ans != nullptr);

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

static mluOpStatus_t mutualInformationForwardParamCheck(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t px_desc, const void *px,
    const mluOpTensorDescriptor_t py_desc, const void *py,
    const mluOpTensorDescriptor_t opt_boundary_desc, const void *opt_boundary,
    const mluOpTensorDescriptor_t p_desc, const void *p, void *workspace,
    const size_t workspace_size, const mluOpTensorDescriptor_t ans_desc,
    void *ans, bool &has_boundary, bool &zero_element) {
  // 1. check handle and tensor_desc
  PARAM_CHECK(API_NAME, handle != nullptr);
  PARAM_CHECK(API_NAME, px_desc != nullptr);
  PARAM_CHECK(API_NAME, py_desc != nullptr);
  PARAM_CHECK(API_NAME, p_desc != nullptr);
  PARAM_CHECK(API_NAME, ans_desc != nullptr);

  // since the layout of all tensor is ARRAY, so skip check tensor layout

  // 2. check mlu platform
  if (handle->arch < 372) {
    LOG(ERROR) << API_NAME << " Only mlu300 and above devices are supported."
               << " Please check the device version!";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  // 3. check tensor dim
  mluOpStatus_t check_status =
      checkTensorDim(px_desc, py_desc, opt_boundary_desc, p_desc, ans_desc);
  if (MLUOP_STATUS_SUCCESS != check_status) {
    return check_status;
  }

  // 4. check tensor shape
  check_status =
      checkTensorShape(px_desc, py_desc, opt_boundary_desc, p_desc, ans_desc);
  if (MLUOP_STATUS_SUCCESS != check_status) {
    return check_status;
  }

  // 5. check tensor stride
  STRIDE_TENSOR_CHECK("[mluOpMutualInformationForward]:", px_desc,
                      "px_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpMutualInformationForward]:", py_desc,
                      "py_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpMutualInformationForward]:", opt_boundary_desc,
                      "opt_boundary_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpMutualInformationForward]:", p_desc,
                      "p_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpMutualInformationForward]:", ans_desc,
                      "ans_desc must be contiguous");

  // 6. check tensor dtype
  check_status = checkTensorDatatype(px_desc, py_desc, opt_boundary_desc,
                                     p_desc, ans_desc);
  if (MLUOP_STATUS_SUCCESS != check_status) {
    return check_status;
  }

  // 7. check scale limit, for large tensor
  check_status = checkTensorScaleLimit(handle, px_desc, py_desc,
                                       opt_boundary_desc, p_desc);
  if (MLUOP_STATUS_SUCCESS != check_status) {
    return check_status;
  }

  const int B = px_desc->dims[0];
  const int S = px_desc->dims[1];
  const int T = py_desc->dims[2];

  // 8. check zero element.
  if (0 == B) {
    zero_element = true;
    VLOG(5) << API_NAME
            << " Skip zero element tensor when px.shape[0] is zero.";
    return MLUOP_STATUS_SUCCESS;
  }

  // 9. check workspace
  if (workspace_size > 0) {
    PARAM_CHECK(API_NAME, workspace != nullptr);
  }

  // 10. check tensor ptr
  check_status = checkTensorPtr(px, py, p, ans, opt_boundary_desc, opt_boundary,
                                S, T, has_boundary);
  if (MLUOP_STATUS_SUCCESS != check_status) {
    return check_status;
  }

  return MLUOP_STATUS_SUCCESS;
}

static void mutualInformationForwardGencase(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t px_desc, const void *px,
    const mluOpTensorDescriptor_t py_desc, const void *py,
    const mluOpTensorDescriptor_t opt_boundary_desc, const void *opt_boundary,
    const mluOpTensorDescriptor_t p_desc, const void *p,
    const mluOpTensorDescriptor_t ans_desc, void *ans) {
  GEN_CASE_START("mutual_information_forward", "MUTUAL_INFORMATION_FORWARD");
  GEN_CASE_HANDLE(handle);

  GEN_CASE_DATA(true, "px", px, px_desc, -1, 1);
  GEN_CASE_DATA(true, "py", py, py_desc, -1, 1);
  if (nullptr != opt_boundary) {
    GEN_CASE_DATA_REAL(true, "opt_boundary", opt_boundary, opt_boundary_desc);
  }
  GEN_CASE_DATA(true, "p", p, p_desc, -1, 1);
  GEN_CASE_DATA(false, "p", p, p_desc, -1, 1);
  GEN_CASE_DATA(false, "ans", ans, ans_desc, -1, 1);
  GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
}

static void policyFunc3Pipeline(const mluOpHandle_t handle, cnrtDim3_t *k_dim,
                                cnrtFunctionType_t *k_type, int batch_size) {
  int core_num = mluop::runtime::getClusterLimitCapability(handle) *
                 mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  *k_type = CNRT_FUNC_TYPE_BLOCK;
  k_dim->x = 1;
  k_dim->y = batch_size < core_num ? batch_size : core_num;
  k_dim->z = 1;
}

static mluOpStatus_t launchMutualInformationForward3PipelineKernel(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t px_desc, const void *px,
    const mluOpTensorDescriptor_t py_desc, const void *py,
    const bool has_boundary, const void *opt_boundary, void *p, void *ans) {
  const int B = px_desc->dims[0];
  const int S = px_desc->dims[1];
  const int T = py_desc->dims[2];

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc3Pipeline(handle, &k_dim, &k_type, B);
  VLOG(5) << "Launch Kernel 3PipelineMutualInformationForward<<<Block "
          << k_dim.x << ", " << k_dim.y << ", " << k_dim.z << ">>>";
  CHECK_RETURN("[MutualInformationForward]",
               kernel3PipelineMutualInformationForward(
                   k_dim, k_type, handle->queue, B, S, T, px, py, has_boundary,
                   opt_boundary, p, ans));

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
      t_block_size[0] = (nram_size / sizeof(float) - 7 * s_block_size[0]) /
                        (3 * s_block_size[0] + 1);
      calComputingDiags(S, T, computing_diag_num, s_block_size, t_block_size,
                        s_repeat, t_repeat, s_remainder, t_remainder, mode);
      // 2) deal_s = S/2; min(s, t) = s;
      mode = 1;
      s_block_size[1] = std::max(S / 2, 1);  // at least 1 number in s_block
      t_block_size[1] = (nram_size / sizeof(float) - 7 * s_block_size[1]) /
                        (3 * s_block_size[1] + 1);
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
      s_block_size[0] = (nram_size / sizeof(float) - 7 * t_block_size[0]) /
                        (3 * t_block_size[0] + 1);
      calComputingDiags(S, T, computing_diag_num, s_block_size, t_block_size,
                        s_repeat, t_repeat, s_remainder, t_remainder, mode);
      // 2) deal_t = T/2; min(s, t) = t;
      mode = 1;
      t_block_size[1] = std::max(T / 2, 1);  // at least 1 number in t_block
      s_block_size[1] = (nram_size / sizeof(float) - 7 * t_block_size[1]) /
                        (3 * t_block_size[1] + 1);
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
      t_block_size[1] = (nram_size / sizeof(float) - 1 * s_block_size[1]) /
                        (3 * s_block_size[1] + 7);
      if (t_block_size[1] > 0) {
        calComputingDiags(S, T, computing_diag_num, s_block_size, t_block_size,
                          s_repeat, t_repeat, s_remainder, t_remainder, mode);
      } else {
        computing_diag_num[1] = -1;  // not support on this partition
      }
      // 3) deal_t = T, deal_s = s; min(s,t) = s;
      mode = 2;
      t_block_size[2] = T;
      s_block_size[2] = (nram_size / sizeof(float) - 1 * t_block_size[2]) /
                        (3 * t_block_size[2] + 7);
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

static mluOpStatus_t launchMutualInformationForwardDefaultKernel(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t px_desc, const void *px,
    const mluOpTensorDescriptor_t py_desc, const void *py,
    const bool has_boundary, const void *opt_boundary, void *p, void *ans) {
  // When S and T is too large, launch default kernel with partition of S and T
  // 1. Compute current arch max N size, according to NRAM size and device RAM
  // 2. Use max_N_size to calculate different partition mode computing diagonal
  //    numbers and choose the partition mode, which has the least computing
  //    diagonal number
  // 3. Launch default kernels by diagonal in parallel, with check of MaxDimX

  const int B = px_desc->dims[0];
  const int S = px_desc->dims[1];
  const int T = py_desc->dims[2];

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  // 1. According to on-chip RAM size, calculate current arch partition block
  // size by square, Use max_N_size to partition on S and T dimension RAM space:
  //   (S+1)*(T+1) + S*T + S*T + 3*min(S,T) + 3*min(S,T)+1
  int max_N_size = (int)(std::sqrt(handle->nram_size / sizeof(float) / 3)) - 2;
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
  // NOTE: p has dimension (S+1, T+1), in function directly use (S, T) instead
  calDefaultPartition(S + 1, T + 1, max_N_size, handle->nram_size, job_diag_num,
                      s_block_size, t_block_size, s_repeat, t_repeat,
                      s_remainder, t_remainder);
  int s_block_num = s_repeat + (int)(s_remainder > 0);
  int t_block_num = t_repeat + (int)(t_remainder > 0);
  int max_s_t_block_num = std::max(s_block_num, t_block_num);
  int min_s_t_block_num = std::min(s_block_num, t_block_num);

  k_type = CNRT_FUNC_TYPE_BLOCK;
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

    VLOG(5) << "Launch Kernel DefaultMutualInformationForward<<< step "
            << step_i << " of Batch Block: " << k_dim.x << ", " << k_dim.y
            << ", " << k_dim.z << ">>>";
    CHECK_RETURN("[MutualInformationForward]",
                 kernelDefaultMutualInformationForward(
                     k_dim, k_type, handle->queue, B, S, T, step_i,
                     job_num_on_step, s_block_num, t_block_num, s_block_size,
                     t_block_size, px, py, has_boundary, opt_boundary, p, ans));
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpMutualInformationForward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t px_desc, const void *px,
    const mluOpTensorDescriptor_t py_desc, const void *py,
    const mluOpTensorDescriptor_t opt_boundary_desc, const void *opt_boundary,
    const mluOpTensorDescriptor_t p_desc, void *p, void *workspace,
    const size_t workspace_size, const mluOpTensorDescriptor_t ans_desc,
    void *ans) {
  // 1. Paramcheck
  bool has_boundary = false;
  bool zero_element = false;
  mluOpStatus_t check_status = mutualInformationForwardParamCheck(
      handle, px_desc, px, py_desc, py, opt_boundary_desc, opt_boundary, p_desc,
      p, workspace, workspace_size, ans_desc, ans, has_boundary, zero_element);

  if (MLUOP_STATUS_SUCCESS != check_status || zero_element) {
    return check_status;
  }

  // 2. Generate case
  if (MLUOP_GEN_CASE_ON_NEW) {
    mutualInformationForwardGencase(handle, px_desc, px, py_desc, py,
                                    opt_boundary_desc, opt_boundary, p_desc, p,
                                    ans_desc, ans);
  }

  // Choose to launch 3pipeline kernel or default kernel
  const int S = px_desc->dims[1];
  const int T = py_desc->dims[2];
  bool is_launch_3pipeline = true;

  // Check 3pipeline kernel scale limit for computing p
  // 9: max_val, mask, temp, ping(py, px, p) and pong(py, px, p)
  // 11: max_val, mask, temp, ping(py, px, p), pong(py, px, p) and 2*(-inf)
  int current_size =
      T * (S + 1) + (T + 1) * S + (T + 1) * (S + 1) + 9 * std::min(S, T) + 11;
  if (current_size > handle->nram_size / sizeof(float)) {
    is_launch_3pipeline = false;
  }

  // 3. Launch kernel
  mluOpStatus_t return_status;
  if (is_launch_3pipeline) {
    // launch 3pipeline kernel when satisfy scale limit
    return_status = launchMutualInformationForward3PipelineKernel(
        handle, px_desc, px, py_desc, py, has_boundary, opt_boundary, p, ans);
  } else {
    // launch default kernel
    return_status = launchMutualInformationForwardDefaultKernel(
        handle, px_desc, px, py_desc, py, has_boundary, opt_boundary, p, ans);
  }

  GEN_CASE_END();
  return return_status;
}
