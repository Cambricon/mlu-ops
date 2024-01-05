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
#include "kernels/utils/cnnl_helper.h"

mluOpStatus_t MLUOP_WIN_API
mluOpMatMulDescCreate(mluOpMatMulDescriptor_t *matmul_desc) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpMatMulDescCreate] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpMatMul", matmul_desc != NULL);
  CHECK_FUNC_RETURN(
      cnnlMatMulDescCreate(matmul_desc), CNNL_STATUS_SUCCESS,
      "[mluOpMatMul] Internal error accured in mluOpMatMulDescCreate.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpMatMulDescDestroy(mluOpMatMulDescriptor_t matmul_desc) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpMatMulDescDestroy] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpMatMul", matmul_desc != NULL);
  CHECK_FUNC_RETURN(
      cnnlMatMulDescDestroy(matmul_desc), CNNL_STATUS_SUCCESS,
      "[mluOpMatMul] Internal error accured in mluOpMatMulDescDestroy.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetMatMulDescAttr(
    mluOpMatMulDescriptor_t matmul_desc, mluOpMatMulDescAttribute_t attr,
    const void *buf, size_t size_in_bytes) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpSetMatMulDescAttr] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpMatMul", matmul_desc != NULL);
  PARAM_CHECK("mluOpMatMul", buf != NULL);
  CHECK_FUNC_RETURN(
      cnnlSetMatMulDescAttr(matmul_desc, cnnlMatMulDescAttribute_t(attr), buf,
                            size_in_bytes),
      CNNL_STATUS_SUCCESS,
      "[mluOpMatMul] Internal error accured in mluOpSetMatMulDescAttr.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetMatMulDescAttr(
    mluOpMatMulDescriptor_t matmul_desc, mluOpMatMulDescAttribute_t attr,
    void *buf, size_t size_in_bytes, size_t *size_written) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpGetMatMulDescAttr] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpMatMul", matmul_desc != NULL);
  PARAM_CHECK("mluOpMatMul", buf != NULL);
  PARAM_CHECK("mluOpMatMul", size_written != NULL);
  CHECK_FUNC_RETURN(
      cnnlGetMatMulDescAttr(matmul_desc, cnnlMatMulDescAttribute_t(attr), buf,
                            size_in_bytes, size_written),
      CNNL_STATUS_SUCCESS,
      "[mluOpMatMul] Internal error accured in mluOpGetMatMulDescAttr.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpCreateMatMulHeuristicResult(mluOpMatMulHeuristicResult_t *result) {
  LOG_FIRST_N(WARNING, 1)
      << "[cnnlCreateMatMulHeuristicResult] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpMatMul", result != NULL);
  CHECK_FUNC_RETURN(cnnlCreateMatMulHeuristicResult(result),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpMatMul] Internal error accured in "
                    "mluOpCreateMatMulHeuristicResult.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpDestroyMatMulHeuristicResult(mluOpMatMulHeuristicResult_t result) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpDestroyMatMulHeuristicResult] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpMatMul", result != NULL);
  CHECK_FUNC_RETURN(cnnlDestroyMatMulHeuristicResult(result),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpMatMul] Internal error accured in "
                    "mluOpDestroyMatMulHeuristicResult.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpGetMatMulHeuristicResult(mluOpMatMulHeuristicResult_t result,
                              mluOpMatMulAlgo_t algo, size_t *workspace_size) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpGetMatMulHeuristicResult] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpMatMul", result != NULL);
  PARAM_CHECK("mluOpMatMul", algo != NULL);
  PARAM_CHECK("mluOpMatMul", workspace_size != NULL);

  CHECK_FUNC_RETURN(cnnlGetMatMulHeuristicResult(result, algo, workspace_size),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpMatMul] Internal error accured in "
                    "mluOpGetMatMulHeuristicResult.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetMatMulAlgoHeuristic(
    mluOpHandle_t handle, mluOpMatMulDescriptor_t matmul_desc,
    mluOpTensorDescriptor_t a_desc, mluOpTensorDescriptor_t b_desc,
    mluOpTensorDescriptor_t c_desc, mluOpTensorDescriptor_t d_desc,
    mluOpMatMulPrefer_t preference, int requested_algo_count,
    mluOpMatMulHeuristicResult_t *result_array, int *return_algo_count) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpGetMatMulAlgoHeuristic] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpMatMul", handle != NULL);
  PARAM_CHECK("mluOpMatMul", matmul_desc != NULL);
  PARAM_CHECK("mluOpMatMul", a_desc != NULL);
  PARAM_CHECK("mluOpMatMul", b_desc != NULL);
  PARAM_CHECK("mluOpMatMul", c_desc != NULL);
  PARAM_CHECK("mluOpMatMul", d_desc != NULL);
  PARAM_CHECK("mluOpMatMul", result_array != NULL);
  PARAM_CHECK("mluOpMatMul", return_algo_count != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(d_desc, cnnl_d_desc);
  CHECK_FUNC_RETURN(
      cnnlGetMatMulAlgoHeuristic(
          cnnl_handle, matmul_desc, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc,
          cnnl_d_desc, cnnlMatMulPrefer_t(preference), requested_algo_count,
          result_array, return_algo_count),
      CNNL_STATUS_SUCCESS,
      "[mluOpMatMul] Internal error accured in mluOpGetMatMulAlgoHeuristic.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_d_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpMatMulAlgoCreate(mluOpMatMulAlgo_t *algo) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpMatMulAlgoCreate] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpMatMul", algo != NULL);
  CHECK_FUNC_RETURN(
      cnnlMatMulAlgoCreate(algo), CNNL_STATUS_SUCCESS,
      "[mluOpMatMul] Internal error accured in mluOpMatMulAlgoCreate.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpMatMulAlgoDestroy(mluOpMatMulAlgo_t algo) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpMatMulAlgoDestroy] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpMatMul", algo != NULL);
  CHECK_FUNC_RETURN(
      cnnlMatMulAlgoDestroy(algo), CNNL_STATUS_SUCCESS,
      "[mluOpMatMul] Internal error accured in mluOpMatMulAlgoDestroy.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetMatMulWorkspaceSize(
    mluOpHandle_t handle, mluOpMatMulDescriptor_t matmul_desc,
    mluOpTensorDescriptor_t a_desc, mluOpTensorDescriptor_t b_desc,
    mluOpTensorDescriptor_t c_desc, mluOpTensorDescriptor_t d_desc,
    mluOpMatMulAlgo_t algo, size_t *workspace_size) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpGetMatMulWorkspaceSize] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpMatMul", handle != NULL);
  PARAM_CHECK("mluOpMatMul", matmul_desc != NULL);
  PARAM_CHECK("mluOpMatMul", a_desc != NULL);
  PARAM_CHECK("mluOpMatMul", b_desc != NULL);
  PARAM_CHECK("mluOpMatMul", c_desc != NULL);
  PARAM_CHECK("mluOpMatMul", d_desc != NULL);
  PARAM_CHECK("mluOpMatMul", algo != NULL);
  PARAM_CHECK("mluOpMatMul", workspace_size != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(d_desc, cnnl_d_desc);

  CHECK_FUNC_RETURN(
      cnnlGetMatMulWorkspaceSize(cnnl_handle, matmul_desc, cnnl_a_desc,
                                 cnnl_b_desc, cnnl_c_desc, cnnl_d_desc, algo,
                                 workspace_size),
      CNNL_STATUS_SUCCESS,
      "[mluOpMatMul] Internal error accured in mluOpGetMatMulWorkspaceSize.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_d_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpMatMul(mluOpHandle_t handle, const bool is_trans_a, const bool is_trans_b,
            const void *alpha, const mluOpTensorDescriptor_t a_desc,
            const void *a, const mluOpTensorDescriptor_t b_desc, const void *b,
            const void *beta, const mluOpTensorDescriptor_t c_desc, void *c) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpMatMul] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpMatMul", handle != NULL);
  PARAM_CHECK("mluOpMatMul", alpha != NULL);
  PARAM_CHECK("mluOpMatMul", a_desc != NULL);
  PARAM_CHECK("mluOpMatMul", a != NULL);
  PARAM_CHECK("mluOpMatMul", b_desc != NULL);
  PARAM_CHECK("mluOpMatMul", b != NULL);
  PARAM_CHECK("mluOpMatMul", beta != NULL);
  PARAM_CHECK("mluOpMatMul", c_desc != NULL);
  PARAM_CHECK("mluOpMatMul", c != NULL);
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);
  CHECK_FUNC_RETURN(
      cnnlMatMul(cnnl_handle, is_trans_a, is_trans_b, alpha, cnnl_a_desc, a,
                 cnnl_b_desc, b, beta, cnnl_c_desc, c),
      CNNL_STATUS_SUCCESS,
      "[mluOpMatMul] Internal error accured in mluOpMatMul.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpMatMul_v2(
    mluOpHandle_t handle, mluOpMatMulDescriptor_t matmul_desc,
    mluOpMatMulAlgo_t algo, const void *alpha,
    const mluOpTensorDescriptor_t a_desc, const void *a,
    const mluOpTensorDescriptor_t b_desc, const void *b, const void *beta,
    const mluOpTensorDescriptor_t c_desc, void *c, void *workspace,
    size_t workspace_size, const mluOpTensorDescriptor_t d_desc, void *d) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpMatMul_v2] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpMatMul_v2", handle != NULL);
  PARAM_CHECK("mluOpMatMul_v2", matmul_desc != NULL);
  PARAM_CHECK("mluOpMatMul_v2", algo != NULL);
  PARAM_CHECK("mluOpMatMul_v2", alpha != NULL);
  PARAM_CHECK("mluOpMatMul_v2", a_desc != NULL);
  PARAM_CHECK("mluOpMatMul_v2", a != NULL);
  PARAM_CHECK("mluOpMatMul_v2", b_desc != NULL);
  PARAM_CHECK("mluOpMatMul_v2", b != NULL);
  PARAM_CHECK("mluOpMatMul_v2", beta != NULL);
  PARAM_CHECK("mluOpMatMul_v2", c_desc != NULL);
  PARAM_CHECK("mluOpMatMul_v2", c != NULL);
  if (workspace_size != 0) {
    PARAM_CHECK("mluOpMatMul", workspace != NULL);
  }
  PARAM_CHECK("mluOpMatMul", d_desc != NULL);
  PARAM_CHECK("mluOpMatMul", d != NULL);
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(d_desc, cnnl_d_desc);
  CHECK_FUNC_RETURN(
      cnnlMatMul_v2(cnnl_handle, matmul_desc, algo, alpha, cnnl_a_desc, a,
                    cnnl_b_desc, b, beta, cnnl_c_desc, c, workspace,
                    workspace_size, cnnl_d_desc, d),
      CNNL_STATUS_SUCCESS,
      "[mluOpMatMul_v2] Internal error accured in mluOpMatMul_v2.",
      MLUOP_STATUS_INTERNAL_ERROR);

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_d_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
