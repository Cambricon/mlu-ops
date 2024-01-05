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

mluOpStatus_t MLUOP_WIN_API mluOpGetBatchMatMulBCastWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t a_desc,
    const mluOpTensorDescriptor_t b_desc, const mluOpTensorDescriptor_t c_desc,
    size_t *workspace_size) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpGetBatchMatMulBCastWorkspaceSize] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpBatchMatMulBCast", handle != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", a_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", b_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", c_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", workspace_size != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);

  CHECK_FUNC_RETURN(
      cnnlGetBatchMatMulBCastWorkspaceSize(
          cnnl_handle, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc, workspace_size),
      CNNL_STATUS_SUCCESS,
      "[mluOpBatchMatMulBCast] Internal error accured in "
      "mluOpGetBatchMatMulBCastWorkspaceSize.",
      MLUOP_STATUS_INTERNAL_ERROR);

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetBatchMatMulHeuristicResult(
    mluOpMatMulHeuristicResult_t result, mluOpMatMulAlgo_t algo,
    size_t *workspace_size) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpGetBatchMatMulHeuristicResult] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpBatchMatMulBCast", result != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", algo != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", workspace_size != NULL);

  CHECK_FUNC_RETURN(
      cnnlGetBatchMatMulHeuristicResult(result, algo, workspace_size),
      CNNL_STATUS_SUCCESS,
      "[mluOpBatchMatMulBCast] Internal error accured in "
      "mluOpGetBatchMatMulHeuristicResult.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetBatchMatMulAlgoHeuristic(
    mluOpHandle_t handle, mluOpMatMulDescriptor_t bmm_bcast_desc,
    mluOpTensorDescriptor_t a_desc, mluOpTensorDescriptor_t b_desc,
    mluOpTensorDescriptor_t c_desc, mluOpMatMulPrefer_t preference,
    int requested_algo_count, mluOpMatMulHeuristicResult_t *result_array,
    int *return_algo_count) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpGetBatchMatMulAlgoHeuristic] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpBatchMatMulBCast", handle != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", bmm_bcast_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", a_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", b_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", c_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", result_array != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", return_algo_count != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);

  CHECK_FUNC_RETURN(cnnlGetBatchMatMulAlgoHeuristic(
                        cnnl_handle, bmm_bcast_desc, cnnl_a_desc, cnnl_b_desc,
                        cnnl_c_desc, cnnlMatMulPrefer_t(preference),
                        requested_algo_count, result_array, return_algo_count),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpBatchMatMulBCast] Internal error accured in "
                    "mluOpGetBatchMatMulAlgoHeuristic.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpBatchMatMulBCastDescCreate(
    mluOpBatchMatMulBCastDescriptor_t *bmm_bcast_desc) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpBatchMatMulBCastDescCreate] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpBatchMatMulBCast", bmm_bcast_desc != NULL);
  CHECK_FUNC_RETURN(cnnlBatchMatMulBCastDescCreate(bmm_bcast_desc),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpBatchMatMulBCast] Internal error accured in "
                    "mluOpBatchMatMulBCastDescCreate.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpBatchMatMulBCastDescDestroy(
    mluOpBatchMatMulBCastDescriptor_t bmm_bcast_desc) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpBatchMatMulBCastDescDestroy] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpBatchMatMulBCast", bmm_bcast_desc != NULL);
  CHECK_FUNC_RETURN(cnnlBatchMatMulBCastDescDestroy(bmm_bcast_desc),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpBatchMatMulBCast] Internal error accured in "
                    "mluOpBatchMatMulBCastDescDestroy.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetBatchMatMulBCastDescAttr(
    mluOpBatchMatMulBCastDescriptor_t bmm_bcast_desc,
    mluOpBatchMatMulBCastDescAttribute_t attr, const void *buf,
    size_t size_in_bytes) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpSetBatchMatMulBCastDescAttr] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpBatchMatMulBCast", bmm_bcast_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", buf != NULL);
  CHECK_FUNC_RETURN(
      cnnlSetBatchMatMulBCastDescAttr(bmm_bcast_desc,
                                      cnnlBatchMatMulBCastDescAttribute_t(attr),
                                      buf, size_in_bytes),
      CNNL_STATUS_SUCCESS,
      "[mluOpBatchMatMulBCast] Internal error accured in "
      "mluOpSetBatchMatMulBCastDescAttr.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetBatchMatMulBCastDescAttr(
    const mluOpBatchMatMulBCastDescriptor_t bmm_bcast_desc,
    mluOpBatchMatMulBCastDescAttribute_t attr, void *buf, size_t size_in_bytes,
    size_t *size_written) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpGetBatchMatMulBCastDescAttr] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpBatchMatMulBCast", bmm_bcast_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", buf != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", size_written != NULL);
  CHECK_FUNC_RETURN(
      cnnlGetBatchMatMulBCastDescAttr(bmm_bcast_desc,
                                      cnnlBatchMatMulBCastDescAttribute_t(attr),
                                      buf, size_in_bytes, size_written),
      CNNL_STATUS_SUCCESS,
      "[mluOpBatchMatMulBCast] Internal error accured in "
      "mluOpGetBatchMatMulBCastDescAttr.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpBatchMatMulBCastAlgoCreate(mluOpBatchMatMulBCastAlgo_t *algo) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpBatchMatMulBCastAlgoCreate] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpBatchMatMulBCast", algo != NULL);
  CHECK_FUNC_RETURN(cnnlBatchMatMulBCastAlgoCreate(algo), CNNL_STATUS_SUCCESS,
                    "[mluOpBatchMatMulBCast] Internal error accured in "
                    "mluOpBatchMatMulBCastAlgoCreate.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpBatchMatMulBCastAlgoDestroy(mluOpBatchMatMulBCastAlgo_t algo) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpBatchMatMulBCastAlgoDestroy] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpBatchMatMulBCast", algo != NULL);
  CHECK_FUNC_RETURN(cnnlBatchMatMulBCastAlgoDestroy(algo), CNNL_STATUS_SUCCESS,
                    "[mluOpBatchMatMulBCast] Internal error accured in "
                    "mluOpBatchMatMulBCastAlgoDestroy.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetQuantizeBatchMatMulBCastAlgorithm(
    mluOpHandle_t handle,
    const mluOpBatchMatMulBCastDescriptor_t bmm_bcast_desc,
    const mluOpTensorDescriptor_t a_desc, const mluOpTensorDescriptor_t b_desc,
    const mluOpTensorDescriptor_t c_desc,
    mluOpBatchMatMulBCastPreference_t preference,
    mluOpBatchMatMulBCastAlgo_t *algo) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpGetQuantizeBatchMatMulBCastAlgorithm] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpBatchMatMulBCast", handle != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", bmm_bcast_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", a_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", b_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", c_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", algo != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);

  CHECK_FUNC_RETURN(
      cnnlGetQuantizeBatchMatMulBCastAlgorithm(
          cnnl_handle, bmm_bcast_desc, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc,
          cnnlBatchMatMulBCastPreference_t(preference), algo),
      CNNL_STATUS_SUCCESS,
      "[mluOpBatchMatMulBCast] Internal error accured in "
      "mluOpGetQuantizeBatchMatMulBCastAlgorithm.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetQuantizeBatchMatMulBCastWorkspaceSize(
    mluOpHandle_t handle, mluOpBatchMatMulBCastDescriptor_t bmm_bcast_desc,
    const mluOpTensorDescriptor_t a_desc, const mluOpTensorDescriptor_t b_desc,
    const mluOpTensorDescriptor_t c_desc, mluOpBatchMatMulBCastAlgo_t algo,
    size_t *workspace_size) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpGetQuantizeBatchMatMulBCastWorkspaceSize] is deprecated and"
      << " will be removed in furture.";
  PARAM_CHECK("mluOpBatchMatMulBCast", handle != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", bmm_bcast_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", a_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", b_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", c_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", algo != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", workspace_size != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);

  CHECK_FUNC_RETURN(cnnlGetQuantizeBatchMatMulBCastWorkspaceSize(
                        cnnl_handle, bmm_bcast_desc, cnnl_a_desc, cnnl_b_desc,
                        cnnl_c_desc, algo, workspace_size),
                    CNNL_STATUS_SUCCESS,
                    "[mluOpBatchMatMulBCast] Internal error accured in "
                    "mluOpGetQuantizeBatchMatMulBCastWorkspaceSize.",
                    MLUOP_STATUS_INTERNAL_ERROR);

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpQuantizeBatchMatMulBCast(
    mluOpHandle_t handle,
    const mluOpBatchMatMulBCastDescriptor_t bmm_bcast_desc, const void *alpha,
    const mluOpTensorDescriptor_t a_desc, const void *a, const void *a_position,
    const void *a_scale, const void *a_offset,
    const mluOpTensorDescriptor_t b_desc, const void *b, const void *b_position,
    const void *b_scale, const void *b_offset, const void *beta,
    const mluOpTensorDescriptor_t c_desc, void *c,
    mluOpBatchMatMulBCastAlgo_t algo, void *workspace, size_t workspace_size) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpQuantizeBatchMatMulBCast] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpBatchMatMulBCast", handle != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", bmm_bcast_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", alpha != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", a_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", a != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", a_position != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", a_scale != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", a_offset != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", b_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", b != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", b_position != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", b_scale != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", b_offset != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", beta != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", c_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", c != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", algo != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", workspace != NULL);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);
  CHECK_FUNC_RETURN(
      cnnlQuantizeBatchMatMulBCast(
          cnnl_handle, bmm_bcast_desc, alpha, cnnl_a_desc, a, a_position,
          a_scale, a_offset, cnnl_b_desc, b, b_position, b_scale, b_offset,
          beta, cnnl_c_desc, c, algo, workspace, workspace_size),
      CNNL_STATUS_SUCCESS,
      "[mluOpBatchMatMulBCast] Internal error accured in "
      "mluOpQuantizeBatchMatMulBCast.",
      MLUOP_STATUS_INTERNAL_ERROR);

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpBatchMatMulBCast(
    mluOpHandle_t handle, const bool is_transa, const bool is_transb,
    const mluOpTensorDescriptor_t a_desc, const void *a,
    const mluOpTensorDescriptor_t b_desc, const void *b, void *workspace,
    size_t workspace_size, const mluOpTensorDescriptor_t c_desc, void *c) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpBatchMatMulBCast] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpBatchMatMulBCast", handle != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", a_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", a != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", b_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", b != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", c_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCast", c != NULL);
  if (workspace_size != 0) {
    PARAM_CHECK("mluOpBatchMatMulBCast", workspace != NULL);
  }

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);

  CHECK_FUNC_RETURN(
      cnnlBatchMatMulBCast(cnnl_handle, is_transa, is_transb, cnnl_a_desc, a,
                           cnnl_b_desc, b, workspace, workspace_size,
                           cnnl_c_desc, c),
      CNNL_STATUS_SUCCESS,
      "[mluOpBatchMatMulBCast] Internal error accured in "
      "mluOpBatchMatMulBCast.",
      MLUOP_STATUS_INTERNAL_ERROR);

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpBatchMatMulBCast_v2(
    mluOpHandle_t handle, mluOpMatMulDescriptor_t bmm_bcast_desc,
    mluOpMatMulAlgo_t algo, const void *alpha,
    const mluOpTensorDescriptor_t a_desc, const void *a,
    const mluOpTensorDescriptor_t b_desc, const void *b, const void *beta,
    const mluOpTensorDescriptor_t c_desc, void *c, void *workspace,
    size_t workspace_size) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpBatchMatMulBCast_v2] is deprecated and"
                          << " will be removed in furture.";
  PARAM_CHECK("mluOpBatchMatMulBCastV2", handle != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCastV2", bmm_bcast_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCastV2", algo != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCastV2", alpha != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCastV2", a_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCastV2", a != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCastV2", b_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCastV2", b != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCastV2", beta != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCastV2", c_desc != NULL);
  PARAM_CHECK("mluOpBatchMatMulBCastV2", c != NULL);
  if (workspace_size != 0) {
    PARAM_CHECK("mluOpBatchMatMulBCast", workspace != NULL);
  }

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);

  CHECK_FUNC_RETURN(
      cnnlBatchMatMulBCast_v2(cnnl_handle, bmm_bcast_desc, algo, alpha,
                              cnnl_a_desc, a, cnnl_b_desc, b, beta, cnnl_c_desc,
                              c, workspace, workspace_size),
      CNNL_STATUS_SUCCESS,
      "[mluOpBatchMatMulBCastV2] Internal error accured in "
      "mluOpBatchMatMulBCastV2.",
      MLUOP_STATUS_INTERNAL_ERROR);

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}
