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
#include <string>

#include "core/logging.h"
#include "core/tensor.h"
#include "core/context.h"
#include "core/mlu_env.h"
#include "core/gen_case.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"
#include "kernels/get_indice_pairs/get_indice_pairs_structs.h"
#include "kernels/get_indice_pairs/normal_get_indice_pairs.h"

static void getIndicePairsGencase(
    mluOpHandle_t handle,
    const mluOpSparseConvolutionDescriptor_t sparse_conv_desc,
    const mluOpTensorDescriptor_t indices_desc, const void *indices,
    const mluOpTensorDescriptor_t indice_pairs_desc, void *indice_pairs,
    const mluOpTensorDescriptor_t out_indices_desc, void *out_indices,
    const mluOpTensorDescriptor_t indice_num_desc, void *indice_num) {
  GEN_CASE_START("get_indice_pairs");
  GEN_CASE_HANDLE(handle);
  GEN_CASE_DATA_REAL(true, "indices", indices, indices_desc);
  GEN_CASE_DATA_REAL(false, "out_indices", out_indices, out_indices_desc);
  GEN_CASE_DATA_REAL(false, "indice_pairs", indice_pairs, indice_pairs_desc);
  GEN_CASE_DATA_REAL(false, "indice_num", indice_num, indice_num_desc);
  GEN_CASE_OP_PARAM_SINGLE(0, "get_indice_pairs", "dimnb",
                           sparse_conv_desc->dimNb);
  GEN_CASE_OP_PARAM_SINGLE(0, "get_indice_pairs", "batch",
                           sparse_conv_desc->batch);
  GEN_CASE_OP_PARAM_ARRAY(1, "get_indice_pairs", "pad", sparse_conv_desc->pad,
                          sparse_conv_desc->dimNb == 4 ? 2 : 3);
  GEN_CASE_OP_PARAM_ARRAY(1, "get_indice_pairs", "stride",
                          sparse_conv_desc->stride,
                          sparse_conv_desc->dimNb == 4 ? 2 : 3);
  GEN_CASE_OP_PARAM_ARRAY(1, "get_indice_pairs", "dilation",
                          sparse_conv_desc->dilation,
                          sparse_conv_desc->dimNb == 4 ? 2 : 3);
  GEN_CASE_OP_PARAM_ARRAY(1, "get_indice_pairs", "input_space",
                          sparse_conv_desc->input_space,
                          sparse_conv_desc->dimNb == 4 ? 2 : 3);
  GEN_CASE_OP_PARAM_ARRAY(1, "get_indice_pairs", "filter_space",
                          sparse_conv_desc->filter_space,
                          sparse_conv_desc->dimNb == 4 ? 2 : 3);
  GEN_CASE_OP_PARAM_ARRAY(1, "get_indice_pairs", "output_space",
                          sparse_conv_desc->output_space,
                          sparse_conv_desc->dimNb == 4 ? 2 : 3);
  GEN_CASE_OP_PARAM_SINGLE(2, "get_indice_pairs", "sub_m",
                           sparse_conv_desc->sub_m);
  GEN_CASE_OP_PARAM_SINGLE(2, "get_indice_pairs", "transpose",
                           sparse_conv_desc->transpose);
  GEN_CASE_OP_PARAM_SINGLE(2, "get_indice_pairs", "inverse",
                           sparse_conv_desc->inverse);
  GEN_CASE_HANDLE_PARAM();
  GEN_CASE_TEST_PARAM_NEW(false, false, true, 0.003, 0.003, 0);
}

static mluOpStatus_t internalGetIndicePairs(
    mluOpHandle_t handle, const std::string interface_name,
    mluOpSparseConvolutionDescriptor_t sparse_conv_desc,
    const mluOpTensorDescriptor_t indices_desc, const void *indices,
    void *workspace, size_t workspace_size,
    const mluOpTensorDescriptor_t indice_pairs_desc, void *indice_pairs,
    const mluOpTensorDescriptor_t out_indices_desc, void *out_indices,
    const mluOpTensorDescriptor_t indice_num_desc, void *indice_num,
    const bool is_get_workspace, size_t *return_ws) {
  PARAM_CHECK(interface_name, handle != NULL);
  PARAM_CHECK(interface_name, sparse_conv_desc != NULL);
  PARAM_CHECK(interface_name, indices_desc != NULL);
  PARAM_CHECK(interface_name, indice_pairs_desc != NULL);
  PARAM_CHECK(interface_name, out_indices_desc != NULL);
  PARAM_CHECK(interface_name, indice_num_desc != NULL);

  // check platform
  if (handle->arch < 372) {
    LOG(ERROR) << interface_name
               << " Only mlu300 and above devices are supported."
               << " Please check the device version!";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  // sparse_conv_desc dimNb  check
  int sparse_conv_dimNb = sparse_conv_desc->dimNb;

  // indices  indice_pairs out_indices indice_num
  // tensor dim check
  PARAM_CHECK(interface_name, indices_desc->dim == 2);
  PARAM_CHECK(interface_name, indice_pairs_desc->dim == 3);
  PARAM_CHECK(interface_name, out_indices_desc->dim == 2);
  PARAM_CHECK(interface_name, indice_num_desc->dim == 1);
  PARAM_CHECK(interface_name, indices_desc->dims[1] == 4);
  PARAM_CHECK(interface_name, out_indices_desc->dims[1] == 4);
  PARAM_CHECK(interface_name, indice_pairs_desc->dims[1] == 2);

  // check shape
  PARAM_CHECK(interface_name,
              indice_pairs_desc->dims[2] == indices_desc->dims[0]);
  PARAM_CHECK(interface_name,
              indice_pairs_desc->dims[0] == indice_num_desc->dims[0]);
  int kernel_volume = 1;
  for (int i = 0; i < sparse_conv_dimNb - 2; i++) {
    kernel_volume *= sparse_conv_desc->filter_space[i];
  }
  int output_spaces = sparse_conv_desc->batch;
  int input_spaces = sparse_conv_desc->batch;
  for (int i = 0; i < sparse_conv_dimNb - 2; i++) {
    output_spaces *= sparse_conv_desc->output_space[i];
    input_spaces *= sparse_conv_desc->input_space[i];
  }
  PARAM_CHECK_LE(interface_name, indices_desc->dims[0], input_spaces);
  for (int i = 0; i < sparse_conv_dimNb - 2; i++) {
    PARAM_CHECK_GE(interface_name, sparse_conv_desc->pad[i], 0);
    PARAM_CHECK_GE(interface_name, sparse_conv_desc->dilation[i], 1);
    PARAM_CHECK_GE(interface_name, sparse_conv_desc->stride[i], 1);
    if (sparse_conv_desc->dilation[i] != 1 &&
        sparse_conv_desc->stride[i] != 1) {
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  PARAM_CHECK(interface_name, indice_pairs_desc->dims[0] == kernel_volume);
  PARAM_CHECK_LE(interface_name, kernel_volume, 4096);
  PARAM_CHECK_LE(interface_name, out_indices_desc->dims[0], output_spaces);

  // large tensor
  if (mluOpGetTensorElementNum(indices_desc) >= LARGE_TENSOR_NUM ||
      mluOpGetTensorElementNum(out_indices_desc) >= LARGE_TENSOR_NUM ||
      mluOpGetTensorElementNum(indice_pairs_desc) >= LARGE_TENSOR_NUM ||
      mluOpGetTensorElementNum(indice_num_desc) >= LARGE_TENSOR_NUM) {
    LOG(ERROR) << interface_name << " Overflow max tensor num."
               << " Currently, MLU-OPS supports tensor num smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // tensor  datatype check
  PARAM_CHECK_EQ(interface_name, indices_desc->dtype, MLUOP_DTYPE_INT32);
  PARAM_CHECK_EQ(interface_name, indice_pairs_desc->dtype, MLUOP_DTYPE_INT32);
  PARAM_CHECK_EQ(interface_name, out_indices_desc->dtype, MLUOP_DTYPE_INT32);
  PARAM_CHECK_EQ(interface_name, indice_num_desc->dtype, MLUOP_DTYPE_INT32);
  // special check
  int sub_m = sparse_conv_desc->sub_m;
  if (sub_m) {
    for (int i = 0; i < sparse_conv_dimNb - 2; i++) {
      PARAM_CHECK_EQ(interface_name, sparse_conv_desc->input_space[i],
                     sparse_conv_desc->output_space[i]);
      PARAM_CHECK_EQ(interface_name, sparse_conv_desc->stride[i], 1);
      PARAM_CHECK_EQ(interface_name, sparse_conv_desc->dilation[i], 1);
    }
  }

  // check zero elment
  if (mluOpGetTensorElementNum(indices_desc) == 0 ||
      mluOpGetTensorElementNum(indice_pairs_desc) == 0 ||
      mluOpGetTensorElementNum(out_indices_desc) == 0 ||
      mluOpGetTensorElementNum(indice_num_desc) == 0) {
    sparse_conv_desc->num_act_out = 0;
    return MLUOP_STATUS_SUCCESS;
  }

  // check nullptr
  if (!is_get_workspace) {
    PARAM_CHECK(interface_name, indices != NULL);
    PARAM_CHECK(interface_name, indice_pairs != NULL);
    PARAM_CHECK(interface_name, out_indices != NULL);
    PARAM_CHECK(interface_name, indice_num != NULL);
    if (workspace_size != 0) {
      PARAM_CHECK(interface_name, workspace != NULL);
    }
  }
  // gencase
  if (!is_get_workspace && MLUOP_GEN_CASE_ON_NEW) {
    getIndicePairsGencase(handle, sparse_conv_desc, indices_desc, indices,
                          indice_pairs_desc, indice_pairs, out_indices_desc,
                          out_indices, indice_num_desc, indice_num);
  }

  // call normal implementaion
  mluOpStatus_t return_status;
  return_status = normalGetIndicePairs(
      handle, interface_name, sparse_conv_desc, indices_desc, indices,
      workspace, workspace_size, indice_pairs_desc, indice_pairs,
      out_indices_desc, out_indices, indice_num_desc, indice_num,
      is_get_workspace, return_ws);

  if (!is_get_workspace) {
    GEN_CASE_END();
  }
  return return_status;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetIndicePairs(
    mluOpHandle_t handle, mluOpSparseConvolutionDescriptor_t sparse_conv_desc,
    const mluOpTensorDescriptor_t indices_desc, const void *indices,
    void *workspace, const size_t workspace_size,
    const mluOpTensorDescriptor_t indice_pairs_desc, void *indice_pairs,
    const mluOpTensorDescriptor_t out_indices_desc, void *out_indices,
    const mluOpTensorDescriptor_t indice_num_desc, void *indice_num) {
  std::string interface_name = "[mluOpGetIndicesPairs]";
  return internalGetIndicePairs(
      handle, interface_name, sparse_conv_desc, indices_desc, indices,
      workspace, workspace_size, indice_pairs_desc, indice_pairs,
      out_indices_desc, out_indices, indice_num_desc, indice_num, false, NULL);
}

mluOpStatus_t MLUOP_WIN_API mluOpGetIndicePairsWorkspaceSize(
    mluOpHandle_t handle, mluOpSparseConvolutionDescriptor_t sparse_conv_desc,
    const mluOpTensorDescriptor_t indices_desc,
    const mluOpTensorDescriptor_t indice_pairs_desc,
    const mluOpTensorDescriptor_t out_indices_desc,
    const mluOpTensorDescriptor_t indice_num_desc, size_t *workspace_size) {
  std::string interface_name = "[mluOpGetIndicePairsWorkspaceSize]";
  PARAM_CHECK(interface_name, handle != NULL);
  PARAM_CHECK(interface_name, sparse_conv_desc != NULL);
  PARAM_CHECK(interface_name, indices_desc != NULL);
  PARAM_CHECK(interface_name, indice_pairs_desc != NULL);
  PARAM_CHECK(interface_name, out_indices_desc != NULL);
  PARAM_CHECK(interface_name, indice_num_desc != NULL);
  PARAM_CHECK(interface_name, workspace_size != NULL);
  if (mluOpGetTensorElementNum(indices_desc) == 0 ||
      mluOpGetTensorElementNum(indice_pairs_desc) == 0 ||
      mluOpGetTensorElementNum(out_indices_desc) == 0 ||
      mluOpGetTensorElementNum(indice_num_desc) == 0) {
    workspace_size[0] = 0;
    return MLUOP_STATUS_SUCCESS;
  }

  return internalGetIndicePairs(handle, interface_name, sparse_conv_desc,
                                indices_desc, NULL, NULL, 0, indice_pairs_desc,
                                NULL, out_indices_desc, NULL, indice_num_desc,
                                NULL, true, workspace_size);
}
