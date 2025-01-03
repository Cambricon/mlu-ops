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
#include "three_nn_forward.h"

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"
#include "kernels/utils/cnnl_helper.h"

mluOpStatus_t MLUOP_WIN_API mluOpGetThreeNNForwardWorkspaceSize(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t known_desc,
    size_t *workspace_size) {
  // handle and desc ptr check null
  PARAM_CHECK("[mluOpThreeNNForwardWorkspace]", handle != NULL);
  PARAM_CHECK("[mluOpThreeNNForwardWorkspace]", known_desc != NULL);
  PARAM_CHECK("[mluOpThreeNNForwardWorkspace]", workspace_size != NULL);

  // check tensor dim
  PARAM_CHECK("[mluOpThreeNNForwardWorkspace]", known_desc->getDim() == 3);

  *workspace_size = known_desc->getTotalTensorSize();
  const int known_dim = known_desc->getDim();
  const int known_permute[3] = {0, 2, 1};
  size_t known_transpose_workspace_size = 0;

  cnnlTransposeDescriptor_t _trans_desc = NULL;
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(known_desc, _known_desc);
  CALL_CNNL(cnnlCreateTransposeDescriptor(&_trans_desc));
  CALL_CNNL(cnnlSetTransposeDescriptor(_trans_desc, known_dim, known_permute));
  CALL_CNNL(cnnlGetTransposeWorkspaceSize(_handle, _known_desc, _trans_desc,
                                          &known_transpose_workspace_size));
  *workspace_size += known_transpose_workspace_size;
  CALL_CNNL(cnnlDestroyTransposeDescriptor(_trans_desc));
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_known_desc);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t transposeTensor(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, const int *permute,
    const mluOpTensorDescriptor_t workspace_dst_desc, void *workspace_dst,
    void *transpose_workspace, size_t transpose_workspace_size) {
  const int input_dim = input_desc->getDim();

  cnnlTransposeDescriptor_t cnnl_trans_desc = NULL;
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(workspace_dst_desc,
                                               cnnl_workspace_dst_desc);
  CALL_CNNL(cnnlCreateTransposeDescriptor(&cnnl_trans_desc));
  CALL_CNNL(cnnlSetTransposeDescriptor(cnnl_trans_desc, input_dim, permute));
  CALL_CNNL(cnnlTranspose_v2(cnnl_handle, cnnl_trans_desc, cnnl_input_desc,
                             input, cnnl_workspace_dst_desc, workspace_dst,
                             transpose_workspace, transpose_workspace_size));
  CALL_CNNL(cnnlDestroyTransposeDescriptor(cnnl_trans_desc));
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_workspace_dst_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t threeNNParamCheck(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t unknown_desc,
    const void *unknown, const mluOpTensorDescriptor_t known_desc,
    const void *known, void *workspace, const size_t workspace_size,
    const mluOpTensorDescriptor_t dist2_desc, void *dist2,
    const mluOpTensorDescriptor_t idx_desc, void *idx) {
  // handle and desc ptr check null
  PARAM_CHECK("[mluOpThreeNNForward]", handle != NULL);
  PARAM_CHECK("[mluOpThreeNNForward]", known_desc != NULL);
  PARAM_CHECK("[mluOpThreeNNForward]", unknown_desc != NULL);
  PARAM_CHECK("[mluOpThreeNNForward]", dist2_desc != NULL);
  PARAM_CHECK("[mluOpThreeNNForward]", idx_desc != NULL);

  // check tensor dim
  PARAM_CHECK("[mluOpThreeNNForward]", unknown_desc->getDim() == 3);
  PARAM_CHECK("[mluOpThreeNNForward]", known_desc->getDim() == 3);
  PARAM_CHECK("[mluOpThreeNNForward]", dist2_desc->getDim() == 3);
  PARAM_CHECK("[mluOpThreeNNForward]", idx_desc->getDim() == 3);

  // check dim0
  PARAM_CHECK("[mluOpThreeNNForward]",
              unknown_desc->getDimIndex(0) == known_desc->getDimIndex(0));
  PARAM_CHECK("[mluOpThreeNNForward]",
              unknown_desc->getDimIndex(0) == dist2_desc->getDimIndex(0));
  PARAM_CHECK("[mluOpThreeNNForward]",
              unknown_desc->getDimIndex(0) == idx_desc->getDimIndex(0));

  // check dim1
  PARAM_CHECK("[mluOpThreeNNForward]",
              unknown_desc->getDimIndex(1) == dist2_desc->getDimIndex(1));
  PARAM_CHECK("[mluOpThreeNNForward]",
              unknown_desc->getDimIndex(1) == idx_desc->getDimIndex(1));

  // check dim2
  PARAM_CHECK("[mluOpThreeNNForward]", unknown_desc->getDimIndex(2) == 3);
  PARAM_CHECK("[mluOpThreeNNForward]", known_desc->getDimIndex(2) == 3);
  PARAM_CHECK("[mluOpThreeNNForward]", dist2_desc->getDimIndex(2) == 3);
  PARAM_CHECK("[mluOpThreeNNForward]", idx_desc->getDimIndex(2) == 3);

  // check tensor datatype，support float16 and float32
  PARAM_CHECK_V2("[mluOpThreeNNForward]",
                 (unknown_desc->getDtype() == MLUOP_DTYPE_HALF) ||
                     (unknown_desc->getDtype() == MLUOP_DTYPE_FLOAT),
                 "Only half and float are supported in input unknown tensor, "
                 "but the data type of tensor is "
                     << mluOpGetNameOfDataType(unknown_desc->getDtype())
                     << ".");
  PARAM_CHECK("[mluOpThreeNNForward]",
              unknown_desc->getDtype() == known_desc->getDtype());
  PARAM_CHECK("[mluOpThreeNNForward]",
              unknown_desc->getDtype() == dist2_desc->getDtype());

  PARAM_CHECK_V2(
      "[mluOpThreeNNForward]", (idx_desc->getDtype() == MLUOP_DTYPE_INT32),
      "Only int32 are supported in output idx, but the data type of tensor is "
          << mluOpGetNameOfDataType(idx_desc->getDtype()) << ".");

  const size_t unknown_element_num = mluOpGetTensorElementNum(unknown_desc);
  const size_t known_element_num = mluOpGetTensorElementNum(known_desc);
  const size_t dist2_element_num = mluOpGetTensorElementNum(dist2_desc);
  const size_t idx_element_num = mluOpGetTensorElementNum(idx_desc);

  // check stride
  STRIDE_TENSOR_CHECK("[mluOpThreeNNForward]:", unknown_desc,
                      "unknown_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpThreeNNForward]:", known_desc,
                      "known_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpThreeNNForward]:", dist2_desc,
                      "dist2_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpThreeNNForward]:", idx_desc,
                      "idx_desc must be contiguous");

  // check large tensor
  TENSOR_NUM_CHECK("[mluOpThreeNNForward]", unknown_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpThreeNNForward]", known_element_num, LARGE_TENSOR_NUM,
                   "");
  TENSOR_NUM_CHECK("[mluOpThreeNNForward]", dist2_element_num, LARGE_TENSOR_NUM,
                   "");
  TENSOR_NUM_CHECK("[mluOpThreeNNForward]", idx_element_num, LARGE_TENSOR_NUM,
                   "");

  // check element num zero
  if (unknown_element_num == 0 || dist2_element_num == 0 ||
      idx_element_num == 0) {
    VLOG(5) << "[mluOpThreeNNForward] Zero element tensor failure.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (known_element_num == 0) {
    VLOG(5) << "[mluOpThreeNNForward] Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  // check workspace ptr
  if (workspace_size > 0) {
    PARAM_CHECK("[mluOpThreeNNForward]", workspace != NULL);
  }

  // input and output ptr check null
  PARAM_CHECK("[mluOpThreeNNForward]", unknown != NULL);
  PARAM_CHECK("[mluOpThreeNNForward]", known != NULL);
  PARAM_CHECK("[mluOpThreeNNForward]", dist2 != NULL);
  PARAM_CHECK("[mluOpThreeNNForward]", idx != NULL);

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpThreeNNForward(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t unknown_desc,
    const void *unknown, const mluOpTensorDescriptor_t known_desc,
    const void *known, void *workspace, const size_t workspace_size,
    const mluOpTensorDescriptor_t dist2_desc, void *dist2,
    const mluOpTensorDescriptor_t idx_desc, void *idx) {
  // params check
  mluOpStatus_t status_paramcheck = threeNNParamCheck(
      handle, unknown_desc, unknown, known_desc, known, workspace,
      workspace_size, dist2_desc, dist2, idx_desc, idx);
  if (status_paramcheck != MLUOP_STATUS_SUCCESS) {
    return status_paramcheck;
  }

  const int b = unknown_desc->getDimIndex(0);
  const int n = unknown_desc->getDimIndex(1);
  const int m = known_desc->getDimIndex(1);

  // generate mluOpThreeNNForward prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("three_nn_forward", "THREE_NN_FORWARD");
    // set handle dump mlu output
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "unknown", unknown, unknown_desc, 100, -100);
    GEN_CASE_DATA(true, "known", known, known_desc, 100, -100);
    GEN_CASE_DATA(false, "dist2", dist2, dist2_desc, 0, 0);
    GEN_CASE_DATA(false, "idx", idx, idx_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  mluOpDataType_t input_dtype = unknown_desc->getDtype();
  void *known_workspace = workspace;
  void *transpose_workspace =
      (int8_t *)known_workspace + known_desc->getTotalTensorSize();

  // start U1 task, occupy all available clusters
  cnrtDim3_t k_dims;
  k_dims.x = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  k_dims.y = mluop::runtime::getClusterLimitCapability(handle);
  k_dims.z = 1;
  cnrtFunctionType_t k_type = cnrtFuncTypeUnion1;

  VLOG(5) << "[mluOpThreeNNForward] cnnlTranspose_v2 feature start.";

  const int known_dim = known_desc->getDim();
  const int known_permute[3] = {0, 2, 1};
  int known_tmp_dims[3] = {0, 0, 0};

  for (int i = 0; i < known_dim; ++i) {
    known_tmp_dims[i] = known_desc->getDimIndex(known_permute[i]);
  }

  mluOpTensorDescriptor_t known_desc_tmp = NULL;
  CHECK_RETURN("[mluOpThreeNNForward]",
               mluOpCreateTensorDescriptor(&known_desc_tmp));
  CHECK_RETURN(
      "[mluOpThreeNNForward]",
      mluOpSetTensorDescriptor(known_desc_tmp, MLUOP_LAYOUT_ARRAY, input_dtype,
                               known_dim, known_tmp_dims));
  CHECK_RETURN(
      "[mluOpThreeNNForward]",
      transposeTensor(handle, known_desc, known, known_permute, known_desc_tmp,
                      known_workspace, transpose_workspace,
                      workspace_size - known_desc->getTotalTensorSize()));
  CHECK_RETURN("[mluOpThreeNNForward]",
               mluOpDestroyTensorDescriptor(known_desc_tmp));

  VLOG(5) << "[mluOpThreeNNForward] cnnlTranspose_v2 feature end.";
  VLOG(5) << "Launch Kernel KernelThreeNNForward<<<Union" << k_type / CORE_DIM
          << ", " << k_dims.x << ", " << k_dims.y << ", " << k_dims.z << ">>>.";
  CHECK_RETURN(
      "[mluOpThreeNNForward]",
      KernelThreeNNForward(k_dims, k_type, handle->queue, input_dtype, b, n, m,
                           unknown, known_workspace, dist2, idx));

  // generate gen_case prototxt
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
