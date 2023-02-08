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
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/debug.h"
#include "kernels/kernel.h"
#include "mlu_op.h"
#include "mlu_op_kernel.h"

static mluOpStatus_t MLUOP_WIN_API initTransposeDescriptor(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    mluOpTransposeDescriptor_t *trans_desc, const int dim, const int *permute,
    size_t *workspace_size) {
  PARAM_CHECK(
      "[mluOpThreeNNForward]",
      MLUOP_STATUS_SUCCESS == mluOpCreateTransposeDescriptor(trans_desc));
  PARAM_CHECK("[mluOpThreeNNForward]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpSetTransposeDescriptor(*trans_desc, dim, permute));
  PARAM_CHECK("[mluOpThreeNNForward]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpGetTransposeWorkspaceSize(handle, input_desc,
                                                 *trans_desc, workspace_size));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetThreeNNForwardWorkspaceSize(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t known_desc,
    size_t *workspace_size) {
  // handle and desc ptr check null
  PARAM_CHECK("[mluOpThreeNNForwardWorkspace]", handle != NULL);
  PARAM_CHECK("[mluOpThreeNNForwardWorkspace]", known_desc != NULL);
  PARAM_CHECK("[mluOpThreeNNForwardWorkspace]", workspace_size != NULL);

  // check tensor dim
  PARAM_CHECK("[mluOpThreeNNForwardWorkspace]", known_desc->dim == 3);

  *workspace_size = known_desc->total_tensor_size;
  mluOpTransposeDescriptor_t trans_desc = NULL;
  const int known_dim = known_desc->dim;
  const int known_permute[3] = {0, 2, 1};
  size_t known_transpose_workspace_size = 0;

  PARAM_CHECK("[mluOpThreeNNForwardWorkspace]",
              MLUOP_STATUS_SUCCESS ==
                  initTransposeDescriptor(handle, known_desc, &trans_desc,
                                          known_dim, known_permute,
                                          &known_transpose_workspace_size));
  *workspace_size += known_transpose_workspace_size;

  PARAM_CHECK(
      "[mluOpThreeNNForwardWorkspace]",
      MLUOP_STATUS_SUCCESS == mluOpDestroyTransposeDescriptor(trans_desc));
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t transposeTensor(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, const int *permute,
    const mluOpTensorDescriptor_t workspace_dst_desc, void *workspace_dst,
    void *transpose_workspace) {
  const int input_dim = input_desc->dim;
  mluOpTransposeDescriptor_t trans_desc = NULL;
  size_t transpose_workspace_size = 0;
  PARAM_CHECK(
      "[mluOpThreeNNForward]",
      MLUOP_STATUS_SUCCESS ==
          initTransposeDescriptor(handle, input_desc, &trans_desc, input_dim,
                                  permute, &transpose_workspace_size));

  PARAM_CHECK(
      "[mluOpThreeNNForward]",
      MLUOP_STATUS_SUCCESS ==
          mluOpTranspose_v2(handle, trans_desc, input_desc, input,
                            workspace_dst_desc, workspace_dst,
                            transpose_workspace, transpose_workspace_size));

  PARAM_CHECK(
      "[mluOpThreeNNForward]",
      MLUOP_STATUS_SUCCESS == mluOpDestroyTransposeDescriptor(trans_desc));
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
  PARAM_CHECK("[mluOpThreeNNForward]", unknown_desc->dim == 3);
  PARAM_CHECK("[mluOpThreeNNForward]", known_desc->dim == 3);
  PARAM_CHECK("[mluOpThreeNNForward]", dist2_desc->dim == 3);
  PARAM_CHECK("[mluOpThreeNNForward]", idx_desc->dim == 3);

  // check dim0
  PARAM_CHECK("[mluOpThreeNNForward]",
              unknown_desc->dims[0] == known_desc->dims[0]);
  PARAM_CHECK("[mluOpThreeNNForward]",
              unknown_desc->dims[0] == dist2_desc->dims[0]);
  PARAM_CHECK("[mluOpThreeNNForward]",
              unknown_desc->dims[0] == idx_desc->dims[0]);

  // check dim1
  PARAM_CHECK("[mluOpThreeNNForward]",
              unknown_desc->dims[1] == dist2_desc->dims[1]);
  PARAM_CHECK("[mluOpThreeNNForward]",
              unknown_desc->dims[1] == idx_desc->dims[1]);

  // check dim2
  PARAM_CHECK("[mluOpThreeNNForward]", unknown_desc->dims[2] == 3);
  PARAM_CHECK("[mluOpThreeNNForward]", known_desc->dims[2] == 3);
  PARAM_CHECK("[mluOpThreeNNForward]", dist2_desc->dims[2] == 3);
  PARAM_CHECK("[mluOpThreeNNForward]", idx_desc->dims[2] == 3);

  // check tensor datatype，support float16 and float32
  PARAM_CHECK_V2("[mluOpThreeNNForward]",
                 (unknown_desc->dtype == MLUOP_DTYPE_HALF) ||
                     (unknown_desc->dtype == MLUOP_DTYPE_FLOAT),
                 "Only half and float are supported in input unknown tensor, "
                 "but the data type of tensor is "
                     << mluop::getNameOfDataType(unknown_desc->dtype) << ".");
  PARAM_CHECK("[mluOpThreeNNForward]",
              unknown_desc->dtype == known_desc->dtype);
  PARAM_CHECK("[mluOpThreeNNForward]",
              unknown_desc->dtype == dist2_desc->dtype);

  PARAM_CHECK_V2(
      "[mluOpThreeNNForward]", (idx_desc->dtype == MLUOP_DTYPE_INT32),
      "Only int32 are supported in output idx, but the data type of tensor is "
          << mluop::getNameOfDataType(idx_desc->dtype) << ".");

  const size_t unknown_element_num = mluOpGetTensorElementNum(unknown_desc);
  const size_t known_element_num = mluOpGetTensorElementNum(known_desc);
  const size_t dist2_element_num = mluOpGetTensorElementNum(dist2_desc);
  const size_t idx_element_num = mluOpGetTensorElementNum(idx_desc);

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

  const int b = unknown_desc->dims[0];
  const int n = unknown_desc->dims[1];
  const int m = known_desc->dims[1];

  // generate mluOpThreeNNForward prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("three_nn_forward");
    // set handle dump mlu output
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "unknown", unknown, unknown_desc, 100, -100);
    GEN_CASE_DATA(true, "known", known, known_desc, 100, -100);
    GEN_CASE_DATA(false, "dist2", dist2, dist2_desc, 0, 0);
    GEN_CASE_DATA(false, "idx", idx, idx_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  mluOpDataType_t input_dtype = unknown_desc->dtype;
  void *known_workspace = workspace;
  void *transpose_workspace =
      (char *)known_workspace + known_desc->total_tensor_size;

  // start U1 task, occupy all available clusters
  cnrtDim3_t k_dims;
  k_dims.x = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  k_dims.y = mluop::runtime::getClusterLimitCapability(handle);
  k_dims.z = 1;
  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;

  VLOG(5) << "[mluOpThreeNNForward] mluOpTranspose_v2 feature start.";

  const int known_dim = known_desc->dim;
  const int known_permute[3] = {0, 2, 1};
  int known_tmp_dims[3] = {0, 0, 0};

  for (int i = 0; i < known_dim; ++i) {
    known_tmp_dims[i] = known_desc->dims[known_permute[i]];
  }

  mluOpTensorDescriptor_t known_desc_tmp = NULL;
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&known_desc_tmp));
  PARAM_CHECK(
      "[mluOpThreeNNForward]",
      MLUOP_STATUS_SUCCESS ==
          mluOpSetTensorDescriptor(known_desc_tmp, MLUOP_LAYOUT_ARRAY,
                                   input_dtype, known_dim, known_tmp_dims));
  PARAM_CHECK("[mluOpThreeNNForward]",
              MLUOP_STATUS_SUCCESS ==
                  transposeTensor(handle, known_desc, known, known_permute,
                                  known_desc_tmp, known_workspace,
                                  transpose_workspace));
  PARAM_CHECK(
      "[mluOpThreeNNForward]",
      MLUOP_STATUS_SUCCESS == mluOpDestroyTensorDescriptor(known_desc_tmp));

  VLOG(5) << "[mluOpThreeNNForward] mluOpTranspose_v2 feature end.";
  VLOG(5) << "Launch Kernel MLUKernelThreeNNForward<<<Union"
          << k_type / CORE_DIM << ", " << k_dims.x << ", " << k_dims.y << ", "
          << k_dims.z << ">>>.";
  switch (input_dtype) {
    case MLUOP_DTYPE_FLOAT: {
      KERNEL_CHECK((mluOpUnion1KernelThreeNNForwardFloat(
          k_dims, k_type, handle->queue, b, n, m, unknown, known_workspace,
          dist2, idx)));
    }; break;
    case MLUOP_DTYPE_HALF: {
      KERNEL_CHECK((mluOpUnion1KernelThreeNNForwardHalf(
          k_dims, k_type, handle->queue, b, n, m, unknown, known_workspace,
          dist2, idx)));
    }; break;
    default: {
      VLOG(5) << "Not implemented.";
      break;
    }
  }

  // generate gen_case prototxt
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
