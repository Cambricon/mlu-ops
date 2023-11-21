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
#include <math.h>
#include <limits.h>
#include <cstdio>

#include <vector>

#include "kernels/utils/cnnl_helper.h"

mluOpStatus_t MLUOP_WIN_API
mluOpCreateReduceDescriptor(mluOpReduceDescriptor_t *reduce_desc) {
  PARAM_CHECK("mluOpReduce", reduce_desc != NULL);
  CHECK_FUNC_RETURN(
      cnnlCreateReduceDescriptor(reduce_desc), CNNL_STATUS_SUCCESS,
      "[mluOpReduce] Internal error accured in mluOpCreateReduceDescriptor.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpDestroyReduceDescriptor(mluOpReduceDescriptor_t reduce_desc) {
  PARAM_CHECK("mluOpReduce", reduce_desc != NULL);
  CHECK_FUNC_RETURN(
      cnnlDestroyReduceDescriptor(reduce_desc), CNNL_STATUS_SUCCESS,
      "[mluOpReduce] Internal error accured in mluOpDestroyReduceDescriptor.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetReduceDescriptor(
    mluOpReduceDescriptor_t reduce_desc, int axis[], int axis_num,
    mluOpReduceOp_t reduce_op, mluOpDataType_t tensor_type,
    mluOpNanPropagation_t nan_propagation, mluOpReduceIndices_t tensor_indices,
    mluOpIndicesType_t indices_type) {
  PARAM_CHECK("mluOpReduce", reduce_desc != NULL);
  PARAM_CHECK_GE("mluOpReduce", reduce_op, 0);
  PARAM_CHECK_GE("mluOpReduce", tensor_type, 0);
  PARAM_CHECK_GE("mluOpReduce", tensor_indices, 0);
  PARAM_CHECK_GE("mluOpReduce", indices_type, 0);
  PARAM_CHECK("mluOpReduce", axis != NULL);
  PARAM_CHECK_GT("mluOpReduce", axis_num, 0);
  CHECK_FUNC_RETURN(
      cnnlSetReduceDescriptor(reduce_desc, axis, axis_num,
                              cnnlReduceOp_t(reduce_op),
                              cnnlDataType_t(tensor_type),
                              cnnlNanPropagation_t(nan_propagation),
                              cnnlReduceIndices_t(tensor_indices),
                              cnnlIndicesType_t(indices_type)),
      CNNL_STATUS_SUCCESS,
      "[mluOpReduce] Internal error accured in mluOpSetReduceDescriptor.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetReduceDescriptor_v2(
    mluOpReduceDescriptor_t reduce_desc, int axis[], int axis_num,
    mluOpReduceOp_t reduce_op, mluOpDataType_t tensor_type,
    mluOpNanPropagation_t nan_propagation, mluOpReduceIndices_t tensor_indices,
    mluOpIndicesType_t indices_type, float p) {
  PARAM_CHECK("mluOpReduce", reduce_desc != NULL);
  PARAM_CHECK_GE("mluOpReduce", reduce_op, 0);
  PARAM_CHECK_GE("mluOpReduce", tensor_type, 0);
  PARAM_CHECK_GE("mluOpReduce", tensor_indices, 0);
  PARAM_CHECK_GE("mluOpReduce", indices_type, 0);
  PARAM_CHECK("mluOpReduce", axis != NULL);
  PARAM_CHECK_GT("mluOpReduce", axis_num, 0);
  PARAM_CHECK("mluOpReduce",
              p != 1.0 && p != 2.0 && p != INFINITY && p != -INFINITY);
  CHECK_FUNC_RETURN(
      cnnlSetReduceDescriptor_v2(
          reduce_desc, axis, axis_num, cnnlReduceOp_t(reduce_op),
          cnnlDataType_t(tensor_type),
          cnnlNanPropagation_t(nan_propagation),
          cnnlReduceIndices_t(tensor_indices),
          cnnlIndicesType_t(indices_type),
          p),
      CNNL_STATUS_SUCCESS,
      "[mluOpReduce] Internal error accured in mluOpSetReduceDescriptor_v2.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetReduceOpWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const mluOpTensorDescriptor_t output_desc,
    const mluOpReduceDescriptor_t reduce_desc, size_t *workspace_size_inbytes) {
  PARAM_CHECK("mluOpReduce", input_desc != NULL);
  PARAM_CHECK("mluOpReduce", output_desc != NULL);
  PARAM_CHECK("mluOpReduce", reduce_desc != NULL);
  CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, _input_desc);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, _output_desc);
  CHECK_FUNC_RETURN(
      cnnlGetReduceOpWorkspaceSize(_handle, _input_desc, _output_desc,
                                   reduce_desc, workspace_size_inbytes),
      CNNL_STATUS_SUCCESS,
      "[mluOpReduce] Internal error accured in mluOpReduce.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_output_desc);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpReduce(mluOpHandle_t handle, const mluOpReduceDescriptor_t reduce_desc,
            void *workspace, size_t workspace_size, const void *alpha,
            const mluOpTensorDescriptor_t input_desc, const void *input,
            const size_t indices_size_inbytes, void *indices, const void *beta,
            const mluOpTensorDescriptor_t output_desc, void *output) {
  PARAM_CHECK("mluOpReduce", handle != NULL);
  PARAM_CHECK("mluOpReduce", reduce_desc != NULL);
  if (workspace_size > 0) {
    PARAM_CHECK("mluOpReduce", workspace != NULL);
  }
  CREATE_AND_SET_CNNL_HANDLE(handle, _handle);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, _input_desc);
  CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, _output_desc);
  CHECK_FUNC_RETURN(
      cnnlReduce(_handle, reduce_desc, workspace, workspace_size, alpha,
                 _input_desc, input, indices_size_inbytes, indices, beta,
                 _output_desc, output),
      CNNL_STATUS_SUCCESS,
      "[mluOpReduce] Internal error accured in mluOpReduce.",
      MLUOP_STATUS_INTERNAL_ERROR);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(_output_desc);
  DESTROY_CNNL_HANDLE(_handle);
  return MLUOP_STATUS_SUCCESS;
}
