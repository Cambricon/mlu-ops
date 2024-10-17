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
#include "masked_col2im_forward.h"

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"
#include "kernels/utils/cnnl_helper.h"

static void policyFunc(const mluOpHandle_t handle, const int mask_cnt,
                       cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  const size_t cluster_limit =
      mluop::runtime::getClusterLimitCapability(handle);
  const size_t core_limit =
      mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  const size_t task_dim = CEIL_ALIGN(mask_cnt, core_limit);
  k_dim->x = core_limit;
  k_dim->y = (task_dim / core_limit) > cluster_limit ? cluster_limit
                                                     : (task_dim / core_limit);
  k_dim->z = 1;
  *k_type = cnrtFuncTypeUnion1;
}

static mluOpStatus_t maskedCol2imForwardPreCheck(
    const mluOpTensorDescriptor_t col_desc,
    const mluOpTensorDescriptor_t mask_h_idx_desc,
    const mluOpTensorDescriptor_t mask_w_idx_desc,
    const mluOpTensorDescriptor_t im_desc) {
  PARAM_CHECK("[mluOpMaskedCol2imForward]", col_desc != NULL);
  PARAM_CHECK("[mluOpMaskedCol2imForward]", mask_h_idx_desc != NULL);
  PARAM_CHECK("[mluOpMaskedCol2imForward]", mask_w_idx_desc != NULL);
  PARAM_CHECK("[mluOpMaskedCol2imForward]", im_desc != NULL);
  PARAM_CHECK("[mluOpMaskedCol2imForward]", col_desc->dim == 2);
  PARAM_CHECK("[mluOpMaskedCol2imForward]", im_desc->dim == 4);
  PARAM_CHECK("[mluOpMaskedCol2imForward]", mask_h_idx_desc->dim == 1);
  PARAM_CHECK("[mluOpMaskedCol2imForward]", mask_w_idx_desc->dim == 1);
  PARAM_CHECK("[mluOpMaskedCol2imForward]",
              im_desc->layout == MLUOP_LAYOUT_NCHW);
  PARAM_CHECK("[mluOpMaskedCol2imForward]",
              col_desc->dtype == MLUOP_DTYPE_FLOAT ||
                  col_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK("[mluOpMaskedCol2imForward]", col_desc->dtype == im_desc->dtype);
  PARAM_CHECK("[mluOpMaskedCol2imForward]",
              mask_h_idx_desc->dtype == MLUOP_DTYPE_INT32);
  PARAM_CHECK("[mluOpMaskedCol2imForward]",
              mask_w_idx_desc->dtype == MLUOP_DTYPE_INT32);
  PARAM_CHECK("[mluOpMaskedCol2imForward]", im_desc->dims[0] == 1);

  PARAM_CHECK("[mluOpMaskedCol2imForward]",
              mask_h_idx_desc->dims[0] == mask_w_idx_desc->dims[0]);
  PARAM_CHECK("[mluOpMaskedCol2imForward]",
              col_desc->dims[1] == mask_h_idx_desc->dims[0]);
  PARAM_CHECK("[mluOpMaskedCol2imForward]",
              col_desc->dims[0] == im_desc->dims[1]);
  // stride check
  STRIDE_TENSOR_CHECK("[mluOpMaskedCol2imForward]:", col_desc,
                      "col_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpMaskedCol2imForward]:", mask_h_idx_desc,
                      "mask_h_idx_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpMaskedCol2imForward]:", mask_w_idx_desc,
                      "mask_w_idx_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpMaskedCol2imForward]:", im_desc,
                      "im_desc must be contiguous");

  const size_t col_element_num = mluOpGetTensorElementNum(col_desc);
  const size_t mask_h_idx_element_num =
      mluOpGetTensorElementNum(mask_h_idx_desc);
  const size_t im_element_num = mluOpGetTensorElementNum(im_desc);
  TENSOR_NUM_CHECK("[mluOpMaskedCol2imForward]", col_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpMaskedCol2imForward]", mask_h_idx_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpMaskedCol2imForward]", im_element_num,
                   LARGE_TENSOR_NUM, "");
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetMaskedCol2imForwardWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t col_desc,
    const mluOpTensorDescriptor_t mask_h_idx_desc,
    const mluOpTensorDescriptor_t mask_w_idx_desc,
    const mluOpTensorDescriptor_t im_desc, size_t *workspace_size) {
  mluOpStatus_t status = MLUOP_STATUS_BAD_PARAM;
  PARAM_CHECK("[mluOpMaskedCol2imForward]", handle != NULL);
  PARAM_CHECK("[mluOpMaskedCol2imForward]", workspace_size != NULL);
  status = maskedCol2imForwardPreCheck(col_desc, mask_h_idx_desc,
                                       mask_w_idx_desc, im_desc);
  if (MLUOP_STATUS_SUCCESS != status) {
    return status;
  }
  if (mluOpGetTensorElementNum(im_desc) == 0 || col_desc->dims[0] == 0) {
    LOG(ERROR) << "[mluOpMaskedCol2imForward] Zero element tensor failure.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (mluOpGetTensorElementNum(mask_h_idx_desc) == 0) {
    VLOG(5) << "[mluOpMaskedCol2imForward] Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }
  *workspace_size = col_desc->total_tensor_size;
  *workspace_size += im_desc->total_tensor_size;

  cnnlTransposeDescriptor_t trans_desc;
  size_t col_transpose_workspace_size = 0;
  int col_dim = col_desc->dim;
  int col_permute[2] = {1, 0};
  int col_MC_dims[2] = {0, 0};
  col_MC_dims[0] = col_desc->dims[1];
  col_MC_dims[1] = col_desc->dims[0];
  mluOpTensorDescriptor_t col_MC_desc_tmp;
  CHECK_RETURN("[mluOpGetMaskedCol2imForwardWorkspaceSize]",
               mluOpCreateTensorDescriptor(&col_MC_desc_tmp));

  PARAM_CHECK(
      "[mluOpMaskedCol2imForward]",
      MLUOP_STATUS_SUCCESS ==
          mluOpSetTensorDescriptor(col_MC_desc_tmp, MLUOP_LAYOUT_ARRAY,
                                   col_desc->dtype, col_dim, col_MC_dims));
  CALL_CNNL(cnnlCreateTransposeDescriptor(&trans_desc));
  CALL_CNNL(cnnlSetTransposeDescriptor(trans_desc, col_dim, col_permute));
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(col_MC_desc_tmp, cnnl_x_desc);
    CALL_CNNL(cnnlGetTransposeWorkspaceSize(
        cnnl_handle, cnnl_x_desc, trans_desc, &col_transpose_workspace_size));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  int im_dim = im_desc->dim;
  int im_permute[4] = {0, 3, 1, 2};
  int NCHW2NHWC_permute[4] = {0, 2, 3, 1};
  int im_NHWC_dims[4] = {0, 0, 0, 0};
  for (int i = 0; i < im_dim; ++i) {
    im_NHWC_dims[i] = im_desc->dims[NCHW2NHWC_permute[i]];
  }
  size_t im_transpose_workspace_size = 0;
  mluOpTensorDescriptor_t im_NHWC_desc_tmp;
  CHECK_RETURN("[mluOpGetMaskedCol2imForwardWorkspaceSize]",
               mluOpCreateTensorDescriptor(&im_NHWC_desc_tmp));

  CHECK_RETURN("[mluOpMaskedCol2imForward]",
               mluOpSetTensorDescriptor(im_NHWC_desc_tmp, MLUOP_LAYOUT_ARRAY,
                                        im_desc->dtype, im_dim, im_NHWC_dims));
  CALL_CNNL(cnnlSetTransposeDescriptor(trans_desc, im_dim, im_permute));
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(im_NHWC_desc_tmp, cnnl_x_desc);
    CALL_CNNL(cnnlGetTransposeWorkspaceSize(
        cnnl_handle, cnnl_x_desc, trans_desc, &im_transpose_workspace_size));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  *workspace_size += im_transpose_workspace_size > col_transpose_workspace_size
                         ? im_transpose_workspace_size
                         : col_transpose_workspace_size;
  CHECK_RETURN("[mluOpMaskedCol2imForward]",
               mluOpDestroyTensorDescriptor(im_NHWC_desc_tmp));
  CALL_CNNL(cnnlDestroyTransposeDescriptor(trans_desc));
  CHECK_RETURN("[mluOpMaskedCol2imForward]",
               mluOpDestroyTensorDescriptor(col_MC_desc_tmp));
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t transposeTensor(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, const int *permute,
    const mluOpTensorDescriptor_t workspace_dst_desc, void *workspace_dst,
    void *transpose_workspace) {
  const int input_dim = input_desc->dim;
  cnnlTransposeDescriptor_t trans_desc;
  size_t transpose_workspace_size = 0;
  CALL_CNNL(cnnlCreateTransposeDescriptor(&trans_desc));
  CALL_CNNL(cnnlSetTransposeDescriptor(trans_desc, input_dim, permute));
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_x_desc);
    CALL_CNNL(cnnlGetTransposeWorkspaceSize(
        cnnl_handle, cnnl_x_desc, trans_desc, &transpose_workspace_size));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_x_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(workspace_dst_desc,
                                                 cnnl_y_desc);
    CALL_CNNL(cnnlTranspose_v2(cnnl_handle, trans_desc, cnnl_x_desc, input,
                               cnnl_y_desc, workspace_dst, transpose_workspace,
                               transpose_workspace_size));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_y_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  CALL_CNNL(cnnlDestroyTransposeDescriptor(trans_desc));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpMaskedCol2imForward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t col_desc,
    const void *col, const mluOpTensorDescriptor_t mask_h_idx_desc,
    const void *mask_h_idx, const mluOpTensorDescriptor_t mask_w_idx_desc,
    const void *mask_w_idx, const size_t workspace_size, void *workspace,
    const mluOpTensorDescriptor_t im_desc, void *im) {
  mluOpStatus_t status = MLUOP_STATUS_BAD_PARAM;
  PARAM_CHECK("[mluOpMaskedCol2imForward]", handle != NULL);
  status = maskedCol2imForwardPreCheck(col_desc, mask_h_idx_desc,
                                       mask_w_idx_desc, im_desc);
  if (MLUOP_STATUS_SUCCESS != status) {
    return status;
  }
  if (mluOpGetTensorElementNum(im_desc) == 0 || col_desc->dims[0] == 0) {
    LOG(ERROR) << "[mluOpMaskedCol2imForward] Zero element tensor failure.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (mluOpGetTensorElementNum(mask_h_idx_desc) == 0) {
    VLOG(5) << "[mluOpMaskedCol2imForward] Skip zero element tensor.";
    uint64_t fill_value = 0x0;
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(im_desc, cnnl_output_desc);
    CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &fill_value,
                          cnnl_output_desc, im));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
    return MLUOP_STATUS_SUCCESS;
  }
  if (workspace_size > 0) {
    PARAM_CHECK("[mluOpMaskedCol2imForward]", workspace != NULL);
  }
  PARAM_CHECK("[mluOpMaskedCol2imForward]", col != NULL);
  PARAM_CHECK("[mluOpMaskedCol2imForward]", mask_h_idx != NULL);
  PARAM_CHECK("[mluOpMaskedCol2imForward]", mask_w_idx != NULL);
  PARAM_CHECK("[mluOpMaskedCol2imForward]", im != NULL);

  const int height = im_desc->dims[2];
  const int width = im_desc->dims[3];
  // generate mluOpMaskedCol2imForward prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("masked_col2im_forward", "MASKED_COL2IM_FORWARD");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "col", col, col_desc, -10, 10);
    GEN_CASE_DATA_REAL(true, "mask_h_idx", mask_h_idx, mask_h_idx_desc);
    GEN_CASE_DATA_REAL(true, "mask_w_idx", mask_w_idx, mask_w_idx_desc);
    GEN_CASE_DATA(false, "im", im, im_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(false, false, true, 0, 0, 0);
  }
  // generate mluOpMaskedCol2imForward prototxt end!
  mluOpDataType_t input_dtype = col_desc->dtype;
  void *col_workspace = workspace;
  void *im_workspace = (int8_t *)workspace + col_desc->total_tensor_size;
  void *transpose_workspace =
      (int8_t *)im_workspace + im_desc->total_tensor_size;

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  const int mask_cnt = mask_h_idx_desc->dims[0];
  policyFunc(handle, mask_cnt, &k_dim, &k_type);

  VLOG(5) << "[mluOpMaskedCol2imForward] cnnlFill_v3 start.";
  const int im_dim = im_desc->dim;
  int NCHW2NHWC_permute[4] = {0, 2, 3, 1};
  int im_NHWC_dims[4] = {0, 0, 0, 0};
  for (int i = 0; i < im_dim; ++i) {
    im_NHWC_dims[i] = im_desc->dims[NCHW2NHWC_permute[i]];
  }
  mluOpTensorDescriptor_t im_NHWC_desc_tmp;
  CHECK_RETURN("[mluOpMaskedCol2imForward]",
               mluOpCreateTensorDescriptor(&im_NHWC_desc_tmp));
  CHECK_RETURN("[mluOpMaskedCol2imForward]",
               mluOpSetTensorDescriptor(im_NHWC_desc_tmp, MLUOP_LAYOUT_ARRAY,
                                        im_desc->dtype, im_dim, im_NHWC_dims));
  uint64_t fill_value = 0x0;
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(im_NHWC_desc_tmp,
                                                 cnnl_output_desc);
    CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &fill_value,
                          cnnl_output_desc, im_workspace));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  VLOG(5) << "[mluOpMaskedCol2imForward] cnnlFill_v3 end.";

  VLOG(5) << "[mluOpMaskedCol2imForward] cnnlTranspose_v2 col start.";

  int col_dim = col_desc->dim;
  int col_permute[2] = {1, 0};
  int col_MC_dims[2] = {0, 0};
  col_MC_dims[0] = col_desc->dims[1];
  col_MC_dims[1] = col_desc->dims[0];
  mluOpTensorDescriptor_t col_MC_desc_tmp;
  CHECK_RETURN("[mluOpMaskedCol2imForward]",
               mluOpCreateTensorDescriptor(&col_MC_desc_tmp));
  CHECK_RETURN("[mluOpMaskedCol2imForward]",
               mluOpSetTensorDescriptor(col_MC_desc_tmp, MLUOP_LAYOUT_ARRAY,
                                        col_desc->dtype, col_dim, col_MC_dims));
  CHECK_RETURN(
      "[mluOpMaskedCol2imForward]",
      transposeTensor(handle, col_desc, col, col_permute, col_MC_desc_tmp,
                      col_workspace, transpose_workspace));
  CHECK_RETURN("[mluOpMaskedCol2imForward]",
               mluOpDestroyTensorDescriptor(col_MC_desc_tmp));
  VLOG(5) << "[mluOpMaskedCol2imForward] cnnlTranspose_v2 col end.";

  const int channels = im_desc->dims[1];
  VLOG(5) << "Launch kernel MLUUnion1MaskedCol2imForward<<<" << k_dim.x << ", "
          << k_dim.y << ", " << k_dim.z << ">>>.";
  CHECK_RETURN("[mluOpMaskedCol2imForward]",
               KernelMaskedCol2imForward(k_dim, k_type, handle->queue,
                                         input_dtype, col_workspace, height,
                                         width, channels, mask_h_idx,
                                         mask_w_idx, mask_cnt, im_workspace));
  VLOG(5) << "Finish launch MLUUnion1MaskedCol2imForward.";

  VLOG(5) << "[mluOpMaskedCol2imForward] cnnlTranspose_v2 im start.";
  int im_permute[4] = {0, 3, 1, 2};
  CHECK_RETURN("[mluOpMaskedCol2imForward]",
               transposeTensor(handle, im_NHWC_desc_tmp, im_workspace,
                               im_permute, im_desc, im, transpose_workspace));
  CHECK_RETURN("[mluOpMaskedCol2imForward]",
               mluOpDestroyTensorDescriptor(im_NHWC_desc_tmp));
  VLOG(5) << "[mluOpMaskedCol2imForward] cnnlTranspose_v2 im end.";
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
