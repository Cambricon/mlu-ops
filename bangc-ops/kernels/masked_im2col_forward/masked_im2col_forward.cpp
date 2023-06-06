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
#include "kernels/masked_im2col_forward/masked_im2col_forward.h"

#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "kernels/kernel.h"

// policy function
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
  *k_type = CNRT_FUNC_TYPE_UNION1;
}

static mluOpStatus_t maskedIm2colForwardPreCheck(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t feature_desc,
    const mluOpTensorDescriptor_t mask_h_idx_desc,
    const mluOpTensorDescriptor_t mask_w_idx_desc,
    const mluOpTensorDescriptor_t data_col_desc, const int kernel_h,
    const int kernel_w) {
  PARAM_CHECK("[mluOpMaskedIm2colForward]", handle != NULL);
  PARAM_CHECK("[mluOpMaskedIm2colForward]", feature_desc != NULL);
  PARAM_CHECK("[mluOpMaskedIm2colForward]", mask_h_idx_desc != NULL);
  PARAM_CHECK("[mluOpMaskedIm2colForward]", mask_w_idx_desc != NULL);
  PARAM_CHECK("[mluOpMaskedIm2colForward]", data_col_desc != NULL);

  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              feature_desc->layout == MLUOP_LAYOUT_NCHW);
  PARAM_CHECK("[mluOpMaskedIm2colForward]", feature_desc->dim == 4);
  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              feature_desc->dtype == MLUOP_DTYPE_FLOAT ||
                  feature_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              feature_desc->dtype == data_col_desc->dtype);
  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              mask_h_idx_desc->dtype == MLUOP_DTYPE_INT32);
  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              mask_w_idx_desc->dtype == MLUOP_DTYPE_INT32);
  PARAM_CHECK("[mluOpMaskedIm2colForward]", feature_desc->dims[0] == 1);
  PARAM_CHECK("[mluOpMaskedIm2colForward]", mask_h_idx_desc->dim == 1);
  PARAM_CHECK("[mluOpMaskedIm2colForward]", mask_w_idx_desc->dim == 1);
  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              mask_h_idx_desc->dims[0] == mask_w_idx_desc->dims[0]);
  PARAM_CHECK("[mluOpMaskedIm2colForward]", data_col_desc->dim == 2);
  PARAM_CHECK(
      "[mluOpMaskedIm2colForward]",
      data_col_desc->dims[0] == feature_desc->dims[1] * kernel_h * kernel_w);
  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              data_col_desc->dims[1] == mask_h_idx_desc->dims[0]);
  PARAM_CHECK("[mluOpMaskedIm2colForward]", kernel_h > 0);
  PARAM_CHECK("[mluOpMaskedIm2colForward]", kernel_w > 0);

  const uint64_t feature_element_num = mluOpGetTensorElementNum(feature_desc);
  const uint64_t mask_h_idx_element_num =
      mluOpGetTensorElementNum(mask_h_idx_desc);
  const uint64_t data_col_element_num = mluOpGetTensorElementNum(data_col_desc);
  TENSOR_NUM_CHECK("[mluOpMaskedIm2colForward]", feature_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpMaskedIm2colForward]", mask_h_idx_element_num,
                   LARGE_TENSOR_NUM, "");
  TENSOR_NUM_CHECK("[mluOpMaskedIm2colForward]", data_col_element_num,
                   LARGE_TENSOR_NUM, "");
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetMaskedIm2colForwardWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t feature_desc,
    const mluOpTensorDescriptor_t mask_h_idx_desc,
    const mluOpTensorDescriptor_t mask_w_idx_desc, const int kernel_h,
    const int kernel_w, const mluOpTensorDescriptor_t data_col_desc,
    size_t *workspace_size) {
  mluOpStatus_t status = MLUOP_STATUS_BAD_PARAM;
  PARAM_CHECK("[mluOpMaskedIm2colForward]", workspace_size != NULL);
  status = maskedIm2colForwardPreCheck(handle, feature_desc, mask_h_idx_desc,
                                       mask_w_idx_desc, data_col_desc, kernel_h,
                                       kernel_w);
  if (MLUOP_STATUS_SUCCESS != status) {
    return status;
  }
  if (mluOpGetTensorElementNum(feature_desc) == 0 ||
      data_col_desc->dims[0] == 0) {
    LOG(ERROR) << "[mluOpMaskedIm2colForward] Zero element tensor failure.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (mluOpGetTensorElementNum(mask_h_idx_desc) == 0) {
    VLOG(5) << "[mluOpMaskedIm2colForward] Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }
  *workspace_size = feature_desc->total_tensor_size;
  *workspace_size += data_col_desc->total_tensor_size;

  mluOpTransposeDescriptor_t trans_desc;
  size_t feature_transpose_workspace_size = 0;
  int feature_dim = feature_desc->dim;
  int feature_permute[4] = {0, 3, 1, 2};

  PARAM_CHECK(
      "[mluOpMaskedIm2colForward]",
      MLUOP_STATUS_SUCCESS == mluOpCreateTransposeDescriptor(&trans_desc));
  PARAM_CHECK(
      "[mluOpMaskedIm2colForward]",
      MLUOP_STATUS_SUCCESS == mluOpSetTransposeDescriptor(
                                  trans_desc, feature_dim, feature_permute));
  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              MLUOP_STATUS_SUCCESS == mluOpGetTransposeWorkspaceSize(
                                          handle, feature_desc, trans_desc,
                                          &feature_transpose_workspace_size));
  if (mluOpGetTensorElementNum(feature_desc) == 0 ||
      data_col_desc->dims[0] == 0) {
    VLOG(5) << "[mluOpMaskedIm2colForward] Zero element tensor failure.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  int data_col_dim = 3;
  int data_col_permute[3] = {2, 1, 0};
  int data_col_HWC_dims[3] = {0, 0, 0};
  int data_col_CHW_dims[3] = {0, 0, 0};
  data_col_HWC_dims[0] = mask_h_idx_desc->dims[0];
  data_col_HWC_dims[1] = kernel_h * kernel_w;
  data_col_HWC_dims[2] = feature_desc->dims[1];
  for (int i = 0; i < data_col_dim; ++i) {
    data_col_CHW_dims[i] = data_col_HWC_dims[data_col_permute[i]];
  }
  size_t data_col_transpose_workspace_size = 0;
  mluOpTensorDescriptor_t data_col_HWC_desc_tmp;
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_col_HWC_desc_tmp));

  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpSetTensorDescriptor(
                      data_col_HWC_desc_tmp, MLUOP_LAYOUT_ARRAY,
                      feature_desc->dtype, data_col_dim, data_col_HWC_dims));
  PARAM_CHECK(
      "[mluOpMaskedIm2colForward]",
      MLUOP_STATUS_SUCCESS == mluOpSetTransposeDescriptor(
                                  trans_desc, data_col_dim, data_col_permute));
  PARAM_CHECK(
      "[mluOpMaskedIm2colForward]",
      MLUOP_STATUS_SUCCESS == mluOpGetTransposeWorkspaceSize(
                                  handle, data_col_HWC_desc_tmp, trans_desc,
                                  &data_col_transpose_workspace_size));
  *workspace_size +=
      data_col_transpose_workspace_size > feature_transpose_workspace_size
          ? data_col_transpose_workspace_size
          : feature_transpose_workspace_size;
  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpDestroyTensorDescriptor(data_col_HWC_desc_tmp));
  PARAM_CHECK(
      "[mluOpMaskedIm2colForward]",
      MLUOP_STATUS_SUCCESS == mluOpDestroyTransposeDescriptor(trans_desc));
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t transposeTensor(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, const int *permute,
    const mluOpTensorDescriptor_t workspace_dst_desc, void *workspace_dst,
    void *transpose_workspace) {
  int input_dim = input_desc->dim;
  mluOpTransposeDescriptor_t trans_desc;
  size_t transpose_workspace_size = 0;
  PARAM_CHECK(
      "[mluOpMaskedIm2colForward]",
      MLUOP_STATUS_SUCCESS == mluOpCreateTransposeDescriptor(&trans_desc));
  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpSetTransposeDescriptor(trans_desc, input_dim, permute));
  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpGetTransposeWorkspaceSize(handle, input_desc, trans_desc,
                                                 &transpose_workspace_size));
  PARAM_CHECK(
      "[mluOpMaskedIm2colForward]",
      MLUOP_STATUS_SUCCESS ==
          mluOpTranspose_v2(handle, trans_desc, input_desc, input,
                            workspace_dst_desc, workspace_dst,
                            transpose_workspace, transpose_workspace_size));
  PARAM_CHECK(
      "[mluOpMaskedIm2colForward]",
      MLUOP_STATUS_SUCCESS == mluOpDestroyTransposeDescriptor(trans_desc));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpMaskedIm2colForward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t feature_desc,
    const void *feature, const mluOpTensorDescriptor_t mask_h_idx_desc,
    const void *mask_h_idx, const mluOpTensorDescriptor_t mask_w_idx_desc,
    const void *mask_w_idx, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, void *workspace,
    const size_t workspace_size, const mluOpTensorDescriptor_t data_col_desc,
    void *data_col) {
  mluOpStatus_t status = MLUOP_STATUS_BAD_PARAM;
  status = maskedIm2colForwardPreCheck(handle, feature_desc, mask_h_idx_desc,
                                       mask_w_idx_desc, data_col_desc, kernel_h,
                                       kernel_w);
  if (MLUOP_STATUS_SUCCESS != status) {
    return status;
  }

  if (mluOpGetTensorElementNum(feature_desc) == 0 ||
      data_col_desc->dims[0] == 0) {
    LOG(ERROR) << "[mluOpMaskedIm2colForward] Zero element tensor failure.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (mluOpGetTensorElementNum(mask_h_idx_desc) == 0) {
    VLOG(5) << "[mluOpMaskedIm2colForward] Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }
  if (workspace_size > 0) {
    PARAM_CHECK("[mluOpMaskedIm2colForward]", workspace != NULL);
  }
  PARAM_CHECK("[mluOpMaskedIm2colForward]", feature != NULL);
  PARAM_CHECK("[mluOpMaskedIm2colForward]", mask_h_idx != NULL);
  PARAM_CHECK("[mluOpMaskedIm2colForward]", mask_w_idx != NULL);
  PARAM_CHECK("[mluOpMaskedIm2colForward]", data_col != NULL);

  // generate mluOpMaskedIm2colForward prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("masked_im2col_forward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "feature", feature, feature_desc, -10, 10);
    GEN_CASE_DATA_REAL(true, "mask_h_idx", mask_h_idx, mask_h_idx_desc);
    GEN_CASE_DATA_REAL(true, "mask_w_idx", mask_w_idx, mask_w_idx_desc);
    GEN_CASE_DATA(false, "data_col", data_col, data_col_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "masked_im2col_forward", "kernel_h", kernel_h);
    GEN_CASE_OP_PARAM_SINGLE(1, "masked_im2col_forward", "kernel_w", kernel_w);
    GEN_CASE_OP_PARAM_SINGLE(1, "masked_im2col_forward", "pad_h", pad_h);
    GEN_CASE_OP_PARAM_SINGLE(2, "masked_im2col_forward", "pad_w", pad_w);
    GEN_CASE_TEST_PARAM_NEW(false, false, true, 0, 0, 0);
  }
  // generate mluOpMaskedIm2colForward prototxt end!
  mluOpDataType_t input_dtype = feature_desc->dtype;
  void *feature_workspace = workspace;
  void *data_col_workspace =
      (char *)workspace + feature_desc->total_tensor_size;
  void *transpose_workspace =
      (char *)data_col_workspace + data_col_desc->total_tensor_size;

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  const int mask_cnt = mask_h_idx_desc->dims[0];
  policyFunc(handle, mask_cnt, &k_dim, &k_type);

  VLOG(5) << "[mluOpMaskedIm2colForward] mluOpFill_v3 start.";
  uint64_t fill_value = 0x0;
  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST, &fill_value,
                               data_col_desc, data_col_workspace));

  VLOG(5) << "[mluOpMaskedIm2colForward] mluOpTranspose_v2 feature start.";

  int feature_dim = feature_desc->dim;
  int feature_permute[4] = {0, 2, 3, 1};
  int feature_tmp_dims[4] = {0, 0, 0, 0};

  for (int i = 0; i < feature_dim; ++i) {
    feature_tmp_dims[i] = feature_desc->dims[feature_permute[i]];
  }

  mluOpTensorDescriptor_t feature_desc_tmp;
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&feature_desc_tmp));
  PARAM_CHECK(
      "[mluOpMaskedIm2colForward]",
      MLUOP_STATUS_SUCCESS ==
          mluOpSetTensorDescriptor(feature_desc_tmp, MLUOP_LAYOUT_ARRAY,
                                   input_dtype, feature_dim, feature_tmp_dims));
  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              MLUOP_STATUS_SUCCESS ==
                  transposeTensor(handle, feature_desc, feature,
                                  feature_permute, feature_desc_tmp,
                                  feature_workspace, transpose_workspace));

  PARAM_CHECK(
      "[mluOpMaskedIm2colForward]",
      MLUOP_STATUS_SUCCESS == mluOpDestroyTensorDescriptor(feature_desc_tmp));

  const int channels = feature_desc->dims[1];
  const int height = feature_desc->dims[2];
  const int width = feature_desc->dims[3];
  VLOG(5) << "Launch kernel MLUUnion1MaskedIm2colForward<<<" << k_dim.x << ", "
          << k_dim.y << ", " << k_dim.z << ">>>.";
  KERNEL_CHECK(KernelMaskedIm2colForward(
      k_dim, k_type, handle->queue, input_dtype, feature_workspace, height,
      width, channels, kernel_h, kernel_w, pad_h, pad_w, mask_h_idx, mask_w_idx,
      mask_cnt, data_col_workspace));

  VLOG(5) << "[mluOpMaskedIm2colForward] mluOpTranspose_v2 data_col start.";
  const int data_col_dim = 3;
  int data_col_permute[3] = {2, 1, 0};
  int data_col_HWC_dims[3] = {0, 0, 0};
  int data_col_CHW_dims[3] = {0, 0, 0};
  data_col_HWC_dims[0] = mask_cnt;
  data_col_HWC_dims[1] = kernel_h * kernel_w;
  data_col_HWC_dims[2] = channels;
  for (int i = 0; i < data_col_dim; ++i) {
    data_col_CHW_dims[i] = data_col_HWC_dims[data_col_permute[i]];
  }

  mluOpTensorDescriptor_t data_col_HWC_desc_tmp;
  mluOpTensorDescriptor_t data_col_CHW_desc_tmp;
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_col_HWC_desc_tmp));
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_col_CHW_desc_tmp));

  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpSetTensorDescriptor(data_col_HWC_desc_tmp,
                                           MLUOP_LAYOUT_ARRAY, input_dtype,
                                           data_col_dim, data_col_HWC_dims));
  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpSetTensorDescriptor(data_col_CHW_desc_tmp,
                                           MLUOP_LAYOUT_ARRAY, input_dtype,
                                           data_col_dim, data_col_CHW_dims));

  PARAM_CHECK(
      "[mluOpMaskedIm2colForward]",
      MLUOP_STATUS_SUCCESS ==
          transposeTensor(handle, data_col_HWC_desc_tmp, data_col_workspace,
                          data_col_permute, data_col_CHW_desc_tmp, data_col,
                          transpose_workspace));
  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpDestroyTensorDescriptor(data_col_HWC_desc_tmp));
  PARAM_CHECK("[mluOpMaskedIm2colForward]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpDestroyTensorDescriptor(data_col_CHW_desc_tmp));
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
