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
#include <algorithm>
#include <string>
#include "mlu_op.h"
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/mlu_env.h"
#include "core/tensor.h"

inline bool isFloatDtype(const mluOpDataType_t &dtype) {
  return (dtype == MLUOP_DTYPE_HALF || dtype == MLUOP_DTYPE_FLOAT);
}

inline mluOpDataType_t getOnchipDataType(
    const mluOpTensorDescriptor_t tensor_desc) {
  if (tensor_desc->onchip_dtype != MLUOP_DTYPE_INVALID) {
    return tensor_desc->onchip_dtype;
  } else {
    return tensor_desc->dtype;
  }
}

inline mluOpStatus_t setMatmulDescInfo(const std::string api_name,
                                       mluOpMatMulDescriptor_t matmul_desc,
                                       const uint32_t is_trans_a_value,
                                       const uint32_t is_trans_b_value,
                                       const uint32_t compute_dtype,
                                       const uint32_t allow_tf32) {
  CHECK_RETURN(api_name,
               mluOpSetMatMulDescAttr(matmul_desc, MLUOP_MATMUL_DESC_TRANSA,
                                      &is_trans_a_value, sizeof(int32_t)));
  CHECK_RETURN(api_name,
               mluOpSetMatMulDescAttr(matmul_desc, MLUOP_MATMUL_DESC_TRANSB,
                                      &is_trans_b_value, sizeof(int32_t)));
  CHECK_RETURN(api_name, mluOpSetMatMulDescAttr(
                             matmul_desc, MLUOP_MATMUL_DESC_COMPUTE_TYPE,
                             &compute_dtype, sizeof(int32_t)));
  CHECK_RETURN(api_name,
               mluOpSetMatMulDescAttr(matmul_desc, MLUOP_MATMUL_ALLOW_TF32,
                                      &(allow_tf32), sizeof(int32_t)));
  return MLUOP_STATUS_SUCCESS;
}

inline std::string getTensorShapeString(const mluOpTensorDescriptor_t desc) {
  std::string res;
  res.push_back('[');
  for (int i = 0; i < desc->dim - 1; i++) {
    res.append(std::to_string(desc->dims[i]) + ',');
  }

  res.append(std::to_string(desc->dims[desc->dim - 1]) + ']');
  return res;
}

static void indiceConvFilterGencase(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t x_indice_desc,
    const void *x_indice, const mluOpTensorDescriptor_t diffy_indice_desc,
    const void *diffy_indice, const mluOpTensorDescriptor_t indice_pairs_desc,
    const void *indice_pairs, const int64_t indice_num[], const int64_t inverse,
    const int64_t subm, void *workspace, size_t workspace_size,
    const mluOpTensorDescriptor_t diffw_desc, void *diffw) {
  double indice_num_max =
      std::max(x_indice_desc->dims[0], diffy_indice_desc->dims[0]);
  GEN_CASE_START("indice_convolution_backward_filter");
  GEN_CASE_HANDLE(handle);
  GEN_CASE_DATA_REAL(true, "x_indice", x_indice, x_indice_desc);
  GEN_CASE_DATA_REAL(true, "diffy_indice", diffy_indice, diffy_indice_desc);
  GEN_CASE_DATA_REAL(true, "indice_pairs_desc", indice_pairs,
                     indice_pairs_desc);
  GEN_CASE_DATA_REAL(false, "diff_w", diffw, diffw_desc);
  GEN_CASE_OP_PARAM_SINGLE(0, "indice_convolution_backward", "inverse",
                           inverse);
  GEN_CASE_OP_PARAM_SINGLE(1, "indice_convolution_backward", "subm", subm);
  GEN_CASE_OP_PARAM_ARRAY(1, "indice_convolution_backward", "indice_num",
                          indice_num, indice_pairs_desc->dims[0]);
  GEN_CASE_HANDLE_PARAM();
  GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
}

// check input and diffy
static mluOpStatus_t indiceConvDtypeVaild(
    const std::string api_name, const mluOpTensorDescriptor_t x_indice_desc,
    const mluOpTensorDescriptor_t diffy_indice_desc,
    const mluOpTensorDescriptor_t indice_pairs_desc,
    const mluOpTensorDescriptor_t diffw_desc) {
  auto input_dtype = x_indice_desc->dtype;
  auto diffy_dtype = diffy_indice_desc->dtype;
  auto diffw_dtype = diffw_desc->dtype;
  auto pairs_dtype = indice_pairs_desc->dtype;
  if (pairs_dtype != MLUOP_DTYPE_INT32) {
    LOG(ERROR) << api_name
               << " indice_pairs_desc only supports data type int32. "
               << "But now the data type is "
               << mluop::getNameOfDataType(pairs_dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if (input_dtype != diffy_dtype || input_dtype != diffw_dtype ||
      !isFloatDtype(input_dtype) || !isFloatDtype(diffy_dtype) ||
      !isFloatDtype(diffw_dtype)) {
    LOG(ERROR) << api_name
               << " The data type of x_indice_desc, diffy_indice_desc "
               << "and diffw_desc should be same and both are half or float."
               << "But now the data types are "
               << mluop::getNameOfDataType(input_dtype) << "-"
               << mluop::getNameOfDataType(diffy_dtype) << "-"
               << mluop::getNameOfDataType(diffw_dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  auto input_on_dtype = x_indice_desc->onchip_dtype;
  auto diffy_on_dtype = diffy_indice_desc->onchip_dtype;
  auto diffw_on_dtype = diffw_desc->onchip_dtype;
  auto pairs_on_dtype = indice_pairs_desc->onchip_dtype;
  if ((MLUOP_DTYPE_INVALID != input_on_dtype &&
       input_on_dtype != input_dtype) ||
      (MLUOP_DTYPE_INVALID != diffy_on_dtype &&
       diffy_on_dtype != diffy_dtype) ||
      (MLUOP_DTYPE_INVALID != pairs_on_dtype &&
       pairs_on_dtype != pairs_dtype)) {
    LOG(ERROR)
        << api_name
        << " For x_indice_desc, diffy_indice_desc and indice_pairs_desc, "
        << "there is no need to set the on-chip data type, and if so, "
        << "it needs to be the same as their off-chip data type. "
        << "But now two data types of x_indice_desc are "
        << mluop::getNameOfDataType(input_dtype) << "-"
        << mluop::getNameOfDataType(input_on_dtype)
        << ", diffy_indice_desc are " << mluop::getNameOfDataType(diffy_dtype)
        << "-" << mluop::getNameOfDataType(diffy_on_dtype)
        << ", and indice_pairs_desc are "
        << mluop::getNameOfDataType(pairs_dtype) << "-"
        << mluop::getNameOfDataType(pairs_on_dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if ((diffw_on_dtype != MLUOP_DTYPE_INVALID &&
       !isFloatDtype(diffw_on_dtype)) ||
      (diffw_dtype == MLUOP_DTYPE_FLOAT &&
       diffw_on_dtype == MLUOP_DTYPE_HALF)) {
    LOG(ERROR)
        << api_name << " The on-chip data type of diffw_desc may not be set, "
        << "if it is set, only half or float types are supported, "
        << "and the bit width of on-chip data type can not be smaller than "
        << "that of off-chip data type. But now two data types of diffw_desc "
           "are "
        << mluop::getNameOfDataType(diffw_dtype) << "-"
        << mluop::getNameOfDataType(diffw_on_dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t baseParamCheck(
    const std::string api_name, mluOpHandle_t handle,
    const mluOpTensorDescriptor_t x_indice_desc,
    const mluOpTensorDescriptor_t diffy_indice_desc,
    const mluOpTensorDescriptor_t indice_pairs_desc,
    const mluOpTensorDescriptor_t diffw_desc, const int64_t indice_num[]) {
  PARAM_CHECK(api_name, handle != nullptr);
  PARAM_CHECK(api_name, x_indice_desc != nullptr);
  PARAM_CHECK(api_name, diffy_indice_desc != nullptr);
  PARAM_CHECK(api_name, indice_pairs_desc != nullptr);
  PARAM_CHECK(api_name, diffw_desc != nullptr);
  PARAM_CHECK(api_name, indice_num != nullptr);

  // check mlu platform
  if (handle->arch != MLUOP_MLU370 && handle->arch != MLUOP_MLU590) {
    LOG(ERROR) << api_name << " Only mlu300 and above devices are supported."
               << "Please check the device version!";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // check data type
  auto dtype_check =
      indiceConvDtypeVaild(api_name, x_indice_desc, diffy_indice_desc,
                           indice_pairs_desc, diffw_desc);
  if (MLUOP_STATUS_SUCCESS != dtype_check) {
    return dtype_check;
  }

  // check diffw layout
  auto diffw_layout = diffw_desc->layout;
  if (!(MLUOP_LAYOUT_HWCN == diffw_layout ||
        MLUOP_LAYOUT_NCHW == diffw_layout ||
        MLUOP_LAYOUT_NHWC == diffw_layout ||
        MLUOP_LAYOUT_NCDHW == diffw_layout ||
        MLUOP_LAYOUT_NDHWC == diffw_layout)) {
    LOG(ERROR) << api_name << " The layout of diffw_desc is "
               << mluop::getNameOfTensorLayout(diffw_layout)
               << ", which is unsupported.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // check shape
  auto ci = mluOpGetTensordimC(diffw_desc);
  auto co = mluOpGetTensordimN(diffw_desc);
  int32_t kd = diffw_desc->dim == 4 ? 1 : mluOpGetTensordimD(diffw_desc);
  int32_t kernel_volume =
      kd * mluOpGetTensordimH(diffw_desc) * mluOpGetTensordimW(diffw_desc);
  bool shape_check = true;
  if (2 != x_indice_desc->dim || 2 != diffy_indice_desc->dim ||
      3 != indice_pairs_desc->dim ||
      (4 != diffw_desc->dim && 5 != diffw_desc->dim)) {
    shape_check = false;  // dimension check failed!
  }

  if (ci != x_indice_desc->dims[1] || co != diffy_indice_desc->dims[1] ||
      kernel_volume != indice_pairs_desc->dims[0] ||
      2 != indice_pairs_desc->dims[1]) {
    shape_check = false;  // interdependent dimension check failed!
  }

  if (!shape_check) {
    LOG(ERROR) << api_name << " Shape check failed! "
               << "Now the shapes are x_indice_desc"
               << getTensorShapeString(x_indice_desc) << ", diffy_indice_desc"
               << getTensorShapeString(diffy_indice_desc)
               << ", indice_pairs_desc"
               << getTensorShapeString(indice_pairs_desc) << ", and diffw_desc"
               << getTensorShapeString(diffw_desc) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t insertTranspose(const std::string api_name,
                                     mluOpHandle_t handle,
                                     const mluOpTensorDescriptor_t diffw_desc,
                                     const void *diffw_temp, void *diffw_buffer,
                                     void *workspace, size_t *size,
                                     const bool is_get_workspace,
                                     const int kernel_volume, const int ci,
                                     const int co) {
  int trans_in_shape[3] = {kernel_volume, ci, co};
  int trans_out_shape[3] = {co, kernel_volume, ci};  // NHWC or NDHWC
  int permute[3] = {2, 0, 1};
  if (MLUOP_LAYOUT_NCHW == diffw_desc->layout ||
      MLUOP_LAYOUT_NCDHW == diffw_desc->layout) {
    trans_out_shape[0] = co;
    trans_out_shape[1] = ci;
    trans_out_shape[2] = kernel_volume;
    permute[0] = 2;
    permute[1] = 1;
    permute[2] = 0;
  }

  size_t transpose_workspace = 0;
  mluOpTensorDescriptor_t trans_in_desc, trans_out_desc;
  mluOpTransposeDescriptor_t trans_desc;
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&trans_in_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&trans_out_desc));
  CHECK_RETURN(api_name, mluOpCreateTransposeDescriptor(&trans_desc));
  CHECK_RETURN(api_name,
               mluOpSetTensorDescriptor(trans_in_desc, MLUOP_LAYOUT_ARRAY,
                                        diffw_desc->dtype, 3, trans_in_shape));
  CHECK_RETURN(api_name,
               mluOpSetTensorDescriptor(trans_out_desc, MLUOP_LAYOUT_ARRAY,
                                        diffw_desc->dtype, 3, trans_out_shape));
  CHECK_RETURN(api_name, mluOpSetTransposeDescriptor(trans_desc, 3, permute));
  CHECK_RETURN(api_name,
               mluOpGetTransposeWorkspaceSize(handle, trans_in_desc, trans_desc,
                                              &transpose_workspace));
  if (is_get_workspace) {  // is get workspace
    *size = transpose_workspace;
  } else {
    auto trans_status = mluOpTranspose_v2(
        handle, trans_desc, trans_in_desc, diffw_temp, trans_out_desc,
        diffw_buffer, workspace, transpose_workspace);
    KERNEL_CALL_CHECK(api_name, "mluOpTranspose_v2", trans_status, "");
  }

  CHECK_RETURN(api_name, mluOpDestroyTensorDescriptor(trans_in_desc));
  CHECK_RETURN(api_name, mluOpDestroyTensorDescriptor(trans_out_desc));
  CHECK_RETURN(api_name, mluOpDestroyTransposeDescriptor(trans_desc));
  return MLUOP_STATUS_SUCCESS;
}

// called by getWorkspace and compute api
//  it's from getWorkspace of workspace_size is not nullptr.
mluOpStatus_t internalIndiceConvBackwardFilter(
    const std::string api_name, mluOpHandle_t handle,
    const mluOpTensorDescriptor_t x_indice_desc, const void *x_indice,
    const mluOpTensorDescriptor_t diffy_indice_desc, const void *diffy_indice,
    const mluOpTensorDescriptor_t indice_pairs_desc, const void *indice_pairs,
    const int64_t indice_num[], void *workspace, size_t *workspace_size,
    const mluOpTensorDescriptor_t diffw_desc, void *diffw) {
  bool is_get_workspace = workspace_size != nullptr ? true : false;
  bool diffw_need_trans = true;
  if (MLUOP_LAYOUT_HWCN == diffw_desc->layout) {
    diffw_need_trans = false;
  }

  // call gather_nd and matmul to finish indice conv.
  int32_t kernel_volume = indice_pairs_desc->dims[0];
  int32_t ci = mluOpGetTensordimC(diffw_desc);
  int32_t co = mluOpGetTensordimN(diffw_desc);
  int32_t max_active_num = 0;
  for (int i = 0; i < kernel_volume; ++i) {
    max_active_num =
        indice_num[i] > max_active_num ? indice_num[i] : max_active_num;
  }

  int64_t max_input_size =
      max_active_num * ci * mluop::getSizeOfDataType(x_indice_desc->dtype);
  int64_t max_diffy_size =
      max_active_num * co * mluop::getSizeOfDataType(x_indice_desc->dtype);
  int64_t diffw_trans_size =
      diffw_need_trans ? diffw_desc->total_tensor_size : 0;

  void *diffw_temp = diffw_need_trans ? workspace : diffw;
  void *input_temp = (char *)workspace + diffw_trans_size;
  void *diffy_temp = (char *)input_temp + max_input_size;
  void *matmul_ws = (char *)diffy_temp + max_diffy_size;

  // create temp tensor for gather and matmul
  mluOpTensorDescriptor_t active_indice_desc;
  mluOpTensorDescriptor_t matmul_a_desc, matmul_b_desc, matmul_c_desc;
  mluOpMatMulDescriptor_t matmul_desc;
  mluOpMatMulAlgo_t matmul_algo;
  mluOpMatMulHeuristicResult_t heuristic_result;
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&active_indice_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_a_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_b_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_c_desc));
  CHECK_RETURN(api_name, mluOpMatMulDescCreate(&matmul_desc));
  CHECK_RETURN(api_name, mluOpMatMulAlgoCreate(&matmul_algo));
  CHECK_RETURN(api_name, mluOpCreateMatMulHeuristicResult(&heuristic_result));
  CHECK_RETURN(api_name,
               setMatmulDescInfo(api_name, matmul_desc, 1, 0,
                                 (uint32_t)getOnchipDataType(diffw_desc), 0));
  int requested_algo_count = 1, return_algo_count = 0;
  float alpha = 1.0, beta = 0.0, fill_value = 0;
  size_t matmul_ws_size = 0, temp_matmul_size = 0;

  // diffw fill for unused kernel
  if (!is_get_workspace) {
    auto fill_status = mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST,
                                    &fill_value, diffw_desc, diffw_temp);
    KERNEL_CALL_CHECK(api_name, "mluOpFill_v3", fill_status, "");
  }

  int64_t in_active_num = indice_pairs_desc->dims[2];
  int64_t cico_size = ci * co * mluop::getSizeOfDataType(diffw_desc->dtype);
  int64_t pair_low_size =
      in_active_num * mluop::getSizeOfDataType(indice_pairs_desc->dtype);

  for (int i = 0; i < kernel_volume; ++i) {
    int32_t active_point_num = indice_num[i];
    if (active_point_num <= 0) {
      continue;
    }

    int32_t active_indices[2] = {active_point_num, 1};
    int32_t a_desc_dims[2] = {active_point_num, ci};
    int32_t b_desc_dims[2] = {active_point_num, co};
    int32_t c_desc_dims[2] = {ci, co};
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               active_indice_desc, MLUOP_LAYOUT_ARRAY,
                               indice_pairs_desc->dtype, 2, active_indices));
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               matmul_a_desc, MLUOP_LAYOUT_ARRAY,
                               x_indice_desc->dtype, 2, a_desc_dims));
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               matmul_b_desc, MLUOP_LAYOUT_ARRAY,
                               diffy_indice_desc->dtype, 2, b_desc_dims));
    CHECK_RETURN(api_name,
                 mluOpSetTensorDescriptor(matmul_c_desc, MLUOP_LAYOUT_ARRAY,
                                          diffw_desc->dtype, 2, c_desc_dims));
    CHECK_RETURN(api_name, mluOpGetMatMulAlgoHeuristic(
                               handle, matmul_desc, matmul_a_desc,
                               matmul_b_desc, matmul_c_desc, matmul_c_desc,
                               nullptr, requested_algo_count, &heuristic_result,
                               &return_algo_count));
    CHECK_RETURN(api_name,
                 mluOpGetMatMulHeuristicResult(heuristic_result, matmul_algo,
                                               &temp_matmul_size));

    if (is_get_workspace) {
      matmul_ws_size =
          temp_matmul_size > matmul_ws_size ? temp_matmul_size : matmul_ws_size;
    } else {
      void *diffw_buffer = (char *)diffw_temp + i * cico_size;
      void *gather_input_indice = (char *)indice_pairs + i * 2 * pair_low_size;
      void *gather_diffy_indice =
          (char *)indice_pairs + i * 2 * pair_low_size + pair_low_size;
      // gather activate input data [n, ci]
      auto gather_x_status =
          mluOpGatherNd(handle, x_indice_desc, x_indice, active_indice_desc,
                        gather_input_indice, matmul_a_desc, input_temp);
      KERNEL_CALL_CHECK(api_name, "mluOpGatherNd", gather_x_status, "");
      // gatehr activate diffy data [n, co]
      auto gather_dy_status = mluOpGatherNd(
          handle, diffy_indice_desc, diffy_indice, active_indice_desc,
          gather_diffy_indice, matmul_b_desc, diffy_temp);
      KERNEL_CALL_CHECK(api_name, "mluOpGatherNd", gather_dy_status, "");
      // get part diffw [ci, co]
      auto matmul_status = mluOpMatMul_v2(
          handle, matmul_desc, matmul_algo, &alpha, matmul_a_desc, input_temp,
          matmul_b_desc, diffy_temp, &beta, matmul_c_desc, diffw_buffer,
          matmul_ws, temp_matmul_size, matmul_c_desc, diffw_buffer);
      KERNEL_CALL_CHECK(api_name, "mluOpMatMul_v2", matmul_status, "");
    }
  }

  // trans temp diffw if needed
  uint64_t trans_ws_size = 0;
  if (diffw_need_trans) {
    void *trans_ws = input_temp;  // multiplexing of space
    CHECK_RETURN(api_name,
                 insertTranspose(api_name, handle, diffw_desc, diffw_temp,
                                 diffw, trans_ws, &trans_ws_size,
                                 is_get_workspace, kernel_volume, ci, co));
  }

  if (is_get_workspace) {
    *workspace_size = diffw_trans_size +
                      std::max(trans_ws_size, max_input_size + max_diffy_size +
                                                  matmul_ws_size);
  }

  CHECK_RETURN(api_name, mluOpDestroyTensorDescriptor(active_indice_desc));
  CHECK_RETURN(api_name, mluOpDestroyTensorDescriptor(matmul_a_desc));
  CHECK_RETURN(api_name, mluOpDestroyTensorDescriptor(matmul_b_desc));
  CHECK_RETURN(api_name, mluOpDestroyTensorDescriptor(matmul_c_desc));
  CHECK_RETURN(api_name, mluOpMatMulDescDestroy(matmul_desc));
  CHECK_RETURN(api_name, mluOpMatMulAlgoDestroy(matmul_algo));
  CHECK_RETURN(api_name, mluOpDestroyMatMulHeuristicResult(heuristic_result));
  return MLUOP_STATUS_SUCCESS;
}

/***************** workspace **************************/
/*| temp diffw | temp input | temp diffy| matmul_ws | */
/*| temp diffw | transpose_ws | */
/* multiplexing of space:(transpose_ws, temp_input + temp_diffy +
 * matmul_ws) */
mluOpStatus_t MLUOP_WIN_API
mluOpGetIndiceConvolutionBackwardFilterWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t x_indice_desc,
    const mluOpTensorDescriptor_t diffy_indice_desc,
    const mluOpTensorDescriptor_t indice_pairs_desc,
    const mluOpTensorDescriptor_t diffw_desc, const int64_t indice_num[],
    const int64_t inverse, const int64_t subm, size_t *size) {
  const std::string api_name =
      "[mluOpGetIndiceConvolutionBackwardFilterWorkspaceSize]";
  auto basic_check =
      baseParamCheck(api_name, handle, x_indice_desc, diffy_indice_desc,
                     indice_pairs_desc, diffw_desc, indice_num);
  if (MLUOP_STATUS_SUCCESS != basic_check) {
    return basic_check;
  }

  // zero element check
  if (0 == x_indice_desc->total_element_num ||
      0 == diffy_indice_desc->total_element_num ||
      0 == indice_pairs_desc->total_element_num ||
      0 == diffw_desc->total_element_num) {
    VLOG(5) << api_name << " Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  CHECK_RETURN(api_name,
               internalIndiceConvBackwardFilter(
                   api_name, handle, x_indice_desc, nullptr, diffy_indice_desc,
                   nullptr, indice_pairs_desc, nullptr, indice_num, nullptr,
                   size, diffw_desc, nullptr));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpIndiceConvolutionBackwardFilter(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t x_indice_desc,
    const void *x_indice, const mluOpTensorDescriptor_t diffy_indice_desc,
    const void *diffy_indice, const mluOpTensorDescriptor_t indice_pairs_desc,
    const void *indice_pairs, const int64_t indice_num[], const int64_t inverse,
    const int64_t subm, void *workspace, size_t workspace_size,
    const mluOpTensorDescriptor_t diffw_desc, void *diffw) {
  const std::string api_name = "[mluOpIndiceConvolutionBackwardFilter]";

  auto basic_check =
      baseParamCheck(api_name, handle, x_indice_desc, diffy_indice_desc,
                     indice_pairs_desc, diffw_desc, indice_num);
  if (MLUOP_STATUS_SUCCESS != basic_check) {
    return basic_check;
  }

  // zero element check
  if (0 == x_indice_desc->total_element_num ||
      0 == diffy_indice_desc->total_element_num ||
      0 == indice_pairs_desc->total_element_num ||
      0 == diffw_desc->total_element_num) {
    VLOG(5) << api_name << " Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  // check data ptr
  PARAM_CHECK(api_name, x_indice != nullptr);
  PARAM_CHECK(api_name, diffy_indice != nullptr);
  PARAM_CHECK(api_name, indice_pairs != nullptr);
  PARAM_CHECK(api_name, diffw != nullptr);
  if (workspace_size > 0) {
    PARAM_CHECK(api_name, workspace != nullptr);
  }

  // gen_case
  if (MLUOP_GEN_CASE_ON_NEW) {
    indiceConvFilterGencase(handle, x_indice_desc, x_indice, diffy_indice_desc,
                            diffy_indice, indice_pairs_desc, indice_pairs,
                            indice_num, inverse, subm, workspace,
                            workspace_size, diffw_desc, diffw);
  }

  CHECK_RETURN(api_name,
               internalIndiceConvBackwardFilter(
                   api_name, handle, x_indice_desc, x_indice, diffy_indice_desc,
                   diffy_indice, indice_pairs_desc, indice_pairs, indice_num,
                   workspace, nullptr, diffw_desc, diffw));

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
