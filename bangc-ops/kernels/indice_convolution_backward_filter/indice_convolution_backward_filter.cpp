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
  for (int32_t i = 0; i < desc->dim - 1; i++) {
    res.append(std::to_string(desc->dims[i]) + ',');
  }
  res.append(std::to_string(desc->dims[desc->dim - 1]) + ']');
  return res;
}

static void indiceConvFilterGencase(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t features_desc,
    const void *features, const mluOpTensorDescriptor_t output_grad_desc,
    const void *output_grad, const mluOpTensorDescriptor_t indice_pairs_desc,
    const void *indice_pairs, const int64_t indice_num[], const int64_t inverse,
    const int64_t subm, void *workspace, size_t workspace_size,
    const mluOpTensorDescriptor_t filters_grad_desc, void *filters_grad) {
  GEN_CASE_START("indice_convolution_backward_filter");
  GEN_CASE_HANDLE(handle);
  GEN_CASE_DATA_REAL(true, "features", features, features_desc);
  GEN_CASE_DATA_REAL(true, "output_grad", output_grad, output_grad_desc);
  GEN_CASE_DATA_REAL(true, "indice_pairs_desc", indice_pairs,
                     indice_pairs_desc);
  GEN_CASE_DATA_REAL(false, "diff_w", filters_grad, filters_grad_desc);
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
    const std::string api_name, const mluOpTensorDescriptor_t features_desc,
    const mluOpTensorDescriptor_t output_grad_desc,
    const mluOpTensorDescriptor_t indice_pairs_desc,
    const mluOpTensorDescriptor_t filters_grad_desc) {
  auto input_dtype = features_desc->dtype;
  auto diffy_dtype = output_grad_desc->dtype;
  auto filters_grad_dtype = filters_grad_desc->dtype;
  auto pairs_dtype = indice_pairs_desc->dtype;
  if (pairs_dtype != MLUOP_DTYPE_INT32) {
    LOG(ERROR) << api_name
               << " indice_pairs_desc only supports data type int32. "
               << "But now the data type is "
               << mluop::getNameOfDataType(pairs_dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if (input_dtype != diffy_dtype || input_dtype != filters_grad_dtype ||
      !isFloatDtype(input_dtype) || !isFloatDtype(diffy_dtype) ||
      !isFloatDtype(filters_grad_dtype)) {
    LOG(ERROR)
        << api_name << " The data type of features_desc, output_grad_desc "
        << "and filters_grad_desc should be the same and the three should "
        << "be either half or float. But now the data types are "
        << mluop::getNameOfDataType(input_dtype) << "-"
        << mluop::getNameOfDataType(diffy_dtype) << "-"
        << mluop::getNameOfDataType(filters_grad_dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  auto input_on_dtype = features_desc->onchip_dtype;
  auto diffy_on_dtype = output_grad_desc->onchip_dtype;
  auto filters_grad_on_dtype = filters_grad_desc->onchip_dtype;
  auto pairs_on_dtype = indice_pairs_desc->onchip_dtype;
  if ((MLUOP_DTYPE_INVALID != input_on_dtype &&
       input_on_dtype != input_dtype) ||
      (MLUOP_DTYPE_INVALID != diffy_on_dtype &&
       diffy_on_dtype != diffy_dtype) ||
      (MLUOP_DTYPE_INVALID != pairs_on_dtype &&
       pairs_on_dtype != pairs_dtype)) {
    LOG(ERROR) << api_name
               << " For features_desc, output_grad_desc and indice_pairs_desc, "
               << "there is no need to set the on-chip data type, and if so, "
               << "it needs to be the same as their off-chip data type. "
               << "But now two data types of features_desc are "
               << mluop::getNameOfDataType(input_dtype) << "-"
               << mluop::getNameOfDataType(input_on_dtype)
               << ", output_grad_desc are "
               << mluop::getNameOfDataType(diffy_dtype) << "-"
               << mluop::getNameOfDataType(diffy_on_dtype)
               << ", and indice_pairs_desc are "
               << mluop::getNameOfDataType(pairs_dtype) << "-"
               << mluop::getNameOfDataType(pairs_on_dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if ((filters_grad_on_dtype != MLUOP_DTYPE_INVALID &&
       !isFloatDtype(filters_grad_on_dtype)) ||
      (filters_grad_dtype == MLUOP_DTYPE_FLOAT &&
       filters_grad_on_dtype == MLUOP_DTYPE_HALF)) {
    LOG(ERROR) << api_name << " The on-chip data type of filters_grad_desc "
               << "may not be set, if it is set, only half or float types are "
               << "supported, and the bit width of on-chip data type can not "
               << "be smaller than that of off-chip data type. But now two "
               << "data types of filters_grad_desc are "
               << mluop::getNameOfDataType(filters_grad_dtype) << "-"
               << mluop::getNameOfDataType(filters_grad_on_dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t baseParamCheck(
    const std::string api_name, mluOpHandle_t handle,
    const mluOpTensorDescriptor_t features_desc,
    const mluOpTensorDescriptor_t output_grad_desc,
    const mluOpTensorDescriptor_t indice_pairs_desc,
    const mluOpTensorDescriptor_t filters_grad_desc,
    const int64_t indice_num[], const int64_t inverse) {
  PARAM_CHECK(api_name, handle != nullptr);
  PARAM_CHECK(api_name, features_desc != nullptr);
  PARAM_CHECK(api_name, output_grad_desc != nullptr);
  PARAM_CHECK(api_name, indice_pairs_desc != nullptr);
  PARAM_CHECK(api_name, filters_grad_desc != nullptr);
  PARAM_CHECK(api_name, indice_num != nullptr);
  PARAM_CHECK(api_name, inverse == 0);

  // check mlu platform
  if (handle->arch < 372) {
    LOG(ERROR) << api_name << " Only mlu300 and above devices are supported."
               << " Please check the device version!";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  // check data type
  auto dtype_check =
      indiceConvDtypeVaild(api_name, features_desc, output_grad_desc,
                           indice_pairs_desc, filters_grad_desc);
  if (MLUOP_STATUS_SUCCESS != dtype_check) {
    return dtype_check;
  }

  if (mluOpGetTensorElementNum(features_desc) >= LARGE_TENSOR_NUM ||
      mluOpGetTensorElementNum(output_grad_desc) >= LARGE_TENSOR_NUM ||
      mluOpGetTensorElementNum(indice_pairs_desc) >= LARGE_TENSOR_NUM ||
      mluOpGetTensorElementNum(filters_grad_desc) >= LARGE_TENSOR_NUM) {
    LOG(ERROR) << api_name << " Overflow max tensor num."
               << " Currently, MLU-OPS supports tensor num smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  bool shape_check = true;
  if (2 != features_desc->dim || 2 != output_grad_desc->dim ||
      3 != indice_pairs_desc->dim ||
      (4 != filters_grad_desc->dim && 5 != filters_grad_desc->dim)) {
    shape_check = false;  // dimension check failed!
  }

  // only DHWCN/HWCN layout of filter_grad is supported, currently
  int32_t filter_dim_len = filters_grad_desc->dim;
  auto ci = filters_grad_desc->dims[filter_dim_len - 2];
  auto co = filters_grad_desc->dims[filter_dim_len - 1];
  auto kd = filter_dim_len == 4 ? 1 : filters_grad_desc->dims[0];
  auto kh = filter_dim_len == 4 ? filters_grad_desc->dims[0]
                                : filters_grad_desc->dims[1];
  auto kw = filter_dim_len == 4 ? filters_grad_desc->dims[1]
                                : filters_grad_desc->dims[2];
  if (ci != features_desc->dims[1] || co != output_grad_desc->dims[1] ||
      features_desc->dims[0] != indice_pairs_desc->dims[2] ||
      2 != indice_pairs_desc->dims[1] ||
      kd * kh * kw != indice_pairs_desc->dims[0]) {
    shape_check = false;  // interdependent dimension check failed!
  }

  if (!shape_check) {
    LOG(ERROR) << api_name << " Shape check failed! "
               << "Now the shapes are features_desc"
               << getTensorShapeString(features_desc) << ", output_grad_desc"
               << getTensorShapeString(output_grad_desc)
               << ", indice_pairs_desc"
               << getTensorShapeString(indice_pairs_desc)
               << ", and filters_grad_desc"
               << getTensorShapeString(filters_grad_desc) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t insertTranspose(
    const std::string api_name, mluOpHandle_t handle,
    const mluOpTensorDescriptor_t filters_grad_desc,
    const void *filters_grad_temp, void *filters_grad_buffer, void *workspace,
    size_t *size, const bool is_get_workspace, const int32_t kernel_volume,
    const int32_t ci, const int32_t co) {
  int32_t trans_in_shape[3] = {kernel_volume, ci, co};
  int32_t trans_out_shape[3] = {co, kernel_volume, ci};  // NHWC or NDHWC
  int32_t permute[3] = {2, 0, 1};
  if (MLUOP_LAYOUT_NCHW == filters_grad_desc->layout ||
      MLUOP_LAYOUT_NCDHW == filters_grad_desc->layout) {
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
  CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                             trans_in_desc, MLUOP_LAYOUT_ARRAY,
                             filters_grad_desc->dtype, 3, trans_in_shape));
  CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                             trans_out_desc, MLUOP_LAYOUT_ARRAY,
                             filters_grad_desc->dtype, 3, trans_out_shape));
  CHECK_RETURN(api_name, mluOpSetTransposeDescriptor(trans_desc, 3, permute));
  CHECK_RETURN(api_name,
               mluOpGetTransposeWorkspaceSize(handle, trans_in_desc, trans_desc,
                                              &transpose_workspace));
  if (is_get_workspace) {  // is get workspace
    *size = transpose_workspace;
  } else {
    auto trans_status = mluOpTranspose_v2(
        handle, trans_desc, trans_in_desc, filters_grad_temp, trans_out_desc,
        filters_grad_buffer, workspace, transpose_workspace);
    KERNEL_CALL_CHECK(api_name, "mluOpTranspose_v2", trans_status, "");
  }

  CHECK_RETURN(api_name, mluOpDestroyTensorDescriptor(trans_in_desc));
  CHECK_RETURN(api_name, mluOpDestroyTensorDescriptor(trans_out_desc));
  CHECK_RETURN(api_name, mluOpDestroyTransposeDescriptor(trans_desc));
  return MLUOP_STATUS_SUCCESS;
}

// called by getWorkspace and compute api
// workspace_size is not nullptr when it's from getWorkspace api.
static mluOpStatus_t internalIndiceConvBackwardFilter(
    const std::string api_name, mluOpHandle_t handle,
    const mluOpTensorDescriptor_t features_desc, const void *features,
    const mluOpTensorDescriptor_t output_grad_desc, const void *output_grad,
    const mluOpTensorDescriptor_t indice_pairs_desc, const void *indice_pairs,
    const int64_t indice_num[], void *workspace, size_t *workspace_size,
    const mluOpTensorDescriptor_t filters_grad_desc, void *filters_grad) {
  bool is_get_workspace = workspace_size != nullptr ? true : false;
  bool filters_grad_need_trans = false;

  // call gather_nd and matmul to finish indice conv.
  int32_t kernel_volume = indice_pairs_desc->dims[0];
  int32_t ci = features_desc->dims[1];
  int32_t co = output_grad_desc->dims[1];
  int32_t max_active_num = 0;
  for (int32_t i = 0; i < kernel_volume; ++i) {
    max_active_num =
        indice_num[i] > max_active_num ? indice_num[i] : max_active_num;
  }

  int64_t max_input_size =
      max_active_num * ci * mluop::getSizeOfDataType(features_desc->dtype);
  int64_t max_diffy_size =
      max_active_num * co * mluop::getSizeOfDataType(features_desc->dtype);
  int64_t filters_grad_trans_size =
      filters_grad_need_trans ? filters_grad_desc->total_tensor_size : 0;

  void *filters_grad_temp = filters_grad_need_trans ? workspace : filters_grad;
  void *input_temp = (char *)workspace + filters_grad_trans_size;
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
  CHECK_RETURN(
      api_name,
      setMatmulDescInfo(api_name, matmul_desc, 1, 0,
                        (uint32_t)getOnchipDataType(filters_grad_desc), 0));
  int32_t requested_algo_count = 1, return_algo_count = 0;
  float alpha = 1.0, beta = 0.0, fill_value = 0;
  size_t matmul_ws_size = 0, temp_matmul_size = 0;

  // filters_grad fill for unused kernel
  if (!is_get_workspace) {
    auto fill_status =
        mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST, &fill_value,
                     filters_grad_desc, filters_grad_temp);
    KERNEL_CALL_CHECK(api_name, "mluOpFill_v3", fill_status, "");
  }

  int64_t in_active_num = indice_pairs_desc->dims[2];
  int64_t cico_size =
      ci * co * mluop::getSizeOfDataType(filters_grad_desc->dtype);
  int64_t pair_low_size =
      in_active_num * mluop::getSizeOfDataType(indice_pairs_desc->dtype);

  for (int32_t i = 0; i < kernel_volume; ++i) {
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
                               features_desc->dtype, 2, a_desc_dims));
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               matmul_b_desc, MLUOP_LAYOUT_ARRAY,
                               output_grad_desc->dtype, 2, b_desc_dims));
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               matmul_c_desc, MLUOP_LAYOUT_ARRAY,
                               filters_grad_desc->dtype, 2, c_desc_dims));
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
      void *filters_grad_buffer = (char *)filters_grad_temp + i * cico_size;
      void *gather_input_indice = (char *)indice_pairs + i * 2 * pair_low_size;
      void *gather_output_grad =
          (char *)indice_pairs + i * 2 * pair_low_size + pair_low_size;
      // gather activate input data [n, ci]
      auto gather_x_status =
          mluOpGatherNd(handle, features_desc, features, active_indice_desc,
                        gather_input_indice, matmul_a_desc, input_temp);
      KERNEL_CALL_CHECK(api_name, "mluOpGatherNd", gather_x_status, "");
      // gatehr activate diffy data [n, co]
      auto gather_dy_status = mluOpGatherNd(
          handle, output_grad_desc, output_grad, active_indice_desc,
          gather_output_grad, matmul_b_desc, diffy_temp);
      KERNEL_CALL_CHECK(api_name, "mluOpGatherNd", gather_dy_status, "");
      // get part filters_grad [ci, co]
      auto matmul_status = mluOpMatMul_v2(
          handle, matmul_desc, matmul_algo, &alpha, matmul_a_desc, input_temp,
          matmul_b_desc, diffy_temp, &beta, matmul_c_desc, filters_grad_buffer,
          matmul_ws, temp_matmul_size, matmul_c_desc, filters_grad_buffer);
      KERNEL_CALL_CHECK(api_name, "mluOpMatMul_v2", matmul_status, "");
    }
  }

  // trans temp filters_grad if needed
  uint64_t trans_ws_size = 0;
  if (filters_grad_need_trans) {
    void *trans_ws = input_temp;  // multiplexing of space
    CHECK_RETURN(
        api_name,
        insertTranspose(api_name, handle, filters_grad_desc, filters_grad_temp,
                        filters_grad, trans_ws, &trans_ws_size,
                        is_get_workspace, kernel_volume, ci, co));
  }

  if (is_get_workspace) {
    *workspace_size = filters_grad_trans_size +
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
/*| temp filters_grad | temp features | temp output_grad| matmul_ws | */
/*| temp filters_grad | transpose_ws | */
/* multiplexing of space:(transpose_ws, temp_input + temp_diffy +
 * matmul_ws) */
mluOpStatus_t MLUOP_WIN_API
mluOpGetIndiceConvolutionBackwardFilterWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t features_desc,
    const mluOpTensorDescriptor_t output_grad_desc,
    const mluOpTensorDescriptor_t indice_pairs_desc,
    const mluOpTensorDescriptor_t filters_grad_desc, const int64_t indice_num[],
    const int64_t inverse, const int64_t subm, size_t *size) {
  const std::string api_name =
      "[mluOpGetIndiceConvolutionBackwardFilterWorkspaceSize]";
  PARAM_CHECK(api_name, size != nullptr);
  auto basic_check =
      baseParamCheck(api_name, handle, features_desc, output_grad_desc,
                     indice_pairs_desc, filters_grad_desc, indice_num, inverse);
  if (MLUOP_STATUS_SUCCESS != basic_check) {
    return basic_check;
  }

  // zero element check
  if (0 == features_desc->total_element_num ||
      0 == output_grad_desc->total_element_num ||
      0 == indice_pairs_desc->total_element_num ||
      0 == filters_grad_desc->total_element_num) {
    VLOG(5) << api_name << " Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  CHECK_RETURN(api_name,
               internalIndiceConvBackwardFilter(
                   api_name, handle, features_desc, nullptr, output_grad_desc,
                   nullptr, indice_pairs_desc, nullptr, indice_num, nullptr,
                   size, filters_grad_desc, nullptr));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpIndiceConvolutionBackwardFilter(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t features_desc,
    const void *features, const mluOpTensorDescriptor_t output_grad_desc,
    const void *output_grad, const mluOpTensorDescriptor_t indice_pairs_desc,
    const void *indice_pairs, const int64_t indice_num[], const int64_t inverse,
    const int64_t subm, void *workspace, size_t workspace_size,
    const mluOpTensorDescriptor_t filters_grad_desc, void *filters_grad) {
  const std::string api_name = "[mluOpIndiceConvolutionBackwardFilter]";

  auto basic_check =
      baseParamCheck(api_name, handle, features_desc, output_grad_desc,
                     indice_pairs_desc, filters_grad_desc, indice_num, inverse);
  if (MLUOP_STATUS_SUCCESS != basic_check) {
    return basic_check;
  }

  // zero element check
  if (0 == features_desc->total_element_num ||
      0 == output_grad_desc->total_element_num ||
      0 == indice_pairs_desc->total_element_num ||
      0 == filters_grad_desc->total_element_num) {
    VLOG(5) << api_name << " Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  // check data ptr
  PARAM_CHECK(api_name, features != nullptr);
  PARAM_CHECK(api_name, output_grad != nullptr);
  PARAM_CHECK(api_name, indice_pairs != nullptr);
  PARAM_CHECK(api_name, filters_grad != nullptr);
  if (workspace_size > 0) {
    PARAM_CHECK(api_name, workspace != nullptr);
  }

  // gen_case
  if (MLUOP_GEN_CASE_ON_NEW) {
    indiceConvFilterGencase(handle, features_desc, features, output_grad_desc,
                            output_grad, indice_pairs_desc, indice_pairs,
                            indice_num, inverse, subm, workspace,
                            workspace_size, filters_grad_desc, filters_grad);
  }

  CHECK_RETURN(api_name,
               internalIndiceConvBackwardFilter(
                   api_name, handle, features_desc, features, output_grad_desc,
                   output_grad, indice_pairs_desc, indice_pairs, indice_num,
                   workspace, nullptr, filters_grad_desc, filters_grad));

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
