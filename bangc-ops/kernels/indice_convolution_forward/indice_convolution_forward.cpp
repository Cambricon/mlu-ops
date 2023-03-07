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
#include "core/type.h"
#include "kernels/kernel.h"

static mluOpStatus_t foolProof(
    const std::string api_name, mluOpHandle_t handle,
    const mluOpTensorDescriptor_t features_desc,
    const mluOpTensorDescriptor_t filters_desc,
    const mluOpTensorDescriptor_t indice_pairs_desc, const int64_t indice_num[],
    const int64_t num_act_out, const int64_t inverse, const int64_t sub_m,
    const mluOpTensorDescriptor_t features_out_desc) {
  // nullptr check
  PARAM_CHECK(api_name, handle != nullptr);
  PARAM_CHECK(api_name, features_desc != nullptr);
  PARAM_CHECK(api_name, filters_desc != nullptr);
  PARAM_CHECK(api_name, indice_pairs_desc != nullptr);
  PARAM_CHECK(api_name, indice_num != nullptr);
  PARAM_CHECK(api_name, features_out_desc != nullptr);

  // platform check
  if (handle->arch < 372) {
    LOG(ERROR) << api_name << "Only mlu300 and above devices are supported."
               << "Please check the device version!";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  // data type check
  PARAM_CHECK(api_name, features_desc->dtype == MLUOP_DTYPE_FLOAT ||
                            features_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK(api_name, filters_desc->dtype == MLUOP_DTYPE_FLOAT ||
                            filters_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK(api_name, indice_pairs_desc->dtype == MLUOP_DTYPE_INT32);
  PARAM_CHECK(api_name, features_out_desc->dtype == MLUOP_DTYPE_FLOAT ||
                            features_out_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK(api_name, features_desc->dtype == features_out_desc->dtype &&
                            features_desc->dtype == filters_desc->dtype);

  // inverse not supported now
  PARAM_CHECK(api_name, sub_m == 0 || sub_m == 1);
  PARAM_CHECK(api_name, inverse == 0 || inverse == 1);
  if (inverse != 0) {
    LOG(ERROR) << api_name << "inverse is: " << inverse
               << ", which is not supported now.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // layout check
  // DHWCN layout not supported yet, use ARRAY temporarily
  // PARAM_CHECK(api_name, filters_desc->layout == MLUOP_LAYOUT_DHWCN);
  if (filters_desc->layout != MLUOP_LAYOUT_NDHWC &&
      filters_desc->layout != MLUOP_LAYOUT_NCDHW &&
      filters_desc->layout != MLUOP_LAYOUT_ARRAY) {
    LOG(ERROR) << api_name << "The layout of filters is: "
               << mluop::getNameOfTensorLayout(filters_desc->layout)
               << ", which is not supported now.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // shape check
  PARAM_CHECK(api_name, features_desc->dim == 2);
  PARAM_CHECK(api_name, indice_pairs_desc->dim == 3);
  PARAM_CHECK(api_name, features_out_desc->dim == 2);
  if (filters_desc->dim != 5) {
    LOG(ERROR) << api_name
               << "The filters dimension number only support 5 currently,"
               << " but filters dimension number is :" << filters_desc->dim
               << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // large tensor
  if (mluOpGetTensorElementNum(features_desc) >= LARGE_TENSOR_NUM ||
      mluOpGetTensorElementNum(filters_desc) >= LARGE_TENSOR_NUM ||
      mluOpGetTensorElementNum(indice_pairs_desc) >= LARGE_TENSOR_NUM ||
      mluOpGetTensorElementNum(features_out_desc) >= LARGE_TENSOR_NUM) {
    LOG(ERROR) << api_name << "Max tensor number overflow. Currently, "
               << "MLU-OPS supports tensor elemenets number smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  auto ci = 0;
  auto num_filter = 0;
  auto co = 0;
  if (filters_desc->layout == MLUOP_LAYOUT_ARRAY) {
    ci = filters_desc->dims[3];
    num_filter =
        filters_desc->dims[0] * filters_desc->dims[1] * filters_desc->dims[2];
    co = filters_desc->dims[4];
  } else {
    ci = mluOpGetTensordimC(filters_desc);
    num_filter = mluOpGetTensordimD(filters_desc) *
                 mluOpGetTensordimH(filters_desc) *
                 mluOpGetTensordimW(filters_desc);
    co = mluOpGetTensordimN(filters_desc);
  }

  // features shape check
  PARAM_CHECK(api_name, features_desc->dims[0] == indice_pairs_desc->dims[2]);
  PARAM_CHECK(api_name, features_desc->dims[1] == ci);

  // indice_pairs shape check
  PARAM_CHECK(api_name, indice_pairs_desc->dims[0] == num_filter);
  PARAM_CHECK(api_name, indice_pairs_desc->dims[1] == 2);

  // features_out shape check
  PARAM_CHECK(api_name, features_out_desc->dims[0] == num_act_out);
  PARAM_CHECK(api_name, features_out_desc->dims[1] == co);

  // indice_num[] check
  for (int i = 0; i < num_filter; ++i) {
    PARAM_CHECK(api_name,
                indice_num[i] >= 0 && indice_num[i] <= features_desc->dims[0]);
  }

  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t mainIndiceConvolutionForward(
    const std::string api_name, mluOpHandle_t handle,
    const mluOpTensorDescriptor_t features_desc, const void *features,
    const mluOpTensorDescriptor_t filters_desc, const void *filters,
    const mluOpTensorDescriptor_t indice_pairs_desc, const void *indice_pairs,
    const int64_t indice_num[], const int64_t num_act_out, void *workspace,
    size_t *workspace_size, const mluOpTensorDescriptor_t features_out_desc,
    void *features_out) {
  // param init
  bool is_workspace_compute = workspace_size != nullptr ? true : false;
  bool filters_need_trans = true;
  int32_t ci = 0;
  int32_t co = 0;
  // MLUOP_LAYOUT_DHWCN not supported yet.
  if (filters_desc->layout == MLUOP_LAYOUT_ARRAY) {
    filters_need_trans = false;
    ci = filters_desc->dims[3];
    co = filters_desc->dims[4];
  } else {
    ci = mluOpGetTensordimC(filters_desc);
    co = mluOpGetTensordimN(filters_desc);
  }
  int32_t num_filter = indice_pairs_desc->dims[0];

  int64_t num_act_in = indice_pairs_desc->dims[2];
  int64_t elementSize_filters =
      ci * co * mluop::getSizeOfDataType(filters_desc->dtype);
  int64_t elementSize_indice_pairs =
      num_act_in * mluop::getSizeOfDataType(indice_pairs_desc->dtype);

  int32_t max_indice_num = 0;
  for (int i = 0; i < num_filter; ++i) {
    max_indice_num =
        indice_num[i] > max_indice_num ? indice_num[i] : max_indice_num;
  }
  size_t workspaceSize_gather =
      max_indice_num * ci * mluop::getSizeOfDataType(features_desc->dtype);
  size_t workspaceSize_matmul =
      max_indice_num * co * mluop::getSizeOfDataType(features_out_desc->dtype);
  size_t workspaceSize_transpose = 0;
  size_t workspaceSize_transposeExtra = 0;
  if (filters_need_trans) {
    workspaceSize_transpose =
        num_filter * ci * co * mluop::getSizeOfDataType(filters_desc->dtype);
  }
  size_t workspaceSize_scatter =
      num_act_out * co * mluop::getSizeOfDataType(features_out_desc->dtype);
  size_t workspaceSize_matmulExtra = 0;
  size_t tempSize_matmulExtra = 0;
  size_t workspaceSize_addNExtra = 0;
  size_t tempSize_addNExtra = 0;
  size_t workspaceSize_maximum = 0;

  float matmul_alpha = 1.0;
  float matmul_beta = 0.0;
  int matmul_requested_algo = 1;
  int matmul_recieved_algo = 0;
  int matmul_is_transA = 0;
  int matmul_is_transB = 0;
  uint32_t matmul_allow_TF32 = 0;
  uint32_t matmul_computetype = (uint32_t)filters_desc->dtype;

  // allocate workspace segment for intermediate data
  void *validFilters_ptr = filters_need_trans ? workspace : (void *)filters;
  void *transposeExtra_ptr = (char *)workspace + workspaceSize_transpose;
  void *matmulResult_ptr = (char *)workspace + workspaceSize_transpose;
  void *gatherResult_ptr = (char *)matmulResult_ptr + workspaceSize_matmul;
  void *matmulExtra_ptr = (char *)gatherResult_ptr + workspaceSize_gather;
  void *scatterResult_ptr = (char *)matmulResult_ptr + workspaceSize_matmul;
  void *addNExtra_ptr = (char *)scatterResult_ptr + workspaceSize_scatter;
  void *addN_ptrs[2] = {scatterResult_ptr, features_out};

  // create intermediate tensor
  mluOpTensorDescriptor_t active_indice_desc;
  mluOpTensorDescriptor_t matmul_a_desc, matmul_b_desc, matmul_c_desc;
  mluOpMatMulDescriptor_t matmul_desc;
  mluOpTensorDescriptor_t addN_descriptors[2] = {features_out_desc,
                                                 features_out_desc};
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
               mluOpSetMatMulDescAttr(matmul_desc, MLUOP_MATMUL_DESC_TRANSA,
                                      &matmul_is_transA, sizeof(int32_t)));
  CHECK_RETURN(api_name,
               mluOpSetMatMulDescAttr(matmul_desc, MLUOP_MATMUL_DESC_TRANSB,
                                      &matmul_is_transB, sizeof(int32_t)));
  CHECK_RETURN(api_name, mluOpSetMatMulDescAttr(
                             matmul_desc, MLUOP_MATMUL_DESC_COMPUTE_TYPE,
                             &matmul_computetype, sizeof(int32_t)));
  CHECK_RETURN(api_name,
               mluOpSetMatMulDescAttr(matmul_desc, MLUOP_MATMUL_ALLOW_TF32,
                                      &matmul_allow_TF32, sizeof(int32_t)));

  // transpose filters to DHWNC layout
  if (filters_need_trans) {
    int trans_in_shape[3] = {0, 0, 0};
    int trans_out_shape[3] = {num_filter, ci, co};
    int permute[3] = {0, 0, 0};
    if (MLUOP_LAYOUT_NDHWC == filters_desc->layout) {
      trans_in_shape[0] = co;
      trans_in_shape[1] = num_filter;
      trans_in_shape[2] = ci;
      permute[0] = 1;
      permute[1] = 2;
      permute[2] = 0;
    } else {
      // MLUOP_LAYOUT_NCDHW == filters_desc->layout
      trans_in_shape[0] = co;
      trans_in_shape[1] = ci;
      trans_in_shape[2] = num_filter;
      permute[0] = 2;
      permute[1] = 1;
      permute[2] = 0;
    }
    mluOpTensorDescriptor_t trans_in_desc, trans_out_desc;
    mluOpTransposeDescriptor_t trans_desc;
    CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&trans_in_desc));
    CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&trans_out_desc));
    CHECK_RETURN(api_name, mluOpCreateTransposeDescriptor(&trans_desc));
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               trans_in_desc, MLUOP_LAYOUT_ARRAY,
                               filters_desc->dtype, 3, trans_in_shape));
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               trans_out_desc, MLUOP_LAYOUT_ARRAY,
                               filters_desc->dtype, 3, trans_out_shape));
    CHECK_RETURN(api_name, mluOpSetTransposeDescriptor(trans_desc, 3, permute));
    CHECK_RETURN(api_name, mluOpGetTransposeWorkspaceSize(
                               handle, trans_in_desc, trans_desc,
                               &workspaceSize_transposeExtra));
    if (!is_workspace_compute) {
      auto trans_status = mluOpTranspose_v2(
          handle, trans_desc, trans_in_desc, filters, trans_out_desc,
          validFilters_ptr, transposeExtra_ptr, workspaceSize_transposeExtra);
      KERNEL_CALL_CHECK(api_name, "mluOpTranspose_v2", trans_status, "");
    }
    CHECK_RETURN(api_name, mluOpDestroyTensorDescriptor(trans_in_desc));
    CHECK_RETURN(api_name, mluOpDestroyTensorDescriptor(trans_out_desc));
    CHECK_RETURN(api_name, mluOpDestroyTransposeDescriptor(trans_desc));
  }

  // invoke gather_nd and matmul to finish indice conv
  int32_t active_point_num = 0;
  int32_t active_indice[2] = {0, 1};
  int32_t matmul_a_shape[2] = {0, ci};
  int32_t matmul_b_shape[2] = {ci, co};
  int32_t matmul_c_shape[2] = {0, co};
  float init_val = 0;

  if (!is_workspace_compute) {
    auto fill_status = mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST, &init_val,
                                    features_out_desc, features_out);
    KERNEL_CALL_CHECK(api_name, "mluOpFill_v3", fill_status, "");
  }

  for (int i = 0; i < num_filter; ++i) {
    active_point_num = indice_num[i];
    if (active_point_num <= 0) {
      continue;
    }
    active_indice[0] = active_point_num;
    matmul_a_shape[0] = active_point_num;
    matmul_c_shape[0] = active_point_num;
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               active_indice_desc, MLUOP_LAYOUT_ARRAY,
                               indice_pairs_desc->dtype, 2, active_indice));
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               matmul_a_desc, MLUOP_LAYOUT_ARRAY,
                               features_desc->dtype, 2, matmul_a_shape));
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               matmul_b_desc, MLUOP_LAYOUT_ARRAY,
                               features_out_desc->dtype, 2, matmul_b_shape));
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               matmul_c_desc, MLUOP_LAYOUT_ARRAY,
                               features_desc->dtype, 2, matmul_c_shape));
    CHECK_RETURN(api_name, mluOpGetMatMulAlgoHeuristic(
                               handle, matmul_desc, matmul_a_desc,
                               matmul_b_desc, matmul_c_desc, matmul_c_desc,
                               nullptr, matmul_requested_algo,
                               &heuristic_result, &matmul_recieved_algo));
    CHECK_RETURN(api_name,
                 mluOpGetMatMulHeuristicResult(heuristic_result, matmul_algo,
                                               &tempSize_matmulExtra));
    CHECK_RETURN(api_name, mluOpGetAddNWorkspaceSize(handle, addN_descriptors,
                                                     2, features_out_desc,
                                                     &tempSize_addNExtra))

    if (is_workspace_compute) {
      workspaceSize_matmulExtra =
          tempSize_matmulExtra > workspaceSize_matmulExtra
              ? tempSize_matmulExtra
              : workspaceSize_matmulExtra;
      workspaceSize_addNExtra = tempSize_addNExtra > workspaceSize_addNExtra
                                    ? tempSize_addNExtra
                                    : workspaceSize_addNExtra;
    } else {
      void *filters_buffer = (char *)validFilters_ptr + i * elementSize_filters;
      void *gatherIndice_buffer =
          (char *)indice_pairs + i * 2 * elementSize_indice_pairs;
      void *scatterAddIndice_buffer =
          (char *)indice_pairs + (i * 2 + 1) * elementSize_indice_pairs;
      // invoke gather to get input data:
      // [num_act_in, ci] -> [indice_pairs_num[i], ci]
      auto gather_x_status =
          mluOpGatherNd(handle, features_desc, features, active_indice_desc,
                        gatherIndice_buffer, matmul_a_desc, gatherResult_ptr);
      KERNEL_CALL_CHECK(api_name, "mluOpGatherNd", gather_x_status, "");
      // invoke matmul to get intermediate result:
      // [indice_pairs_num[i], ci] * [ci, co] = [indice_pairs_num[i], co]
      auto matmul_status = mluOpMatMul_v2(
          handle, matmul_desc, matmul_algo, &matmul_alpha, matmul_a_desc,
          gatherResult_ptr, matmul_b_desc, filters_buffer, &matmul_beta,
          matmul_c_desc, matmulResult_ptr, matmulExtra_ptr,
          tempSize_matmulExtra, matmul_c_desc, matmulResult_ptr);
      KERNEL_CALL_CHECK(api_name, "mluOpMatMul_v2", matmul_status, "");

      auto fill_status =
          mluOpFill_v3(handle, MLUOP_POINTER_MODE_HOST, &init_val,
                       features_out_desc, scatterResult_ptr);
      KERNEL_CALL_CHECK(api_name, "mluOpFill_v3", fill_status, "");

      // invoke scatter_add to add intermediate result to final result:
      // [indice_num[i], co] -> [num_act_out, co]
      auto scatter_add_status = mluOpScatterNd_v2(
          handle, MLUOP_SCATTERND_UPDATE, active_indice_desc,
          scatterAddIndice_buffer, matmul_c_desc, matmulResult_ptr,
          features_out_desc, scatterResult_ptr, features_out_desc,
          scatterResult_ptr);
      KERNEL_CALL_CHECK(api_name, "mluOpScatterNd_v2", scatter_add_status, "");

      auto addN_status = mluOpAddN_v2(handle, addN_descriptors, addN_ptrs, 2,
                                      features_out_desc, features_out,
                                      addNExtra_ptr, tempSize_addNExtra);
    }
  }
  if (is_workspace_compute) {
    workspaceSize_maximum = std::max(
        workspaceSize_matmul + workspaceSize_gather + workspaceSize_matmulExtra,
        workspaceSize_transposeExtra);
    workspaceSize_maximum = std::max(
        workspaceSize_matmul + workspaceSize_scatter + workspaceSize_addNExtra,
        workspaceSize_maximum);
    *workspace_size = workspaceSize_transpose + workspaceSize_maximum;
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

// workspace composition:
// | transposed filters | transpose_extra |
//                      ||
//                      \/
// | transposed filters | matmul_result | gather_result | matmul_extra |
//                      ||
//                      \/
// | transposed filters | matmul_result | scatter_result | addN_extra |
mluOpStatus_t MLUOP_WIN_API mluOpGetIndiceConvolutionForwardWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t features_desc,
    const mluOpTensorDescriptor_t filters_desc,
    const mluOpTensorDescriptor_t indice_pairs_desc,
    const mluOpTensorDescriptor_t features_out_desc, const int64_t indice_num[],
    const int64_t num_act_out, const int64_t inverse, const int64_t sub_m,
    size_t *size) {
  const std::string api_name =
      "[mluOpGetIndiceConvolutionForwardWorkspaceSize]";

  // foolproof check
  auto fool_proof = foolProof(api_name, handle, features_desc, filters_desc,
                              indice_pairs_desc, indice_num, num_act_out,
                              inverse, sub_m, features_out_desc);
  if (fool_proof != MLUOP_STATUS_SUCCESS) {
    return fool_proof;
  }

  // zero element
  if (mluOpGetTensorElementNum(features_desc) == 0 ||
      mluOpGetTensorElementNum(indice_pairs_desc) == 0 ||
      mluOpGetTensorElementNum(filters_desc) == 0 ||
      mluOpGetTensorElementNum(features_out_desc) == 0) {
    VLOG(5) << api_name << "Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  // nullptr check
  PARAM_CHECK(api_name, size != nullptr);

  // main process
  CHECK_RETURN(api_name,
               mainIndiceConvolutionForward(
                   api_name, handle, features_desc, nullptr, filters_desc,
                   nullptr, indice_pairs_desc, nullptr, indice_num, num_act_out,
                   nullptr, size, features_out_desc, nullptr));
  VLOG(5) << api_name << "workspace size: " << *size << ".";
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpIndiceConvolutionForward(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t features_desc,
    const void *features, const mluOpTensorDescriptor_t filters_desc,
    const void *filters, const mluOpTensorDescriptor_t indice_pairs_desc,
    const void *indice_pairs, const int64_t indice_num[],
    const int64_t num_act_out, const int64_t inverse, const int64_t sub_m,
    void *workspace, const size_t workspace_size,
    const mluOpTensorDescriptor_t features_out_desc, void *features_out) {
  const std::string api_name = "[mluOpIndiceConvolutionForward]";

  // foolproof check
  auto fool_proof = foolProof(api_name, handle, features_desc, filters_desc,
                              indice_pairs_desc, indice_num, num_act_out,
                              inverse, sub_m, features_out_desc);
  if (fool_proof != MLUOP_STATUS_SUCCESS) {
    return fool_proof;
  }

  // zero element
  if (mluOpGetTensorElementNum(filters_desc) == 0 ||
      mluOpGetTensorElementNum(features_desc) == 0 ||
      mluOpGetTensorElementNum(indice_pairs_desc) == 0 ||
      mluOpGetTensorElementNum(features_out_desc) == 0) {
    VLOG(5) << api_name << "Skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  // data pointer nullptr check
  PARAM_CHECK(api_name, features != nullptr);
  PARAM_CHECK(api_name, filters != nullptr);
  PARAM_CHECK(api_name, indice_pairs != nullptr);
  PARAM_CHECK(api_name, features_out != nullptr);
  if (workspace_size > 0) {
    PARAM_CHECK(api_name, workspace != nullptr);
  }

  // gen_case
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("indice_convolution_forward");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA_REAL(true, "features", features, features_desc);
    GEN_CASE_DATA_REAL(true, "filters", filters, filters_desc);
    GEN_CASE_DATA_REAL(true, "indice_pairs_desc", indice_pairs,
                       indice_pairs_desc);
    GEN_CASE_DATA_REAL(false, "features_out", features_out, features_out_desc);
    GEN_CASE_OP_PARAM_SINGLE(0, "indice_convolution_forward", "inverse",
                             inverse);
    GEN_CASE_OP_PARAM_SINGLE(1, "indice_convolution_forward", "sub_m", sub_m);
    GEN_CASE_OP_PARAM_ARRAY(1, "indice_convolution_forward", "indice_num",
                            indice_num, indice_pairs_desc->dims[0]);
    GEN_CASE_OP_PARAM_SINGLE(1, "indice_convolution_forward", "num_active_out",
                             num_act_out);
    GEN_CASE_HANDLE_PARAM();
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  // main process
  CHECK_RETURN(api_name, mainIndiceConvolutionForward(
                             api_name, handle, features_desc, features,
                             filters_desc, filters, indice_pairs_desc,
                             indice_pairs, indice_num, num_act_out, workspace,
                             nullptr, features_out_desc, features_out));

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_END();
  }
  return MLUOP_STATUS_SUCCESS;
}
