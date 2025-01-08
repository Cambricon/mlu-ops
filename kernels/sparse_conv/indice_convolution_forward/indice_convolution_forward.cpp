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

#include "core/cnnl_helper.h"
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/mlu_env.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"
#include "mlu_op.h"
#include "kernels/sparse_conv/get_indice_pairs/get_indice_pairs_structs.h"

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
  PARAM_CHECK(api_name, features_desc->getDtype() == MLUOP_DTYPE_FLOAT ||
                            features_desc->getDtype() == MLUOP_DTYPE_HALF);
  PARAM_CHECK(api_name, filters_desc->getDtype() == MLUOP_DTYPE_FLOAT ||
                            filters_desc->getDtype() == MLUOP_DTYPE_HALF);
  PARAM_CHECK(api_name, indice_pairs_desc->getDtype() == MLUOP_DTYPE_INT32);
  PARAM_CHECK(api_name, features_out_desc->getDtype() == MLUOP_DTYPE_FLOAT ||
                            features_out_desc->getDtype() == MLUOP_DTYPE_HALF);
  PARAM_CHECK(api_name,
              features_desc->getDtype() == features_out_desc->getDtype() &&
                  features_desc->getDtype() == filters_desc->getDtype());

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
  // PARAM_CHECK(api_name, filters_desc->getLayout() == MLUOP_LAYOUT_DHWCN);
  if (filters_desc->getLayout() != MLUOP_LAYOUT_NDHWC &&
      filters_desc->getLayout() != MLUOP_LAYOUT_NCDHW &&
      filters_desc->getLayout() != MLUOP_LAYOUT_ARRAY) {
    LOG(ERROR) << api_name << "The layout of filters is: "
               << mluOpGetNameOfTensorLayout(filters_desc->getLayout())
               << ", which is not supported now.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // shape check
  PARAM_CHECK(api_name, features_desc->getDim() == 2);
  PARAM_CHECK(api_name, indice_pairs_desc->getDim() == 3);
  PARAM_CHECK(api_name, features_out_desc->getDim() == 2);
  if (indice_pairs_desc->getDimIndex(2) > INDICE_IN_LARGE_TENSOR_NUM) {
    LOG(ERROR) << api_name << " Check failed: "
               << "indice_pairs_desc->getDimIndex(2) cannot be greater than "
               << INDICE_IN_LARGE_TENSOR_NUM << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  if (filters_desc->getDim() != 5) {
    LOG(ERROR) << api_name
               << "The filters dimension number only support 5 currently,"
               << " but filters dimension number is :" << filters_desc->getDim()
               << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // check stride
  STRIDE_TENSOR_CHECK(api_name + ":", features_desc,
                      "features_desc must be contiguous");
  STRIDE_TENSOR_CHECK(api_name + ":", filters_desc,
                      "filters_desc must be contiguous");
  STRIDE_TENSOR_CHECK(api_name + ":", indice_pairs_desc,
                      "indice_pairs_desc must be contiguous");
  STRIDE_TENSOR_CHECK(api_name + ":", features_out_desc,
                      "features_out_desc must be contiguous");

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
  if (filters_desc->getLayout() == MLUOP_LAYOUT_ARRAY) {
    ci = filters_desc->getDimIndex(3);
    num_filter = filters_desc->getDimIndex(0) * filters_desc->getDimIndex(1) *
                 filters_desc->getDimIndex(2);
    co = filters_desc->getDimIndex(4);
  } else {
    ci = mluOpGetTensordimC(filters_desc);
    num_filter = mluOpGetTensordimD(filters_desc) *
                 mluOpGetTensordimH(filters_desc) *
                 mluOpGetTensordimW(filters_desc);
    co = mluOpGetTensordimN(filters_desc);
  }

  // features shape check
  PARAM_CHECK(api_name, features_desc->getDimIndex(0) ==
                            indice_pairs_desc->getDimIndex(2));
  PARAM_CHECK(api_name, features_desc->getDimIndex(1) == ci);

  // indice_pairs shape check
  PARAM_CHECK(api_name, indice_pairs_desc->getDimIndex(0) == num_filter);
  PARAM_CHECK(api_name, indice_pairs_desc->getDimIndex(1) == 2);

  // features_out shape check
  PARAM_CHECK(api_name, features_out_desc->getDimIndex(0) == num_act_out);
  PARAM_CHECK(api_name, features_out_desc->getDimIndex(1) == co);

  // indice_num[] check
  for (int i = 0; i < num_filter; ++i) {
    std::string i_str = "i: " + std::to_string(i) + ".";
    PARAM_CHECK_V2(
        api_name,
        indice_num[i] >= 0 && indice_num[i] <= features_desc->getDimIndex(0),
        << i_str);
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
  if (filters_desc->getLayout() == MLUOP_LAYOUT_ARRAY) {
    filters_need_trans = false;
    ci = filters_desc->getDimIndex(3);
    co = filters_desc->getDimIndex(4);
  } else {
    ci = mluOpGetTensordimC(filters_desc);
    co = mluOpGetTensordimN(filters_desc);
  }
  int32_t num_filter = indice_pairs_desc->getDimIndex(0);

  int64_t num_act_in = indice_pairs_desc->getDimIndex(2);
  int64_t elementSize_filters =
      ci * co * mluop::getSizeOfDataType(filters_desc->getDtype());
  int64_t elementSize_indice_pairs =
      num_act_in * mluop::getSizeOfDataType(indice_pairs_desc->getDtype());

  int32_t max_indice_num = 0;
  for (int i = 0; i < num_filter; ++i) {
    max_indice_num =
        indice_num[i] > max_indice_num ? indice_num[i] : max_indice_num;
  }
  size_t workspaceSize_gather =
      max_indice_num * ci * mluop::getSizeOfDataType(features_desc->getDtype());
  size_t workspaceSize_matmul =
      max_indice_num * co *
      mluop::getSizeOfDataType(features_out_desc->getDtype());
  size_t workspaceSize_transpose = 0;
  size_t workspaceSize_transposeExtra = 0;
  if (filters_need_trans) {
    workspaceSize_transpose =
        num_filter * ci * co *
        mluop::getSizeOfDataType(filters_desc->getDtype());
  }
  size_t workspaceSize_scatter =
      num_act_out * co *
      mluop::getSizeOfDataType(features_out_desc->getDtype());
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
  uint32_t matmul_computetype = (uint32_t)filters_desc->getDtype();

  // allocate workspace segment for intermediate data
  void *validFilters_ptr = filters_need_trans ? workspace : (void *)filters;
  void *transposeExtra_ptr = (int8_t *)workspace + workspaceSize_transpose;
  void *matmulResult_ptr = (int8_t *)workspace + workspaceSize_transpose;
  void *gatherResult_ptr = (int8_t *)matmulResult_ptr + workspaceSize_matmul;
  void *matmulExtra_ptr = (int8_t *)gatherResult_ptr + workspaceSize_gather;
  void *scatterResult_ptr = (int8_t *)matmulResult_ptr + workspaceSize_matmul;
  void *addNExtra_ptr = (int8_t *)scatterResult_ptr + workspaceSize_scatter;
  void *addN_ptrs[2] = {scatterResult_ptr, features_out};

  // create intermediate tensor
  mluOpTensorDescriptor_t active_indice_desc;
  mluOpTensorDescriptor_t matmul_a_desc, matmul_b_desc, matmul_c_desc;
  cnnlMatMulDescriptor_t matmul_desc;
  // mluOpTensorDescriptor_t addN_descriptors[2] = {features_out_desc,
  //                                                features_out_desc};
  cnnlMatMulAlgo_t matmul_algo;
  cnnlMatMulHeuristicResult_t heuristic_result;
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&active_indice_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_a_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_b_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_c_desc));
  CALL_CNNL(cnnlMatMulDescCreate(&matmul_desc));
  CALL_CNNL(cnnlMatMulAlgoCreate(&matmul_algo));
  CALL_CNNL(cnnlCreateMatMulHeuristicResult(&heuristic_result));

  CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_TRANSA,
                                  &matmul_is_transA, sizeof(int32_t)));
  CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_TRANSB,
                                  &matmul_is_transB, sizeof(int32_t)));
  CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_COMPUTE_TYPE,
                                  &matmul_computetype, sizeof(int32_t)));
  CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_ALLOW_TF32,
                                  &matmul_allow_TF32, sizeof(int32_t)));

  // transpose filters to DHWNC layout
  if (filters_need_trans) {
    int trans_in_shape[3] = {0, 0, 0};
    int trans_out_shape[3] = {num_filter, ci, co};
    int permute[3] = {0, 0, 0};
    if (MLUOP_LAYOUT_NDHWC == filters_desc->getLayout()) {
      trans_in_shape[0] = co;
      trans_in_shape[1] = num_filter;
      trans_in_shape[2] = ci;
      permute[0] = 1;
      permute[1] = 2;
      permute[2] = 0;
    } else {
      // MLUOP_LAYOUT_NCDHW == filters_desc->getLayout()
      trans_in_shape[0] = co;
      trans_in_shape[1] = ci;
      trans_in_shape[2] = num_filter;
      permute[0] = 2;
      permute[1] = 1;
      permute[2] = 0;
    }
    mluOpTensorDescriptor_t trans_in_desc, trans_out_desc;
    cnnlTransposeDescriptor_t trans_desc;
    CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&trans_in_desc));
    CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&trans_out_desc));
    CALL_CNNL(cnnlCreateTransposeDescriptor(&trans_desc));
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               trans_in_desc, MLUOP_LAYOUT_ARRAY,
                               filters_desc->getDtype(), 3, trans_in_shape));
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               trans_out_desc, MLUOP_LAYOUT_ARRAY,
                               filters_desc->getDtype(), 3, trans_out_shape));
    CALL_CNNL(cnnlSetTransposeDescriptor(trans_desc, 3, permute));
    {
      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(trans_in_desc, cnnl_x_desc);
      CALL_CNNL(cnnlGetTransposeWorkspaceSize(
          cnnl_handle, cnnl_x_desc, trans_desc, &workspaceSize_transposeExtra));
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }
    if (!is_workspace_compute) {
      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(trans_in_desc, cnnl_x_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(trans_out_desc, cnnl_y_desc);
      CALL_CNNL(cnnlTranspose_v2(
          cnnl_handle, trans_desc, cnnl_x_desc, filters, cnnl_y_desc,
          validFilters_ptr, transposeExtra_ptr, workspaceSize_transposeExtra));
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_y_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }
    CHECK_RETURN(api_name, mluOpDestroyTensorDescriptor(trans_in_desc));
    CHECK_RETURN(api_name, mluOpDestroyTensorDescriptor(trans_out_desc));
    CALL_CNNL(cnnlDestroyTransposeDescriptor(trans_desc));
  }

  // invoke gather_nd and matmul to finish indice conv
  int32_t active_point_num = 0;
  int32_t active_indice[2] = {0, 1};
  int32_t matmul_a_shape[2] = {0, ci};
  int32_t matmul_b_shape[2] = {ci, co};
  int32_t matmul_c_shape[2] = {0, co};
  float init_val = 0;

  if (!is_workspace_compute) {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(features_out_desc,
                                                 cnnl_output_desc);
    CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &init_val,
                          cnnl_output_desc, features_out));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }

  for (int i = 0; i < num_filter; ++i) {
    active_point_num = indice_num[i];
    if (active_point_num <= 0) {
      continue;
    }
    active_indice[0] = active_point_num;
    matmul_a_shape[0] = active_point_num;
    matmul_c_shape[0] = active_point_num;
    CHECK_RETURN(api_name,
                 mluOpSetTensorDescriptor(
                     active_indice_desc, MLUOP_LAYOUT_ARRAY,
                     indice_pairs_desc->getDtype(), 2, active_indice));
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               matmul_a_desc, MLUOP_LAYOUT_ARRAY,
                               features_desc->getDtype(), 2, matmul_a_shape));
    CHECK_RETURN(api_name,
                 mluOpSetTensorDescriptor(matmul_b_desc, MLUOP_LAYOUT_ARRAY,
                                          features_out_desc->getDtype(), 2,
                                          matmul_b_shape));
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               matmul_c_desc, MLUOP_LAYOUT_ARRAY,
                               features_desc->getDtype(), 2, matmul_c_shape));
    {
      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_a_desc, cnnl_a_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_b_desc, cnnl_b_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_c_desc, cnnl_c_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_c_desc, cnnl_d_desc);
      CALL_CNNL(cnnlGetMatMulAlgoHeuristic(
          cnnl_handle, matmul_desc, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc,
          cnnl_d_desc, nullptr, matmul_requested_algo, &heuristic_result,
          &matmul_recieved_algo));
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_d_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }
    CALL_CNNL(cnnlGetMatMulHeuristicResult(heuristic_result, matmul_algo,
                                           &tempSize_matmulExtra));
    uint32_t addn_num = 2;
    {
      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
      cnnlTensorDescriptor_t *cnnl_input_descs =
          (cnnlTensorDescriptor_t *)malloc(sizeof(cnnlTensorDescriptor_t) *
                                           addn_num);
      for (int i = 0; i < addn_num; i++) {
        CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(features_out_desc,
                                              cnnl_input_descs[i]);
      }
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(features_out_desc,
                                                   cnnl_output_desc);
      CALL_CNNL(cnnlGetAddNWorkspaceSize(cnnl_handle, cnnl_input_descs,
                                         addn_num, cnnl_output_desc,
                                         &tempSize_addNExtra));
      for (int i = 0; i < addn_num; i++) {
        DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_descs[i]);
      }
      free(cnnl_input_descs);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }

    if (is_workspace_compute) {
      workspaceSize_matmulExtra =
          tempSize_matmulExtra > workspaceSize_matmulExtra
              ? tempSize_matmulExtra
              : workspaceSize_matmulExtra;
      workspaceSize_addNExtra = tempSize_addNExtra > workspaceSize_addNExtra
                                    ? tempSize_addNExtra
                                    : workspaceSize_addNExtra;
    } else {
      void *filters_buffer =
          (int8_t *)validFilters_ptr + i * elementSize_filters;
      void *gatherIndice_buffer =
          (int8_t *)indice_pairs + i * 2 * elementSize_indice_pairs;
      void *scatterAddIndice_buffer =
          (int8_t *)indice_pairs + (i * 2 + 1) * elementSize_indice_pairs;
      // invoke gather to get input data:
      // [num_act_in, ci] -> [indice_pairs_num[i], ci]
      {
        DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
        DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(features_desc,
                                                     cnnl_params_desc);
        DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(active_indice_desc,
                                                     cnnl_indices_desc);
        DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_a_desc,
                                                     cnnl_output_desc);
        CALL_CNNL(cnnlGatherNd(cnnl_handle, cnnl_params_desc, features,
                               cnnl_indices_desc, gatherIndice_buffer,
                               cnnl_output_desc, gatherResult_ptr));
        DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_params_desc);
        DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_indices_desc);
        DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
        DESTROY_CNNL_HANDLE(cnnl_handle);
      }
      // invoke matmul to get intermediate result:
      // [indice_pairs_num[i], ci] * [ci, co] = [indice_pairs_num[i], co]
      {
        DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
        DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_a_desc,
                                                     cnnl_a_desc);
        DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_b_desc,
                                                     cnnl_b_desc);
        DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_c_desc,
                                                     cnnl_c_desc);
        DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_c_desc,
                                                     cnnl_d_desc);
        CALL_CNNL(cnnlMatMul_v2(
            cnnl_handle, matmul_desc, matmul_algo, &matmul_alpha, cnnl_a_desc,
            gatherResult_ptr, cnnl_b_desc, filters_buffer, &matmul_beta,
            cnnl_c_desc, matmulResult_ptr, matmulExtra_ptr,
            tempSize_matmulExtra, cnnl_d_desc, matmulResult_ptr));

        DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
        DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
        DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
        DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_d_desc);
        DESTROY_CNNL_HANDLE(cnnl_handle);
      }

      {
        DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
        DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(features_out_desc,
                                                     cnnl_output_desc);
        CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &init_val,
                              cnnl_output_desc, scatterResult_ptr));
        DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
        DESTROY_CNNL_HANDLE(cnnl_handle);
      }

      // invoke scatter_add to add intermediate result to final result:
      // [indice_num[i], co] -> [num_act_out, co]
      {
        DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
        DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(active_indice_desc,
                                                     cnnl_indices_desc);
        DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_c_desc,
                                                     cnnl_updates_desc);
        DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(features_out_desc,
                                                     cnnl_input_desc);
        DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(features_out_desc,
                                                     cnnl_output_desc);

        CALL_CNNL(cnnlScatterNd_v2(cnnl_handle, CNNL_SCATTERND_UPDATE,
                                   cnnl_indices_desc, scatterAddIndice_buffer,
                                   cnnl_updates_desc, matmulResult_ptr,
                                   cnnl_input_desc, scatterResult_ptr,
                                   cnnl_output_desc, scatterResult_ptr));
        DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_indices_desc);
        DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_updates_desc);
        DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
        DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
        DESTROY_CNNL_HANDLE(cnnl_handle);
      }

      {
        int addn_num = 2;
        DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
        cnnlTensorDescriptor_t *cnnl_input_descs =
            (cnnlTensorDescriptor_t *)malloc(sizeof(cnnlTensorDescriptor_t) *
                                             addn_num);
        for (int i = 0; i < addn_num; i++) {
          CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(features_out_desc,
                                                cnnl_input_descs[i]);
        }
        DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(features_out_desc,
                                                     cnnl_output_desc);

        CALL_CNNL(cnnlAddN_v2(cnnl_handle, cnnl_input_descs, addN_ptrs,
                              addn_num, cnnl_output_desc, features_out,
                              addNExtra_ptr, tempSize_addNExtra));
        for (int i = 0; i < addn_num; i++) {
          DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_descs[i]);
        }
        free(cnnl_input_descs);
        DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
        DESTROY_CNNL_HANDLE(cnnl_handle);
      }
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
  CALL_CNNL(cnnlMatMulDescDestroy(matmul_desc));
  CALL_CNNL(cnnlMatMulAlgoDestroy(matmul_algo));
  CALL_CNNL(cnnlDestroyMatMulHeuristicResult(heuristic_result));
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
    GEN_CASE_START("indice_convolution_forward", "INDICE_CONVOLUTION_FORWARD");
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
                            indice_num, indice_pairs_desc->getDimIndex(0));
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

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
