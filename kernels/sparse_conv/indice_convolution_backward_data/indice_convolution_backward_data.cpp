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
#include "kernels/sparse_conv/indice_convolution_backward_data/indice_convolution_backward_data.h"

#include <algorithm>
#include <string>

#include "core/context.h"
#include "core/gen_case.h"
#include "kernels/sparse_conv/get_indice_pairs/get_indice_pairs_structs.h"
#include "kernels/utils/cnnl_helper.h"
#include "mlu_op.h"

static mluOpStatus_t foolCheckNoPtr(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t output_grad_desc,
    const mluOpTensorDescriptor_t filters_desc,
    const mluOpTensorDescriptor_t indice_pairs_desc, const int64_t indice_num[],
    const int64_t inverse, const int64_t sub_m,
    const mluOpTensorDescriptor_t input_grad_desc, bool *is_zero) {
  std::string api = "[mluOpIndiceConvolutionBackwardData]";
  // check nullptr
  PARAM_CHECK(api, handle != NULL);
  PARAM_CHECK(api, output_grad_desc != NULL);
  PARAM_CHECK(api, filters_desc != NULL);
  PARAM_CHECK(api, indice_pairs_desc != NULL);
  PARAM_CHECK(api, input_grad_desc != NULL);

  // check platform
  if (handle->arch < 372) {
    LOG(ERROR) << api << " Only support hardware over MLU300 .";
    return MLUOP_STATUS_ARCH_MISMATCH;
  }

  // check dim
  PARAM_CHECK_EQ(api, output_grad_desc->dim, 2);
  PARAM_CHECK(api, filters_desc->dim == 4 || filters_desc->dim == 5);
  PARAM_CHECK_EQ(api, indice_pairs_desc->dim, 3);
  PARAM_CHECK_EQ(api, input_grad_desc->dim, 2);

  // check shape
  PARAM_CHECK(api, indice_pairs_desc->dims[1] == 2);
  if (indice_pairs_desc->dims[2] > INDICE_IN_LARGE_TENSOR_NUM) {
    LOG(ERROR) << api << " Check failed: "
               << "indice_pairs_desc->dims[2] cannot be greater than "
               << INDICE_IN_LARGE_TENSOR_NUM << ".";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // check dtype
  PARAM_CHECK(api, output_grad_desc->dtype == MLUOP_DTYPE_FLOAT ||
                       output_grad_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK(api, filters_desc->dtype == MLUOP_DTYPE_FLOAT ||
                       filters_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK(api, input_grad_desc->dtype == MLUOP_DTYPE_FLOAT ||
                       input_grad_desc->dtype == MLUOP_DTYPE_HALF);
  PARAM_CHECK(api, indice_pairs_desc->dtype == MLUOP_DTYPE_INT32);

  // check layout
  bool layout_check = filters_desc->layout == MLUOP_LAYOUT_NHWC ||
                      filters_desc->layout == MLUOP_LAYOUT_NCHW ||
                      filters_desc->layout == MLUOP_LAYOUT_HWCN ||
                      filters_desc->layout == MLUOP_LAYOUT_NCDHW ||
                      filters_desc->layout == MLUOP_LAYOUT_NDHWC ||
                      filters_desc->layout == MLUOP_LAYOUT_ARRAY;
  if (!layout_check) {
    LOG(ERROR) << api
               << " The filters tensor only supports "
                  "NHWC/NCHW/HWCN/NCDHW/NDHWC/ARRAY layout.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // get filters params
  int kd = 1, kh = 1, kw = 1, dyc = 1, dxc = 1;
  if (filters_desc->layout != MLUOP_LAYOUT_ARRAY) {
    kh = mluOpGetTensordimH(filters_desc);
    kw = mluOpGetTensordimW(filters_desc);
    dyc = mluOpGetTensordimN(filters_desc);
    dxc = mluOpGetTensordimC(filters_desc);
    if (filters_desc->dim == 5) {
      kd = mluOpGetTensordimD(filters_desc);
    }
  } else {
    if (filters_desc->dim == 5) {
      kd = filters_desc->dims[0];
    }
    int _dim = filters_desc->dim;
    kh = filters_desc->dims[_dim - 4];
    kw = filters_desc->dims[_dim - 3];
    dxc = filters_desc->dims[_dim - 2];
    dyc = filters_desc->dims[_dim - 1];
  }
  int K = kd * kh * kw;

  // check param
  PARAM_CHECK(api, inverse == 0 || inverse == 1);
  PARAM_CHECK(api, sub_m == 0 || sub_m == 1);
  for (int kk = 0; kk < K; ++kk) {
    PARAM_CHECK(api, indice_num[kk] >= 0);
  }
  if (inverse == 1) {
    LOG(ERROR) << api << " Not support inverse == 1 yet.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // check algorithm, relationship between params
  if (K != indice_pairs_desc->dims[0]) {
    LOG(ERROR) << api
               << " The dims[0] of indice_pairs should be equal to the "
                  "multiple of kd, kh and kw.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (output_grad_desc->dims[1] != dyc) {
    LOG(ERROR) << api
               << " The dims[1] of output_grad should be equal to dyc of "
                  "filters tensor.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (input_grad_desc->dims[1] != dxc) {
    LOG(ERROR) << api
               << " The dims[1] of input_grad should be equal to dxc of "
                  "filters tensor.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (input_grad_desc->dims[0] != indice_pairs_desc->dims[2]) {
    LOG(ERROR) << api
               << " The dims[0] of input_grad should be equal to the dims[2] "
                  "of indice_pairs.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  int max_indice_num = getMaxNumInArray(indice_num, K);

  if (indice_pairs_desc->dims[2] < max_indice_num) {
    VLOG(5) << "indice_pairs_desc->dims[2] " << indice_pairs_desc->dims[2]
            << " max_indice_num " << max_indice_num;
    LOG(ERROR) << api
               << " The data in indice_num array should be smaller or equal to"
               << " the dims[2] of indice_pairs.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (sub_m == 1) {
    if (input_grad_desc->dims[0] != output_grad_desc->dims[0]) {
      LOG(ERROR) << api
                 << " The dims[0] of input_grad should be equal to the dims[0]"
                 << " of output_grad when sub_m is 1.";
      return MLUOP_STATUS_BAD_PARAM;
    }

    if (indice_num[K / 2] < max_indice_num) {
      LOG(ERROR) << api
                 << " The middle number of the indice_num array should be the "
                 << "maximum of the array when sub_m is 1. Now the maximum is "
                 << max_indice_num << " while the middle number of the array "
                 << "is " << indice_num[K / 2] << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  if (output_grad_desc->dims[0] < max_indice_num) {
    LOG(ERROR)
        << api
        << " The dims[0] of output_grad should be larger than or equal to the"
        << " maximum number of indice_num.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if (sub_m == 1 && K % 2 == 0) {
    LOG(ERROR) << api << " When sub_m value is 1, the filters dims (Kd, Kh & "
               << "Kw) should be odd numbers.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  PARAM_CHECK(api, output_grad_desc->dtype == input_grad_desc->dtype);
  PARAM_CHECK(api, output_grad_desc->dtype == filters_desc->dtype);

  // check constraints: not support large tensor
  uint64_t input_grad_count = mluOpGetTensorElementNum(input_grad_desc);
  TENSOR_NUM_CHECK(api, input_grad_count, LARGE_TENSOR_NUM,
                   "input_grad tensor num is too large. ");
  uint64_t output_grad_count = mluOpGetTensorElementNum(output_grad_desc);
  TENSOR_NUM_CHECK(api, output_grad_count, LARGE_TENSOR_NUM,
                   "output_grad tensor num is too large. ");
  uint64_t filter_count = mluOpGetTensorElementNum(filters_desc);
  TENSOR_NUM_CHECK(api, filter_count, LARGE_TENSOR_NUM,
                   "filters tensor num is too large. ");
  uint64_t indice_pairs_count = mluOpGetTensorElementNum(indice_pairs_desc);
  TENSOR_NUM_CHECK(api, indice_pairs_count, LARGE_TENSOR_NUM,
                   "indice_pairs tensor num is too large. ");

  // check zero element
  if (input_grad_count == 0) {
    LOG(INFO) << "input_grad is a zero-element tensor.";
    *is_zero = true;
    return MLUOP_STATUS_SUCCESS;
  }
  if (output_grad_count == 0) {
    LOG(INFO) << "output_grad is a zero-element tensor.";
    *is_zero = true;
    return MLUOP_STATUS_SUCCESS;
  }
  if (filter_count == 0) {
    LOG(INFO) << "filters is a zero-element tensor.";
    *is_zero = true;
    return MLUOP_STATUS_SUCCESS;
  }
  if (indice_pairs_count == 0) {
    LOG(INFO) << "indice_pairs is a zero-element tensor.";
    *is_zero = true;
    return MLUOP_STATUS_SUCCESS;
  }
  return MLUOP_STATUS_SUCCESS;
}

static void getPermuteArray(const mluOpTensorLayout_t filter_layout,
                            int *permute) {
  // transpose to (D)HWCN, (kd-)kh-kw-dxc-dyc
  switch (filter_layout) {
    case MLUOP_LAYOUT_NHWC: {
      permute[0] = 1;
      permute[1] = 2;
      permute[2] = 3;
      permute[3] = 0;
    }; break;
    case MLUOP_LAYOUT_NCHW: {
      permute[0] = 2;
      permute[1] = 3;
      permute[2] = 1;
      permute[3] = 0;
    }; break;
    case MLUOP_LAYOUT_NDHWC: {
      permute[0] = 1;
      permute[1] = 2;
      permute[2] = 3;
      permute[3] = 4;
      permute[4] = 0;
    }; break;
    case MLUOP_LAYOUT_NCDHW: {
      permute[0] = 2;
      permute[1] = 3;
      permute[2] = 4;
      permute[3] = 1;
      permute[4] = 0;
    }; break;
    case MLUOP_LAYOUT_HWCN:
    default:
      break;
  }
}

static mluOpStatus_t foolCheck(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t output_grad_desc,
    const void *output_grad, const mluOpTensorDescriptor_t filters_desc,
    const void *filters, const mluOpTensorDescriptor_t indice_pairs_desc,
    const void *indice_pairs, const int64_t indice_num[], const int64_t inverse,
    const int64_t sub_m, void *workspace, size_t workspace_size,
    const mluOpTensorDescriptor_t input_grad_desc, void *input_grad,
    bool *is_zero) {
  std::string api = "[mluOpIndiceConvolutionBackwardData]";
  mluOpStatus_t ret =
      foolCheckNoPtr(handle, output_grad_desc, filters_desc, indice_pairs_desc,
                     indice_num, inverse, sub_m, input_grad_desc, is_zero);
  if (ret != MLUOP_STATUS_SUCCESS) {
    return ret;
  }
  if (*is_zero) {
    return MLUOP_STATUS_SUCCESS;
  }

  // check workspace & other space
  PARAM_CHECK(api, output_grad != NULL);
  PARAM_CHECK(api, filters != NULL);
  PARAM_CHECK(api, indice_pairs != NULL);
  PARAM_CHECK(api, input_grad != NULL);
  if (workspace_size > 0) {
    PARAM_CHECK(api, workspace != NULL);
  }
  return MLUOP_STATUS_SUCCESS;
}

static void spconvbpdataGencase(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t output_grad_desc,
    const void *output_grad, const mluOpTensorDescriptor_t filters_desc,
    const void *filters, const mluOpTensorDescriptor_t indice_pairs_desc,
    const void *indice_pairs, const int64_t indice_num[], const int64_t inverse,
    const int64_t sub_m, void *workspace, size_t workspace_size,
    const mluOpTensorDescriptor_t input_grad_desc, void *input_grad) {
  GEN_CASE_START("indice_convolution_backward_data",
                 "INDICE_CONVOLUTION_BACKWARD_DATA");
  GEN_CASE_HANDLE(handle);
  GEN_CASE_DATA_REAL(true, "output_grad", output_grad, output_grad_desc);
  GEN_CASE_DATA_REAL(true, "filters", filters, filters_desc);
  GEN_CASE_DATA_REAL(true, "indice_pairs_desc", indice_pairs,
                     indice_pairs_desc);
  GEN_CASE_DATA_REAL(false, "input_grad", input_grad, input_grad_desc);
  GEN_CASE_OP_PARAM_SINGLE(0, "indice_convolution_backward_data", "inverse",
                           inverse);
  GEN_CASE_OP_PARAM_SINGLE(1, "indice_convolution_backward_data", "sub_m",
                           sub_m);
  GEN_CASE_OP_PARAM_ARRAY(1, "indice_convolution_backward_data", "indice_num",
                          indice_num, indice_pairs_desc->dims[0]);
  GEN_CASE_HANDLE_PARAM();
  GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
}

/*
 *   [output_grad]              [filters]
 *         |                       |
 *         | cnnlGatherNd()        | cnnlTranspose_v2()
 *         |                       |
 *         V                       V
 * [output_grad_condence]       [filter_transpose]
 *         |_______________________|
 *                     |
 *                     | cnnlMatMul_v2()
 *                     |
 *                     V
 *           [input_grad_condence]
 *                     |
 *                     | cnnlScatterNd_v2(CNNL_SCATTERND_UPDATE)
 *                     |
 *                     V
 *           [workspace_input_grad_tmp]
 *                     |
 *                     | cnnlAddN_v2()
 *                     |
 *                     V
 *               [input_grad]
 */
mluOpStatus_t MLUOP_WIN_API mluOpGetIndiceConvolutionBackwardDataWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t output_grad_desc,
    const mluOpTensorDescriptor_t filters_desc,
    const mluOpTensorDescriptor_t indice_pairs_desc,
    const mluOpTensorDescriptor_t input_grad_desc, const int64_t indice_num[],
    const int64_t inverse, size_t *workspace_size) {
  const char *api_name = "[mluOpGetIndiceConvolutionBackwardDataWorkspaceSize]";
  bool is_zero_element = false;
  if (workspace_size == NULL) {
    LOG(ERROR) << api_name
               << " The pointer workspace_size should not be nullptr.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  mluOpStatus_t ret =
      foolCheckNoPtr(handle, output_grad_desc, filters_desc, indice_pairs_desc,
                     indice_num, inverse, 0, input_grad_desc, &is_zero_element);
  if (ret != MLUOP_STATUS_SUCCESS) {
    return ret;
  }
  if (is_zero_element) {
    return MLUOP_STATUS_SUCCESS;
  }

  int kd = 1, kh = 1, kw = 1, dyc = 1, dxc = 1;
  if (filters_desc->layout != MLUOP_LAYOUT_ARRAY) {
    kh = mluOpGetTensordimH(filters_desc);
    kw = mluOpGetTensordimW(filters_desc);
    dyc = mluOpGetTensordimN(filters_desc);
    dxc = mluOpGetTensordimC(filters_desc);
    if (filters_desc->dim == 5) {
      kd = mluOpGetTensordimD(filters_desc);
    }
  } else {
    if (filters_desc->dim == 5) {
      kd = filters_desc->dims[0];
    }
    int _dim = filters_desc->dim;
    kh = filters_desc->dims[_dim - 4];
    kw = filters_desc->dims[_dim - 3];
    dxc = filters_desc->dims[_dim - 2];
    dyc = filters_desc->dims[_dim - 1];
  }
  int K = kd * kh * kw;
  int max_indice_num = getMaxNumInArray(indice_num, K);
  uint64_t filter_transpose_size = 0;
  uint64_t transpose_workspace_size = 0;
  uint64_t output_grad_condence_size = 0;
  uint64_t input_grad_condence_size = 0;
  uint64_t matmul_workspace_size = 0;
  if (!(filters_desc->layout == MLUOP_LAYOUT_HWCN ||
        filters_desc->layout == MLUOP_LAYOUT_ARRAY)) {
    filter_transpose_size = mluOpGetTensorElementNum(filters_desc) *
                            mluOpDataTypeBytes(filters_desc->dtype);
    // get cnnlTranspose_v2 workspace workspace_size
    size_t transpose_workspace_size_ = 0;
    cnnlTransposeDescriptor_t trans_desc;
    CALL_CNNL(cnnlCreateTransposeDescriptor(&trans_desc));
    int permute[5] = {0, 1, 2, 3, 4};
    getPermuteArray(filters_desc->layout, permute);
    CALL_CNNL(
        cnnlSetTransposeDescriptor(trans_desc, filters_desc->dim, permute));
    {
      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(filters_desc, cnnl_x_desc);
      CALL_CNNL(cnnlGetTransposeWorkspaceSize(
          cnnl_handle, cnnl_x_desc, trans_desc, &transpose_workspace_size_));
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }
    CALL_CNNL(cnnlDestroyTransposeDescriptor(trans_desc));
    transpose_workspace_size = (uint64_t)transpose_workspace_size_;
  }
  output_grad_condence_size = max_indice_num * output_grad_desc->dims[1] *
                              mluOpDataTypeBytes(filters_desc->dtype);
  input_grad_condence_size = max_indice_num * input_grad_desc->dims[1] *
                             mluOpDataTypeBytes(filters_desc->dtype);

  // matmul workspace
  {
    mluOpTensorDescriptor_t sub_filters_desc;
    mluOpTensorDescriptor_t output_grad_condence_desc;
    mluOpTensorDescriptor_t input_grad_condence_desc;

    cnnlMatMulDescriptor_t cnnl_matmul_desc;
    cnnlMatMulHeuristicResult_t cnnl_heuristic_result;
    cnnlMatMulAlgo_t cnnl_matmul_algo;

    CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&sub_filters_desc));
    // CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&sub_filters_desc));
    int sub_filter_dims[2] = {(int)(dxc), (int)(dyc)};
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               sub_filters_desc, MLUOP_LAYOUT_ARRAY,
                               filters_desc->dtype, 2, sub_filter_dims));
    int is_trans_a = 0, is_trans_b = 1;
    int tf32_flag_int = 0;
    CALL_CNNL(cnnlMatMulDescCreate(&cnnl_matmul_desc));
    CALL_CNNL(cnnlSetMatMulDescAttr(cnnl_matmul_desc, CNNL_MATMUL_DESC_TRANSA,
                                    &(is_trans_a), sizeof(is_trans_a)));
    CALL_CNNL(cnnlSetMatMulDescAttr(cnnl_matmul_desc, CNNL_MATMUL_DESC_TRANSB,
                                    &(is_trans_b), sizeof(is_trans_b)));
    CALL_CNNL(cnnlSetMatMulDescAttr(cnnl_matmul_desc, CNNL_MATMUL_ALLOW_TF32,
                                    &(tf32_flag_int), sizeof(tf32_flag_int)));
    CHECK_RETURN(api_name,
                 mluOpCreateTensorDescriptor(&output_grad_condence_desc));
    int output_grad_condence_dims[2] = {(int)(max_indice_num), (int)(dyc)};
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(output_grad_condence_desc,
                                                    MLUOP_LAYOUT_ARRAY,
                                                    output_grad_desc->dtype, 2,
                                                    output_grad_condence_dims));
    CHECK_RETURN(api_name,
                 mluOpCreateTensorDescriptor(&input_grad_condence_desc));
    int input_grad_condence_dims[2] = {(int)(max_indice_num), (int)(dxc)};
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(input_grad_condence_desc,
                                                    MLUOP_LAYOUT_ARRAY,
                                                    input_grad_desc->dtype, 2,
                                                    input_grad_condence_dims));

    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(sub_filters_desc,
                                                 cnnl_sub_filters_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(
        output_grad_condence_desc, cnnl_output_grad_condence_desc);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_grad_condence_desc,
                                                 cnnl_input_grad_condence_desc);

    CALL_CNNL(cnnlCreateMatMulHeuristicResult(&cnnl_heuristic_result));
    CALL_CNNL(cnnlMatMulAlgoCreate(&cnnl_matmul_algo));

    // set matmul heuristic_result & algorithm
    int requested_algo_count = 1, return_algo_count = 0;
    CALL_CNNL(cnnlGetMatMulAlgoHeuristic(
        cnnl_handle, cnnl_matmul_desc, cnnl_output_grad_condence_desc,
        cnnl_sub_filters_desc, cnnl_input_grad_condence_desc,
        cnnl_input_grad_condence_desc, NULL, requested_algo_count,
        &cnnl_heuristic_result, &return_algo_count));

    // launch matmul
    size_t workspace_size_matmul = 0;
    float alpha_gemm = 1.0f, beta_gemm = 0.0f;
    CALL_CNNL(cnnlGetMatMulHeuristicResult(
        cnnl_heuristic_result, cnnl_matmul_algo, &workspace_size_matmul));

    // destroy descriptors
    CALL_CNNL(cnnlDestroyMatMulHeuristicResult(cnnl_heuristic_result));
    CALL_CNNL(cnnlMatMulDescDestroy(cnnl_matmul_desc));
    CALL_CNNL(cnnlMatMulAlgoDestroy(cnnl_matmul_algo));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_grad_condence_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_sub_filters_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_grad_condence_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);

    CHECK_RETURN(api_name,
                 mluOpDestroyTensorDescriptor(output_grad_condence_desc));
    CHECK_RETURN(api_name, mluOpDestroyTensorDescriptor(sub_filters_desc));
    CHECK_RETURN(api_name,
                 mluOpDestroyTensorDescriptor(input_grad_condence_desc));
    matmul_workspace_size = (uint64_t)workspace_size_matmul;
  }
  // scatter to input_grad_tmp_workspace_size workspace
  uint64_t input_grad_tmp_workspace_size =
      mluOpGetTensorElementNum(input_grad_desc) *
      mluOpDataTypeBytes(input_grad_desc->dtype);

  // addn workspace
  uint32_t addn_num = 2;
  size_t addn_workspace_size = 0;
  {
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    cnnlTensorDescriptor_t *cnnl_input_descs = (cnnlTensorDescriptor_t *)malloc(
        sizeof(cnnlTensorDescriptor_t) * addn_num);
    for (int i = 0; i < addn_num; i++) {
      CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_grad_desc,
                                            cnnl_input_descs[i]);
    }
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_grad_desc,
                                                 cnnl_output_desc);
    CALL_CNNL(
        cnnlGetAddNWorkspaceSize(cnnl_handle, cnnl_input_descs, addn_num,
                                 cnnl_output_desc, &addn_workspace_size));
    for (int i = 0; i < addn_num; i++) {
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_descs[i]);
    }
    free(cnnl_input_descs);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }

  *workspace_size =
      (size_t)(filter_transpose_size + transpose_workspace_size +
               output_grad_condence_size + input_grad_condence_size +
               matmul_workspace_size + input_grad_tmp_workspace_size +
               addn_workspace_size);
  VLOG(5) << "[mluOpIndiceConvolutionBackwardData] filter_transpose_size="
          << filter_transpose_size
          << ", transpose_workspace_size=" << transpose_workspace_size
          << ", output_grad_condence_size=" << output_grad_condence_size
          << ", input_grad_condence_size=" << input_grad_condence_size
          << ", matmul_workspace_size=" << matmul_workspace_size
          << ", input_grad_tmp_workspace_size=" << input_grad_tmp_workspace_size
          << ", addn_workspace_size=" << addn_workspace_size;
  VLOG(5) << "[mluOpIndiceConvolutionBackwardData] workspace workspace_size: "
          << *workspace_size;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpIndiceConvolutionBackwardData(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t output_grad_desc,
    const void *output_grad, const mluOpTensorDescriptor_t filters_desc,
    const void *filters, const mluOpTensorDescriptor_t indice_pairs_desc,
    const void *indice_pairs, const int64_t indice_num[], const int64_t inverse,
    const int64_t sub_m, void *workspace, const size_t workspace_size,
    const mluOpTensorDescriptor_t input_grad_desc, void *input_grad) {
  const char *api_name = "[mluOpIndiceConvolutionBackwardData]";
  // fool check
  {
    bool is_zero_element = false;
    mluOpStatus_t ret = foolCheck(
        handle, output_grad_desc, output_grad, filters_desc, filters,
        indice_pairs_desc, indice_pairs, indice_num, inverse, sub_m, workspace,
        workspace_size, input_grad_desc, input_grad, &is_zero_element);
    if (ret != MLUOP_STATUS_SUCCESS) {
      return ret;
    }
    if (is_zero_element) {
      return MLUOP_STATUS_SUCCESS;
    }
  }

  // gen_case
  if (MLUOP_GEN_CASE_ON_NEW) {
    spconvbpdataGencase(handle, output_grad_desc, output_grad, filters_desc,
                        filters, indice_pairs_desc, indice_pairs, indice_num,
                        inverse, sub_m, workspace, workspace_size,
                        input_grad_desc, input_grad);
  }

  // get filters params
  int kd = 1, kh = 1, kw = 1, dyc = 1, dxc = 1;
  if (filters_desc->layout != MLUOP_LAYOUT_ARRAY) {
    kh = mluOpGetTensordimH(filters_desc);
    kw = mluOpGetTensordimW(filters_desc);
    dyc = mluOpGetTensordimN(filters_desc);
    dxc = mluOpGetTensordimC(filters_desc);
    if (filters_desc->dim == 5) {
      kd = mluOpGetTensordimD(filters_desc);
    }
  } else {
    if (filters_desc->dim == 5) {
      kd = filters_desc->dims[0];
    }
    int _dim = filters_desc->dim;
    kh = filters_desc->dims[_dim - 4];
    kw = filters_desc->dims[_dim - 3];
    dxc = filters_desc->dims[_dim - 2];
    dyc = filters_desc->dims[_dim - 1];
  }
  int K = kd * kh * kw;
  int cal_dwidth = mluOpDataTypeBytes(filters_desc->dtype);
  uint64_t filter_transpose_size = 0, output_grad_condence_size = 0,
           input_grad_condence_size = 0;
  if (!(filters_desc->layout == MLUOP_LAYOUT_HWCN)) {
    filter_transpose_size = mluOpGetTensorElementNum(filters_desc) * cal_dwidth;
    VLOG(5) << "host invoke: filter_transpose_size " << filter_transpose_size;
  }
  output_grad_condence_size =
      getMaxNumInArray(indice_num, K) * output_grad_desc->dims[1] * cal_dwidth;
  input_grad_condence_size =
      getMaxNumInArray(indice_num, K) * input_grad_desc->dims[1] * cal_dwidth;
  char *filter_transpose = (char *)filters;
  char *workspace_base = (char *)workspace;

  // transpose filters to layout XHWCN
  mluOpTensorDescriptor_t filter_transpose_desc;
  if (filters_desc->layout != MLUOP_LAYOUT_HWCN &&
      filters_desc->layout != MLUOP_LAYOUT_ARRAY) {
    filter_transpose = (char *)workspace;
    workspace_base += filter_transpose_size;
    cnnlTransposeDescriptor_t trans_desc;
    CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&filter_transpose_desc));
    CALL_CNNL(cnnlCreateTransposeDescriptor(&trans_desc));
    int permute[5] = {0, 1, 2, 3, 4};
    int filter_transpose_dims[5];
    getPermuteArray(filters_desc->layout, permute);
    for (int i = 0; i < filters_desc->dim; ++i) {
      filter_transpose_dims[i] = filters_desc->dims[permute[i]];
      VLOG(5) << "permute " << permute[i];
    }
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               filter_transpose_desc, MLUOP_LAYOUT_ARRAY,
                               filters_desc->dtype, filters_desc->dim,
                               filter_transpose_dims));
    CALL_CNNL(
        cnnlSetTransposeDescriptor(trans_desc, filters_desc->dim, permute));
    size_t transpose_workspace_size = 0;
    {
      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(filters_desc, cnnl_x_desc);
      CALL_CNNL(cnnlGetTransposeWorkspaceSize(
          cnnl_handle, cnnl_x_desc, trans_desc, &transpose_workspace_size));
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }
    char *transpose_workspace = workspace_base;
    workspace_base += transpose_workspace_size;
    {
      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(filters_desc, cnnl_x_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(filter_transpose_desc,
                                                   cnnl_y_desc);
      CALL_CNNL(cnnlTranspose_v2(
          cnnl_handle, trans_desc, cnnl_x_desc, filters, cnnl_y_desc,
          filter_transpose, transpose_workspace, transpose_workspace_size));
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_y_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }
    CALL_CNNL(cnnlDestroyTransposeDescriptor(trans_desc));
    CHECK_RETURN(api_name, mluOpDestroyTensorDescriptor(filter_transpose_desc));
  } else {
    filter_transpose_desc = filters_desc;
  }
  char *output_grad_condence = workspace_base;
  workspace_base += output_grad_condence_size;
  char *input_grad_condence = workspace_base;
  workspace_base += input_grad_condence_size;

  // filters calculate desc
  mluOpTensorDescriptor_t sub_filters_desc;
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&sub_filters_desc));
  int sub_filter_dims[2] = {(int)(dxc), (int)(dyc)};
  CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                             sub_filters_desc, MLUOP_LAYOUT_ARRAY,
                             filters_desc->dtype, 2, sub_filter_dims));
  float fill_value = 0;
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_grad_desc,
                                               cnnl_output_desc);
  CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &fill_value,
                        cnnl_output_desc, input_grad));
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);

  void *workspace_matmul = NULL;
  char *workspace_input_grad_tmp = NULL;
  char *workspace_addn = NULL;

  // filters DHW dim loop
  int kk_count = 0;
  for (size_t kk = 0; kk < K; ++kk) {
    VLOG(5) << "indice_num " << indice_num[kk];
    if (indice_num[kk] == 0) {
      continue;
    }
    const int int_dwidth = 4;
    char *sub_filter = filter_transpose + kk * dyc * dxc * cal_dwidth;

    // gather output_grad
    mluOpTensorDescriptor_t gather_indices_desc;
    mluOpTensorDescriptor_t output_grad_condence_desc;
    CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&gather_indices_desc));
    int gather_indices_dims[2] = {(int)(indice_num[kk]), (int)(1)};
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                               gather_indices_desc, MLUOP_LAYOUT_ARRAY,
                               MLUOP_DTYPE_INT32, 2, gather_indices_dims));
    CHECK_RETURN(api_name,
                 mluOpCreateTensorDescriptor(&output_grad_condence_desc));
    int output_grad_condence_dims[2] = {(int)(indice_num[kk]), (int)(dyc)};
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(output_grad_condence_desc,
                                                    MLUOP_LAYOUT_ARRAY,
                                                    output_grad_desc->dtype, 2,
                                                    output_grad_condence_dims));
    uint64_t gather_indices_offset =
        (kk * 2 + 1) * int(indice_pairs_desc->dims[2]) * int_dwidth;
    char *gather_indices =
        (char *)(const_cast<void *>(indice_pairs)) + gather_indices_offset;
    {
      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_grad_desc,
                                                   cnnl_params_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(gather_indices_desc,
                                                   cnnl_indices_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_grad_condence_desc,
                                                   cnnl_output_desc);
      CALL_CNNL(cnnlGatherNd(cnnl_handle, cnnl_params_desc, output_grad,
                             cnnl_indices_desc, gather_indices,
                             cnnl_output_desc, output_grad_condence));
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_params_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_indices_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }

    // matmul
    cnnlMatMulDescriptor_t matmul_desc;
    int is_trans_a = 0, is_trans_b = 1;
    int tf32_flag_int = 0;
    CALL_CNNL(cnnlMatMulDescCreate(&matmul_desc));
    CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_TRANSA,
                                    &(is_trans_a), sizeof(is_trans_a)));
    CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_TRANSB,
                                    &(is_trans_b), sizeof(is_trans_b)));
    CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_ALLOW_TF32,
                                    &(tf32_flag_int), sizeof(tf32_flag_int)));
    mluOpTensorDescriptor_t input_grad_condence_desc;
    CHECK_RETURN(api_name,
                 mluOpCreateTensorDescriptor(&input_grad_condence_desc));
    int input_grad_condence_dims[2] = {(int)(indice_num[kk]), (int)(dxc)};
    CHECK_RETURN(api_name, mluOpSetTensorDescriptor(input_grad_condence_desc,
                                                    MLUOP_LAYOUT_ARRAY,
                                                    input_grad_desc->dtype, 2,
                                                    input_grad_condence_dims));
    cnnlMatMulHeuristicResult_t heuristic_result;
    CALL_CNNL(cnnlCreateMatMulHeuristicResult(&heuristic_result));
    cnnlMatMulAlgo_t matmul_algo;
    CALL_CNNL(cnnlMatMulAlgoCreate(&matmul_algo));

    // set matmul heuristic_result & algorithm
    int requested_algo_count = 1, return_algo_count = 0;
    {
      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_grad_condence_desc,
                                                   cnnl_a_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(sub_filters_desc,
                                                   cnnl_b_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_grad_condence_desc,
                                                   cnnl_c_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_grad_condence_desc,
                                                   cnnl_d_desc);
      CALL_CNNL(cnnlGetMatMulAlgoHeuristic(
          cnnl_handle, matmul_desc, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc,
          cnnl_d_desc, NULL, requested_algo_count, &heuristic_result,
          &return_algo_count));
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_d_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }

    // launch matmul
    size_t workspace_size_matmul = 0;
    float alpha_gemm = 1.0f, beta_gemm = 0.0f;
    CALL_CNNL(cnnlGetMatMulHeuristicResult(heuristic_result, matmul_algo,
                                           &workspace_size_matmul));
    if (kk_count == 0) {
      workspace_matmul = workspace_size_matmul == 0
                             ? NULL
                             : reinterpret_cast<void *>(workspace_base);
      workspace_base += workspace_size_matmul;
    }
    {
      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_grad_condence_desc,
                                                   cnnl_a_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(sub_filters_desc,
                                                   cnnl_b_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_grad_condence_desc,
                                                   cnnl_c_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_grad_condence_desc,
                                                   cnnl_d_desc);
      CALL_CNNL(cnnlMatMul_v2(
          cnnl_handle, matmul_desc, matmul_algo, &alpha_gemm, cnnl_a_desc,
          output_grad_condence, cnnl_b_desc, sub_filter, &beta_gemm,
          cnnl_c_desc, input_grad_condence, workspace_matmul,
          workspace_size_matmul, cnnl_d_desc, input_grad_condence));

      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_d_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }
    // destroy descriptors
    CALL_CNNL(cnnlDestroyMatMulHeuristicResult(heuristic_result));
    CALL_CNNL(cnnlMatMulDescDestroy(matmul_desc));
    CALL_CNNL(cnnlMatMulAlgoDestroy(matmul_algo));

    // fill workspace_input_grad_tmp
    uint64_t input_grad_tmp_workspace_size =
        mluOpGetTensorElementNum(input_grad_desc) *
        mluOpDataTypeBytes(input_grad_desc->dtype);
    if (kk_count == 0) {
      workspace_input_grad_tmp = workspace_base;
      workspace_base += input_grad_tmp_workspace_size;
    }
    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_grad_desc,
                                                 cnnl_output_desc);
    CALL_CNNL(cnnlFill_v3(cnnl_handle, CNNL_POINTER_MODE_HOST, &fill_value,
                          cnnl_output_desc, workspace_input_grad_tmp));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);

    // scatter input_grad
    uint64_t scatter_indices_offset =
        (kk * 2) * int(indice_pairs_desc->dims[2]) * int_dwidth;
    char *scatter_indices =
        (char *)(const_cast<void *>(indice_pairs)) + scatter_indices_offset;
    {
      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(gather_indices_desc,
                                                   cnnl_indices_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_grad_condence_desc,
                                                   cnnl_updates_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_grad_desc,
                                                   cnnl_input_desc);
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_grad_desc,
                                                   cnnl_output_desc);

      CALL_CNNL(cnnlScatterNd_v2(cnnl_handle, CNNL_SCATTERND_UPDATE,
                                 cnnl_indices_desc, scatter_indices,
                                 cnnl_updates_desc, input_grad_condence,
                                 cnnl_input_desc, workspace_input_grad_tmp,
                                 cnnl_output_desc, workspace_input_grad_tmp));
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_indices_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_updates_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }

    // add workspace_input_grad_tmp tensor back to input_grad
    if (kk_count == 0) {
      workspace_addn = workspace_base;
    }
    void *addn_array[2] = {reinterpret_cast<void *>(workspace_input_grad_tmp),
                           input_grad};
    size_t addn_workspace_size = 0;
    uint32_t addn_num = 2;

    {
      DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
      cnnlTensorDescriptor_t *cnnl_input_descs =
          (cnnlTensorDescriptor_t *)malloc(sizeof(cnnlTensorDescriptor_t) *
                                           addn_num);
      for (int i = 0; i < addn_num; i++) {
        CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_grad_desc,
                                              cnnl_input_descs[i]);
      }
      DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_grad_desc,
                                                   cnnl_output_desc);
      CALL_CNNL(
          cnnlGetAddNWorkspaceSize(cnnl_handle, cnnl_input_descs, addn_num,
                                   cnnl_output_desc, &addn_workspace_size));

      CALL_CNNL(cnnlAddN_v2(cnnl_handle, cnnl_input_descs, addn_array, addn_num,
                            cnnl_output_desc, input_grad, workspace_addn,
                            addn_workspace_size));
      for (int i = 0; i < addn_num; i++) {
        DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_descs[i]);
      }
      free(cnnl_input_descs);
      DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_output_desc);
      DESTROY_CNNL_HANDLE(cnnl_handle);
    }

    CHECK_RETURN(api_name,
                 mluOpDestroyTensorDescriptor(input_grad_condence_desc));
    CHECK_RETURN(api_name, mluOpDestroyTensorDescriptor(gather_indices_desc));
    CHECK_RETURN(api_name,
                 mluOpDestroyTensorDescriptor(output_grad_condence_desc));
    kk_count++;
  }
  CHECK_RETURN(api_name, mluOpDestroyTensorDescriptor(sub_filters_desc));
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
