/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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

#include "fft_basic_ops.h"

#ifndef FFT_STOCK_BATCH_LIMIT
#define FFT_STOCK_BATCH_LIMIT 512
#endif

bool fftIsIntDtype(const mluOpDataType_t dtype) {
  if (dtype == MLUOP_DTYPE_INT8 || dtype == MLUOP_DTYPE_INT16 ||
      dtype == MLUOP_DTYPE_INT31) {
    return true;
  } else {
    return false;
  }
}

bool fftIsFloatDtype(const mluOpDataType_t dtype) {
  if (dtype == MLUOP_DTYPE_HALF || dtype == MLUOP_DTYPE_FLOAT) {
    return true;
  } else {
    return false;
  }
}

mluOpStatus_t fftGetQuantizeParamWorkspaceSize(mluOpHandle_t handle,
                                               size_t &required_size,
                                               int array_length,
                                               mluOpDataType_t data_type,
                                               mluOpDataType_t compute_type,
                                               const std::string api) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  // size_t required_size = 0;
  if (data_type != compute_type) {
    // create descriptor
    mluOpTensorDescriptor_t input_desc;
    status = mluOpCreateTensorDescriptor(&input_desc);
    CHECK_RETURN(api, status);

    // set descriptor
    int64_t input_dims[1] = {array_length};
    status = mluOpSetTensorDescriptor_v2(input_desc, MLUOP_LAYOUT_ARRAY,
                                         data_type, 1, input_dims);
    CHECK_RETURN(api, status);

    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                      cnnl_handle);  // convert to cnnl_handle
    // convert to cnnl_tensor_descriptor
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);

    // get quantize param workspace
    CALL_CNNL(cnnlGetQuantizeParamWorkspaceSize(cnnl_handle, cnnl_input_desc,
                                                &required_size));

    status = mluOpDestroyTensorDescriptor(input_desc);
    CHECK_RETURN(api, status);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  return status;
}

mluOpStatus_t fftQuantizePositionScale(mluOpHandle_t handle, int array_length,
                                       mluOpDataType_t data_type,
                                       mluOpDataType_t compute_type,
                                       const void *input, void *position,
                                       void *scale, void *workspace,
                                       size_t workspace_size,
                                       const std::string api) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  if (data_type != compute_type) {
    // create descriptor
    mluOpTensorDescriptor_t quant_desc;
    status = mluOpCreateTensorDescriptor(&quant_desc);
    CHECK_RETURN(api, status);

    // set descriptor
    int64_t quant_dims[1] = {array_length};
    status = mluOpSetTensorDescriptor_v2(quant_desc, MLUOP_LAYOUT_ARRAY,
                                         data_type, 1, quant_dims);
    CHECK_RETURN(api, status);

    DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                      cnnl_handle);  // convert to cnnl_handle
    // convert to cnnl_tensor_descriptor
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(quant_desc, cnnl_quant_desc);

    // get quantize param
    int bit_width;
    mluop::castDtypeToBitwidth(compute_type, &bit_width);
    cnnlQuantizeMode_t mode = CNNL_QUANTIZE_POSITION_SCALE;
    CALL_CNNL(cnnlQuantizeParam(cnnl_handle, mode, cnnl_quant_desc, input,
                                bit_width, workspace, workspace_size, position,
                                scale, nullptr));
    status = mluOpDestroyTensorDescriptor(quant_desc);
    CHECK_RETURN(api, status);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_quant_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  return status;
}

mluOpStatus_t fftGetQuantizeMatMulWorkspaceSize(
    mluOpHandle_t handle, size_t &workspace_size, int m, int k, int n,
    bool is_trans_a, bool is_trans_b, mluOpDataType_t a_compute_type,
    mluOpDataType_t b_compute_type, mluOpDataType_t data_type,
    const std::string api) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  int trans_a_int = (int)is_trans_a;
  int trans_b_int = (int)is_trans_b;

  // create descriptor
  mluOpTensorDescriptor_t a_desc = nullptr;
  mluOpTensorDescriptor_t b_desc = nullptr;
  mluOpTensorDescriptor_t c_desc = nullptr;
  status = mluOpCreateTensorDescriptor(&a_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&b_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&c_desc);
  CHECK_RETURN(api, status);

  // set descriptor
  int64_t a_dims[2];
  int64_t b_dims[2];
  int64_t c_dims[2] = {m, n};
  if (is_trans_a) {
    a_dims[0] = k;
    a_dims[1] = m;
  } else {
    a_dims[0] = m;
    a_dims[1] = k;
  }
  if (is_trans_b) {
    b_dims[0] = n;
    b_dims[1] = k;
  } else {
    b_dims[0] = k;
    b_dims[1] = n;
  }
  status = mluOpSetTensorDescriptor_v2(a_desc, MLUOP_LAYOUT_ARRAY, data_type, 2,
                                       a_dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptorOnchipDataType(a_desc, a_compute_type);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(b_desc, MLUOP_LAYOUT_ARRAY, data_type, 2,
                                       b_dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptorOnchipDataType(b_desc, b_compute_type);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(c_desc, MLUOP_LAYOUT_ARRAY, data_type, 2,
                                       c_dims);
  CHECK_RETURN(api, status);
  if (a_compute_type == MLUOP_DTYPE_INT31 ||
      b_compute_type == MLUOP_DTYPE_INT31) {
    status = mluOpSetTensorDescriptorOnchipDataType(c_desc, MLUOP_DTYPE_FLOAT);
    CHECK_RETURN(api, status);
  } else if (fftIsIntDtype(a_compute_type) && fftIsIntDtype(b_compute_type) &&
             c_desc->getDtype() == MLUOP_DTYPE_HALF) {
    status = mluOpSetTensorDescriptorOnchipDataType(c_desc, MLUOP_DTYPE_FLOAT);
    CHECK_RETURN(api, status);
  } else {
    status = mluOpSetTensorDescriptorOnchipDataType(c_desc, data_type);
    CHECK_RETURN(api, status);
  }

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                    cnnl_handle);  // convert to cnnl_handle

  // convert to cnnl_tensor_descriptor
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);

  // get matmul workspace
  if (fftIsIntDtype(a_compute_type) && fftIsIntDtype(b_compute_type)) {
    cnnlMatMulDescriptor_t matmul_desc;
    CALL_CNNL(cnnlMatMulDescCreate(&matmul_desc));
    CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_COMPUTE_TYPE,
                                    &data_type, sizeof(int32_t)));
    CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_TRANSA,
                                    &trans_a_int, sizeof(int32_t)));
    CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_TRANSB,
                                    &trans_b_int, sizeof(int32_t)));

    cnnlMatMulAlgo_t matmul_algo;
    CALL_CNNL(cnnlMatMulAlgoCreate(&matmul_algo));
    cnnlMatMulPreference_t preference = CNNL_MATMUL_FASTEST;
    CALL_CNNL(cnnlGetQuantizeMatMulAlgorithm(
        cnnl_handle, matmul_desc, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc,
        preference, &matmul_algo));

    CALL_CNNL(cnnlGetQuantizeMatMulWorkspaceSize(
        cnnl_handle, matmul_desc, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc,
        matmul_algo, &workspace_size));

    CALL_CNNL(cnnlMatMulDescDestroy(matmul_desc));
    CALL_CNNL(cnnlMatMulAlgoDestroy(matmul_algo));
  } else {
    // workspace_size = 0;  // mluOpMatmul doesn't need workspace.
    cnnlMatMulDescriptor_t matmul_desc;
    cnnlMatMulAlgo_t matmul_algo;
    cnnlMatMulHeuristicResult_t heuristic_result;
    size_t matmul_ws_size = 0, workspace_size = 0;
    bool allow_tf32 = false;
    cnnlDataType_t cnnl_compute_type = CNNL_DTYPE_FLOAT;  // (TODO)

    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_d_desc);
    CALL_CNNL(cnnlMatMulDescCreate(&matmul_desc));
    CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_TRANSA,
                                    &trans_a_int, sizeof(int32_t)));
    CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_TRANSB,
                                    &trans_b_int, sizeof(int32_t)));
    CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_ALLOW_TF32,
                                    &allow_tf32, sizeof(int32_t)));
    CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_COMPUTE_TYPE,
                                    &cnnl_compute_type,
                                    sizeof(cnnl_compute_type)));
    CALL_CNNL(cnnlMatMulAlgoCreate(&matmul_algo));
    CALL_CNNL(cnnlCreateMatMulHeuristicResult(&heuristic_result));
    int32_t requested_algo_count = 1, return_algo_count = 0;

    CALL_CNNL(cnnlGetMatMulAlgoHeuristic(
        cnnl_handle, matmul_desc, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc,
        cnnl_d_desc, nullptr, requested_algo_count, &heuristic_result,
        &return_algo_count));
    CALL_CNNL(cnnlGetMatMulHeuristicResult(heuristic_result, matmul_algo,
                                           &workspace_size));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_d_desc);
    CALL_CNNL(cnnlMatMulDescDestroy(matmul_desc));
    CALL_CNNL(cnnlMatMulAlgoDestroy(matmul_algo));
    CALL_CNNL(cnnlDestroyMatMulHeuristicResult(heuristic_result));
  }

  status = mluOpDestroyTensorDescriptor(a_desc);
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(b_desc);
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(c_desc);
  CHECK_RETURN(api, status);
  // destroy cnnl descriptor
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);

  DESTROY_CNNL_HANDLE(cnnl_handle);
  return status;
}

mluOpStatus_t fftQuantMatMul(mluOpHandle_t handle, int m, int k, int n,
                             void *a_ptr, void *a_pos, void *a_scale,
                             void *b_ptr, void *b_pos, void *b_scale,
                             void *c_ptr, bool is_trans_a, bool is_trans_b,
                             float alpha, float beta,
                             mluOpDataType_t a_compute_type,
                             mluOpDataType_t b_compute_type,
                             mluOpDataType_t data_type, void *workspace,
                             size_t workspace_size, const std::string api) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  int trans_a_int = (int)is_trans_a;
  int trans_b_int = (int)is_trans_b;

  // create descriptor
  mluOpTensorDescriptor_t a_desc = nullptr;
  mluOpTensorDescriptor_t b_desc = nullptr;
  mluOpTensorDescriptor_t c_desc = nullptr;
  status = mluOpCreateTensorDescriptor(&a_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&b_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&c_desc);
  CHECK_RETURN(api, status);

  // set descriptor
  int64_t a_dims[2];
  int64_t b_dims[2];
  int64_t c_dims[2] = {m, n};
  if (is_trans_a) {
    a_dims[0] = k;
    a_dims[1] = m;
  } else {
    a_dims[0] = m;
    a_dims[1] = k;
  }
  if (is_trans_b) {
    b_dims[0] = n;
    b_dims[1] = k;
  } else {
    b_dims[0] = k;
    b_dims[1] = n;
  }

  status = mluOpSetTensorDescriptor_v2(a_desc, MLUOP_LAYOUT_ARRAY, data_type, 2,
                                       a_dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptorOnchipDataType(a_desc, a_compute_type);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(b_desc, MLUOP_LAYOUT_ARRAY, data_type, 2,
                                       b_dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptorOnchipDataType(b_desc, b_compute_type);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(c_desc, MLUOP_LAYOUT_ARRAY, data_type, 2,
                                       c_dims);
  CHECK_RETURN(api, status);
  if (a_compute_type == MLUOP_DTYPE_INT31 ||
      b_compute_type == MLUOP_DTYPE_INT31) {
    status = mluOpSetTensorDescriptorOnchipDataType(c_desc, MLUOP_DTYPE_FLOAT);
    CHECK_RETURN(api, status);
  } else if (fftIsIntDtype(a_compute_type) && fftIsIntDtype(b_compute_type) &&
             c_desc->getDtype() == MLUOP_DTYPE_HALF) {
    status = mluOpSetTensorDescriptorOnchipDataType(c_desc, MLUOP_DTYPE_FLOAT);
    CHECK_RETURN(api, status);
  } else {
    status = mluOpSetTensorDescriptorOnchipDataType(c_desc, data_type);
    CHECK_RETURN(api, status);
  }

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                    cnnl_handle);  // convert to cnnl_handle

  // convert to cnnl_tensor_descriptor
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_d_desc);

  // compute matmul result
  if (fftIsIntDtype(a_compute_type) && fftIsIntDtype(b_compute_type)) {
    cnnlMatMulDescriptor_t matmul_desc;
    CALL_CNNL(cnnlMatMulDescCreate(&matmul_desc));
    CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_COMPUTE_TYPE,
                                    &data_type, sizeof(int32_t)));
    CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_TRANSA,
                                    &trans_a_int, sizeof(int32_t)));
    CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_TRANSB,
                                    &trans_b_int, sizeof(int32_t)));

    cnnlMatMulAlgo_t matmul_algo;
    CALL_CNNL(cnnlMatMulAlgoCreate(&matmul_algo));
    cnnlMatMulPreference_t preference = CNNL_MATMUL_FASTEST;
    CALL_CNNL(cnnlGetQuantizeMatMulAlgorithm(
        cnnl_handle, matmul_desc, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc,
        preference, &matmul_algo));

    const float one = 1.0;
    const float zero = 0.0;
    CALL_CNNL(cnnlQuantizeMatMul(
        cnnl_handle, matmul_desc, &one, cnnl_a_desc, a_ptr, a_pos, a_scale,
        nullptr, cnnl_b_desc, b_ptr, b_pos, b_scale, nullptr, &zero,
        cnnl_c_desc, c_ptr, matmul_algo, workspace, workspace_size));

    if ((alpha != 1.0) || (beta != 0.0)) {
      CALL_CNNL(cnnlTransform_v2(cnnl_handle, CNNL_POINTER_MODE_HOST, &alpha,
                                 cnnl_c_desc, c_ptr, &beta, cnnl_c_desc,
                                 c_ptr));
    }

    CALL_CNNL(cnnlMatMulDescDestroy(matmul_desc));
    CALL_CNNL(cnnlMatMulAlgoDestroy(matmul_algo));
  } else {
    c_desc->setOnchipDtype(MLUOP_DTYPE_FLOAT);
    cnnlMatMulDescriptor_t matmul_desc;
    cnnlMatMulAlgo_t matmul_algo;
    cnnlMatMulHeuristicResult_t heuristic_result;
    size_t matmul_ws_size = 0, workspace_size = 0;
    bool allow_tf32 = false;
    cnnlDataType_t cnnl_compute_type = CNNL_DTYPE_FLOAT;  // (TODO)

    CALL_CNNL(cnnlMatMulDescCreate(&matmul_desc));
    CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_TRANSA,
                                    &trans_a_int, sizeof(int32_t)));
    CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_TRANSB,
                                    &trans_b_int, sizeof(int32_t)));
    CALL_CNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_ALLOW_TF32,
                                    &allow_tf32, sizeof(int32_t)));
    CALL_CNNL(cnnlMatMulAlgoCreate(&matmul_algo));
    CALL_CNNL(cnnlCreateMatMulHeuristicResult(&heuristic_result));
    int32_t requested_algo_count = 1, return_algo_count = 0;

    CALL_CNNL(cnnlGetMatMulAlgoHeuristic(
        cnnl_handle, matmul_desc, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc,
        cnnl_d_desc, nullptr, requested_algo_count, &heuristic_result,
        &return_algo_count));
    CALL_CNNL(cnnlGetMatMulHeuristicResult(heuristic_result, matmul_algo,
                                           &workspace_size));
    float *workspace = nullptr;
    if (workspace_size > 0) {
      CNRT_CHECK(cnrtMalloc((void **)&workspace, workspace_size));
    }
    CALL_CNNL(cnnlMatMul_v2(cnnl_handle, matmul_desc, matmul_algo, &alpha,
                            cnnl_a_desc, a_ptr, cnnl_b_desc, b_ptr, &beta,
                            cnnl_c_desc, c_ptr, workspace, workspace_size,
                            cnnl_d_desc, c_ptr));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_d_desc);
    CALL_CNNL(cnnlMatMulDescDestroy(matmul_desc));
    CALL_CNNL(cnnlMatMulAlgoDestroy(matmul_algo));
    CALL_CNNL(cnnlDestroyMatMulHeuristicResult(heuristic_result));
  }

  status = mluOpDestroyTensorDescriptor(a_desc);
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(b_desc);
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(c_desc);
  CHECK_RETURN(api, status);
  // destroy cnnl descriptor
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);

  DESTROY_CNNL_HANDLE(cnnl_handle);

  return status;
}

mluOpStatus_t fftGetBatchMatMulBcastWorkspaceSize(
    mluOpHandle_t handle,
    int m,  // 2 * L = 750
    int k,  // L = 375
    int n,  // 2^m = 128
    int batch, void *a_ptr, void *a_pos, void *a_scale, void *b_ptr,
    void *b_pos, void *b_scale, void *c_ptr, bool is_trans_a, bool is_trans_b,
    float alpha, float beta, mluOpDataType_t a_compute_type,
    mluOpDataType_t b_compute_type, mluOpDataType_t data_type, void *workspace,
    size_t workspace_size, const std::string api) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  int trans_a_int = (int)is_trans_a;
  int trans_b_int = (int)is_trans_b;

  // create descriptor
  mluOpTensorDescriptor_t a_desc = nullptr;
  mluOpTensorDescriptor_t b_desc = nullptr;
  mluOpTensorDescriptor_t c_desc = nullptr;
  status = mluOpCreateTensorDescriptor(&a_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&b_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&c_desc);
  CHECK_RETURN(api, status);

  // set descriptor
  int64_t a_dims[2];
  int64_t b_dims[3] = {batch, k, n};
  int64_t c_dims[3] = {batch, m, n};
  if (is_trans_a) {
    a_dims[0] = k;
    a_dims[1] = m;
  } else {
    a_dims[0] = m;
    a_dims[1] = k;
  }
  if (is_trans_b) {
    b_dims[1] = n;
    b_dims[2] = k;
  } else {
    b_dims[1] = k;
    b_dims[2] = n;
  }
  status = mluOpSetTensorDescriptor_v2(a_desc, MLUOP_LAYOUT_ARRAY, data_type, 2,
                                       a_dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptorOnchipDataType(a_desc, a_compute_type);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(b_desc, MLUOP_LAYOUT_ARRAY, data_type, 3,
                                       b_dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptorOnchipDataType(b_desc, b_compute_type);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(c_desc, MLUOP_LAYOUT_ARRAY, data_type, 3,
                                       c_dims);
  CHECK_RETURN(api, status);
  c_desc->setOnchipDtype(MLUOP_DTYPE_FLOAT);
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                    cnnl_handle);  // convert to cnnl_handle

  // convert to cnnl_tensor_descriptor
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);
  cnnlMatMulAlgo_t algo;
  CALL_CNNL(cnnlMatMulAlgoCreate(&algo));
  cnnlMatMulDescriptor_t bmm_bcast_desc;
  bool use_stride = false;
  auto cast_mode = CNNL_MATMUL_BYPASS_QUANTIZE;
  CALL_CNNL(cnnlMatMulDescCreate(&bmm_bcast_desc));
  CALL_CNNL(cnnlSetMatMulDescAttr(bmm_bcast_desc, CNNL_MATMUL_DESC_TRANSA,
                                  &trans_a_int, sizeof(int32_t)));
  CALL_CNNL(cnnlSetMatMulDescAttr(bmm_bcast_desc, CNNL_MATMUL_DESC_TRANSB,
                                  &trans_b_int, sizeof(int32_t)));

  cnnlMatMulHeuristicResult_t heuristic_result;
  CALL_CNNL(cnnlCreateMatMulHeuristicResult(&heuristic_result));
  int requested_algo_count = 1, return_algo_count = 0;
  cnnlGetBatchMatMulExAlgoHeuristic(
      cnnl_handle, bmm_bcast_desc, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc, NULL,
      requested_algo_count, &heuristic_result, &return_algo_count);
  cnnlGetBatchMatMulExHeuristicResult(heuristic_result, algo, &workspace_size);
  // destroy descriptor
  status = mluOpDestroyTensorDescriptor(a_desc);
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(b_desc);
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(c_desc);
  CHECK_RETURN(api, status);
  // destroy cnnl descriptor
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  CALL_CNNL(cnnlMatMulDescDestroy(bmm_bcast_desc));
  CALL_CNNL(cnnlMatMulAlgoDestroy(algo));
  CALL_CNNL(cnnlDestroyMatMulHeuristicResult(heuristic_result));

  DESTROY_CNNL_HANDLE(cnnl_handle);

  return status;
}

mluOpStatus_t fftBatchMatMulBcast(
    mluOpHandle_t handle,
    int m,  // 2 * L = 750
    int k,  // L = 375
    int n,  // 2^m = 128
    int batch, void *a_ptr, void *a_pos, void *a_scale, void *b_ptr,
    void *b_pos, void *b_scale, void *c_ptr, bool is_trans_a, bool is_trans_b,
    float alpha, float beta, mluOpDataType_t a_compute_type,
    mluOpDataType_t b_compute_type, mluOpDataType_t data_type, void *workspace,
    size_t workspace_size, const std::string api) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  int trans_a_int = (int)is_trans_a;
  int trans_b_int = (int)is_trans_b;

  // create descriptor
  mluOpTensorDescriptor_t a_desc = nullptr;
  mluOpTensorDescriptor_t b_desc = nullptr;
  mluOpTensorDescriptor_t c_desc = nullptr;
  status = mluOpCreateTensorDescriptor(&a_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&b_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&c_desc);
  CHECK_RETURN(api, status);

  // set descriptor
  int64_t a_dims[2];
  int64_t b_dims[3] = {batch, k, n};
  int64_t c_dims[3] = {batch, m, n};
  if (is_trans_a) {
    a_dims[0] = k;
    a_dims[1] = m;
  } else {
    a_dims[0] = m;
    a_dims[1] = k;
  }
  if (is_trans_b) {
    b_dims[1] = n;
    b_dims[2] = k;
  } else {
    b_dims[1] = k;
    b_dims[2] = n;
  }
  status = mluOpSetTensorDescriptor_v2(a_desc, MLUOP_LAYOUT_ARRAY, data_type, 2,
                                       a_dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptorOnchipDataType(a_desc, a_compute_type);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(b_desc, MLUOP_LAYOUT_ARRAY, data_type, 3,
                                       b_dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptorOnchipDataType(b_desc, b_compute_type);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(c_desc, MLUOP_LAYOUT_ARRAY, data_type, 3,
                                       c_dims);
  CHECK_RETURN(api, status);
  c_desc->setOnchipDtype(MLUOP_DTYPE_FLOAT);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                    cnnl_handle);  // convert to cnnl_handle

  // convert to cnnl_tensor_descriptor
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);

  cnnlMatMulAlgo_t algo;
  CALL_CNNL(cnnlMatMulAlgoCreate(&algo));
  cnnlMatMulDescriptor_t bmm_bcast_desc;
  bool use_stride = false;
  auto cast_mode = CNNL_MATMUL_BYPASS_QUANTIZE;
  CALL_CNNL(cnnlMatMulDescCreate(&bmm_bcast_desc));
  CALL_CNNL(cnnlSetMatMulDescAttr(bmm_bcast_desc, CNNL_MATMUL_DESC_TRANSA,
                                  &trans_a_int, sizeof(int32_t)));
  CALL_CNNL(cnnlSetMatMulDescAttr(bmm_bcast_desc, CNNL_MATMUL_DESC_TRANSB,
                                  &trans_b_int, sizeof(int32_t)));

  cnnlMatMulHeuristicResult_t heuristic_result;
  CALL_CNNL(cnnlCreateMatMulHeuristicResult(&heuristic_result));
  alpha = 1.0;
  beta = 0.0;
  int requested_algo_count = 1, return_algo_count = 0;
  cnnlGetBatchMatMulExAlgoHeuristic(
      cnnl_handle, bmm_bcast_desc, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc, NULL,
      requested_algo_count, &heuristic_result, &return_algo_count);
  cnnlGetBatchMatMulExHeuristicResult(heuristic_result, algo, &workspace_size);
  if (workspace_size > 0) {
    CNRT_CHECK(cnrtMalloc((void **)&workspace, workspace_size));
  } else {
    CNRT_CHECK(cnrtMalloc((void **)&workspace, m * n * sizeof(float)));
  }

  CALL_CNNL(cnnlBatchMatMulEx(cnnl_handle, bmm_bcast_desc, algo, &alpha,
                              cnnl_a_desc, a_ptr, cnnl_b_desc, b_ptr, &beta,
                              cnnl_c_desc, c_ptr, (void *)workspace,
                              workspace_size));
  // destroy descriptor
  status = mluOpDestroyTensorDescriptor(a_desc);
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(b_desc);
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(c_desc);
  CHECK_RETURN(api, status);
  // destroy cnnl descriptor
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  CALL_CNNL(cnnlMatMulDescDestroy(bmm_bcast_desc));
  CALL_CNNL(cnnlMatMulAlgoDestroy(algo));
  CALL_CNNL(cnnlDestroyMatMulHeuristicResult(heuristic_result));

  DESTROY_CNNL_HANDLE(cnnl_handle);

  return status;
}

mluOpStatus_t fftGetTransposeWorkspaceSize(mluOpHandle_t handle,
                                           size_t &workspace_size, int dim_num,
                                           int64_t ori_dims[], int permute[],
                                           mluOpDataType_t data_type,
                                           const std::string api) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  // create descriptor
  mluOpTensorDescriptor_t input_desc = nullptr;
  status = mluOpCreateTensorDescriptor(&input_desc);
  CHECK_RETURN(api, status);

  cnnlTransposeDescriptor_t trans_desc = nullptr;
  CALL_CNNL(cnnlCreateTransposeDescriptor(&trans_desc));

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                    cnnl_handle);  // convert to cnnl_handle
  // set descriptor
  status = mluOpSetTensorDescriptor_v2(input_desc, MLUOP_LAYOUT_ARRAY,
                                       data_type, dim_num, ori_dims);
  CHECK_RETURN(api, status);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR_v2(input_desc, cnnl_input_desc);
  CALL_CNNL(cnnlSetTransposeDescriptor(trans_desc, dim_num, permute));
  // get workspace
  CALL_CNNL(cnnlGetTransposeWorkspaceSize(cnnl_handle, cnnl_input_desc,
                                          trans_desc, &workspace_size));

  status = mluOpDestroyTensorDescriptor(input_desc);
  CHECK_RETURN(api, status);
  // destroy descriptor
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  CALL_CNNL(cnnlDestroyTransposeDescriptor(trans_desc));
  DESTROY_CNNL_HANDLE(cnnl_handle);
  return status;
}

mluOpStatus_t fftTranspose(mluOpHandle_t handle, int dim_num,
                           int64_t ori_dims[], int64_t transed_dims[],
                           int permute[], void *ori_ptr, void *transed_ptr,
                           mluOpDataType_t data_type, void *workspace,
                           size_t workspace_size, const std::string api) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  // create descriptor
  mluOpTensorDescriptor_t input_desc = nullptr;
  mluOpTensorDescriptor_t transed_input_desc = nullptr;
  status = mluOpCreateTensorDescriptor(&input_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&transed_input_desc);
  CHECK_RETURN(api, status);

  // set descriptor
  status = mluOpSetTensorDescriptor_v2(input_desc, MLUOP_LAYOUT_ARRAY,
                                       data_type, dim_num, ori_dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(transed_input_desc, MLUOP_LAYOUT_ARRAY,
                                       data_type, dim_num, transed_dims);
  CHECK_RETURN(api, status);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                    cnnl_handle);  // convert to cnnl_handle

  // convert to cnnl_tensor_descriptor
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR_v2(input_desc, cnnl_input_desc);
  // cnnlTensorDescriptor_t cnnl_input_desc = NULL;
  // CALL_CNNL(cnnlSetTensorDescriptor_v2(cnnl_input_desc, CNNL_LAYOUT_ARRAY,
  // data_type,
  //                                   dim_num, ori_dims));
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR_v2(transed_input_desc,
                                                  cnnl_transed_input_desc);
  // compute transpose
  cnnlTransposeDescriptor_t trans_desc = nullptr;
  CALL_CNNL(cnnlCreateTransposeDescriptor(&trans_desc));
  CALL_CNNL(cnnlSetTransposeDescriptor(trans_desc, dim_num, permute));

  CALL_CNNL(cnnlTranspose_v2(cnnl_handle, trans_desc, cnnl_input_desc, ori_ptr,
                             cnnl_transed_input_desc, transed_ptr, workspace,
                             workspace_size));

  status = mluOpDestroyTensorDescriptor(input_desc);
  CHECK_RETURN(api, status);
  // destroy descriptor
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_transed_input_desc);
  CALL_CNNL(cnnlDestroyTransposeDescriptor(trans_desc));

  DESTROY_CNNL_HANDLE(cnnl_handle);

  return status;
}

mluOpStatus_t fftGetOptensorWorkspaceSize(mluOpHandle_t handle,
                                          size_t &workspace_size, int elem_num,
                                          mluOpDataType_t data_type,
                                          const std::string api) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  // create descriptor
  mluOpTensorDescriptor_t in1_desc = nullptr;
  mluOpTensorDescriptor_t in2_desc = nullptr;
  mluOpTensorDescriptor_t out_desc = nullptr;
  status = mluOpCreateTensorDescriptor(&in1_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&in2_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&out_desc);
  CHECK_RETURN(api, status);

  // set descriptor
  int64_t dims[1] = {elem_num};
  status = mluOpSetTensorDescriptor_v2(in1_desc, MLUOP_LAYOUT_ARRAY, data_type,
                                       1, dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(in2_desc, MLUOP_LAYOUT_ARRAY, data_type,
                                       1, dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(out_desc, MLUOP_LAYOUT_ARRAY, data_type,
                                       1, dims);
  CHECK_RETURN(api, status);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                    cnnl_handle);  // convert to cnnl_handle

  // convert to cnnl_tensor_descriptor
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(in1_desc, cnnl_in1_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(in2_desc, cnnl_in2_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(out_desc, cnnl_out_desc);

  // get workspace
  CALL_CNNL(cnnlGetOpTensorWorkspaceSize(cnnl_handle, cnnl_in1_desc,
                                         cnnl_in2_desc, cnnl_out_desc,
                                         &workspace_size));
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(in1_desc);
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(in2_desc);
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(out_desc);
  CHECK_RETURN(api, status);
  // destroy descriptor
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_in1_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_in2_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_out_desc);

  DESTROY_CNNL_HANDLE(cnnl_handle);

  return status;
}

mluOpStatus_t fftOptensor(mluOpHandle_t handle, int elem_num, void *in1_ptr,
                          void *in2_ptr, void *out_ptr, float alpha1,
                          float alpha2, float beta, mluOpDataType_t data_type,
                          cnnlOpTensorDesc_t op_type, void *workspace,
                          size_t workspace_size, const std::string api) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  // create descriptor
  mluOpTensorDescriptor_t in1_desc = nullptr;
  mluOpTensorDescriptor_t in2_desc = nullptr;
  mluOpTensorDescriptor_t out_desc = nullptr;
  status = mluOpCreateTensorDescriptor(&in1_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&in2_desc);
  CHECK_RETURN(api, status);
  status = mluOpCreateTensorDescriptor(&out_desc);
  CHECK_RETURN(api, status);

  // set descriptor
  int64_t dims[1] = {elem_num};
  status = mluOpSetTensorDescriptor_v2(in1_desc, MLUOP_LAYOUT_ARRAY, data_type,
                                       1, dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(in2_desc, MLUOP_LAYOUT_ARRAY, data_type,
                                       1, dims);
  CHECK_RETURN(api, status);
  status = mluOpSetTensorDescriptor_v2(out_desc, MLUOP_LAYOUT_ARRAY, data_type,
                                       1, dims);
  CHECK_RETURN(api, status);

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle,
                                    cnnl_handle);  // convert to cnnl_handle

  // convert to cnnl_tensor_descriptor
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(in1_desc, cnnl_in1_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(in2_desc, cnnl_in2_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(out_desc, cnnl_out_desc);
  // compute optensor
  cnnlOpTensorDescriptor_t opTensor_desc = nullptr;
  CALL_CNNL(cnnlCreateOpTensorDescriptor(&opTensor_desc));
  CALL_CNNL(cnnlSetOpTensorDescriptor(opTensor_desc, op_type,
                                      (cnnlDataType_t)data_type,
                                      CNNL_NOT_PROPAGATE_NAN));

  CALL_CNNL(cnnlOpTensor(cnnl_handle, opTensor_desc, &alpha1, cnnl_in1_desc,
                         in1_ptr, &alpha2, cnnl_in2_desc, in2_ptr, workspace,
                         workspace_size, &beta, cnnl_out_desc, out_ptr));

  // destroy descriptor
  CALL_CNNL(cnnlDestroyOpTensorDescriptor(opTensor_desc));

  status = mluOpDestroyTensorDescriptor(in1_desc);
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(in2_desc);
  CHECK_RETURN(api, status);
  status = mluOpDestroyTensorDescriptor(out_desc);
  CHECK_RETURN(api, status);
  // destroy cnnl descriptor
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_in1_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_in2_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_out_desc);

  DESTROY_CNNL_HANDLE(cnnl_handle);

  return status;
}

//
static void initBasicParam(const int n, int &L, int &m) {
  // split n into 2^m * L
  m = 0;
  L = n;
  while (1) {
    int rem = L % 2;
    if (rem != 0) {
      break;
    }
    m++;
    L = L / 2;
  }

  // when L is smaller than 64, encrease L to ensure IO efficiency.
  while (L < 64 && m > 1) {
    L = L * 2;
    m--;
  }
}

static bool findStockham(mluOpHandle_t handle, int &L, int &m, int &L_sub,
                         bool &find_stockham) {
  if (find_stockham) {
    int NFU_ALIGN_NUM = NFU_ALIGN_SIZE / sizeof(float);
    int max_nram_size = handle->nram_size + REM_FOR_STACK - 32 * 1024;
    L_sub = PAD_UP(L, NFU_ALIGN_NUM);
    int L_tmp = L;
    int m_tmp = m;
    // one calculation requires 14 copies of space as follows:
    // input(4): y_in_r, z_in_r, y_in_i, z_in_i,
    // output(4): x_out1_r, x_out2_r, x_out1_i, x_out2_i,
    // w matrix(6): w_r, w_i, wz_rr, wz_ri, wz_ir, wz_ii
    // 2 represents ping_pong for pipeline.
    // 1 represents a public space stores the incremental sequence shared by
    // ping_pong.
    int cal_unit_tmp = 14 * 2 + 1;
    size_t cal_unit_once_tmp =
        L_sub * pow(2, m_tmp - 1) * cal_unit_tmp * sizeof(float);
    while (cal_unit_once_tmp > max_nram_size) {
      if (L_sub >= NFU_ALIGN_NUM * 2) {
        L_sub -= NFU_ALIGN_NUM;
      } else if (m_tmp > 1) {
        L_tmp = L_tmp * 2;
        m_tmp--;
      } else {
        break;
      }
      cal_unit_once_tmp =
          L_sub * pow(2, m_tmp - 1) * cal_unit_tmp * sizeof(float);
    }
    if (cal_unit_once_tmp < max_nram_size && L_tmp <= 4096) {
      L = L_tmp;
      m = m_tmp;
      L_sub = std::min(L, PAD_UP(L_sub, NFU_ALIGN_NUM));
      VLOG(5) << "m: " << m << ", L: " << L << ", L_sub: " << L_sub;
      return true;
    }
  }
  return false;
}

static bool findCooleyTukey(mluOpHandle_t handle, int &L, int &m, int &s) {
  int cal_unit = 14;
  int cal_unit_once =
      PAD_UP(L, NFU_ALIGN_SIZE / sizeof(float)) * cal_unit * sizeof(float);

  // calculate s
  s = m;
  if (cal_unit_once <= handle->nram_size) {
    // split m
    for (int i = m; i >= 0; i--) {
      size_t space_use = pow(2, i) * cal_unit_once;
      if (space_use < handle->nram_size) {
        s = i;
        break;
      }
    }
  } else {
    m = 0;
    return -1;
  }
  if (s == m) {
    s--;
  }

  VLOG(5) << "m: " << m << ", L: " << L << ", s: " << s;

  return true;
}

// Find the most suitable parameters for Cooley-Tukey or Stockham algorithm.
int findFFTOptLimit(mluOpHandle_t handle, const int n, const int batch, int &m,
                    int &L, int &s, int &L_sub, bool &find_stockham) {
  initBasicParam(n, L, m);
  int flag = 0;
  int flag_stockham;
  int flag_cooley_tukey;
  flag_stockham = findStockham(handle, L, m, L_sub, find_stockham);
  if (flag_stockham && batch > FFT_STOCK_BATCH_LIMIT &&
      L > 30 * std::pow(2, m)) {
      // FFT_STOCK_BATCH_LIMIT & L > 30 * 2^m : Numerical
      // values derived from testing experience
    flag_cooley_tukey = findCooleyTukey(handle, L, m, s);
    // try Cooley-Tukey algo, which may has better performace
    if (flag_cooley_tukey) {
      flag = 1;  // Cooley-Tukey algo has better performance
    }
  }
  if (!flag_stockham) {  // if cannot deal by Stockham algo, try Cooley-Tukey
                         // algo
    flag = findCooleyTukey(handle, L, m, s);
  }
  return flag;
}
