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

#include "cholesky.h"
#include <cstdio>
#include <algorithm>
#include <string>

// calculates the required workspace size for performing the Cholesky
// decomposition on a given matrix or batch of matrices.
mluOpStatus_t MLUOP_WIN_API mluOpGetCholeskyWorkspaceSize(
    mluOpTensorDescriptor_t input_desc, size_t* size) {
  PARAM_CHECK("mluOpCholesky", input_desc != NULL);

  PARAM_CHECK("mluOpCholesky", input_desc->dim == 2 || input_desc->dim == 3);
  PARAM_CHECK("mluOpCholesky", input_desc->dims[0] > 0);
  PARAM_CHECK("mluOpCholesky", input_desc->dims[1] > 0);

  if (input_desc->dim == 3) {
    PARAM_CHECK("mluOpCholesky", input_desc->dims[2] > 0);
  }

  mluOpDataType_t dtype = input_desc->dtype;
  PARAM_CHECK("mluOpCholesky",
              dtype == MLUOP_DTYPE_FLOAT || dtype == MLUOP_DTYPE_COMPLEX_FLOAT);

  uint64_t type_size;
  MLUOP_CHECK(mluOpGetSizeOfDataType(dtype, &type_size));
  int64_t size_a = 0, lda = 0, size_c = 0, ldc = 0;
  int64_t batch_size = 1;
  int dim = input_desc->dim;
  if (dim == 2) {
    size_a = input_desc->dims[0];
  } else if (dim == 3) {
    batch_size = input_desc->dims[0];
    size_a = input_desc->dims[1];
  }

  if (dtype == MLUOP_DTYPE_FLOAT) {
    *size = size_a * size_a * sizeof(float) * batch_size * 3;
  } else {
    *size = size_a * size_a * sizeof(float) * 2 * batch_size * 3;
  }
  return MLUOP_STATUS_SUCCESS;
}

// performs the necessary operations to compute matrix transformations,
// potentially involving Cholesky decomposition or matrix transposition,
// depending on the input parameters.
mluOpStatus_t MLUOP_WIN_API
calculate_body(mluOpHandle_t handle, int batch_size,
               const mluOpTensorDescriptor_t input_desc, float* d_input,
               const mluOpTensorDescriptor_t output_desc, float* d_output,
               bool upper, float* workspace) {
  mluOpDataType_t dtype = input_desc->dtype;

  int recnb = RECNB;
  int gbstep = 0;
  int dim = input_desc->dim;
  bool is_row_major = (input_desc->strides)[dim - 1] == 1;

  uint64_t type_size;
  MLUOP_CHECK(mluOpGetSizeOfDataType(dtype, &type_size));
  int size_a = 0, lda = 0, size_c = 0, ldc = 0;
  if (dim == 2) {
    size_a = input_desc->dims[0];
    lda = input_desc->dims[1];
    size_c = output_desc->dims[0];
    ldc = output_desc->dims[1];
  } else if (dim == 3) {
    size_a = input_desc->dims[1];
    lda = input_desc->dims[2];
    size_c = output_desc->dims[1];
    ldc = output_desc->dims[2];
  }

  PARAM_CHECK("mluOpCholesky", lda >= size_a);
  PARAM_CHECK("mluOpCholesky", ldc >= size_c);

  cnrtQueue_t queue;
  mluOpGetQueue(handle, &queue);

  int jb;
  const float s_one = 1.0;
  const float s_neg_one = -1.0;

  if (dtype == MLUOP_DTYPE_FLOAT) {
    if (upper == true) {
      CHECK_RETURN("mluOpCholesky",
                   transpose(batch_size, size_a, size_a, d_input, d_output,
                             handle, dtype, workspace));
    } else {
      KernelMyCnrtMemcpy1D(d_input, d_output,
        size_a * lda * ((uint64_t)batch_size), queue, 0);
    }
  } else {
    CHECK_RETURN("mluOpCholesky",
                 transpose(batch_size, size_a * size_a, 2, d_input, d_output,
                           handle, MLUOP_DTYPE_FLOAT, workspace));
  }
  cnrtQueueSync(queue);
  int stride = size_a * lda;

  if (dtype == MLUOP_DTYPE_FLOAT) {
    int row = is_row_major ? lda : size_a;
    int nb = NB;
    set_half_zero(batch_size, stride, d_output, lda, lda, handle);
    cnrtQueueSync(queue);
    for (int j = 0; j < row; j += nb) {
      jb = std::min(nb, row - j);
      CHECK_RETURN("mluOpCholesky",
                   ssyrk(batch_size, stride, false, is_row_major, jb, j,
                         OFFSET_ROW(d_output, j, 0), lda,
                         OFFSET_ROW(d_output, j, j), lda, handle, workspace));
      cnrtQueueSync(queue);
      CHECK_RETURN("mluOpCholesky",
                   spotrf_recursion(batch_size, stride, is_row_major, false,
                                      jb, recnb, OFFSET_ROW(d_output, j, j),
                                      lda, j, handle, workspace));
      if (j + jb < row) {
        CHECK_RETURN(
            "mluOpCholesky",
            sgemm(batch_size, !is_row_major, is_row_major, row - j - jb, jb, j,
                  -1.0f, 1.0f, OFFSET_ROW(d_output, j + jb, 0), lda, stride,
                  OFFSET_ROW(d_output, j, 0), lda, stride,
                  OFFSET_ROW(d_output, j + jb, j), lda, stride, handle,
                  workspace));
        cnrtQueueSync(queue);
      }
      if (j + jb < row) {
        CHECK_RETURN(
            "mluOpCholesky",
            strsm(batch_size, stride, false, is_row_major, jb, row - j - jb,
                  OFFSET_ROW(d_output, j, j), lda,
                  OFFSET_ROW(d_output, j + jb, j), lda, handle, workspace));
        cnrtQueueSync(queue);
      }
    }

    if (upper) {
      cnrtQueueSync(queue);
      CHECK_RETURN("mluOpCholesky",
                   transpose(batch_size, size_a, size_a, d_output, workspace,
                             handle, dtype, workspace));
      cnrtQueueSync(queue);
      KernelMyCnrtMemcpy1D(workspace, d_output,
        size_a * lda * ((uint64_t)batch_size), queue, 0);
    }
  } else {
    recnb = CRECNB;
    int nb = CNB;
    int row = lda;
    float* r_start = d_output;
    float* i_start = d_output + size_a * lda;
    stride *= 2;

    set_half_zero(batch_size, stride, r_start, lda, lda, handle);
    set_half_zero(batch_size, stride, i_start, lda, lda, handle);
    cnrtQueueSync(queue);

    for (int j = 0; j < row; j += nb) {
      jb = std::min(nb, row - j);
      CHECK_RETURN("mluOpCholesky",
                   cherk(batch_size, stride, jb, j, r_start + j * lda,
                         i_start + j * lda, lda, r_start + j * lda + j,
                         i_start + j * lda + j, lda, handle, workspace));
      cnrtQueueSync(queue);
      CHECK_RETURN("mluOpCholesky",
                   cpotrf_recursion(
                       batch_size, stride, jb, recnb, r_start + j * lda + j,
                       i_start + j * lda + j, lda, handle, workspace));
      cnrtQueueSync(queue);
      if (j + jb < row) {
        CHECK_RETURN("mluOpCholesky",
                     cgemm(batch_size, false, true, row - j - jb, jb, j, -1.0f,
                           1.0f, OFFSET_ROW(r_start, j + jb, 0),
                           OFFSET_ROW(i_start, j + jb, 0), lda, stride,
                           OFFSET_ROW(r_start, j, 0), OFFSET_ROW(i_start, j, 0),
                           lda, stride, OFFSET_ROW(r_start, j + jb, j),
                           OFFSET_ROW(i_start, j + jb, j), lda, stride, handle,
                           workspace));
        cnrtQueueSync(queue);
      }
      if (j + jb < row) {
        CHECK_RETURN(
            "mluOpCholesky",
            ctrsm(batch_size, stride, jb, row - j - jb,
                  OFFSET_ROW(r_start, j, j), OFFSET_ROW(i_start, j, j), lda,
                  OFFSET_ROW(r_start, j + jb, j),
                  OFFSET_ROW(i_start, j + jb, j), lda, handle, workspace));
        cnrtQueueSync(queue);
      }
    }

    CHECK_RETURN("mluOpCholesky",
                 transpose(batch_size, 2, size_a * size_a, d_output, workspace,
                           handle, MLUOP_DTYPE_FLOAT, workspace));
    cnrtQueueSync(queue);
    if (upper) {
      cnrtQueueSync(queue);
      CHECK_RETURN("mluOpCholesky",
                   transpose(batch_size, size_a, size_a, workspace, d_output,
                             handle, dtype, workspace));
      cnrtQueueSync(queue);
      CHECK_RETURN("mluOpCholesky", conj_complex(batch_size, size_a, size_a,
                                                 d_output, d_output, handle));
      cnrtQueueSync(queue);
    } else {
        KernelMyCnrtMemcpy1D(workspace,
          d_output, size_a * lda * 16 * 2, queue, 0);
        KernelMyCnrtMemcpy1D(workspace + type_size / 4 * size_a * lda * 16,
          d_output + type_size / 4 * size_a * lda * 16,
          size_a * lda * ((uint64_t)batch_size - 16) * 2, queue, 0);
    }
  }

  cnrtQueueSync(queue);
  return MLUOP_STATUS_SUCCESS;
}

// computes the Cholesky decomposition.
// This function is designed to handle both single and batch processing of
// matrices in either 2D or 3D formats. The function ensures that the input
// matrices are either float or complex float types and performs the
// decomposition either on the upper or lower triangular part of the matrix,
// based on the 'upper' boolean flag.
mluOpStatus_t MLUOP_WIN_API
mluOpCholesky(mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
              float* d_input, const mluOpTensorDescriptor_t output_desc,
              float* d_output, bool upper, void* workspace) {
  PARAM_CHECK("mluOpCholesky", handle != NULL);
  PARAM_CHECK("mluOpCholesky", input_desc != NULL);
  PARAM_CHECK("mluOpCholesky", d_input != NULL);
  PARAM_CHECK("mluOpCholesky", output_desc != NULL);
  PARAM_CHECK("mluOpCholesky", d_output != NULL);
  PARAM_CHECK("mluOpCholesky", input_desc->layout == MLUOP_LAYOUT_ARRAY);
  PARAM_CHECK("mluOpCholesky", output_desc->layout == MLUOP_LAYOUT_ARRAY);

  PARAM_CHECK("mluOpCholesky", input_desc->dim == 2 || input_desc->dim == 3);
  PARAM_CHECK("mluOpCholesky", output_desc->dim == input_desc->dim);
  PARAM_CHECK("mluOpCholesky", input_desc->dims[0] > 0);
  PARAM_CHECK("mluOpCholesky", input_desc->dims[1] > 0);
  PARAM_CHECK("mluOpCholesky", output_desc->dims[0] > 0);
  PARAM_CHECK("mluOpCholesky", output_desc->dims[1] > 0);
  if (input_desc->dim == 2) {
    PARAM_CHECK("mluOpCholesky", input_desc->dims[0] == input_desc->dims[1]);
    PARAM_CHECK("mluOpCholesky", output_desc->dims[0] == output_desc->dims[1]);
  } else {
    PARAM_CHECK("mluOpCholesky", input_desc->dims[1] == input_desc->dims[2]);
    PARAM_CHECK("mluOpCholesky", output_desc->dims[1] == output_desc->dims[2]);
  }

  cnrtQueue_t queue;
  mluOpGetQueue(handle, &queue);

  if (input_desc->dim == 3) {
    PARAM_CHECK("mluOpCholesky", input_desc->dims[2] > 0);
    PARAM_CHECK("mluOpCholesky", output_desc->dims[2] > 0);
  }

  mluOpDataType_t dtype = input_desc->dtype;
  PARAM_CHECK("mluOpCholesky", dtype == output_desc->dtype);
  PARAM_CHECK("mluOpCholesky",
              dtype == MLUOP_DTYPE_FLOAT || dtype == MLUOP_DTYPE_COMPLEX_FLOAT);

  int dim = input_desc->dim;
  int size_a = 0, lda = 0, size_c = 0, ldc = 0;

  int batch_size = 1;
  if (dim == 2) {
    size_a = input_desc->dims[0];
    lda = input_desc->dims[1];
    size_c = output_desc->dims[0];
    ldc = output_desc->dims[1];
  } else if (dim == 3) {
    batch_size = input_desc->dims[0];
    size_a = input_desc->dims[1];
    lda = input_desc->dims[2];
    size_c = output_desc->dims[1];
    ldc = output_desc->dims[2];
  }
  calculate_body(handle, ((uint64_t)batch_size), input_desc, d_input,
    output_desc, d_output, upper, (float*)workspace);
  return MLUOP_STATUS_SUCCESS;
}


// m * n
mluOpStatus_t transpose(int batch, int m, int n, float* d_input,
                        float* d_output, mluOpHandle_t handle,
                        mluOpDataType_t type, float* workspace) {
  if (m == 0) return MLUOP_STATUS_SUCCESS;
  cnrtQueue_t queue;
  mluOpGetQueue(handle, &queue);

  mluOpTensorDescriptor_t trans_input_desc, trans_output_desc;
  std::string api_name = "Cholesky";
  const int input_dim = 3;

  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&trans_input_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&trans_output_desc));

  int32_t transpose_input_shape[3] = {batch, m, n};
  int32_t transpose_output_shape[3] = {batch, n, m};

  CHECK_RETURN(api_name,
               mluOpSetTensorDescriptor(trans_input_desc, MLUOP_LAYOUT_ARRAY,
                                        type, 3, transpose_input_shape));

  CHECK_RETURN(api_name,
               mluOpSetTensorDescriptor(trans_output_desc, MLUOP_LAYOUT_ARRAY,
                                        type, 3, transpose_output_shape));

  int permute[3] = {0, 2, 1};

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(trans_input_desc, cnnl_in_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(trans_output_desc,
                                               cnnl_out_desc);

  cnnlTransposeDescriptor_t cnnl_trans_desc = NULL;

  CALL_CNNL(cnnlCreateTransposeDescriptor(&cnnl_trans_desc));

  CALL_CNNL(cnnlSetTransposeDescriptor(cnnl_trans_desc, input_dim, permute));

  size_t size = 0;

  CALL_CNNL(cnnlGetTransposeWorkspaceSize(cnnl_handle, cnnl_in_desc,
                                          cnnl_trans_desc, &size));

  CALL_CNNL(cnnlTranspose_v2(cnnl_handle, cnnl_trans_desc, cnnl_in_desc,
                             d_input, cnnl_out_desc, d_output, workspace,
                             size));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t sgemm(int batch, bool trans_a, bool trans_b, int m, int n, int k,
                    float alpha, float beta, float* d_a, int lda, int stride_a,
                    float* d_b, int ldb, int stride_b, float* d_c, int ldc,
                    int stride_c, mluOpHandle_t handle, float* workspace) {
  if (k == 0) return MLUOP_STATUS_SUCCESS;

  int32_t batch_size_arr[1] = {batch};
  int64_t stride_a_arr[1] = {stride_a};
  int64_t stride_b_arr[1] = {stride_b};
  int64_t stride_c_arr[1] = {stride_c};

  std::string api_name = "Cholesky";

  cnrtQueue_t queue;
  mluOpGetQueue(handle, &queue);

  cnnlStrideBatchMatMulAlgo_t algo;
  CALL_CNNL(cnnlStrideBatchMatMulAlgoCreate(&algo));

  cnnlStrideBatchMatMulHeuristicResult_t heuristic_result;
  CALL_CNNL(cnnlCreateStrideBatchMatMulHeuristicResult(&heuristic_result));

  cnnlStrideBatchMatMulDescriptor_t stride_bmm_desc;
  CALL_CNNL(cnnlStrideBatchMatMulDescCreate(&stride_bmm_desc));
  int32_t allow_tf32 = 0, max_batch_dim = 1;
  CALL_CNNL(cnnlSetStrideBatchMatMulDescAttr(stride_bmm_desc,
                                             CNNL_STRIDE_BMM_ALLOW_TF32,
                                             &(allow_tf32), sizeof(int32_t)));
  CALL_CNNL(cnnlSetStrideBatchMatMulDescAttr(
      stride_bmm_desc, CNNL_STRIDE_BMM_MAX_BATCH_DIM, &(max_batch_dim),
      sizeof(int32_t)));

  mluOpTensorDescriptor_t matmul_a_desc, matmul_b_desc, matmul_c_desc;

  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_a_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_b_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_c_desc));

  int32_t matmul_a_shape[2] = {batch, stride_a};
  int32_t matmul_b_shape[2] = {batch, stride_b};
  int32_t matmul_c_shape[2] = {batch, stride_c};

  CHECK_RETURN(api_name,
               mluOpSetTensorDescriptor(matmul_a_desc, MLUOP_LAYOUT_ARRAY,
                                        MLUOP_DTYPE_FLOAT, 2, matmul_a_shape));
  CHECK_RETURN(api_name,
               mluOpSetTensorDescriptor(matmul_b_desc, MLUOP_LAYOUT_ARRAY,
                                        MLUOP_DTYPE_FLOAT, 2, matmul_b_shape));
  CHECK_RETURN(api_name,
               mluOpSetTensorDescriptor(matmul_c_desc, MLUOP_LAYOUT_ARRAY,
                                        MLUOP_DTYPE_FLOAT, 2, matmul_c_shape));

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_c_desc, cnnl_c_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_c_desc, cnnl_d_desc);

  int requested_algo_count = 1, return_algo_count = 0;
  size_t workspace_size;

  cnnlGetStrideBatchMatMulAlgoHeuristic_v2(
      cnnl_handle, stride_bmm_desc, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc,
      cnnl_d_desc, trans_a, trans_b,  &(alpha), &(beta), m, n, k, lda,
      ldb, ldc, ldc, batch_size_arr, stride_a_arr, stride_b_arr, stride_c_arr,
      stride_c_arr, nullptr, requested_algo_count, &heuristic_result,
      &return_algo_count);

  cnnlGetStrideBatchMatMulHeuristicResult(heuristic_result, &algo,
                                          &workspace_size);

  if (workspace_size > 0) {
    MLULOG("sgemm workspace size: %ld\n", workspace_size);
  }

  CALL_CNNL(cnnlStrideBatchMatMul_v3(
      cnnl_handle, stride_bmm_desc, algo, trans_a, trans_b, m, n, k,
      batch_size_arr, &(alpha), cnnl_a_desc, d_a, lda, stride_a_arr,
      cnnl_b_desc, d_b, ldb, stride_b_arr, &(beta), cnnl_c_desc, d_c, ldc,
      stride_c_arr, workspace, workspace_size, cnnl_d_desc, d_c, ldc,
      stride_c_arr));

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t conj_complex(int batch, int m, int n, float* d_input,
                           float* d_output, mluOpHandle_t handle) {
  if (m == 0) return MLUOP_STATUS_SUCCESS;
  cnrtQueue_t queue;
  mluOpGetQueue(handle, &queue);

  mluOpTensorDescriptor_t input_desc, output_desc;
  std::string api_name = "Cholesky";

  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&input_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&output_desc));

  int32_t input_shape[3] = {batch, m, n};
  int32_t output_shape[3] = {batch, m, n};

  CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                             input_desc, MLUOP_LAYOUT_ARRAY,
                             MLUOP_DTYPE_COMPLEX_FLOAT, 3, input_shape));

  CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                             output_desc, MLUOP_LAYOUT_ARRAY,
                             MLUOP_DTYPE_COMPLEX_FLOAT, 3, output_shape));

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_in_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, cnnl_out_desc);

  CALL_CNNL(
      cnnlConj(cnnl_handle, cnnl_in_desc, d_input, cnnl_out_desc, d_output));

  return MLUOP_STATUS_SUCCESS;
}
