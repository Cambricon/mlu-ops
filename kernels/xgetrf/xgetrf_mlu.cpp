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
#include <time.h>
#include "xgetrf.h"
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/unary_op/unary_op_host.h"
#include "kernels/utils/cnnl_helper.h"

void policyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                cnrtFunctionType_t *k_type, int batch, FUNCTYPE func) {
  int dim_x;
  int max_core_num = mluop::runtime::getCoreNumOfJobLimitCapability(handle);
  if (func == INV) {
    if (batch > 1) {
      *k_type = mluop::runtime::getJobLimitCapabilityCnrtFuncType(handle);
      dim_x = ROUNDUP(handle->core_num_per_cluster * batch, max_core_num);
    } else {
      *k_type = CNRT_FUNC_TYPE_UNION1;
      dim_x = handle->core_num_per_cluster * 1;
    }
    k_dim->x = dim_x;
    k_dim->y = 1;
    k_dim->z = 1;
  } else if (func == MEMCPY) {
    if (batch > 1) {
      *k_type = mluop::runtime::getJobLimitCapabilityCnrtFuncType(handle);
      dim_x = ROUNDUP(handle->core_num_per_cluster * batch, max_core_num);
    } else {
      *k_type = CNRT_FUNC_TYPE_UNION1;
      dim_x = handle->core_num_per_cluster * 1;
    }
    k_dim->x = dim_x;
    k_dim->y = 1;
    k_dim->z = 1;
  } else if (func == SWAP) {
    if (batch > 1) {
      *k_type = mluop::runtime::getJobLimitCapabilityCnrtFuncType(handle);
      dim_x = ROUNDUP(handle->core_num_per_cluster * batch, max_core_num);
    } else {
      if (taskType == 1) {
        *k_type = CNRT_FUNC_TYPE_UNION1;
        dim_x = handle->core_num_per_cluster * 1;
      } else if (taskType == 8) {
        *k_type = CNRT_FUNC_TYPE_UNION8;
        dim_x = handle->core_num_per_cluster * 8;
      }
    }
    k_dim->x = dim_x;
    k_dim->y = 1;
    k_dim->z = 1;
  }
}
void policyFunc2(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                 cnrtFunctionType_t *k_type, mluOpDataType_t dtype, int m,
                 int mode, int batch, FUNCTYPE func) {
  int dim_x;
  int max_core_num = mluop::runtime::getCoreNumOfJobLimitCapability(handle);
  if (func == SCALGER) {
    if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
      if (batch > 1) {
        *k_type = mluop::runtime::getJobLimitCapabilityCnrtFuncType(handle);
        dim_x = ROUNDUP(handle->core_num_per_cluster * batch, max_core_num);
      } else {
        if (taskType == 1) {
          *k_type = CNRT_FUNC_TYPE_UNION1;
          dim_x = handle->core_num_per_cluster * 1;
        } else if (taskType == 8) {
          if (mode == 1) {
            if (m <= MAX_M_SIZE_COMPLEX_PIVOT * TaskUnion1 ||
                m > MAX_M_SIZE_COMPLEX_PIVOT * TaskUnion8) {
              *k_type = CNRT_FUNC_TYPE_UNION1;
              dim_x = handle->core_num_per_cluster * 1;
            } else {
              *k_type =
                  mluop::runtime::getJobLimitCapabilityCnrtFuncType(handle);
              dim_x =
                  ROUNDUP(handle->core_num_per_cluster * batch, max_core_num);
            }
          } else {
            *k_type = mluop::runtime::getJobLimitCapabilityCnrtFuncType(handle);
            dim_x = ROUNDUP(handle->core_num_per_cluster * batch, max_core_num);
          }
        }
      }
    } else if (dtype == MLUOP_DTYPE_FLOAT) {
      if (batch > 1) {
        *k_type = mluop::runtime::getJobLimitCapabilityCnrtFuncType(handle);
        dim_x = ROUNDUP(handle->core_num_per_cluster * batch, max_core_num);
      } else {
        if (taskType == 1) {
          *k_type = CNRT_FUNC_TYPE_UNION1;
          dim_x = handle->core_num_per_cluster * 1;
        } else if (taskType == 8) {
          if (mode == 1) {
            if (m <= MAX_M_SIZE_PIVOT * TaskUnion1 ||
                m > MAX_M_SIZE_PIVOT * TaskUnion8) {
              *k_type = CNRT_FUNC_TYPE_UNION1;
              dim_x = handle->core_num_per_cluster * 1;
            } else {
              *k_type =
                  mluop::runtime::getJobLimitCapabilityCnrtFuncType(handle);
              dim_x =
                  ROUNDUP(handle->core_num_per_cluster * batch, max_core_num);
            }
          } else {
            *k_type = mluop::runtime::getJobLimitCapabilityCnrtFuncType(handle);
            dim_x = ROUNDUP(handle->core_num_per_cluster * batch, max_core_num);
          }
        }
      }
    }
    k_dim->x = dim_x;
    k_dim->y = 1;
    k_dim->z = 1;
  }
}
mluOpStatus_t MLUOP_WIN_API cnrtMemcpy2D(mluOpHandle_t handle, int batch, int m,
                                         int n, float *dA, int ldda,
                                         int stride_a, float *dB, int lddb,
                                         int stride_b, int mode,
                                         cnrtQueue_t queue) {
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  policyFunc(handle, &k_dim, &k_type, batch, MEMCPY);
  CHECK_RETURN(
      "[KernelcnrtMemcpy2D]",
      KernelcnrtMemcpy2D(k_dim, k_type, queue, MLUOP_DTYPE_FLOAT, batch, m, n,
                         dA, ldda, stride_a, dB, lddb, stride_b, mode));

  CNRT_CHECK(cnrtQueueSync(queue));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API MatrixInverse(mluOpHandle_t handle,
                                          mluOpDataType_t dtype, int batch,
                                          int m, float *d_input, int ld_input,
                                          int stride_input, float *d_output,
                                          int ld_output, int stride_output,
                                          cnrtQueue_t queue) {
  cnrtDim3_t dim;
  cnrtFunctionType_t func_type;

  policyFunc(handle, &dim, &func_type, batch, INV);

  CHECK_RETURN(
      "kernelInverse",
      KernelInverse(dim, func_type, queue, dtype, batch, d_input, ld_input,
                    stride_input, d_output, ld_output, stride_output, m));
  CNRT_CHECK(cnrtQueueSync(queue));

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
CMatrixInverse(mluOpHandle_t handle, mluOpDataType_t dtype, int batch, int m,
               float *rd_input, float *id_input, int ld_input, int stride_input,
               float *rd_output, float *id_output, int ld_output,
               int stride_output, cnrtQueue_t queue) {
  cnrtDim3_t dim;
  cnrtFunctionType_t func_type;

  policyFunc(handle, &dim, &func_type, batch, INV);

  CHECK_RETURN(
      "kernelInverse",
      KernelComplexInverse(dim, func_type, queue, dtype, batch, rd_input,
                           id_input, ld_input, stride_input, rd_output,
                           id_output, ld_output, stride_output, m));
  CNRT_CHECK(cnrtQueueSync(queue));

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t trsm(mluOpHandle_t handle, mluOpDataType_t dtype, int batch,
                   int M, int N, int m, int n, float *d_a, int lda, float *d_b,
                   int ldb, float *work_space, cnrtQueue_t queue) {
  if (n == 0) return MLUOP_STATUS_BAD_PARAM;

  int32_t *info;
  CNRT_CHECK(cnrtMalloc((void **)&info, sizeof(int32_t)));

  cnnlStrideBatchMatMulDescriptor_t matmul_desc;
  int tf32_flag_int = 1;
  int max_batch_dim = 1;
  CALL_CNNL(cnnlStrideBatchMatMulDescCreate(&matmul_desc));
  CALL_CNNL(
      cnnlSetStrideBatchMatMulDescAttr(matmul_desc, CNNL_STRIDE_BMM_ALLOW_TF32,
                                       &(tf32_flag_int), sizeof(int32_t)));
  CALL_CNNL(cnnlSetStrideBatchMatMulDescAttr(
      matmul_desc, CNNL_STRIDE_BMM_MAX_BATCH_DIM, &(max_batch_dim),
      sizeof(int32_t)));

  cnnlStrideBatchMatMulHeuristicResult_t cnnl_heuristic_result;
  cnnlStrideBatchMatMulAlgo_t cnnl_matmul_algo;
  CALL_CNNL(cnnlCreateStrideBatchMatMulHeuristicResult(&cnnl_heuristic_result));
  CALL_CNNL(cnnlStrideBatchMatMulAlgoCreate(&cnnl_matmul_algo));

  mluOpTensorDescriptor_t matmul_a_desc, matmul_b_desc, matmul_d_desc,
      info_desc;
  std::string api_name = "LU";
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_a_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_b_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_d_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&info_desc));
  int32_t matmul_a_shape[2] = {batch * m, m};
  int32_t matmul_b_shape[2] = {batch * M, ldb};
  int32_t matmul_d_shape[2] = {batch * M, ldb};  // not for real use
  int32_t info_shape[1] = {1};

  CHECK_RETURN(api_name,
               mluOpSetTensorDescriptor(matmul_a_desc, MLUOP_LAYOUT_ARRAY,
                                        MLUOP_DTYPE_FLOAT, 2, matmul_a_shape));
  CHECK_RETURN(api_name,
               mluOpSetTensorDescriptor(matmul_b_desc, MLUOP_LAYOUT_ARRAY,
                                        MLUOP_DTYPE_FLOAT, 2, matmul_b_shape));
  CHECK_RETURN(api_name,
               mluOpSetTensorDescriptor(matmul_d_desc, MLUOP_LAYOUT_ARRAY,
                                        MLUOP_DTYPE_FLOAT, 2, matmul_d_shape));
  CHECK_RETURN(api_name,
               mluOpSetTensorDescriptor(info_desc, MLUOP_LAYOUT_ARRAY,
                                        MLUOP_DTYPE_INT32, 1, info_shape));

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_d_desc, cnnl_d_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(info_desc, cnnl_info_desc);

  MatrixInverse(handle, dtype, batch, m, d_a, lda, M * lda, work_space, m,
                m * m, queue);

  const int32_t batch_size_arr[1] = {batch};
  float alpha_gemm = 1.0f, beta_gemm = 0.0f;
  const int64_t stride_a_arr[1] = {m * m};
  const int64_t stride_b_arr[1] = {M * ldb};
  const int64_t stride_c_arr[1] = {M * ldb};
  const int64_t stride_d_arr[1] = {M * ldb};

  VLOG(5) << "kernel cnnlStrideBatchMatMul_v3 for trsm";
  cnnlStrideBatchMatMul_v3(
      cnnl_handle, matmul_desc, cnnl_matmul_algo, false, false, m, n, m,
      batch_size_arr, &alpha_gemm, cnnl_a_desc, work_space, m, stride_a_arr,
      cnnl_b_desc, d_b, ldb, stride_b_arr, &beta_gemm, cnnl_b_desc, d_b, ldb,
      stride_c_arr, NULL, 0, cnnl_d_desc, d_b, ldb, stride_d_arr);

  CNRT_CHECK(cnrtQueueSync(queue));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t ctrsm(mluOpHandle_t handle, mluOpDataType_t dtype, int batch,
                    int M, int N, int m, int n, float *d_ra, float *d_ia,
                    int lda, float *d_rb, float *d_ib, int ldb,
                    float *work_space, cnrtQueue_t queue) {
  if (n == 0) return MLUOP_STATUS_BAD_PARAM;
  mluOpTensorDescriptor_t matmul_a_desc, matmul_b_desc, matmul_c_desc,
      info_desc;
  std::string api_name = "LU";

  float *inv_d_ra, *inv_d_ia;
  inv_d_ra = work_space;
  inv_d_ia = inv_d_ra + batch * m * m;

  CNRT_CHECK(cnrtMemset(inv_d_ra, 0, batch * m * m * sizeof(float)));
  CNRT_CHECK(cnrtMemset(inv_d_ia, 0, batch * m * m * sizeof(float)));

  float *d_rb1, *d_ib1;
  d_rb1 = inv_d_ia + batch * m * m;
  d_ib1 = d_rb1 + batch * m * n;

  cnrtMemcpy2D(handle, batch, m, n, d_rb, ldb, M * ldb, d_rb1, n, m * n, 1,
               queue);
  cnrtMemcpy2D(handle, batch, m, n, d_ib, ldb, M * ldb, d_ib1, n, m * n, 1,
               queue);

  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_a_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_b_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_c_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&info_desc));
  int32_t matmul_a_shape[2] = {batch * m, m};
  int32_t matmul_b_shape[2] = {batch * m, n};
  int32_t matmul_c_shape[2] = {batch * M, lda};
  int stride_a = m * m;
  int stride_b = m * n;
  int32_t info_shape[1] = {1};

  CHECK_RETURN(api_name,
               mluOpSetTensorDescriptor(matmul_a_desc, MLUOP_LAYOUT_ARRAY,
                                        MLUOP_DTYPE_FLOAT, 2, matmul_a_shape));
  CHECK_RETURN(api_name,
               mluOpSetTensorDescriptor(matmul_b_desc, MLUOP_LAYOUT_ARRAY,
                                        MLUOP_DTYPE_FLOAT, 2, matmul_b_shape));
  CHECK_RETURN(api_name,
               mluOpSetTensorDescriptor(matmul_c_desc, MLUOP_LAYOUT_ARRAY,
                                        MLUOP_DTYPE_FLOAT, 2, matmul_c_shape));
  CHECK_RETURN(api_name,
               mluOpSetTensorDescriptor(info_desc, MLUOP_LAYOUT_ARRAY,
                                        MLUOP_DTYPE_INT32, 1, info_shape));

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_c_desc, cnnl_c_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(info_desc, cnnl_info_desc);

  CMatrixInverse(handle, dtype, batch, m, d_ra, d_ia, lda, M * lda, inv_d_ra,
                 inv_d_ia, m, m * m, queue);

  int is_trans_a = 0, is_trans_b = 0;
  gemm_for_ctrsm(handle, MLUOP_DTYPE_FLOAT, m, n, m, 1.0, 0.0, batch, M, N,
                 inv_d_ra, d_rb1, d_rb, d_rb, lda, queue);
  cnrtQueueSync(queue);

  gemm_for_ctrsm(handle, MLUOP_DTYPE_FLOAT, m, n, m, -1.0, 1.0, batch, M, N,
                 inv_d_ia, d_ib1, d_rb, d_rb, lda, queue);
  cnrtQueueSync(queue);

  gemm_for_ctrsm(handle, MLUOP_DTYPE_FLOAT, m, n, m, 1.0, 0.0, batch, M, N,
                 inv_d_ra, d_ib1, d_ib, d_ib, lda, queue);
  cnrtQueueSync(queue);

  gemm_for_ctrsm(handle, MLUOP_DTYPE_FLOAT, m, n, m, 1.0, 1.0, batch, M, N,
                 inv_d_ia, d_rb1, d_ib, d_ib, lda, queue);
  CNRT_CHECK(cnrtQueueSync(queue));

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API gemm(mluOpHandle_t handle, mluOpDataType_t dtype,
                                 int m_, int n_, int k_, float alpha,
                                 float beta, int batch, int m, int n,
                                 float *dev_a, float *dev_b, float *dev_c,
                                 float *dev_d, int ldda, cnrtQueue_t queue) {
  if (m_ <= 0 || n_ <= 0) return MLUOP_STATUS_BAD_PARAM;

  int dim_mn[2] = {batch * m_, n_};
  int dim_kn[2] = {k_, n_};
  int dim_mk[2] = {m_, k_};
  int dim_MN[2] = {batch * m, ldda};

  int is_trans_a = 0, is_trans_b = 0;
  int tf32_flag_int = 1;
  bool use_beta = true;
  bool use_stride = true;
  int max_batch_dim = 1;
  float alpha_gemm = alpha, beta_gemm = beta;

  cnnlStrideBatchMatMulDescriptor_t matmul_desc;
  CALL_CNNL(cnnlStrideBatchMatMulDescCreate(&matmul_desc));
  CALL_CNNL(
      cnnlSetStrideBatchMatMulDescAttr(matmul_desc, CNNL_STRIDE_BMM_ALLOW_TF32,
                                       &(tf32_flag_int), sizeof(int32_t)));
  CALL_CNNL(cnnlSetStrideBatchMatMulDescAttr(
      matmul_desc, CNNL_STRIDE_BMM_MAX_BATCH_DIM, &(max_batch_dim),
      sizeof(int32_t)));

  mluOpTensorDescriptor_t a_desc, b_desc, c_desc, d_desc;
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&a_desc));
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&b_desc));
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&c_desc));
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&d_desc));

  mluOpSetTensorDescriptor(a_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
                           dim_MN);
  mluOpSetTensorDescriptor(b_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
                           dim_MN);
  mluOpSetTensorDescriptor(c_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
                           dim_MN);
  mluOpSetTensorDescriptor(d_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
                           dim_MN);

  // launch matmul
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_d_desc);

  const int32_t batch_size_arr[1] = {batch};
  int stride_a = m * ldda;
  const int64_t stride_a_arr[1] = {stride_a};
  const int64_t stride_b_arr[1] = {stride_a};
  const int64_t stride_c_arr[1] = {stride_a};
  const int64_t stride_d_arr[1] = {stride_a};
  int requested_algo_count = 1, return_algo_count = 0;

  size_t workspace_size_matmul = 0;
  void *workspace = NULL;

  cnnlStrideBatchMatMulHeuristicResult_t cnnl_heuristic_result;
  cnnlStrideBatchMatMulAlgo_t cnnl_matmul_algo;
  CALL_CNNL(cnnlCreateStrideBatchMatMulHeuristicResult(&cnnl_heuristic_result));
  CALL_CNNL(cnnlStrideBatchMatMulAlgoCreate(&cnnl_matmul_algo));
  cnnlGetStrideBatchMatMulAlgoHeuristic_v2(
      cnnl_handle, matmul_desc, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc,
      cnnl_d_desc, is_trans_a, is_trans_b, &(alpha_gemm), &(beta_gemm), m_, n_,
      k_, ldda, ldda, ldda, ldda, batch_size_arr, stride_a_arr, stride_b_arr,
      stride_c_arr, stride_d_arr, NULL, requested_algo_count,
      &cnnl_heuristic_result, &return_algo_count);
  cnnlGetStrideBatchMatMulHeuristicResult(
      cnnl_heuristic_result, &cnnl_matmul_algo, &workspace_size_matmul);
  VLOG(5) << "kernel cnnlStrideBatchMatMul_v3 for gemm";
  cnnlStrideBatchMatMul_v3(
      cnnl_handle, matmul_desc, cnnl_matmul_algo, is_trans_a, is_trans_b, m_,
      n_, k_, batch_size_arr, &(alpha_gemm), cnnl_a_desc, dev_a, ldda,
      stride_a_arr, cnnl_b_desc, dev_b, ldda, stride_b_arr, &(beta_gemm),
      cnnl_c_desc, dev_c, ldda, stride_c_arr, workspace, workspace_size_matmul,
      cnnl_d_desc, dev_d, ldda, stride_d_arr);

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_d_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);

  return MLUOP_STATUS_SUCCESS;
}
mluOpStatus_t MLUOP_WIN_API gemm_for_ctrsm(
    mluOpHandle_t handle, mluOpDataType_t dtype, int m_, int n_, int k_,
    float alpha, float beta, int batch, int m, int n, float *dev_a,
    float *dev_b, float *dev_c, float *dev_d, int ldda, cnrtQueue_t queue) {
  if (m_ <= 0 || n_ <= 0) return MLUOP_STATUS_BAD_PARAM;
  int dim_mn[2] = {batch * m_, n_};
  int dim_kn[2] = {batch * k_, n_};
  int dim_mk[2] = {m_, k_};
  int dim_mm[2] = {batch * m_, m_};
  int dim_MN[2] = {batch * m, ldda};

  int is_trans_a = 0, is_trans_b = 0;
  int tf32_flag_int = 1;
  bool use_beta = true;
  bool use_stride = true;
  int max_batch_dim = 1;
  float alpha_gemm = alpha, beta_gemm = beta;

  cnnlStrideBatchMatMulDescriptor_t matmul_desc;
  CALL_CNNL(cnnlStrideBatchMatMulDescCreate(&matmul_desc));
  CALL_CNNL(
      cnnlSetStrideBatchMatMulDescAttr(matmul_desc, CNNL_STRIDE_BMM_ALLOW_TF32,
                                       &(tf32_flag_int), sizeof(int32_t)));
  CALL_CNNL(cnnlSetStrideBatchMatMulDescAttr(
      matmul_desc, CNNL_STRIDE_BMM_MAX_BATCH_DIM, &(max_batch_dim),
      sizeof(int32_t)));

  mluOpTensorDescriptor_t a_desc, b_desc, c_desc, d_desc;
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&a_desc));
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&b_desc));
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&c_desc));
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&d_desc));

  mluOpSetTensorDescriptor(a_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
                           dim_mm);
  mluOpSetTensorDescriptor(b_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
                           dim_kn);
  mluOpSetTensorDescriptor(c_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
                           dim_MN);
  mluOpSetTensorDescriptor(d_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
                           dim_MN);

  // launch matmul
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_c_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc, cnnl_d_desc);

  const int32_t batch_size_arr[1] = {batch};
  int stride_a = m_ * m_;
  int stride_b = m_ * n_;
  int stride_c = m * ldda;
  const int64_t stride_a_arr[1] = {stride_a};
  const int64_t stride_b_arr[1] = {stride_b};
  const int64_t stride_c_arr[1] = {stride_c};
  const int64_t stride_d_arr[1] = {stride_c};
  int requested_algo_count = 1, return_algo_count = 0;

  size_t workspace_size_matmul = 0;
  void *workspace = NULL;

  cnnlStrideBatchMatMulHeuristicResult_t cnnl_heuristic_result;
  cnnlStrideBatchMatMulAlgo_t cnnl_matmul_algo;
  CALL_CNNL(cnnlCreateStrideBatchMatMulHeuristicResult(&cnnl_heuristic_result));
  CALL_CNNL(cnnlStrideBatchMatMulAlgoCreate(&cnnl_matmul_algo));

  cnnlGetStrideBatchMatMulAlgoHeuristic_v2(
      cnnl_handle, matmul_desc, cnnl_a_desc, cnnl_b_desc, cnnl_c_desc,
      cnnl_d_desc, is_trans_a, is_trans_b, &(alpha_gemm), &(beta_gemm), m_, n_,
      k_, m_, n_, ldda, ldda, batch_size_arr, stride_a_arr, stride_b_arr,
      stride_c_arr, stride_d_arr, NULL, requested_algo_count,
      &cnnl_heuristic_result, &return_algo_count);
  cnnlGetStrideBatchMatMulHeuristicResult(
      cnnl_heuristic_result, &cnnl_matmul_algo, &workspace_size_matmul);
  VLOG(5) << "kernel cnnlStrideBatchMatMul_v3 for gemm_for_ctrsm";
  cnnlStrideBatchMatMul_v3(
      cnnl_handle, matmul_desc, cnnl_matmul_algo, is_trans_a, is_trans_b, m_,
      n_, k_, batch_size_arr, &(alpha_gemm), cnnl_a_desc, dev_a, m_,
      stride_a_arr, cnnl_b_desc, dev_b, n_, stride_b_arr, &(beta_gemm),
      cnnl_c_desc, dev_c, ldda, stride_c_arr, workspace, workspace_size_matmul,
      cnnl_d_desc, dev_d, ldda, stride_d_arr);

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_d_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);

  return MLUOP_STATUS_SUCCESS;
}
mluOpStatus_t cgemm(mluOpHandle_t handle, mluOpDataType_t dtype, int m_, int n_,
                    int k_, int batch, int m, int n, float *d_ra, float *d_ia,
                    int lda, float *d_rb, float *d_ib, int ldb, float *d_rc,
                    float *d_ic, int ldc, int ldda, cnrtQueue_t queue) {
  if (m_ <= 0 || n_ <= 0) return MLUOP_STATUS_SUCCESS;

  float *r_c, *i_c;
  r_c = d_rc;
  i_c = d_ic;
  float alpha = -1;
  float beta = 1;

  gemm(handle, MLUOP_DTYPE_FLOAT, m_, n_, k_, alpha, beta, batch, m, n, d_ra,
       d_rb, r_c, r_c, ldda, queue);
  cnrtQueueSync(queue);
  gemm(handle, MLUOP_DTYPE_FLOAT, m_, n_, k_, -alpha, beta, batch, m, n, d_ia,
       d_ib, r_c, r_c, ldda, queue);
  cnrtQueueSync(queue);

  gemm(handle, MLUOP_DTYPE_FLOAT, m_, n_, k_, alpha, beta, batch, m, n, d_ra,
       d_ib, i_c, i_c, ldda, queue);
  cnrtQueueSync(queue);

  gemm(handle, MLUOP_DTYPE_FLOAT, m_, n_, k_, alpha, beta, batch, m, n, d_ia,
       d_rb, i_c, i_c, ldda, queue);
  cnrtQueueSync(queue);

  return MLUOP_STATUS_SUCCESS;
}
mluOpStatus_t MLUOP_WIN_API transpose(mluOpHandle_t handle,
                                      mluOpDataType_t dtype, int batch, int m,
                                      int n, float *input, float *output,
                                      cnrtQueue_t queue) {
  int input_dim = 2;
  const int permute[2] = {1, 0};
  int src_dims[2] = {batch * m * n, 2};
  int dst_dims[2] = {2, batch * m * n};
  size_t transpose_workspace_size = 0;

  mluOpTensorDescriptor_t input_desc;
  mluOpTensorDescriptor_t output_desc;
  cnnlTransposeDescriptor_t trans_desc;
  void *transpose_workspace = NULL;

  mluOpCreateTensorDescriptor(&input_desc);
  mluOpCreateTensorDescriptor(&output_desc);
  mluOpSetTensorDescriptor(input_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
                           src_dims);
  mluOpSetTensorDescriptor(output_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                           2, dst_dims);

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
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, cnnl_y_desc);
    VLOG(5) << "kernel cnnlTranspose_v2 for transpose";
    CALL_CNNL(cnnlTranspose_v2(cnnl_handle, trans_desc, cnnl_x_desc, input,
                               cnnl_y_desc, output, transpose_workspace,
                               transpose_workspace_size));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_y_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  CALL_CNNL(cnnlDestroyTransposeDescriptor(trans_desc));
  return MLUOP_STATUS_SUCCESS;
}
mluOpStatus_t MLUOP_WIN_API transpose_back(mluOpHandle_t handle,
                                           mluOpDataType_t dtype, int batch,
                                           int m, int n, float *output,
                                           void *workspace, cnrtQueue_t queue) {
  mluOpTensorDescriptor_t input_desc;
  mluOpTensorDescriptor_t output_desc;
  cnnlTransposeDescriptor_t trans_desc;
  float *input = (float *)workspace;
  // CNRT_CHECK(cnrtMemcpy(input, output, batch * m * n * 2 * sizeof(float),
  //                       cnrtMemcpyDevToDev));
  cnrtMemcpy2D(handle, batch, m, n, (float *)output, n, m * n, (float *)input,
               n, m * n, 2, queue);
  void *transpose_workspace = NULL;
  int input_dim = 2;
  const int permute[2] = {1, 0};
  int src_dims[2] = {2, batch * m * n};
  int dst_dims[2] = {batch * m * n, 2};
  size_t transpose_workspace_size = 0;

  mluOpCreateTensorDescriptor(&input_desc);
  mluOpCreateTensorDescriptor(&output_desc);
  mluOpSetTensorDescriptor(input_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
                           src_dims);
  mluOpSetTensorDescriptor(output_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                           2, dst_dims);

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
    DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(output_desc, cnnl_y_desc);
    VLOG(5) << "kernel cnnlTranspose_v2 for transpose_back";
    CALL_CNNL(cnnlTranspose_v2(cnnl_handle, trans_desc, cnnl_x_desc, input,
                               cnnl_y_desc, output, transpose_workspace,
                               transpose_workspace_size));
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_x_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_y_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);
  }
  CALL_CNNL(cnnlDestroyTransposeDescriptor(trans_desc));
  cnrtQueueSync(queue);

  return MLUOP_STATUS_SUCCESS;
}

int get_xgetrf_native_nb(int m, int n) {
  int nb;
  int minmn = MIN(m, n);

  // recommended 16、32、64
  if (minmn <= 65536)
    nb = 32;
  else
    nb = 64;

  return nb;
}

int xgetrf_mlu(mluOpHandle_t handle, mluOpDataType_t dtype, int batch, int m,
               int n, float *dA, float *d_rA, float *d_iA, int ldda, int *dipiv,
               int *info, int mode, void *workspace) {
  int nb;
  int minmn, liwork;
  int i, j, jb, rows;
  int *diwork, *dinfo;
  int gbm = m;

  /* Check arguments */
  *info = 0;
  if (m < 0)
    *info = -1;
  else if (n < 0)
    *info = -2;
  else if (ldda < MAX(1, n))
    *info = -4;

  if (*info != 0) {
    return *info;
  }

  // Quick return if possible
  if (m == 0 || n == 0) {
    return *info;
  }

  minmn = MIN(m, n);
  nb = get_xgetrf_native_nb(m, n);
  // nb = 32;

  liwork = 1;
  diwork = (dtype == MLUOP_DTYPE_COMPLEX_FLOAT)
               ? (int *)workspace + 2 * (m * n + m * m) * batch + m
               : (int *)workspace + batch * 64 * 64 + m;
  dinfo = diwork;  // dinfo size = 1
  CNRT_CHECK(cnrtMemset(dinfo, 0, sizeof(int)));

  if (nb <= 1 || nb >= MIN(m, n)) {
    if (dtype == MLUOP_DTYPE_FLOAT)
      xgetrf2_native(handle, dtype, batch, m, n, dA, NULL, NULL, ldda, m, n,
                     dipiv, dinfo, 0, mode, workspace, handle->queue);
    else if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT)
      xgetrf2_native(handle, dtype, batch, m, n, dA, d_rA(0, 0), d_iA(0, 0),
                     ldda, m, n, dipiv, dinfo, 0, mode, workspace,
                     handle->queue);
  } else {
    cnrtQueueSync(handle->queue);  //

    for (j = 0; j < minmn - nb; j += nb) {
      cnrtQueueSync(handle->queue);

      rows = m - j;
      if (dtype == MLUOP_DTYPE_FLOAT)
        xgetrf2_native(handle, dtype, batch, rows, nb, dA(j, j), NULL, NULL,
                       ldda, m, n, dipiv + j, dinfo, j, mode, workspace,
                       handle->queue);
      else if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT)
        xgetrf2_native(handle, dtype, batch, rows, nb, dA(j, j), d_rA(j, j),
                       d_iA(j, j), ldda, m, n, dipiv + j, dinfo, j, mode,
                       workspace, handle->queue);

      cnrtQueueSync(handle->queue);

      // do the small non-parallel computations (next panel update)
      if (j + nb < minmn - nb) {
        if (dtype == MLUOP_DTYPE_FLOAT) {
          trsm(handle, dtype, batch, m, n, nb, n - j - nb, dA(j, j), ldda,
               dA(j, j + nb), ldda, (float *)workspace, handle->queue);

          gemm(handle, dtype, m - (j + nb), n - j - nb, nb, -1, 1, batch, m, n,
               dA(j + nb, j), dA(j, j + nb), dA(j + nb, j + nb),
               dA(j + nb, j + nb), ldda, handle->queue);
        } else if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
          ctrsm(handle, dtype, batch, m, n, nb, n - j - nb, d_rA(j, j),
                d_iA(j, j), ldda, d_rA(j, j + nb), d_iA(j, j + nb), ldda,
                (float *)workspace, handle->queue);
          cgemm(handle, dtype, m - (j + nb), n - j - nb, nb, batch, m, n,
                d_rA(j + nb, j), d_iA(j + nb, j), ldda, d_rA(j, j + nb),
                d_iA(j, j + nb), ldda, d_rA(j + nb, j + nb),
                d_iA(j + nb, j + nb), ldda, ldda, handle->queue);
        }
      } else {
        if (dtype == MLUOP_DTYPE_FLOAT) {
          trsm(handle, dtype, batch, m, n, nb, n - (j + nb), dA(j, j), ldda,
               dA(j, j + nb), ldda, (float *)workspace, handle->queue);

          gemm(handle, dtype, m - (j + nb), n - (j + nb), nb, -1, 1, batch, m,
               n, dA(j + nb, j), dA(j, j + nb), dA(j + nb, j + nb),
               dA(j + nb, j + nb), ldda, handle->queue);
        } else if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
          ctrsm(handle, dtype, batch, m, n, nb, n - (j + nb), d_rA(j, j),
                d_iA(j, j), ldda, d_rA(j, j + nb), d_iA(j, j + nb), ldda,
                (float *)workspace, handle->queue);
          cgemm(handle, dtype, m - (j + nb), n - (j + nb), nb, batch, m, n,
                d_rA(j + nb, j), d_iA(j + nb, j), ldda, d_rA(j, j + nb),
                d_iA(j, j + nb), ldda, d_rA(j + nb, j + nb),
                d_iA(j + nb, j + nb), ldda, ldda, handle->queue);
        }
      }
    }
    jb = MIN(m - j, n - j);

    if (jb > 0) {
      rows = m - j;

      if (dtype == MLUOP_DTYPE_FLOAT) {
        xgetrf2_native(handle, dtype, batch, rows, jb, dA(j, j), NULL, NULL,
                       ldda, m, n, dipiv + j, dinfo, j, mode, workspace,
                       handle->queue);

        trsm(handle, dtype, batch, m, n, jb, n - j - jb, dA(j, j), ldda,
             dA(j, j + jb), ldda, (float *)workspace, handle->queue);
      } else if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
        xgetrf2_native(handle, dtype, batch, rows, jb, dA(j, j), d_rA(j, j),
                       d_iA(j, j), ldda, m, n, dipiv + j, dinfo, j, mode,
                       workspace, handle->queue);

        ctrsm(handle, dtype, batch, m, n, jb, n - j - jb, d_rA(j, j),
              d_iA(j, j), ldda, d_rA(j, j + jb), d_iA(j, j + jb), ldda,
              (float *)workspace, handle->queue);
      }
    }
  }

  cnrtMemcpy(info, dinfo, sizeof(int), cnrtMemcpyDevToHost);
  cnrtQueueQuery(handle->queue);

  return *info;
}
