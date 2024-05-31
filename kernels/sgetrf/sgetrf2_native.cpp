#include "sgetrf.h"

#include <time.h>
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/unary_op/unary_op_host.h"
#include "kernels/utils/cnnl_helper.h"

static mluOpStatus_t MLUOP_WIN_API MatrixInverse(
    mluOpHandle_t handle,
    int batch,
    int m,
    float *d_input, int ld_input, int stride_input,
    float *d_output, int ld_output, int stride_output,
    cnrtQueue_t queue)
{

  cnrtDim3_t dim;
  cnrtFunctionType_t func_type = CNRT_FUNC_TYPE_UNION1;
  dim.x = 4;
  dim.y = 1;
  dim.z = 1;

  CHECK_RETURN("kernelInverse", KernelInverse(
                                    dim, func_type, queue,
                                    batch,
                                    d_input, ld_input, stride_input,
                                    d_output, ld_output, stride_output,
                                    m));
  CNRT_CHECK(cnrtQueueSync(queue));

  return MLUOP_STATUS_SUCCESS;
}

/* m n是原矩阵的尺寸*/
static mluOpStatus_t trsm3(mluOpHandle_t handle, int m, int n, float *d_a, int lda, float *d_b, int ldb, float *work_space, cnrtQueue_t queue)
{
  if (n == 0)
    return MLUOP_STATUS_BAD_PARAM;
  mluOpTensorDescriptor_t matmul_a_desc, matmul_b_desc, info_desc;
  std::string api_name = "LU";

  int32_t *info;
  CNRT_CHECK(cnrtMalloc((void **)&info, sizeof(int32_t)));

  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_a_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&matmul_b_desc));
  CHECK_RETURN(api_name, mluOpCreateTensorDescriptor(&info_desc));
  int32_t matmul_a_shape[2] = {m, m};
  int32_t matmul_b_shape[2] = {m, ldb};
  int32_t info_shape[1] = {1};

  CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                             matmul_a_desc, MLUOP_LAYOUT_ARRAY,
                             MLUOP_DTYPE_FLOAT, 2, matmul_a_shape));
  CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                             matmul_b_desc, MLUOP_LAYOUT_ARRAY,
                             MLUOP_DTYPE_FLOAT, 2, matmul_b_shape));
  CHECK_RETURN(api_name, mluOpSetTensorDescriptor(
                             info_desc, MLUOP_LAYOUT_ARRAY,
                             MLUOP_DTYPE_INT32, 1, info_shape));

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_a_desc, cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(matmul_b_desc, cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(info_desc, cnnl_info_desc);

  MatrixInverse(handle,
                1,
                m,
                d_a, lda, m * lda,
                work_space, m, m * m,
                queue);

  cnnlStrideBatchMatMul(cnnl_handle, false, false, m, n, m,
                        1, 1.0,
                        cnnl_a_desc, work_space, m, m * m,
                        cnnl_b_desc, d_b, ldb, m * n,
                        0.0f,
                        cnnl_b_desc, d_b, ldb, m * n);

  CNRT_CHECK(cnrtQueueSync(queue));
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t MLUOP_WIN_API scal_ger(
    mluOpHandle_t handle,
    int M_size, int N_size, int ib, int J,
    int m, int n, int step,
    float *dA, int lda,
    int *info, int gbstep,
    cnrtQueue_t queue)
{
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  k_type = CNRT_FUNC_TYPE_UNION8;
  int dim_x = handle->core_num_per_cluster * 8;
  k_dim.x = dim_x;
  k_dim.y = 1;
  k_dim.z = 1;

  CHECK_RETURN("[KernelScal_ger]", KernelScal_ger(
                                       k_dim, k_type, queue, MLUOP_DTYPE_FLOAT,
                                       M_size, N_size, ib, J,
                                       m, n, step, dA, lda, info, gbstep));
  CNRT_CHECK(cnrtQueueSync(queue));

  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t MLUOP_WIN_API MatrixAdd(
    mluOpHandle_t handle,
    int m, int n,
    float *dA, int ldda,
    float *dB, int lddb,
    float *dC, int lddc,
    cnrtQueue_t queue)
{
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  k_type = CNRT_FUNC_TYPE_UNION1;
  int dim_x = handle->core_num_per_cluster * 1;
  k_dim.x = dim_x;
  k_dim.y = 1;
  k_dim.z = 1;

  CHECK_RETURN("[KernelMatrixAdd]", KernelMatrixAdd(
                                        k_dim, k_type, queue, MLUOP_DTYPE_FLOAT,
                                        m, n, dA, ldda, dB, lddb, dC, lddc));

  CNRT_CHECK(cnrtQueueSync(queue));
  return MLUOP_STATUS_SUCCESS;
};

static mluOpStatus_t MLUOP_WIN_API gemm3(
    mluOpHandle_t handle,
    int m_, int n_, int k_,
    int m, int n,
    float *dev_a, float *dev_b, float *dev_c, float *dev_d,
    int ldda,
    cnrtQueue_t queue)
{
  if (m_ <= 0 || n_ <= 0)
    return MLUOP_STATUS_BAD_PARAM;

  int dim_mn[2] = {m_, n_};
  int dim_kn[2] = {k_, n_};
  int dim_mk[2] = {m_, k_};
  int dim_mn2[2] = {m_ + k_, ldda};

  float *test_dA3;

  cnrtMalloc((void **)&test_dA3, m_ * n_ * sizeof(float));
  dev_c = test_dA3;

  mluOpTensorDescriptor_t a_desc, b_desc, c_desc, d_desc;

  MLUOP_CHECK(mluOpCreateTensorDescriptor(&a_desc));
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&b_desc));
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&c_desc));
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&d_desc));

  mluOpSetTensorDescriptor(a_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2, dim_mn2);
  mluOpSetTensorDescriptor(b_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2, dim_mn2);
  mluOpSetTensorDescriptor(c_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2, dim_mn);
  mluOpSetTensorDescriptor(d_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2, dim_mn2);

  int is_trans_a = 0, is_trans_b = 0;
  int tf32_flag_int = 1;
  bool use_beta = true;
  bool use_stride = true;

  float alpha_gemm = -1.0f, beta_gemm = 0.0f;

  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(a_desc,
                                               cnnl_a_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(b_desc,
                                               cnnl_b_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc,
                                               cnnl_c_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(c_desc,
                                               cnnl_d_desc);
  CALL_CNNL(
      cnnlStrideBatchMatMul(cnnl_handle, is_trans_a, is_trans_b, m_, n_, k_,
                            1, alpha_gemm,
                            cnnl_a_desc, dev_a, (ldda), (m_ * k_),
                            cnnl_b_desc, dev_b, (ldda), (k_ * n_), beta_gemm,
                            cnnl_c_desc, dev_c, (n_), (m_ * n_)));

  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_d_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);

  MatrixAdd(handle, m_, n_, (float *)dev_c, n_, (float *)dev_d, (ldda), (float *)dev_d, (ldda), queue);

  return MLUOP_STATUS_SUCCESS;
}

int sgetrf2_native(
    mluOpHandle_t handle,
    int m, int n,
    float *dA, int ldda,
    int *dipiv, int *dinfo,
    int gbstep, cnrtQueue_t queue)
{

  int arginfo = 0;
  if (m < 0)
  {
    arginfo = -1;
  }
  else if (n < 0)
  {
    arginfo = -2;
  }
  else if (ldda < MAX(1, n))
  {
    arginfo = -4;
  }

  if (arginfo != 0)
  {

    return arginfo;
  }

  // Quick return if possible
  if (m == 0 || n == 0)
  {
    return arginfo;
  }
  int nb = 16;
  int min_mn = MIN(m, n);
  int gbj, j, step, ib;
  float *workspace;
  cnrtMalloc((void **)&workspace, nb * nb * sizeof(float));

  for (j = 0; j < min_mn; j += nb)
  {
    ib = MIN(nb, min_mn - j);
    arginfo = scal_ger(handle,
                       m, n, ib, j,
                       m - j, ib, j,
                       dA, ldda,
                       dinfo, gbstep, queue);

    if ((n - j - ib) > 0)
    {
      trsm3(handle,
            ib, n - j - ib,
            dA(j, j), ldda,
            dA(j, j + ib), ldda, workspace, queue);

      gemm3(handle,
            m - (j + ib), n - (j + ib), ib,
            m, n,
            dA(ib + j, j), dA(j, j + ib), dA(j + ib, j + ib), dA(j + ib, j + ib),
            ldda, queue);
    }
  }
  return 0;
}
