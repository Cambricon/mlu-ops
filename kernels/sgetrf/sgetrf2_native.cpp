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

static mluOpStatus_t MLUOP_WIN_API trsm(
    mluOpHandle_t handle,
    int m, int n,
    float *dA, int ldda,
    float *dB, int lddb,
    cnrtQueue_t queue)
{
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
  int dim_x = roundup(m, handle->core_num_per_cluster);
  k_dim.x = dim_x;
  k_dim.y = dim_x;
  k_dim.z = 1;

  CHECK_RETURN("[KernelTrsm]", KernelTrsm(
                                   k_dim, k_type, queue, MLUOP_DTYPE_FLOAT,
                                   m, n, dA, ldda, dB, lddb));

  CNRT_CHECK(cnrtQueueSync(queue));

  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t MLUOP_WIN_API scal_ger(
    mluOpHandle_t handle,
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

  k_type = CNRT_FUNC_TYPE_UNION8;
  int dim_x = handle->core_num_per_cluster * 8;
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

  // int m_=m-n1, n_=n2,k_=n1;
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
  int nb = 4;
  int min_mn = MIN(m, n);
  int gbj, j, step, ib;
  float *workspace;
  cnrtMalloc((void **)&workspace, nb * nb * sizeof(float));

  printf("      sgetrf2_native m n %d %d \n", m, n);
  for (j = 0; j < min_mn; j += nb)
  {
    // printf("-----------------------------------\n");
    // printf("第%d次循环\n",j/nb);
    ib = MIN(nb, min_mn - j);
    for (step = 0; step < ib; step++)
    {
      gbj = j + step;
      
      if (gbj < m)
      {
        // printf("scal_dger m n gbj %d %d %d \n",m-gbj,ib-step,gbj);
        arginfo = scal_ger(handle, m - gbj, ib - step, gbj, dA, ldda, dinfo, gbstep, queue);
        if (arginfo != 0)
        {
          printf("kernel failed with %d in scal_ger\n", arginfo);
          // break;
        }
      }
    }
    if ((n - j - ib) > 0)
    {
      // printf("      trsm %d %d\n",ib,n-j-ib);
      trsm(handle,
           ib, n - j - ib,
           dA(j, j), ldda,
           dA(j, j + ib), ldda, queue);
      // printf("      gemm %d %d %d\n",m-(j+ib),n-(j+ib),ib);
      gemm3(handle,
            m - (j + ib), n - (j + ib), ib,
            m, n,
            dA(ib + j, j), dA(j, j + ib), dA(j + ib, j + ib), dA(j + ib, j + ib),
            ldda, queue);
    }
  }
  return 0;
}
