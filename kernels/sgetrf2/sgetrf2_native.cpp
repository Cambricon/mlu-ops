#include "sgetrf2.h"

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
    mluOpDataType_t dtype,
    int batch,
    int m,
    float *d_input, int ld_input, int stride_input,
    float *d_output, int ld_output, int stride_output,
    cnrtQueue_t queue)
{

  cnrtDim3_t dim;
  cnrtFunctionType_t func_type = CNRT_FUNC_TYPE_UNION8;
  int dim_x;

  if (batch > 1)
  {
    func_type = CNRT_FUNC_TYPE_UNION8;
    dim_x = ROUNDUP(handle->core_num_per_cluster * batch, 32);
  }
  else
  {
    if (taskType == 1)
    {
      func_type = CNRT_FUNC_TYPE_UNION1;
      dim_x = handle->core_num_per_cluster * 1;
    }
    else if (taskType == 8)
    {
      func_type = CNRT_FUNC_TYPE_UNION1;
      dim_x = handle->core_num_per_cluster * 1;
    }
  }
  dim.x = dim_x;
  dim.y = 1;
  dim.z = 1;

  CHECK_RETURN("kernelInverse", KernelInverse(
                                    dim, func_type, queue, dtype,
                                    batch,
                                    d_input, ld_input, stride_input,
                                    d_output, ld_output, stride_output,
                                    m));
  CNRT_CHECK(cnrtQueueSync(queue));

  return MLUOP_STATUS_SUCCESS;
}
static mluOpStatus_t MLUOP_WIN_API CMatrixInverse(
    mluOpHandle_t handle, mluOpDataType_t dtype,
    int batch,
    int m,
    float *rd_input, float *id_input, int ld_input, int stride_input,
    float *rd_output, float *id_output, int ld_output, int stride_output,
    cnrtQueue_t queue)
{

  cnrtDim3_t dim;
  cnrtFunctionType_t func_type = CNRT_FUNC_TYPE_UNION8;
  int dim_x;

  if (batch > 1)
  {
    func_type = CNRT_FUNC_TYPE_UNION8;
    dim_x = ROUNDUP(handle->core_num_per_cluster * batch, 32);
  }
  else
  {
    if (taskType == 1)
    {
      func_type = CNRT_FUNC_TYPE_UNION1;
      dim_x = handle->core_num_per_cluster * 1;
    }
    else if (taskType == 8)
    {
      func_type = CNRT_FUNC_TYPE_UNION1;
      dim_x = handle->core_num_per_cluster * 1;
    }
  }
  dim.x = dim_x;
  dim.y = 1;
  dim.z = 1;

  CHECK_RETURN("kernelInverse", KernelComplexInverse(
                                    dim, func_type, queue, dtype, batch,
                                    rd_input, id_input, ld_input, stride_input,
                                    rd_output, id_output, ld_output, stride_output,
                                    m));
  CNRT_CHECK(cnrtQueueSync(queue));

  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t trsm3(mluOpHandle_t handle, mluOpDataType_t dtype,
                           int batch, int M, int N, int m, int n, float *d_a, int lda, float *d_b, int ldb, float *work_space, cnrtQueue_t queue)
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
  int32_t matmul_a_shape[2] = {batch * m, m};
  int32_t matmul_b_shape[2] = {batch * M, ldb};
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

  MatrixInverse(handle, dtype,
                batch,
                m,
                d_a, lda, M * lda,
                work_space, m, m * m,
                queue);
  cnnlStrideBatchMatMul(cnnl_handle, false, false, m, n, m,
                        batch, 1.0,
                        cnnl_a_desc, work_space, m, m * m,
                        cnnl_b_desc, d_b, ldb, M * ldb,
                        0.0f,
                        cnnl_b_desc, d_b, ldb, M * ldb);

  CNRT_CHECK(cnrtQueueSync(queue));
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t MLUOP_WIN_API scal_ger(
    mluOpHandle_t handle, mluOpDataType_t dtype,
    int batch, int M_size, int N_size, int ib, int J,
    int m, int n, int step,
    float *dA, int lda, int stride_a, float *workspace,
    int *dipiv, int *dipiv2, int *info, int gbstep, int mode,
    cnrtQueue_t queue)
{
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  int dim_x;

  if (batch > 1)
  {
    k_type = CNRT_FUNC_TYPE_UNION8;
    dim_x = ROUNDUP(handle->core_num_per_cluster * batch, 32);
  }
  else
  {
    if (taskType == 1)
    {
      k_type = CNRT_FUNC_TYPE_UNION1;
      dim_x = handle->core_num_per_cluster * 1;
    }
    else if (taskType == 8)
    {
      if (mode == 1)
      {
        if (m <= MAX_M_SIZE1 * TaskUnion1 || m > MAX_M_SIZE1 * TaskUnion8)
        {
          k_type = CNRT_FUNC_TYPE_UNION1;
          dim_x = handle->core_num_per_cluster * 1;
        }
        else
        {
          k_type = CNRT_FUNC_TYPE_UNION8;
          dim_x = handle->core_num_per_cluster * 8;
        }
      }
      else
      {
        k_type = CNRT_FUNC_TYPE_UNION8;
        dim_x = handle->core_num_per_cluster * 8;
      }
    }
  }

  k_dim.x = dim_x;
  k_dim.y = 1;
  k_dim.z = 1;

  CHECK_RETURN("[KernelScal_ger]", KernelScal_ger(
                                       k_dim, k_type, queue, dtype,
                                       batch, M_size, N_size, ib, J,
                                       m, n, step,
                                       dA, lda, stride_a, workspace,
                                       dipiv, dipiv2, info, gbstep, mode));
  CNRT_CHECK(cnrtQueueSync(queue));
  return MLUOP_STATUS_SUCCESS;
}

static mluOpStatus_t MLUOP_WIN_API ccal_ger(
    mluOpHandle_t handle, mluOpDataType_t dtype,
    int batch, int M_size, int N_size, int ib, int J,
    int m, int n, int step,
    float *d_rA, float *d_iA, int lda, int stride_a, float *workspace,
    int *dipiv, int *dipiv2, int *info, int gbstep, int mode,
    cnrtQueue_t queue)
{
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  int dim_x;

  if (batch > 1)
  {
    k_type = CNRT_FUNC_TYPE_UNION8;
    dim_x = ROUNDUP(handle->core_num_per_cluster * batch, 32);
  }
  else
  {
    if (taskType == 1)
    {
      k_type = CNRT_FUNC_TYPE_UNION1;
      dim_x = handle->core_num_per_cluster * 1;
    }
    else if (taskType == 8)
    {
      if (mode == 1)
      {
        if (m <= MAX_M_SIZE_COMPLEX_PIVOT * TaskUnion1 || m > MAX_M_SIZE1 * TaskUnion8)
        {
          k_type = CNRT_FUNC_TYPE_UNION1;
          dim_x = handle->core_num_per_cluster * 1;
        }
        else
        {
          k_type = CNRT_FUNC_TYPE_UNION8;
          dim_x = handle->core_num_per_cluster * 8;
        }
      }
      else
      {
        k_type = CNRT_FUNC_TYPE_UNION8;
        dim_x = handle->core_num_per_cluster * 8;
      }
    }
  }

  k_dim.x = dim_x;
  k_dim.y = 1;
  k_dim.z = 1;

  CHECK_RETURN("[KernelScal_ger]", KernelCcal_ger(
                                       k_dim, k_type, queue, dtype,
                                       batch, M_size, N_size, ib, J,
                                       m, n, step, d_rA, d_iA, lda, stride_a, workspace,
                                       dipiv, dipiv2, info, gbstep, mode));
  CNRT_CHECK(cnrtQueueSync(queue));

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API swap(
    mluOpHandle_t handle, mluOpDataType_t dtype,
    int batch, int M_size, int N_size, int ib, int J,
    int m, int n, int step,
    float *dA, float *d_rA, float *d_iA, int lda, int stride_a,
    int *dipiv, int *dipiv2, int *info, int gbstep,
    cnrtQueue_t queue)
{
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  int dim_x;

  if (batch > 1)
  {
    k_type = CNRT_FUNC_TYPE_UNION8;
    dim_x = ROUNDUP(handle->core_num_per_cluster * batch, 32);
  }
  else
  {
    if (taskType == 1)
    {
      k_type = CNRT_FUNC_TYPE_UNION1;
      dim_x = handle->core_num_per_cluster * 1;
    }
    else if (taskType == 8)
    {
      k_type = CNRT_FUNC_TYPE_UNION8;
      dim_x = handle->core_num_per_cluster * 8;
    }
  }

  k_dim.x = dim_x;
  k_dim.y = 1;
  k_dim.z = 1;

  CHECK_RETURN("[KernelScal_ger]", KernelSwap(
                                       k_dim, k_type, queue, dtype,
                                       batch, M_size, N_size, ib, J,
                                       m, n, step,
                                       dA, d_rA, d_iA, lda, stride_a,
                                       dipiv, dipiv2, info, gbstep));
  CNRT_CHECK(cnrtQueueSync(queue));
  return MLUOP_STATUS_SUCCESS;
}

int sgetrf2_native(
    mluOpHandle_t handle,
    mluOpDataType_t dtype,
    int batch, int m, int n,
    float *dA, float *d_rA, float *d_iA, int ldda, int gbm, int gbn,
    int *dipiv, int *dinfo,
    int gbstep, int mode, cnrtQueue_t queue)
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
  int *dipiv2;
  if (mode == 1)
  {
    cnrtMalloc((void **)&dipiv2, batch * m * sizeof(int));
    cnrtMemset(dipiv2, 0, batch * m * sizeof(int));
  }
  if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT)
    cnrtMalloc((void **)&workspace, batch * nb * nb * 2 * sizeof(float));
  else
    cnrtMalloc((void **)&workspace, batch * nb * nb * sizeof(float));

  for (j = 0; j < min_mn; j += nb)
  {

    ib = MIN(nb, min_mn - j);
    if (dtype == MLUOP_DTYPE_FLOAT)
    {
      arginfo = scal_ger(handle, dtype,
                         batch, m, n, ib, j,
                         m - j, ib, j,
                         dA, ldda, gbm * ldda, workspace,
                         dipiv + j, dipiv2, dinfo, gbstep, mode, queue);
      if (mode == 1)
      {

        if (gbn - (j + ib) - gbstep > 0)
        {

          swap(handle, dtype,
               batch, m, n, ib, j,
               m - j, gbn - (j + ib) - gbstep, j,
               dA + j + j * ldda + ib, NULL, NULL, ldda, gbm * ldda,
               dipiv + j, dipiv2, dinfo, gbstep, queue);
        }

        if (gbstep + j > 0)
        {
          swap(handle, dtype,
               batch, m, n, ib, j,
               m - j, gbstep + j, j,
               dA + j + j * ldda - gbstep - j, NULL, NULL, ldda, gbm * ldda,
               dipiv + j, dipiv2, dinfo, gbstep, queue);
        }
      }
    }
    else if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT)
    {
      arginfo = ccal_ger(handle, dtype,
                         batch, m, n, ib, j,
                         m - j, ib, j,
                         d_rA, d_iA, ldda, gbm * ldda, workspace,
                         dipiv + j, dipiv2, dinfo, gbstep, mode, queue);
      if (mode == 1)
      {

        if (gbn - (j + ib) - gbstep > 0)
        {
          swap(handle, dtype,
               batch, m, n, ib, j,
               m - j, gbn - (j + ib) - gbstep, j,
               NULL, d_rA + j + j * ldda + ib, d_iA + j + j * ldda + ib,
               ldda, gbm * ldda,
               dipiv + j, dipiv2, dinfo, gbstep, queue);
        }

        if (gbstep + j > 0)
        {
          swap(handle, dtype,
               batch, m, n, ib, j,
               m - j, gbstep + j, j,
               NULL, d_rA + j + j * ldda - gbstep - j, d_iA + j + j * ldda - gbstep - j,
               ldda, gbm * ldda,
               dipiv + j, dipiv2, dinfo, gbstep, queue);
        }
      }
    }

    if ((n - j - ib) > 0)
    {
      if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT)
      {

        ctrsm(handle, dtype,
              batch, gbm, ldda,
              ib, n - j - ib,
              d_rA(j, j), d_iA(j, j), ldda,
              d_rA(j, j + ib), d_iA(j, j + ib), ldda,
              workspace, queue);

        cgemm(handle, dtype,
              m - (j + ib), n - (j + ib), ib,
              batch, gbm, n,
              d_rA(ib + j, j), d_iA(ib + j, j), ldda,
              d_rA(j, j + ib), d_iA(j, j + ib), ldda,
              d_rA(j + ib, j + ib), d_iA(j + ib, j + ib), ldda,
              ldda, queue);
      }
      else if (dtype == MLUOP_DTYPE_FLOAT)
      {
        trsm3(handle, dtype,
              batch, gbm, ldda,
              ib, n - j - ib,
              dA(j, j), ldda,
              dA(j, j + ib), ldda, workspace, queue);

        gemm3(handle, dtype,
              m - (j + ib), n - (j + ib), ib,
              -1, 1,
              batch, gbm, n,
              dA(ib + j, j), dA(j, j + ib), dA(j + ib, j + ib), dA(j + ib, j + ib),
              ldda, queue);
      }
    }
  }

  cnrtFree(workspace);
  return 0;
}
