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

  // cnnlInverse(cnnl_handle, cnnl_a_desc,work_space,false,cnnl_a_desc,work_space,cnnl_info_desc,info);
  
  MatrixInverse(handle,
                1,
                m,
                d_a, lda, m * lda,
                work_space, m, m * m,
                queue);
  // printf("    after inversing\n");
  cnnlStrideBatchMatMul(cnnl_handle, false, false, m, n, m,
                        1, 1.0,
                        cnnl_a_desc, work_space, m, m * m,
                        cnnl_b_desc, d_b, ldb, m * n,
                        0.0f,
                        cnnl_b_desc, d_b, ldb, m * n);
  // cnnlStrideBatchMatMul(cnnl_handle, false, true, n,m, m, 1, 1.0, cnnl_b_desc, d_b, ldb, n*ldb, cnnl_a_desc, work_space, m, m*m, 0.0f, cnnl_b_desc, d_b, ldb, n*ldb);

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

static mluOpStatus_t MLUOP_WIN_API MyCnrtMemcpy2D(
    mluOpHandle_t handle,
    int m, int n,
    float *dA, int ldda,
    float *dB, int lddb,
    int mode,
    cnrtQueue_t queue)
{
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  k_type = CNRT_FUNC_TYPE_UNION1;
  int dim_x = handle->core_num_per_cluster;
  k_dim.x = dim_x;
  k_dim.y = 1;
  k_dim.z = 1;

  CHECK_RETURN("[KernelMyCnrtMemcpy2D]", KernelMyCnrtMemcpy2D(
                                             k_dim, k_type, queue, MLUOP_DTYPE_FLOAT,
                                             m, n, dA, ldda, dB, lddb, mode));

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
  int dim_MN[2] = {m_ + k_, ldda};

  float *workspace;

  cnrtMalloc((void **)&workspace, m_ * n_ * sizeof(float));
  dev_c = workspace;

  mluOpTensorDescriptor_t a_desc, b_desc, c_desc, d_desc;

  MLUOP_CHECK(mluOpCreateTensorDescriptor(&a_desc));
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&b_desc));
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&c_desc));
  MLUOP_CHECK(mluOpCreateTensorDescriptor(&d_desc));

  mluOpSetTensorDescriptor(a_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2, dim_MN);
  mluOpSetTensorDescriptor(b_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2, dim_MN);
  mluOpSetTensorDescriptor(c_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2, dim_mn);
  mluOpSetTensorDescriptor(d_desc, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2, dim_MN);

  int is_trans_a = 0, is_trans_b = 0;
  int tf32_flag_int = 1;
  bool use_beta = true;
  bool use_stride = true;

  // launch matmul

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

  MLUOP_CHECK(mluOpDestroyTensorDescriptor(a_desc));
  MLUOP_CHECK(mluOpDestroyTensorDescriptor(b_desc));
  MLUOP_CHECK(mluOpDestroyTensorDescriptor(c_desc));
  MLUOP_CHECK(mluOpDestroyTensorDescriptor(d_desc));
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_d_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);

  MatrixAdd(handle, m_, n_, (float *)dev_c, n_, (float *)dev_d, (ldda), (float *)dev_d, (ldda), queue);

  return MLUOP_STATUS_SUCCESS;
}

int sgetrf_recpanel_native(
    mluOpHandle_t handle,
    int m, int n,
    float *dA, int ldda,
    int *dipiv, int *dipivinfo,
    int *dinfo, int gbstep,
    cnrtQueue_t queue, cnrtQueue_t update_queue)
{
  printf("    sgetrf_recpanel_native m n %d %d\n", m, n);
  int recpnb = 32;
  if (m == 0 || n == 0)
  {
    return 0;
  }
  else if (m < 0 || n < 0)
  {
    return -1;
  }

  int panel_nb = n;
  if (panel_nb <= recpnb)
  {
    sgetrf2_native(handle, m, n, dA, ldda, dipiv, dinfo, gbstep, queue);
    return 0;
  }
  else
  {
    // split A over two [A A2]
    // panel on A1, update on A2 then panel on A1
    int n1 = (n / 2);
    int n2 = n - n1;

    // panel on A1
    sgetrf_recpanel_native(handle, m, n1, dA(0, 0), ldda, dipiv, dipivinfo, dinfo, gbstep, queue, update_queue);

    // update A2
    int m1 = MIN(n1, m);

    float *workspace;
    cnrtMalloc((void **)&workspace, m1 * m1 * sizeof(float));

    printf("    -----------------------------------\n");
    printf("    trsm3 %d %d\n", m1, n2);

    trsm3(handle,
          m1, n2,
          dA(0, 0), ldda,
          dA(0, n1), ldda, workspace, queue);

    printf("    gemm3 %d %d %d\n", m - n1, n2, n1);

    int m_ = m - n1, n_ = n2, k_ = n1;

    gemm3(handle,
          m_, n_, k_,
          m, n,
          dA(n1, 0), dA(0, n1), dA(n1, n1), dA(n1, n1),
          ldda, queue);

    // panel on A2
    sgetrf_recpanel_native(handle, m - n1, n2, dA(n1, n1), ldda, dipiv + n1, dipivinfo + n1, dinfo, n1, queue, update_queue);
  }
  return 0;
}