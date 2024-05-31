
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

    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_a_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_b_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_c_desc);
    DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_d_desc);
    DESTROY_CNNL_HANDLE(cnnl_handle);

    MatrixAdd(handle, m_, n_, (float *)dev_c, n_, (float *)dev_d, (ldda), (float *)dev_d, (ldda), queue);

    return MLUOP_STATUS_SUCCESS;
}

int get_sgetrf_native_nb(int m, int n)
{
    int nb;
    int minmn = MIN(m, n);

    if (minmn <= 4096)
        nb = 64;
    else if (minmn <= 10240)
        nb = 128;
    else if (minmn <= 20480)
        nb = 512;
    else
        nb = 1024;

    return nb;
}

int sgetrf_mlu(
    mluOpHandle_t handle,
    int m, int n,
    float *dA, int ldda,
    int *ipiv,
    int *info, int mode)
{
    int nb;
    int maxm, maxn, minmn, liwork;
    int i, j, jb, rows, lddat;
    float *dAT = NULL, *dAP = NULL;
    int *diwork, *dipiv, *dipivinfo, *dinfo;

    minmn = MIN(m, n);
    nb = get_sgetrf_native_nb(m, n);

    float *workspace;
    cnrtMalloc((void **)&workspace, nb * nb * sizeof(float));

    cnrtQueue_t queue[2] = {NULL};
    // CNRT_CHECK(cnrtQueueCreate(&queue[0]));
    CNRT_CHECK(cnrtQueueCreate(&queue[1]));
    mluOpGetQueue(handle, &(queue[0]));

    liwork = m + minmn + 1;
    cnrtMalloc((void **)&diwork, liwork * sizeof(int));
    dipivinfo = diwork;    // dipivinfo size = m
    dipiv = dipivinfo + m; // dipiv size = minmn
    dinfo = dipiv + minmn; // dinfo size = 1
    cnrtMemsetAsync(dinfo, 0, sizeof(int), queue[0]);

    if (nb <= 1 || nb >= MIN(m, n))
    {
        sgetrf2_native(handle, m, n, dA, ldda, dipiv, dinfo, 0, queue[0]);
    }
    else
    {
        maxm = roundup(m, 32);
        maxn = roundup(n, 32);

        cnrtMalloc((void **)&dAP, nb * maxm * sizeof(float));
        if (m == n)
        {
            dAT = dA;
            lddat = ldda;
        }
        else
        {
            lddat = maxm; 
            cnrtMalloc((void **)&dAT, lddat * maxn);
        }
        cnrtQueueSync(queue[0]); 

        for (j = 0; j < minmn - nb; j += nb)
        {
            MyCnrtMemcpy2D(handle, m - j, nb, dA(j, j), ldda, dAP(0, 0), nb, 1, queue[0]);

            cnrtQueueSync(queue[0]); 

            rows = m - j;
            sgetrf2_native(handle, rows, nb, dAP(0,0), nb, dipiv+j, dinfo, j, queue[0]);

            cnrtQueueSync(queue[0]); 

            MyCnrtMemcpy2D(handle, m - j, nb, dAP(0, 0), nb, dA(j, j), ldda, 1, queue[0]);

            if (j + nb < minmn - nb)
            {
                trsm3(handle, nb, n - j - nb,
                      dA(j, j), ldda,
                      dA(j, j + nb), ldda,
                      workspace,
                      queue[0]);

                gemm3(handle,
                      m - (j + nb), n - j - nb, nb,
                      m, n,
                      dA(j + nb, j), dA(j, j + nb), dA(j + nb, j + nb), dA(j + nb, j + nb), ldda, queue[0]);
            }
            else
            {
                trsm3(handle,
                      nb, n - (j + nb),
                      dA(j, j), ldda,
                      dA(j, j + nb), ldda,
                      workspace, queue[0]);

                gemm3(handle,
                      m - (j + nb), n - (j + nb), nb,
                      m, n,
                      dA(j + nb, j), dA(j, j + nb), dA(j + nb, j + nb), dA(j + nb, j + nb), ldda, queue[0]);
            }
        }
        jb = MIN(m - j, n - j);

        if (jb > 0)
        {
            rows = m - j;

            MyCnrtMemcpy2D(handle, rows, jb, dA(j, j), ldda, dAP(0, 0), jb, 1, queue[0]);

            sgetrf2_native(handle, rows, jb, dAP(0,0), jb, dipiv+j, dinfo, j, queue[0]);

            MyCnrtMemcpy2D(handle, rows, jb, dAP(0, 0), jb, dA(j, j), ldda, 1, queue[0]);

            trsm3(handle,
                  jb, n - j - jb,
                  dA(j, j), ldda,
                  dA(j, j + jb), ldda,
                  workspace, queue[0]);
        }
    }

    cnrtQueueQuery(queue[0]);
    cnrtQueueQuery(queue[1]);

    cnrtFree(diwork);
    cnrtFree(dAP);
    if (m != n)
    {
        cnrtFree(dAT);
    }
    return *info;
}