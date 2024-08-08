#include "sgetrf2.h"
#include <time.h>
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/unary_op/unary_op_host.h"
mluOpStatus_t MLUOP_WIN_API mluOpGetLUWorkspace(mluOpHandle_t handle,
                                                const mluOpTensorDescriptor_t x_desc,
                                                int *workspace_size,
                                                void **workspace)
{
    /* sgetrf参数转换*/
    int m, n, batch = 1;
    mluOpDataType_t dtype = x_desc->dtype;

    if (x_desc->dim == 2)
    {
        m = x_desc->dims[0];
        n = x_desc->dims[1];
    }
    else if (x_desc->dim == 3)
    {
        batch = x_desc->dims[0];
        m = x_desc->dims[1];
        n = x_desc->dims[2];
    }
    else if (x_desc->dim == 4)
    {
        batch = x_desc->dims[0] * x_desc->dims[1];
        m = x_desc->dims[2];
        n = x_desc->dims[3];
    }
    int tol = 1024;
    if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT)
        *workspace_size = 2 * (m * n + m * m) * batch + m + 2 * m + tol;
    else if (dtype == MLUOP_DTYPE_FLOAT)
        *workspace_size = batch * 64 * 64 + m + 2 * m + tol;

    if (*workspace_size)
    {
        CNRT_CHECK(cnrtMalloc((void **)workspace, (*workspace_size) * sizeof(float)));
    }

    return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpFreeLUWorkspace(void **workspace)
{
    PARAM_CHECK("mluOpSgetrf2", workspace != NULL);
    if (*workspace != NULL)
    {
        CNRT_CHECK(cnrtFree((void *)(*workspace)));
        *workspace = NULL;
    }
    return MLUOP_STATUS_SUCCESS;
}
mluOpStatus_t MLUOP_WIN_API mluOpSgetrf2(mluOpHandle_t handle,
                                         const mluOpTensorDescriptor_t x_desc,
                                         void *x,
                                         const mluOpTensorDescriptor_t y_desc,
                                         void *y,
                                         void *workspace,
                                         int *ipiv,
                                         int *info,
                                         int mode)
{
    /* sgetrf参数转换*/
    int m, n, batch = 1;
    mluOpDataType_t dtype = x_desc->dtype;

    if (x_desc->dim == 2)
    {
        m = x_desc->dims[0];
        n = x_desc->dims[1];
    }
    else if (x_desc->dim == 3)
    {
        batch = x_desc->dims[0];
        m = x_desc->dims[1];
        n = x_desc->dims[2];
    }
    else if (x_desc->dim == 4)
    {
        batch = x_desc->dims[0] * x_desc->dims[1];
        m = x_desc->dims[2];
        n = x_desc->dims[3];
    }
    mluOpGetQueue(handle, &(handle->queue));
    int trans = x_desc->strides[x_desc->dim - 1] == 1 ? 0 : 1;
    int ldda = n;
    if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT)
    {

        transpose(handle, MLUOP_DTYPE_COMPLEX_FLOAT, batch, m, n, (float *)x, (float *)y, handle->queue);

    }

    else
        cnrtMemcpy((float *)y, (float *)x, batch * m * n * sizeof(float), CNRT_MEM_TRANS_DIR_DEV2DEV);

    if (mode == 0) 
    {
        if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT)
            sgetrf_mlu(handle, dtype, batch, m, n,
                       (float *)y, (float *)y, (float *)y + batch * m * ldda, ldda,
                       ipiv, info, mode, workspace);
        else if (dtype == MLUOP_DTYPE_FLOAT)
            sgetrf_mlu(handle, dtype, batch, m, n,
                       (float *)y, NULL, NULL, ldda,
                       ipiv, info, mode, workspace);
    }
    else
    {
        if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT)
        {
            for (int b = 0; b < batch; b++)
            {
                sgetrf_mlu(handle, dtype, 1, m, n,
                           NULL, (float *)y + b * m * n, (float *)y + batch * m * ldda + b * m * n, ldda,
                           ipiv + b * m, info, mode, workspace);
            }
        }
        else if (dtype == MLUOP_DTYPE_FLOAT)
        {
            for (int b = 0; b < batch; b++)
            {
                sgetrf_mlu(handle, dtype, 1, m, n,
                           (float *)y + b * m * n, NULL, NULL, ldda,
                           ipiv + b * m, info, mode, workspace);
            }
        }
    }

    if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT)
    {

        transpose_back(handle, MLUOP_DTYPE_COMPLEX_FLOAT, batch, m, n, (float *)y, workspace, handle->queue);
    }

    return MLUOP_STATUS_SUCCESS;
}
