#include "sgetrf.h"
#include <time.h>
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/unary_op/unary_op_host.h"

mluOpStatus_t MLUOP_WIN_API mluOpSgetrf(mluOpHandle_t handle,
                                        const mluOpTensorDescriptor_t x_desc,
                                        void *x,
                                        const mluOpTensorDescriptor_t y_desc,
                                        void *y,
                                        int *ipiv,
                                        int *info,
                                        int mode)
{
    /* sgetrf参数转换*/
    int m = x_desc->dims[x_desc->dim - 2];
    int n = x_desc->dims[x_desc->dim - 1];
    int trans = x_desc->strides[x_desc->dim - 1] == 1 ? 0 : 1;
    int ldda = n;

    cnrtMemcpy(y, (void *)x, m * n * sizeof(float), CNRT_MEM_TRANS_DIR_DEV2DEV);

    sgetrf_mlu(handle, m, n, (float *)y, ldda, ipiv, info, mode);

    return MLUOP_STATUS_SUCCESS;
}
