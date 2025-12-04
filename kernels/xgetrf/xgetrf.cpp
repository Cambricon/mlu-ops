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
#include <time.h>
#include "xgetrf.h"
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/unary_op/unary_op_host.h"
#define op_name "[mluOpXgetrf]"

mluOpStatus_t MLUOP_WIN_API mluOpGetXgetrfWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t x_desc,
    size_t *workspace_size) {
  PARAM_CHECK("mluOpXgetrf", x_desc != NULL);

  PARAM_CHECK("mluOpXgetrf",
              x_desc->dim == 2 || x_desc->dim == 3 || x_desc->dim == 4);

  size_t m, n, batch = 1;
  mluOpDataType_t dtype = x_desc->dtype;

  PARAM_CHECK("mluOpXgetrf",
              dtype == MLUOP_DTYPE_FLOAT || dtype == MLUOP_DTYPE_COMPLEX_FLOAT);

  uint64_t type_size;
  MLUOP_CHECK(mluOpGetSizeOfDataType(dtype, &type_size));

  if (x_desc->dim == 2) {
    PARAM_CHECK("mluOpXgetrf", x_desc->dims[0] >= 0);
    PARAM_CHECK("mluOpXgetrf", x_desc->dims[1] >= 0);
    m = x_desc->dims[0];
    n = x_desc->dims[1];
  } else if (x_desc->dim == 3) {
    PARAM_CHECK("mluOpXgetrf", x_desc->dims[1] >= 0);
    PARAM_CHECK("mluOpXgetrf", x_desc->dims[2] >= 0);
    batch = x_desc->dims[0];
    m = x_desc->dims[1];
    n = x_desc->dims[2];
  } else if (x_desc->dim == 4) {
    PARAM_CHECK("mluOpXgetrf", x_desc->dims[2] >= 0);
    PARAM_CHECK("mluOpXgetrf", x_desc->dims[3] >= 0);
    batch = x_desc->dims[0] * x_desc->dims[1];
    m = x_desc->dims[2];
    n = x_desc->dims[3];
  }
  int tol = 1024;
  if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT)
    *workspace_size = 2 * (m * n + m * m) * batch + m + 2 * m + tol;
  else if (dtype == MLUOP_DTYPE_FLOAT)
    *workspace_size = batch * 64 * 64 + m + 2 * m + tol;
  *workspace_size *= type_size;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpXgetrf(mluOpHandle_t handle, const mluOpTensorDescriptor_t x_desc,
            const void *x, const mluOpTensorDescriptor_t y_desc, void *y,
            void *workspace, const mluOpTensorDescriptor_t pivots_desc,
            int *pivots, int *info, int mode) {
  /* parameter check*/
  size_t m, n;

  int batch;
  mluOpDataType_t dtype = x_desc->dtype;
  PARAM_CHECK("mluOpXgetrf", x_desc != NULL);
  PARAM_CHECK("mluOpXgetrf", y_desc != NULL);
  PARAM_CHECK("mluOpXgetrf", pivots_desc != NULL);
  PARAM_CHECK("mluOpXgetrf", x != NULL);
  PARAM_CHECK("mluOpXgetrf", y != NULL);
  PARAM_CHECK("mluOpXgetrf", pivots != NULL);
  PARAM_CHECK("mluOpXgetrf",
              x_desc->dim == 2 || x_desc->dim == 3 || x_desc->dim == 4);

  PARAM_CHECK("mluOpXgetrf",
              dtype == MLUOP_DTYPE_FLOAT || dtype == MLUOP_DTYPE_COMPLEX_FLOAT);
  if (x_desc->dim == 2) {
    PARAM_CHECK("mluOpXgetrf", x_desc->dims[0] >= 0);
    PARAM_CHECK("mluOpXgetrf", x_desc->dims[1] >= 0);
    batch = 1;
    m = x_desc->dims[0];
    n = x_desc->dims[1];
  } else if (x_desc->dim == 3) {
    PARAM_CHECK("mluOpXgetrf", x_desc->dims[1] >= 0);
    PARAM_CHECK("mluOpXgetrf", x_desc->dims[2] >= 0);
    batch = x_desc->dims[0];
    m = x_desc->dims[1];
    n = x_desc->dims[2];
  } else if (x_desc->dim == 4) {
    PARAM_CHECK("mluOpXgetrf", x_desc->dims[2] >= 0);
    PARAM_CHECK("mluOpXgetrf", x_desc->dims[3] >= 0);
    batch = x_desc->dims[0] * x_desc->dims[1];
    m = x_desc->dims[2];
    n = x_desc->dims[3];
  }
  // check 0 element
  if (mluOpGetTensorElementNum(x_desc) == 0) {
    VLOG(5) << op_name << "skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }
  // check largetensor
  if (handle->arch < MLUOP_MLU590) {
    uint64_t num_input = mluOpGetTensorElementNum(x_desc);
    TENSOR_NUM_CHECK(op_name, num_input, LARGE_TENSOR_NUM,
                     "input tensor num is too large. ");
  }

  mluOpGetQueue(handle, &(handle->queue));

  size_t ldda = n;
  size_t minmn = MIN(m, n);
  if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
    transpose(handle, MLUOP_DTYPE_COMPLEX_FLOAT, batch, m, n, (float *)x,
              (float *)y, handle->queue);
  } else {
    cnrtMemcpy2D(handle, batch, m, n, (float *)x, n, m * n, (float *)y, n,
                 m * n, 0, handle->queue);
    // cnrtMemcpy((float *)y, (float *)x, batch * m * n * sizeof(float),
    //            CNRT_MEM_TRANS_DIR_DEV2DEV);
  }
  if (mode == 0) {
    if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT)
      xgetrf_mlu(handle, dtype, batch, m, n, (float *)y, (float *)y,
                 (float *)y + batch * m * ldda, ldda, pivots, info, mode,
                 workspace);
    else if (dtype == MLUOP_DTYPE_FLOAT)
      xgetrf_mlu(handle, dtype, batch, m, n, (float *)y, NULL, NULL, ldda,
                 pivots, info, mode, workspace);
  } else {
    if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
      for (int b = 0; b < batch; b++) {
        xgetrf_mlu(handle, dtype, 1, m, n, NULL, (float *)y + b * m * n,
                   (float *)y + batch * m * ldda + b * m * n, ldda,
                   pivots + b * minmn, info, mode, workspace);
      }
    } else if (dtype == MLUOP_DTYPE_FLOAT) {
      for (int b = 0; b < batch; b++) {
        xgetrf_mlu(handle, dtype, 1, m, n, (float *)y + b * m * n, NULL, NULL,
                   ldda, pivots + b * minmn, info, mode, workspace);
      }
    }
  }
  if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
    transpose_back(handle, MLUOP_DTYPE_COMPLEX_FLOAT, batch, m, n, (float *)y,
                   workspace, handle->queue);
  }

  return MLUOP_STATUS_SUCCESS;
}
