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
#include "kernels/utils/cnnl_helper.h"

mluOpStatus_t MLUOP_WIN_API scal_ger(mluOpHandle_t handle,
                                     mluOpDataType_t dtype, int batch,
                                     int M_size, int N_size, int ib, int J,
                                     int m, int n, int step, float *dA,
                                     float *d_rA, float *d_iA, int lda,
                                     int stride_a, float *workspace, int *dipiv,
                                     int *dipiv2, int *pivot, int *info,
                                     int gbstep, int mode, cnrtQueue_t queue) {
  if (m == 0 || n == 0) return MLUOP_STATUS_BAD_PARAM;

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc2(handle, &k_dim, &k_type, dtype, m, mode, batch, SCALGER);
  if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
    CHECK_RETURN(
        "[KernelCcal_ger]",
        KernelCcal_ger(k_dim, k_type, queue, dtype, batch, M_size, N_size, ib,
                       J, m, n, step, d_rA, d_iA, lda, stride_a, workspace,
                       dipiv, dipiv2, pivot, info, gbstep, mode));
  } else if (dtype == MLUOP_DTYPE_FLOAT) {
    CHECK_RETURN(
        "[KernelScal_ger]",
        KernelScal_ger(k_dim, k_type, queue, dtype, batch, M_size, N_size, ib,
                       J, m, n, step, dA, lda, stride_a, workspace, dipiv,
                       dipiv2, pivot, info, gbstep, mode));
  }

  CNRT_CHECK(cnrtQueueSync(queue));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API swap(mluOpHandle_t handle, mluOpDataType_t dtype,
                                 int batch, int M_size, int N_size, int ib,
                                 int J, int m, int n, int step, float *dA,
                                 float *d_rA, float *d_iA, int lda,
                                 int stride_a, int *dipiv, int *dipiv2,
                                 int *info, int gbstep, cnrtQueue_t queue) {
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(handle, &k_dim, &k_type, batch, SWAP);

  CHECK_RETURN("[KernelSwap]",
               KernelSwap(k_dim, k_type, queue, dtype, batch, M_size, N_size,
                          ib, J, m, n, step, dA, d_rA, d_iA, lda, stride_a,
                          dipiv, dipiv2, info, gbstep));
  // CNRT_CHECK(cnrtQueueSync(queue));
  return MLUOP_STATUS_SUCCESS;
}

int xgetrf2_native(mluOpHandle_t handle, mluOpDataType_t dtype, int batch,
                   int m, int n, float *dA, float *d_rA, float *d_iA, int ldda,
                   int gbm, int gbn, int *dipiv, int *pivot, int *dinfo,
                   int gbstep, int mode, void *workspace, cnrtQueue_t queue) {
  int arginfo = 0;
  if (m < 0) {
    arginfo = -1;
  } else if (n < 0) {
    arginfo = -2;
  } else if (ldda < MAX(1, n)) {
    arginfo = -4;
  }

  if (arginfo != 0) {
    return arginfo;
  }

  // Quick return if possible
  if (m == 0 || n == 0) {
    return arginfo;
  }
  int nb = 16;  // recommended nb 16、32
  int min_mn = MIN(m, n);
  int gbj, j, step, ib;
  int *dipiv2;
  if (mode == 1) {
    // dipiv2 = (dtype == MLUOP_DTYPE_COMPLEX_FLOAT)
    //              ? (int *)workspace + 2 * (gbm * gbn + gbm * gbm) * batch
    //              : (int *)workspace + batch * 64 * 64;
    // pivot = dipiv2 + gbm;
    dipiv2 = dipiv + gbm;
    CNRT_CHECK(cnrtMemset(dipiv2, 0, m * sizeof(int)));
  }

  for (j = 0; j < min_mn; j += nb) {
    ib = MIN(nb, min_mn - j);
    if (dtype == MLUOP_DTYPE_FLOAT) {
      arginfo =
          scal_ger(handle, dtype, batch, m, n, ib, j, m - j, ib, j, dA, NULL,
                   NULL, ldda, gbm * ldda, (float *)workspace, dipiv + j,
                   dipiv2, pivot + j, dinfo, gbstep, mode, queue);
      if (mode == 1) {
        if (gbn - (j + ib) - gbstep > 0) {
          swap(handle, dtype, batch, m, n, ib, j, m - j,
               gbn - (j + ib) - gbstep, j, dA + j + j * ldda + ib, NULL, NULL,
               ldda, gbm * ldda, dipiv + j, dipiv2, dinfo, gbstep, queue);
        }

        if (gbstep + j > 0) {
          swap(handle, dtype, batch, m, n, ib, j, m - j, gbstep + j, j,
               dA + j + j * ldda - gbstep - j, NULL, NULL, ldda, gbm * ldda,
               dipiv + j, dipiv2, dinfo, gbstep, queue);
        }
      }
    } else if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
      arginfo =
          scal_ger(handle, dtype, batch, m, n, ib, j, m - j, ib, j, NULL, d_rA,
                   d_iA, ldda, gbm * ldda, (float *)workspace, dipiv + j,
                   dipiv2, pivot + j, dinfo, gbstep, mode, queue);
      if (mode == 1) {
        if (gbn - (j + ib) - gbstep > 0) {
          swap(handle, dtype, batch, m, n, ib, j, m - j,
               gbn - (j + ib) - gbstep, j, NULL, d_rA + j + j * ldda + ib,
               d_iA + j + j * ldda + ib, ldda, gbm * ldda, dipiv + j, dipiv2,
               dinfo, gbstep, queue);
        }

        if (gbstep + j > 0) {
          swap(handle, dtype, batch, m, n, ib, j, m - j, gbstep + j, j, NULL,
               d_rA + j + j * ldda - gbstep - j,
               d_iA + j + j * ldda - gbstep - j, ldda, gbm * ldda, dipiv + j,
               dipiv2, dinfo, gbstep, queue);
        }
      }
    }

    if ((n - j - ib) > 0) {
      if (dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
        ctrsm(handle, dtype, batch, gbm, ldda, ib, n - j - ib, d_rA(j, j),
              d_iA(j, j), ldda, d_rA(j, j + ib), d_iA(j, j + ib), ldda,
              (float *)workspace, queue);
        cgemm(handle, dtype, m - (j + ib), n - (j + ib), ib, batch, gbm, n,
              d_rA(ib + j, j), d_iA(ib + j, j), ldda, d_rA(j, j + ib),
              d_iA(j, j + ib), ldda, d_rA(j + ib, j + ib), d_iA(j + ib, j + ib),
              ldda, ldda, queue);
      } else if (dtype == MLUOP_DTYPE_FLOAT) {
        trsm(handle, dtype, batch, gbm, ldda, ib, n - j - ib, dA(j, j), ldda,
             dA(j, j + ib), ldda, (float *)workspace, queue);

        gemm(handle, dtype, m - (j + ib), n - (j + ib), ib, -1, 1, batch, gbm,
             n, dA(ib + j, j), dA(j, j + ib), dA(j + ib, j + ib),
             dA(j + ib, j + ib), ldda, queue);
      }
    }
  }

  return MLUOP_STATUS_SUCCESS;
}
