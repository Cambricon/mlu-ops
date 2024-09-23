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

#ifndef __CHOLESKY_H
#define __CHOLESKY_H

#define DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <vector>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
// #include <bang.h>
#include "mlu_op.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"
#include "kernels/debug.h"
#include "kernels/utils/cnnl_helper.h"

#define CNB (32)
#define REC_NB (16)
#define POTF_NB ((REC_NB) / 4)
#define CREC_NB (16)
#define CPOTF_NB ((CREC_NB) / 4)
// #define CPOTF_NB ((CREC_NB))
#define __CNRT_FUNC_TYPE__ CNRT_FUNC_TYPE_UNION1
#define TASK_NUM (4)
#define NB (32)

#define CLUSTER_NUM 1
#define M (TASK_NUM * POTF_NB)
#define ZERO 0.0
#define SHARED_MEM_SIZE (((M * POTF_NB / TASK_NUM * 4) + (POTF_NB * POTF_NB)))
#define OFFSET_ROW(A, i, j) A + ((i) * (lda) + (j))
#define OFFSET_B_ROW(B, i, j) B + ((i) * (ldb) + (j))

mluOpStatus_t mlu_spotrf_rectile(int batch, int stride, bool trans, bool uplo,
                                 int n, int recnb, float* dA, int ldda,
                                 int gbstep, mluOpHandle_t handle,
                                 float* workspace);

mluOpStatus_t ssyrk(int batch, int stride, bool upper, bool trans, int n, int k,
                    float* d_a, int ldda, float* d_c, int lddc,
                    mluOpHandle_t handle, float* workspace);

mluOpStatus_t sgemm(int batch, bool trans_a, bool trans_b, int m, int n, int k,
                    float alpha, float beta, float* d_a, int lda, int stride_a,
                    float* d_b, int ldb, int stride_b, float* d_c, int ldc,
                    int stride_c, mluOpHandle_t handle, float* workspace);

// side:true->right
//     false->left
mluOpStatus_t strsm(int batch, int stride, bool upper, bool trans, int m, int n,
                    float* d_a, int ldda, float* d_b, int lddb,
                    mluOpHandle_t handle, float* workspace);

mluOpStatus_t transpose(int batch, int m, int n, float* d_input,
                        float* d_output, mluOpHandle_t handle,
                        mluOpDataType_t type, float* workspace);

mluOpStatus_t conj_complex(int batch, int m, int n, float* d_input,
                           float* d_output, mluOpHandle_t handle);

mluOpStatus_t mlu_cpotrf_rectile(int batch, int stride, int n, int recnb,
                                 float* drA, float* diA, int lda,
                                 mluOpHandle_t handle, float* workspace);

mluOpStatus_t cgemm(int batch, bool trans_a, bool trans_b, int m, int n, int k,
                    float alpha, float beta, float* d_ra, float* d_ia, int lda,
                    int stride_a, float* d_rb, float* d_ib, int ldb,
                    int stride_b, float* d_rc, float* d_ic, int ldc,
                    int stride_c, mluOpHandle_t handle, float* workspace);

mluOpStatus_t workspace_free(float** workspace);

mluOpStatus_t set_half_zero(int batch, int stride, float* d_a, int lda, int m,
                            mluOpHandle_t handle);

mluOpStatus_t ctrsm(int batch, int stride, int m, int n, float* rd_a,
                    float* id_a, int lda, float* rd_b, float* id_b, int ldb,
                    mluOpHandle_t handle, float* workspace);

mluOpStatus_t cherk(int batch, int stride, int n, int k, float* rd_a,
                    float* id_a, int lda, float* rd_c, float* id_c, int ldc,
                    mluOpHandle_t handle, float* workspace);

#endif
