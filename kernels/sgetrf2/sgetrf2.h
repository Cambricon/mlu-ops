/*************************************************************************
 * Copyright (C) [2022] by Cambricon, Inc.
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
#include "mlu_op.h"
#include "kernels/kernel.h"

#define TaskUnion1 4
#define TaskUnion2 8
#define TaskUnion4 16
#define TaskUnion8 32
#define MAX_DIM 65532
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define dA(i_, j_) (dA + (i_)*ldda + (j_))
#define d_rA(i_, j_) (d_rA + (i_)*ldda + (j_))
#define d_iA(i_, j_) (d_iA + (i_)*ldda + (j_))
#define dAT(i_, j_) (dAT + (i_) + (j_)*lddat)
#define dAP(i_, j_) (dAP + (i_)*nb + (j_))
#define c_one 1.0
#define c_neg_one -1.0
#define taskType 8
#define MAX_M_SIZE_COMPLEX 1024
#define MAX_M_SIZE_COMPLEX_PIVOT 2048
#define MAX_M_SIZE1 2048
#define CEILDIV(x, y) ((x + y - 1) / y)
#define ROUNDUP(x, y) (CEILDIV(x, y) * y)

mluOpStatus_t MLUOP_WIN_API KernelScal_ger(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, int batch, int M_size, int N_size, int ib, int J,
    int m, int n, int step, float *dA, int lda, int stride_a, float *workspace,
    int *dipiv, int *dipiv2, int *info, int gbstep, int mode);
mluOpStatus_t MLUOP_WIN_API KernelCcal_ger(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, int batch, int M_size, int N_size, int ib, int J,
    int m, int n, int step, float *d_rA, float *d_iA, int lda, int stride_a,
    float *workspace, int *dipiv, int *dipiv2, int *info, int gbstep, int mode);

mluOpStatus_t MLUOP_WIN_API
KernelMatrixAdd(cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
                mluOpDataType_t d_type, int batch, int m, int n, float *dA,
                int ldda, int stride_a, float *dB, int lddb, int stride_b,
                float *dC, int lddc, int stride_c);

mluOpStatus_t MLUOP_WIN_API KernelMyCnrtMemcpy2D(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, int batch, int m, int n, float *dA, int ldda,
    int stride_a, float *dB, int lddb, int stride_b, int mode);

mluOpStatus_t MLUOP_WIN_API KernelInverse(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t dtype, int batch, float *d_input, int ld_input,
    int stride_input, float *d_output, int ld_output, int stride_output, int m);

mluOpStatus_t MLUOP_WIN_API KernelComplexInverse(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t dtype, int batch, float *rd_input, float *id_input,
    int ld_input, int stride_input, float *rd_output, float *id_output,
    int ld_output, int stride_output, int m);

mluOpStatus_t MLUOP_WIN_API KernelSwap(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, int batch, int M_size, int N_size, int ib, int J,
    int m, int n, int step, float *dA, float *d_rA, float *d_iA, int lda,
    int stride_a, int *dipiv, int *dipiv2, int *info, int gbstep);

mluOpStatus_t cgemm(mluOpHandle_t handle, mluOpDataType_t dtype, int m_, int n_,
                    int k_, int batch, int m, int n, float *d_ra, float *d_ia,
                    int lda, float *d_rb, float *d_ib, int ldb, float *d_rc,
                    float *d_ic, int ldc, int ldda, cnrtQueue_t queue);
mluOpStatus_t complex_inverse(mluOpHandle_t handle, mluOpDataType_t dtype,
                              int batch, float *rd_input, float *id_input,
                              int ld_input, int stride_input, float *rd_output,
                              float *id_output, int ld_output,
                              int stride_output, int m, cnrtQueue_t queue);
mluOpStatus_t ctrsm(mluOpHandle_t handle, mluOpDataType_t dtype, int batch,
                    int M, int N, int m, int n, float *d_ra, float *d_ia,
                    int lda, float *d_rb, float *d_ib, int ldb,
                    float *work_space, cnrtQueue_t queue);

mluOpStatus_t MLUOP_WIN_API gemm(mluOpHandle_t handle, mluOpDataType_t dtype,
                                 int m_, int n_, int k_, float alpha,
                                 float beta, int batch, int m, int n,
                                 float *dev_a, float *dev_b, float *dev_c,
                                 float *dev_d, int ldda, cnrtQueue_t queue);
mluOpStatus_t MLUOP_WIN_API gemm_for_ctrsm(
    mluOpHandle_t handle, mluOpDataType_t dtype, int m_, int n_, int k_,
    float alpha, float beta, int batch, int m, int n, float *dev_a,
    float *dev_b, float *dev_c, float *dev_d, int ldda, cnrtQueue_t queue);

mluOpStatus_t MLUOP_WIN_API transpose(mluOpHandle_t handle,
                                      mluOpDataType_t dtype, int batch, int m,
                                      int n, float *input, float *workspace_dst,
                                      cnrtQueue_t queue);
mluOpStatus_t MLUOP_WIN_API transpose_back(mluOpHandle_t handle,
                                           mluOpDataType_t dtype, int batch,
                                           int m, int n, float *output,
                                           void *workspace, cnrtQueue_t queue);
mluOpStatus_t MLUOP_WIN_API MyCnrtMemcpy2D(mluOpHandle_t handle, int batch,
                                           int m, int n, float *dA, int ldda,
                                           int stride_a, float *dB, int lddb,
                                           int stride_b, int mode,
                                           cnrtQueue_t queue);

int sgetrf2_native(mluOpHandle_t handle, mluOpDataType_t dtype, int batch,
                   int m, int n, float *dA, float *d_rA, float *d_iA, int ldda,
                   int gbm, int gbn, int *dipiv, int *dinfo, int gbstep,
                   int mode, void *workspace, cnrtQueue_t queue);

int sgetrf_mlu(mluOpHandle_t handle, mluOpDataType_t dtype, int batch, int m,
               int n, float *dA, float *d_rA, float *d_iA, int ldda, int *ipiv,
               int *info, int mode, void *workspace);
