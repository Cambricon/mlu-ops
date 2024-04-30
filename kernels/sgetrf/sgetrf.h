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

#define TaskUnion1 4
#define TaskUnion2 8
#define TaskUnion4 16
#define TaskUnion8 32
#define MAX_DIM 65532
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define dA(i_, j_) (dA + (i_)*ldda + (j_))
#define dAT(i_, j_) (dAT + (i_) + (j_)*lddat)
#define dAP(i_, j_) (dAP + (i_)*nb + (j_))
#define c_one 1.0;
#define c_neg_one -1.0;

#include "mlu_op.h"

static inline int ceildiv(int x, int y)
{
    return (x + y - 1) / y;
}

static inline int roundup(int x, int y)
{
    return ceildiv(x, y) * y;
}

mluOpStatus_t MLUOP_WIN_API KernelTrsm(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                                       cnrtQueue_t queue, mluOpDataType_t d_type,
                                       int m, int n,
                                       float *dA, int32_t ldda,
                                       float *dB, int32_t lddb);

mluOpStatus_t MLUOP_WIN_API KernelScal_ger(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                                           cnrtQueue_t queue, mluOpDataType_t d_type,
                                           int m, int n, int step,
                                           float *dA, int lda,
                                           int *info, int gbstep);

mluOpStatus_t MLUOP_WIN_API KernelMatrixAdd(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
    cnrtQueue_t queue, mluOpDataType_t d_type,
    int m, int n,
    float *dA, int ldda,
    float *dB, int lddb,
    float *dC, int lddc);

mluOpStatus_t MLUOP_WIN_API KernelMyCnrtMemcpy2D(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
    cnrtQueue_t queue, mluOpDataType_t d_type,
    int m, int n,
    float *dA, int ldda,
    float *dB, int lddb,
    int mode);

mluOpStatus_t MLUOP_WIN_API KernelInverse(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
    cnrtQueue_t queue,
    int batch,
    float *d_input, int ld_input, int stride_input,
    float *d_output, int ld_output, int stride_output,
    int m);

int sgetrf2_native(
    mluOpHandle_t handle,
    int m, int n,
    float *dA, int ldda,
    int *dipiv, int *dinfo,
    int gbstep, cnrtQueue_t queue);

int sgetrf_recpanel_native(
    mluOpHandle_t handle,
    int m, int n,
    float *dA, int ldda,
    int *dipiv, int *dipivinfo,
    int *dinfo, int gbstep,
    cnrtQueue_t queue, cnrtQueue_t update_queue);

int sgetrf_mlu(
    mluOpHandle_t handle,
    int m, int n,
    float *dA, int ldda,
    int *ipiv,
    int *info, int mode);
