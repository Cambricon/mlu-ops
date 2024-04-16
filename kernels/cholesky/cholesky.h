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
// #include <bang.h>
#include "mlu_op.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"
#include "kernels/utils/cnnl_helper.h"


#define REC_NB (8)
#define POTF_NB ((REC_NB)/4)
#define __CNRT_FUNC_TYPE__ CNRT_FUNC_TYPE_UNION1
#define TASK_NUM (4)
#define NB (16)
#define CLUSTER_NUM 1
#define M (TASK_NUM * POTF_NB) //POTF边长
#define ZERO 0.0
#define SHARED_MEM_SIZE (((M*POTF_NB/TASK_NUM * 4)+(POTF_NB * POTF_NB)))
#define OFFSET_ROW(A, i, j) A + ((i) * (lda) + (j))
#define OFFSET_B_ROW(B, i, j) B + ((i) * (ldb) + (j))


mluOpStatus_t mlu_spotrf_rectile(bool trans, bool uplo, int n, int recnb, float* dA, int ldda, int gbstep, mluOpHandle_t handle);
// void mluOpCholesky(bool trans, bool uplo, int n, float* dA, float* dC, int ldda);

mluOpStatus_t ssyrk(bool upper, bool trans,int n, int k, float* d_a, int ldda, float* d_c, int lddc, mluOpHandle_t handle);

mluOpStatus_t sgemm(bool trans_a, bool trans_b, int m, int n, int k, float alpha, float beta, float* d_a,int lda, float* d_b, int ldb,  float* d_c, int ldc, mluOpHandle_t handle);

//side:true->right
//     false->left
mluOpStatus_t strsm(bool upper, bool trans, int m, int n, float* d_a, int ldda, float* d_b, int lddb, mluOpHandle_t handle);

mluOpStatus_t transpose(int m, float* d_input,float* d_output, mluOpHandle_t handle);

#endif