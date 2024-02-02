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

#ifndef KERNELS_FFT_COMMON_FFT_BASIC_OPS_H_
#define KERNELS_FFT_COMMON_FFT_BASIC_OPS_H_

#include <algorithm>
#include <string>
#include "core/tensor.h"
#include "core/context.h"
#include "core/tool.h"
#include "kernels/kernel.h"
#include "kernels/utils/cnnl_helper.h"
#include "mlu_op.h"

bool fftIsIntDtype(const mluOpDataType_t dtype);

bool fftIsFloatDtype(const mluOpDataType_t dtype);

mluOpStatus_t fftGetQuantizeParamWorkspaceSize(mluOpHandle_t handle,
                                               size_t &required_size,
                                               int array_length,
                                               mluOpDataType_t data_type,
                                               mluOpDataType_t compute_type,
                                               const std::string api);

mluOpStatus_t fftQuantizePositionScale(
    mluOpHandle_t handle, int array_length, mluOpDataType_t data_type,
    mluOpDataType_t compute_type, const void *input, void *position,
    void *scale, void *workspace, size_t workspace_size, const std::string api);

mluOpStatus_t fftGetQuantizeMatMulWorkspaceSize(
    mluOpHandle_t handle, size_t &workspace_size, int m, int k, int n,
    bool is_trans_a, bool is_trans_b, mluOpDataType_t a_compute_type,
    mluOpDataType_t b_compute_type, mluOpDataType_t data_type,
    const std::string api);

mluOpStatus_t fftQuantMatMul(mluOpHandle_t handle, int m, int k, int n,
                             void *a_ptr, void *a_pos, void *a_scale,
                             void *b_ptr, void *b_pos, void *b_scale,
                             void *c_ptr, bool is_trans_a, bool is_trans_b,
                             float alpha, float beta,
                             mluOpDataType_t a_compute_type,
                             mluOpDataType_t b_compute_type,
                             mluOpDataType_t data_type, void *workspace,
                             size_t workspace_size, const std::string api);

mluOpStatus_t fftBatchMatMulBcast(mluOpHandle_t handle, int m, int k, int n,
                                  int batch, void *a_ptr, void *a_pos,
                                  void *a_scale, void *b_ptr, void *b_pos,
                                  void *b_scale, void *c_ptr, bool is_trans_a,
                                  bool is_trans_b, float alpha, float beta,
                                  mluOpDataType_t a_compute_type,
                                  mluOpDataType_t b_compute_type,
                                  mluOpDataType_t data_type, void *workspace,
                                  size_t workspace_size, const std::string api);

mluOpStatus_t fftGetTransposeWorkspaceSize(mluOpHandle_t handle,
                                           size_t &workspace_size, int dim_num,
                                           int ori_dims[], int permute[],
                                           mluOpDataType_t data_type,
                                           const std::string api);

mluOpStatus_t fftTranspose(mluOpHandle_t handle, int dim_num, int ori_dims[],
                           int transed_dims[], int permute[], void *ori_ptr,
                           void *transed_ptr, mluOpDataType_t data_type,
                           void *workspace, size_t workspace_size,
                           const std::string api);

mluOpStatus_t fftGetOptensorWorkspaceSize(mluOpHandle_t handle,
                                          size_t &workspace_size, int elem_num,
                                          mluOpDataType_t data_type,
                                          const std::string api);

mluOpStatus_t fftOptensor(mluOpHandle_t handle, int elem_num, void *in1_ptr,
                          void *in2_ptr, void *out_ptr, float alpha1,
                          float alpha2, float beta, mluOpDataType_t data_type,
                          cnnlOpTensorDesc_t op_type, void *workspace,
                          size_t workspace_size, const std::string api);

int findFFTOptLimit(mluOpHandle_t handle, const int n, int &m, int &L, int &s,
                    int &L_sub, bool &find_stockham);
#endif  // KERNELS_FFT_COMMON_FFT_BASIC_OPS_H_
