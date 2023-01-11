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
#include "kernels/kernel_wrapper/wrapper.h"

mluOpStatus_t MLUOP_WIN_API mluOpMatMul(mluOpHandle_t handle,
                                        const bool is_trans_a,
                                        const bool is_trans_b,
                                        const void *alpha,
                                        const mluOpTensorDescriptor_t a_desc,
                                        const void *a,
                                        const mluOpTensorDescriptor_t b_desc,
                                        const void *b,
                                        const void *beta,
                                        const mluOpTensorDescriptor_t c_desc,
                                        void *c) {
  matmulWrapper wrapper;
  mluOpStatus_t ret = wrapper.invoke(handle, is_trans_a, is_trans_b,
                                     alpha, a_desc, a, b_desc, b, beta,
                                     c_desc, c);
  return ret;
}

mluOpStatus_t MLUOP_WIN_API mluOpMatMul_v2(mluOpHandle_t handle,
                                           mluOpMatMulDescriptor_t matmul_desc,
                                           mluOpMatMulAlgo_t algo,
                                           const void *alpha,
                                           const mluOpTensorDescriptor_t a_desc,
                                           const void *a,
                                           const mluOpTensorDescriptor_t b_desc,
                                           const void *b,
                                           const void *beta,
                                           const mluOpTensorDescriptor_t c_desc,
                                           void *c,
                                           void *workspace,
                                           size_t workspace_size,
                                           const mluOpTensorDescriptor_t d_desc,
                                           void *d) {
  matmulV2Wrapper wrapper;
  mluOpStatus_t ret = wrapper.invoke(handle, matmul_desc, algo, alpha,
                                     a_desc, a, b_desc, b, beta, c_desc, c,
                                     workspace, workspace_size, d_desc, d);
  return ret;
}
