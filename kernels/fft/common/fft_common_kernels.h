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
#ifndef KERNELS_FFT_COMMON_FFT_COMMON_KERNELS_H_
#define KERNELS_FFT_COMMON_FFT_COMMON_KERNELS_H_

#include "kernels/fft/fft.h"

mluOpStatus_t MLUOP_WIN_API kernelGenerateRFFTHalfDFTMatrix(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpFFTPlan_t fft_plan, mluOpDataType_t in_r_dtype, int n);

mluOpStatus_t MLUOP_WIN_API kernelGenerateRFFTFullDFTMatrix(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpFFTPlan_t fft_plan, mluOpDataType_t in_r_dtype, int row, int n);

mluOpStatus_t MLUOP_WIN_API kernelGenerateIRFFTHalfDFTMatrix(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpFFTPlan_t fft_plan, mluOpDataType_t in_r_dtype, int n);

mluOpStatus_t MLUOP_WIN_API kernelGenerateIRFFTFullDFTMatrix(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpFFTPlan_t fft_plan, mluOpDataType_t in_r_dtype, int n);

#endif  // KERNELS_FFT_COMMON_FFT_COMMON_KERNELS_H_
