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
#ifndef KERNELS_TENSOR_STRIDE_PROCESS_TENSOR_STRIDE_PROCESS_MLU_H_
#define KERNELS_TENSOR_STRIDE_PROCESS_TENSOR_STRIDE_PROCESS_MLU_H_

#include "kernels/kernel.h"
#include "mlu_op.h"
#include "tensor_stride_process.h"

void MLUOP_WIN_API mluOpUnion1KernelTensorStrideIn(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, TensorShape input_shape, void *output,
    mluOpDataType_t dtype);

void MLUOP_WIN_API mluOpUnion1KernelTensorStrideOut(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, void *output, TensorShape output_shape,
    mluOpDataType_t dtype);

#endif  // KERNELS_TENSOR_STRIDE_PROCESS_TENSOR_STRIDE_PROCESS_MLU_H_
