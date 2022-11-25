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
#ifndef KERNELS_FILL_FILL_MLU_H_
#define KERNELS_FILL_FILL_MLU_H_

#include <stdint.h>

#include "kernels/kernel.h"
#include "mlu_op.h"
#include "kernels/tensor_stride_process/tensor_stride_process.h"

// FillDeviceValue
void MLUOP_WIN_API mluOpUnion1KernelFillDeviceValue(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t k_datatype, void *output, size_t size, const void *value);

// FillHostValue
void MLUOP_WIN_API mluOpUnion1KernelFillHostValue(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t k_datatype, void *output, size_t size, uint32_t value,
    uint32_t value_high, uint32_t value_low);

// FillDeviceValueWithStride
void MLUOP_WIN_API mluOpUnion1KernelFillDeviceValueWithStride(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t k_datatype, void *output, TensorShape output_shape,
    size_t size, const void *value);

// FillHostValueWithStride
void MLUOP_WIN_API mluOpUnion1KernelFillHostValueWithStride(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t k_datatype, void *output, TensorShape output_shape,
    size_t size, uint32_t value, uint32_t value_high, uint32_t value_low);

#endif  // KERNELS_FILL_FILL_MLU_H_
