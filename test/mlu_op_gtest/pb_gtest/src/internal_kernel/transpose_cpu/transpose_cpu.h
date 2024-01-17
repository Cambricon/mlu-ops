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
#ifndef TEST_MLU_OP_GTEST_SRC_INTERNAL_KERNEL_TRANSPOSE_CPU_TRANSPOSE_CPU_H_
#define TEST_MLU_OP_GTEST_SRC_INTERNAL_KERNEL_TRANSPOSE_CPU_TRANSPOSE_CPU_H_

#include <vector>
#include "core/tensor.h"
#include "kernels/kernel.h"
#include "kernels/debug.h"

#define TRANSPOSE_MAX_DIM 8

mluOpStatus_t mluOpTransposeCpu(const int64_t dim,
                                const std::vector<int> permute,
                                const mluOpTensorDescriptor_t x_desc,
                                const void *x,
                                const mluOpTensorDescriptor_t y_desc, void *y);

#endif  // TEST_MLU_OP_GTEST_SRC_INTERNAL_KERNEL_TRANSPOSE_CPU_TRANSPOSE_CPU_H_
