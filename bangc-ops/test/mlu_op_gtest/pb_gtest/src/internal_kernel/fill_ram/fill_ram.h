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
#ifndef TEST_MLU_OP_GTEST_PB_GTEST_SRC_INTERNAL_KERNEL_FILL_RAM_FILL_RAM_H_
#define TEST_MLU_OP_GTEST_PB_GTEST_SRC_INTERNAL_KERNEL_FILL_RAM_FILL_RAM_H_

#include "mlu_op.h"
#include "core/tensor.h"
#include "kernels/kernel.h"

enum nram_value {
  NAN_HALF = 0,
  NAN_FLOAT = 1,
  INF_HALF = 2,
  INF_FLOAT = 3,
  NO_FILL = 4
};

void mluBlockFillRam(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                     cnrtQueue_t queue, nram_value value);

mluOpStatus_t mluOpFillRam(mluOpHandle_t handle, nram_value value);

#endif  // TEST_MLU_OP_GTEST_PB_GTEST_SRC_INTERNAL_KERNEL_FILL_RAM_FILL_RAM_H_
