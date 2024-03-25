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
#ifndef KERNELS_CARAFE_CARAFE_H
#define KERNELS_CARAFE_CARAFE_H

#include "mlu_op.h"

#define CARAFE_FORWARD_API "[mluOpCarafeForward]: "
#define CARAFE_BACKWARD_API "[mluOpCarafeBackward]: "

#define CARAFE_CHECK_RETURN(api, returned_status, ...)      \
  if (returned_status != MLUOP_STATUS_SUCCESS) {            \
    LOG(ERROR) << api << " BAD return value. " __VA_ARGS__; \
    return returned_status;                                 \
  }

#define INDEX3(n, h, w, c, strN, strH, strW) strN *n + strH *h + strW *w + c
#define CARAFE_BACKWARD_NRAM_SIZE MAX_NRAM_SIZE
#define NRAM_BLOCK PAD_DOWN(MAX_NRAM_SIZE / 5, 64)

struct mluOpCarafeStruct {
  int dimNb;
  int kernel_size;
  int group_size;
  int scale_factor;
};

mluOpStatus_t MLUOP_WIN_API KernelCarafeForward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, const void *input, const void *mask, void *output,
    int input_dimN, int input_dimH, int input_dimW, int input_dimC,
    int kernel_size, int group_size, int scale_factor, int block_dimH,
    int block_dimW, int block_dimG, int block_dimC, int grid_dimH,
    int grid_dimW, int grid_dimG, int grid_dimC);

mluOpStatus_t MLUOP_WIN_API KernelCarafeBackward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, void *input, void *mask, void *grad_output,
    void *grad_input, void *grad_mask, int n, int hi, int wi, int c, int k_up,
    int group, int scale);

#endif  // KERNELS_CARAFE_CARAFE_H
