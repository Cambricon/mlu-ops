/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
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
#ifndef KERNELS_DIFF_IOU_ROTATED_SORT_VERTICES_FORWARD_\
DIFF_IOU_ROTATED_SORT_VERTICES_FORWARD_H
#define KERNELS_DIFF_IOU_ROTATED_SORT_VERTICES_FORWARD_\
DIFF_IOU_ROTATED_SORT_VERTICES_FORWARD_H

#include "mlu_op.h"

#define MAX_NUM_VERT_IDX 9
#define INTERSECTION_OFFSET 8
#define EPSILON 1e-8

#define DIFF_IOU_ROTATED_DEBUG 1

#if (DIFF_IOU_ROTATED_DEBUG == 1)
#define print __##bang_printf
#define KLOG(fmt, ...)          \
  do {                          \
    print("[task%d] ", taskId); \
    print(fmt, ##__VA_ARGS__);  \
  } while (0)
#else
#define print(...)
#define KLOG(fmt, ...)
#endif

void MLUOP_WIN_API KernelDiffIouRotatedSortVerticesForward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *vertices, const void *mask, const void *num_valid, void *idx,
    const int32_t dim_b, const int32_t dim_n, const int32_t dim_m);

#endif
