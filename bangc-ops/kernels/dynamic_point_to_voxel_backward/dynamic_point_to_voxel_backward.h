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
#ifndef KERNELS_DYNAMIC_POINT_TO_VOXEL_BACKWARD_DYNAMIC_POINT_TO_VOXEL_BACKWARD_H
#define KERNELS_DYNAMIC_POINT_TO_VOXEL_BACKWARD_DYNAMIC_POINT_TO_VOXEL_BACKWARD_H

#include "mlu_op.h"

typedef enum {
  REDUCE_MEAN = 0,
  REDUCE_MAX = 1,
} ReduceMode;

void MLUOP_WIN_API KernelDynamicPointToVoxelBackward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    ReduceMode reduce_mode, const float *feats, int32_t num_points,
    int32_t num_feats, int32_t num_voxel, int32_t *point2voxel_map,
    int32_t *voxel_points_count, float *voxel_feats);

#endif  // KERNELS_DYNAMIC_POINT_TO_VOXEL_BACKWARD_DYNAMIC_POINT_TO_VOXEL_BACKWARD_H