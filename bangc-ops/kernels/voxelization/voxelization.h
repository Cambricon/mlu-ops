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
#ifndef KERNELS_VOXELIZATION_VOXELIZATION_H
#define KERNELS_VOXELIZATION_VOXELIZATION_H

#include "mlu_op.h"

mluOpStatus_t MLUOP_WIN_API KernelDynamicVoxelize(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *points, const void *voxel_size, const void *coors_range,
    void *coors, const int32_t num_points, const int32_t num_features);

mluOpStatus_t MLUOP_WIN_API KernelPoint2Voxel(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue, void *coors,
    void *point_to_pointidx, void *point_to_voxelidx, const int32_t num_points,
    const int32_t max_points);

mluOpStatus_t MLUOP_WIN_API KernelCalcPointsPerVoxel(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    void *point_to_pointidx, void *point_to_voxelidx, void *coor_to_voxelidx,
    void *num_points_per_voxel, void *voxel_num, const int32_t max_voxels,
    const int32_t num_points);

mluOpStatus_t MLUOP_WIN_API KernelAssignVoxelsCoors(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *points, void *temp_coors, void *point_to_voxelidx,
    void *coor_to_voxelidx, void *voxels, void *coors, const int32_t max_points,
    const int32_t num_points, const int32_t num_features);

#endif  // KERNELS_VOXELIZATION_VOXELIZATION_H
