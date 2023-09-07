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
#ifndef KERNELS_ROIAWARE_POOL3D_ROIAWARE_POOL3D_H
#define KERNELS_ROIAWARE_POOL3D_ROIAWARE_POOL3D_H

#include "mlu_op.h"

mluOpStatus_t MLUOP_WIN_API KernelPtsIdxOfVoxels(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, const int pool_method, const int boxes_num,
    const int pts_num, const int max_pts_each_voxel, const int out_x,
    const int out_y, const int out_z, const void *rois, const void *pts,
    void *pts_idx_of_voxels);

mluOpStatus_t MLUOP_WIN_API KernelRoiawarePool3dForward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, const int pool_method, const int boxes_num,
    const int pts_num, const int channels, const int max_pts_each_voxel,
    const int out_x, const int out_y, const int out_z, const void *pts_feature,
    const void *pts_idx_of_voxels, void *pooled_features, void *argmax);

mluOpStatus_t MLUOP_WIN_API KernelRoiawarePool3dBackward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, const int pool_method, const int boxes_num,
    const int out_x, const int out_y, const int out_z, const int channels,
    const int max_pts_each_voxel, const void *pts_idx_of_voxels,
    const void *argmax, const void *grad_out, void *grad_in);

#endif  // KERNELS_ROIAWARE_POOL3D_ROIAWARE_POOL3D_H
