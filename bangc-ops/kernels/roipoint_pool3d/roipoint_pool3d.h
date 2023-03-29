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
#ifndef KERNELS_ROIPOINT_POOL3D_ROIPOINT_POOL3D_H
#define KERNELS_ROIPOINT_POOL3D_ROIPOINT_POOL3D_H

#include "mlu_op.h"

void MLUOP_WIN_API KernelRoipointPool3d(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, const int batch_size, const int pts_num,
    const int boxes_num, const int feature_in_len, const int sampled_pts_num,
    const char *points_xyz_gdram, const char *point_features_gdram,
    const char *boxes3d_gdram, char *pooled_features_gdram,
    char *pooled_empty_flag_gdram);

void MLUOP_WIN_API KernelRoipointPool3dLargeBoxesNum(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    mluOpDataType_t d_type, const int batch_size, const int pts_num,
    const int boxes_num, const int feature_in_len, const int sampled_pts_num,
    const char *points_xyz_gdram, const char *point_features_gdram,
    const char *boxes3d_gdram, char *pooled_features_gdram,
    char *pooled_empty_flag_gdram);

#endif  // KERNELS_ROIPOINT_POOL3D_ROIPOINT_POOL3D_H
