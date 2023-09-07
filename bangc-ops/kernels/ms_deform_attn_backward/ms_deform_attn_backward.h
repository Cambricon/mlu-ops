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
#ifndef KERNELS_MS_DEFORM_ATTN_BACKWARD_MS_DEFORM_ATTN_BACKWARD_H
#define KERNELS_MS_DEFORM_ATTN_BACKWARD_MS_DEFORM_ATTN_BACKWARD_H

#include "mlu_op.h"

#define FAST_KERNEL_MAX_NLP (128)
#define FAST_KERNEL_MAX_NLPC (16384)

mluOpStatus_t MLUOP_WIN_API KernelMsDeformAttnBackwardSmallChannels(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const float *data_value, const int32_t *spatial_shapes,
    const int32_t *data_level_start_index, const float *data_sampling_loc,
    const float *data_attn_weight, const float *grad_output,
    const int32_t batch, const int32_t spatial_size, const int32_t num_heads,
    const int32_t channels, const int32_t num_levels, const int32_t num_query,
    const int32_t num_points, float *grad_value, float *grad_sampling_loc,
    float *grad_attn_weight);

mluOpStatus_t MLUOP_WIN_API KernelMsDeformAttnBackwardDefault(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const float *data_value, const int32_t *spatial_shapes,
    const int32_t *data_level_start_index, const float *data_sampling_loc,
    const float *data_attn_weight, const float *grad_output,
    const int32_t batch, const int32_t spatial_size, const int32_t num_heads,
    const int32_t channels, const int32_t num_levels, const int32_t num_query,
    const int32_t num_points, float *grad_value, float *grad_sampling_loc,
    float *grad_attn_weight);

mluOpStatus_t MLUOP_WIN_API KernelMsDeformAttnBackwardFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const float *data_value, const int32_t *spatial_shapes,
    const int32_t *data_level_start_index, const float *data_sampling_loc,
    const float *data_attn_weight, const float *grad_output,
    const int32_t batch, const int32_t spatial_size, const int32_t num_heads,
    const int32_t channels, const int32_t num_levels, const int32_t num_query,
    const int32_t num_points, float *grad_value, float *grad_sampling_loc,
    float *grad_attn_weight);

#endif  // KERNELS_MS_DEFORM_ATTN_BACKWARD_MS_DEFORM_ATTN_BACKWARD_H
