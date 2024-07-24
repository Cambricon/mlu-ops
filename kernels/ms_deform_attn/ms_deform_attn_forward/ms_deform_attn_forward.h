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
#ifndef KERNELS_MS_DEFORM_ATTN_FORWARD_MS_DEFORM_ATTN_FORWARD_H_
#define KERNELS_MS_DEFORM_ATTN_FORWARD_MS_DEFORM_ATTN_FORWARD_H_

#include "kernels/kernel.h"
#include "mlu_op.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MS_DEFORM_ATTN_FORWARD_HEADVECTOR 1

template <typename T>
__mlu_func__ void __mluop_floor(T *dst_ram, T* src_ram, int size) {
// #if (__BANG_ARCH__ >= 322)
//   __bang_floor(dst_ram, src_ram, size);  // This bang interface is for nan/inf
//                                   // temp.
// #else
  if(sizeof(T) == sizeof(float)) {
    int16 *mid = (int16 *)(dst_ram + size / 2);
    __bang_float2int16_dn(mid, (float *)src_ram, size, 0);
    __bang_int162float((float *)dst_ram, (int16_t*)mid, size, 0);
  } else {
    __bang_half2int16_dn((int16_t *)dst_ram, (half *)src_ram, size, 0);
    __bang_int162half((half *)dst_ram, (int16_t*)dst_ram, size, 0);
  }
// #endif
}

template <typename T>
__mlu_global__ void MLUKernelMsDeformAttnForwardDefault(
    const char *data_value_gdram, const char *data_spatial_shapes_gdram,
    const char *data_level_start_index_gdram,
    const char *data_sampling_loc_gdram, const char *data_attn_weight_gdram,
    const int batch_size, const int num_keys, const int num_heads,
    const int channels, const int num_levels, const int num_queries,
    const int num_points, char *data_col_gdram);

template <typename T>
__mlu_global__ void MLUKernelMsDeformAttnForwardSmallChannel(
    const char *data_value_gdram, const char *data_spatial_shapes_gdram,
    const char *data_level_start_index_gdram,
    const char *data_sampling_loc_gdram, const char *data_attn_weight_gdram,
    const int batch_size, const int num_keys, const int num_heads,
    const int channels, const int num_levels, const int num_queries,
    const int num_points, char *data_col_gdram);

template <typename T>
__mlu_global__ void MLUKernelMsDeformAttnForwardFast(
    const char *data_value_gdram, const char *data_spatial_shapes_gdram,
    const char *data_level_start_index_gdram,
    const char *data_sampling_loc_gdram, const char *data_attn_weight_gdram,
    const int32_t batch_size, const int32_t num_keys, const int32_t num_heads,
    const int32_t channels, const int32_t num_levels, const int32_t num_queries,
    const int32_t num_points, char *data_col_gdram);

#endif  // KERNELS_MS_DEFORM_ATTN_FORWARD_MS_DEFORM_ATTN_FORWARD_H_
