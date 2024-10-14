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
#ifndef KERNELS_MS_DEFORM_ATTN_FORWARD_MS_DEFORM_ATTN_UTILS_H_
#define KERNELS_MS_DEFORM_ATTN_FORWARD_MS_DEFORM_ATTN_UTILS_H_

#include <math.h>
#include <algorithm>

#include "kernels/kernel.h"
#include "kernels/utils/common.h"

#define BIT_COLLECT_PAD (8)
#define BACKWARD_MAX_NQ_NL_NP (1024)

#if (__BANG_ARCH__ >= 372)

__mlu_func__ void broadcastSpatialHW(
    float* spatial_offset_bd_nram,  // (num_levels, num_points)
    float* spatial_h_bd_nram,       // (num_levels, num_points)
    float* spatial_w_bd_nram,       // (num_levels, num_points)
    int32_t* spatial_shapes_nram,   // (num_levels, 2)
    int32_t* spatial_offset_nram,   // (num_levels)
    const int32_t num_levels, const int32_t num_points) {
  __bang_int322float((float*)spatial_shapes_nram, spatial_shapes_nram,
                     num_levels * 2, 0);
  __memcpy(spatial_h_bd_nram, spatial_shapes_nram, sizeof(float), NRAM2NRAM,
           sizeof(float), num_points - 1, num_points * sizeof(float),
           num_levels - 1, 0, num_points - 1, 2 * sizeof(float),
           num_levels - 1);
  __memcpy(spatial_w_bd_nram, (float*)spatial_shapes_nram + 1, sizeof(float),
           NRAM2NRAM, sizeof(float), num_points - 1, num_points * sizeof(float),
           num_levels - 1, 0, num_points - 1, 2 * sizeof(float),
           num_levels - 1);
  __bang_int322float((float*)spatial_offset_nram, spatial_offset_nram,
                     num_levels, 0);
  __memcpy(spatial_offset_bd_nram, spatial_offset_nram, sizeof(float),
           NRAM2NRAM, sizeof(float), num_points - 1, num_points * sizeof(float),
           num_levels - 1, 0, num_points - 1, sizeof(float), num_levels - 1);
}

template <typename T>
__mlu_func__ void prepareLoopV2(
    int32_t* seq_nram, T* zeros_nram, int32_t* spatial_offset_nram,
    int32_t* spatial_hw_nram, int8_t* mask_x_nram, int8_t* mask_y_nram,
    T* spatial_offset_bd_nram, T* spatial_h_bd_nram, T* spatial_w_bd_nram,
    T* value_sram, const void* data_level_start_index_gdram,
    const void* data_spatial_shapes_gdram, const int32_t num_keys,
    const int32_t num_levels, const int32_t num_points,
    const int32_t max_deal_n, const int32_t mask_size, const int32_t channels) {
  if (seq_nram != nullptr) {
    for (int i = 0; i < 8; i++) {
      seq_nram[i] = i;
    }
    __bang_add_scalar(seq_nram + 8, seq_nram, 8, 8);     // [0, 7] + 8
    __bang_add_scalar(seq_nram + 16, seq_nram, 16, 16);  // [0, 15] + 16
    __bang_add_scalar(seq_nram + 32, seq_nram, 32, 32);  // [0, 31] + 32
    __bang_add_scalar(seq_nram + 64, seq_nram, 64, 64);
    __bang_add_scalar(seq_nram + 128, seq_nram, 128, 128);
    __bang_add_scalar(seq_nram + 256, seq_nram, 256, 256);
    __bang_add_scalar(seq_nram + 512, seq_nram, 512, 512);  // [0, 511] + 512
  }
  __bang_write_value(zeros_nram, channels, (T)0);
  __bang_write_value(mask_x_nram, mask_size, (int8_t)0x55);
  __bang_write_value(mask_y_nram, mask_size, (int8_t)0xAA);
  __memcpy_async(spatial_offset_nram, data_level_start_index_gdram,
                 num_levels * sizeof(int32_t), GDRAM2NRAM);
  __memcpy_async(spatial_hw_nram, data_spatial_shapes_gdram,
                 num_levels * 2 * sizeof(int32_t), GDRAM2NRAM);
  __sync_io_move_compute();
  broadcastSpatialHW(spatial_offset_bd_nram, spatial_h_bd_nram,
                     spatial_w_bd_nram, spatial_hw_nram, spatial_offset_nram,
                     num_levels, num_points);
}

/*
  Split batch*head into taskDimY, the split num_queries into coreDim.
  This plan is used to staying data_value on SRAM.
*/
__mlu_func__ void splitTaskV1(
    int32_t& cluster_begin_batch_head, int32_t& cluster_act_batch_head,
    int32_t& cluster_end_batch_head, int32_t& core_begin_query,
    int32_t& core_act_query, int32_t& core_loop_num, int32_t& core_step_query,
    const int32_t max_deal_n, const int32_t batch_size, const int32_t num_keys,
    const int32_t num_heads, const int32_t channels, const int32_t num_levels,
    const int32_t num_queries, const int32_t num_points) {
  // split batch*head into taskDimY
  int32_t batch_head = batch_size * num_heads;
  int32_t cluster_avg_batch_head = (batch_head + taskDimY - 1) / taskDimY;
  cluster_begin_batch_head = taskIdY * cluster_avg_batch_head;
  cluster_act_batch_head =
      std::min(cluster_avg_batch_head, batch_head - cluster_begin_batch_head);
  cluster_end_batch_head = cluster_begin_batch_head + cluster_act_batch_head;
  // split query into coreDim
  int32_t core_avg_query = (num_queries + coreDim - 1) / coreDim;
  core_begin_query = coreId * core_avg_query;
  core_act_query = std::min(num_queries - core_begin_query, core_avg_query);
  core_loop_num = (core_act_query + max_deal_n - 1) / max_deal_n;
  core_step_query = core_loop_num > 0
                        ? (core_act_query + core_loop_num - 1) / core_loop_num
                        : 0;
}

/*
  Split num_queries into taskDim.
  Each core iterate in batch * head
*/
__mlu_func__ void splitTaskV2(
    int32_t& cluster_begin_batch_head, int32_t& cluster_act_batch_head,
    int32_t& cluster_end_batch_head, int32_t& core_begin_query,
    int32_t& core_act_query, int32_t& core_loop_num, int32_t& core_step_query,
    const int32_t max_deal_n, const int32_t batch_size, const int32_t num_keys,
    const int32_t num_heads, const int32_t channels, const int32_t num_levels,
    const int32_t num_queries, const int32_t num_points) {
  // not split batch*head
  int32_t batch_head = batch_size * num_heads;
  cluster_begin_batch_head = 0;
  cluster_act_batch_head = batch_head;
  cluster_end_batch_head = batch_head;
  // split query into taskDim
  int32_t core_avg_query = (num_queries + taskDim - 1) / taskDim;
  core_begin_query = taskId * core_avg_query;
  core_act_query = std::min(num_queries - core_begin_query, core_avg_query);
  core_loop_num = (core_act_query + max_deal_n - 1) / max_deal_n;
  core_step_query = core_loop_num > 0
                        ? (core_act_query + core_loop_num - 1) / core_loop_num
                        : 0;
}

template <typename T>
__mlu_func__ void computePolationWeightOffsetCond(
    int32_t* data_offset_nram, T* weight_polation_nram,
    T* cond_point_polation_nram, T* cond_point_valid_nram, T* loc_nram,
    int8_t* mask_x_nram, int8_t* mask_y_nram, T* spatial_offset_bd_nram,
    T* spatial_w_bd_nram, T* spatial_h_bd_nram, T* delata_xy_nram, T* buf_nram,
    const bool cached_delta_xy, const int32_t deal_n, const int32_t num_levels,
    const int32_t num_points, const int32_t num_heads, const int32_t channels) {
  int32_t total_points = deal_n * num_levels * num_points;
  int32_t block_points = num_levels * num_points;
  T* buf_x_nram = buf_nram;
  T* buf_y_nram = buf_nram + total_points;
  T* buf_cond_nram = buf_nram + 2 * total_points;
  T* buf_x_floor = buf_nram + 2 * total_points;
  T* buf_x_ceil = buf_nram + 3 * total_points;
  T* buf_y_floor = buf_nram + 4 * total_points;
  T* buf_y_ceil = buf_nram + 5 * total_points;
  //================================================================================================
  int32_t total_coord_pad = PAD_UP(total_points * 2, BIT_COLLECT_PAD);
  __bang_filter_bitindex(buf_x_nram, loc_nram, mask_x_nram, total_coord_pad);
  __bang_filter_bitindex(buf_y_nram, loc_nram, mask_y_nram, total_coord_pad);
  // x = loc_x * spatial_w - 0.5; y = loc_y * spatial_h - 0.5;
  __bang_fusion(FUSION_FMS, buf_x_nram, buf_x_nram, spatial_w_bd_nram, (T)0.5,
                total_points, block_points);
  __bang_fusion(FUSION_FMS, buf_y_nram, buf_y_nram, spatial_h_bd_nram, (T)0.5,
                total_points, block_points);
  //================================================================================================
  // get point condition. use buf0, buf1, buf2
  // (x > -1 && y > -1 && y < spatial_h && x < spatial_w)
  __bang_gt_scalar(cond_point_valid_nram, buf_x_nram, (T)-1.0, total_points);
  __bang_gt_scalar(buf_cond_nram, buf_y_nram, (T)-1.0, total_points);
  __bang_and(cond_point_valid_nram, cond_point_valid_nram, buf_cond_nram,
             total_points);
  __bang_cycle_lt(buf_cond_nram, buf_x_nram, spatial_w_bd_nram, total_points,
                  block_points);
  __bang_and(cond_point_valid_nram, cond_point_valid_nram, buf_cond_nram,
             total_points);
  __bang_cycle_lt(buf_cond_nram, buf_y_nram, spatial_h_bd_nram, total_points,
                  block_points);
  __bang_and(cond_point_valid_nram, cond_point_valid_nram, buf_cond_nram,
             total_points);
  //================================================================================================
  __bang_floor(buf_x_floor, buf_x_nram, total_points);
  __bang_add_scalar(buf_x_ceil, buf_x_floor, 1.0, total_points);
  __bang_floor(buf_y_floor, buf_y_nram, total_points);
  __bang_add_scalar(buf_y_ceil, buf_y_floor, 1.0, total_points);
  T* cond_point_polation_nram_tl = cond_point_polation_nram;
  T* cond_point_polation_nram_bl = cond_point_polation_nram + total_points;
  T* cond_point_polation_nram_tr = cond_point_polation_nram + 2 * total_points;
  T* cond_point_polation_nram_br = cond_point_polation_nram + 3 * total_points;
  T* cond_point_polation_nram_cond1 = weight_polation_nram;
  T* cond_point_polation_nram_cond2 = weight_polation_nram + total_points;
  T* cond_point_polation_nram_cond3 = weight_polation_nram + 2 * total_points;
  T* cond_point_polation_nram_cond4 = weight_polation_nram + 3 * total_points;
  __bang_ge_scalar(cond_point_polation_nram_cond1, buf_x_floor, (T)0,
                   total_points);
  __bang_cycle_lt(cond_point_polation_nram_cond2, buf_x_ceil, spatial_w_bd_nram,
                  total_points, block_points);
  __bang_ge_scalar(cond_point_polation_nram_cond3, buf_y_floor, (T)0,
                   total_points);
  __bang_cycle_lt(cond_point_polation_nram_cond4, buf_y_ceil, spatial_h_bd_nram,
                  total_points, block_points);
  __bang_and(cond_point_polation_nram_tl, cond_point_polation_nram_cond1,
             cond_point_polation_nram_cond4, total_points);
  __bang_and(cond_point_polation_nram_bl, cond_point_polation_nram_cond1,
             cond_point_polation_nram_cond3, total_points);
  __bang_and(cond_point_polation_nram_tr, cond_point_polation_nram_cond2,
             cond_point_polation_nram_cond4, total_points);
  __bang_and(cond_point_polation_nram_br, cond_point_polation_nram_cond2,
             cond_point_polation_nram_cond3, total_points);
  //================================================================================================
  // get polation weight.
  T* buf_dx = (T*)data_offset_nram;
  T* buf_dy = buf_dx + total_points;
  T* buf_dx_1 = buf_dy + total_points;
  T* buf_dy_1 = buf_dx_1 + total_points;
  T* weight_polation_nram_1 = weight_polation_nram;
  T* weight_polation_nram_2 = weight_polation_nram + 1 * total_points;
  T* weight_polation_nram_3 = weight_polation_nram + 2 * total_points;
  T* weight_polation_nram_4 = weight_polation_nram + 3 * total_points;
  // T* weight_polation_nram_buf = buf_nram + 4 * total_points;
  __bang_sub(buf_dx, buf_x_floor, buf_x_nram, total_points);  // -dx
  __bang_sub(buf_dy, buf_y_floor, buf_y_nram, total_points);  // -dy
  __bang_fusion(FUSION_FSS, buf_dx_1, buf_x_nram, buf_x_floor,
                (T)1.0,  // dx - 1
                total_points, total_points);
  __bang_fusion(FUSION_FSS, buf_dy_1, buf_y_nram, buf_y_floor,
                (T)1.0,  // dy - 1
                total_points, total_points);
  __bang_mul(weight_polation_nram_1, buf_dx_1, buf_dy,
             total_points);  // (-dy)(dx-1)
  __bang_mul(weight_polation_nram_2, buf_dx_1, buf_dy_1,
             total_points);  // (dx-1)*(dy-1)
  __bang_mul(weight_polation_nram_3, buf_dx, buf_dy,
             total_points);  // (-dx)*(-dy)
  __bang_mul(weight_polation_nram_4, buf_dx, buf_dy_1,
             total_points);  // (-dx)*(dy-1)
  if (cached_delta_xy) {
    __bang_sub(delata_xy_nram, buf_x_nram, buf_x_floor, total_points);  // dx
    __bang_add_scalar(delata_xy_nram + total_points, buf_dx, 1,
                      total_points);  // 1-dx
    __bang_sub(delata_xy_nram + 2 * total_points, buf_y_nram, buf_y_floor,
               total_points);  // dy
    __bang_add_scalar(delata_xy_nram + 3 * total_points, buf_dy, 1,
                      total_points);  // 1-dy
  }
  //================================================================================================
  // correct the x,y in [0, w-1] and [0, h-1]
  T* spatial_w1_bd_nram = buf_nram;
  T* spatial_h1_bd_nram = buf_nram + block_points;
  __bang_sub_scalar(spatial_w1_bd_nram, spatial_w_bd_nram, (T)1, block_points);
  __bang_sub_scalar(spatial_h1_bd_nram, spatial_h_bd_nram, (T)1, block_points);
  __bang_maxeq_scalar(buf_x_floor, buf_x_floor, (T)0, total_points);
  __bang_maxeq_scalar(buf_x_ceil, buf_x_ceil, (T)0, total_points);
  __bang_cycle_minequal(buf_x_floor, buf_x_floor, spatial_w1_bd_nram,
                        total_points, block_points);
  __bang_cycle_minequal(buf_x_ceil, buf_x_ceil, spatial_w1_bd_nram,
                        total_points, block_points);
  __bang_maxeq_scalar(buf_y_floor, buf_y_floor, (T)0, total_points);
  __bang_maxeq_scalar(buf_y_ceil, buf_y_ceil, (T)0, total_points);
  __bang_cycle_minequal(buf_y_floor, buf_y_floor, spatial_h1_bd_nram,
                        total_points, block_points);
  __bang_cycle_minequal(buf_y_ceil, buf_y_ceil, spatial_h1_bd_nram,
                        total_points, block_points);
  //================================================================================================
  // offset = y*w + x
  T* buf_hw_offset = buf_nram;
  T* data_offset_nram_tl = (T*)data_offset_nram;
  T* data_offset_nram_bl = data_offset_nram_tl + total_points;
  T* data_offset_nram_tr = data_offset_nram_bl + total_points;
  T* data_offset_nram_br = data_offset_nram_tr + total_points;
  // y_ceil*w + offset + x_floor
  __bang_fusion(FUSION_FMA, buf_hw_offset, buf_y_ceil, spatial_w_bd_nram,
                spatial_offset_bd_nram, total_points, block_points);
  __bang_add(data_offset_nram_tl, buf_hw_offset, buf_x_floor, total_points);
  // y_ceil*w + offset + x_ceil
  __bang_add(data_offset_nram_tr, buf_hw_offset, buf_x_ceil, total_points);
  // y_floor*w + offset + x_foor
  __bang_fusion(FUSION_FMA, buf_hw_offset, buf_y_floor, spatial_w_bd_nram,
                spatial_offset_bd_nram, total_points, block_points);
  __bang_add(data_offset_nram_bl, buf_hw_offset, buf_x_floor, total_points);
  // y_floor*w + offset + x_ceil
  __bang_add(data_offset_nram_br, buf_hw_offset, buf_x_ceil, total_points);
  __bang_float2int32(data_offset_nram, (T*)data_offset_nram, total_points * 4,
                     0);
  int32_t stride = num_heads * channels * sizeof(T);
  __bang_mul_scalar(data_offset_nram, data_offset_nram, stride,
                    total_points * 4);
  //================================================================================================
  // merge conditions and clear weight, cast conditions to bits
  T* cond_point_polation_nram_tmp = buf_nram;
  __bang_cycle_and(cond_point_polation_nram, cond_point_polation_nram,
                   cond_point_valid_nram, 4 * total_points, total_points);
  __bang_float2int32((int32_t*)cond_point_polation_nram_tmp,
                     cond_point_polation_nram, total_points * 4, 0);
  __bang_mul_scalar((int32_t*)cond_point_polation_nram_tmp,
                    (int32_t*)cond_point_polation_nram_tmp, (int32_t)0xffffffff,
                    total_points * 4);
  __bang_band((int8_t*)weight_polation_nram, (int8_t*)weight_polation_nram,
              (int8_t*)cond_point_polation_nram_tmp,
              total_points * 4 * sizeof(float));
}

/*
  compute condition, polation_weight, offset and store to SRAM.
  cache_delta_xy and cache_point_valid is true in backward, false in forward.
*/
template <typename T>
__mlu_func__ void stageOneLoop(
    T* sampling_loc_gdram, T* weight_attn_gdram, int32_t* data_offset_nram,
    void* delata_xy_nram, T* weight_polation_nram, T* cond_point_polation_nram,
    T* cond_point_valid_nram, T* loc_nram, T* buf_nram, T* buf_nram_end,
    int8_t* mask_x_nram, int8_t* mask_y_nram, T* spatial_offset_bd_nram,
    T* spatial_w_bd_nram, T* spatial_h_bd_nram, int32_t* spatial_offset_nram,
    int32_t* spatial_hw_nram, int32_t* data_offset_sram, void* delta_xy_sram,
    T* weight_polation_sram, T* weight_attn_sram, T* cond_point_polation_sram,
    const bool cache_delta_xy, const bool cache_point_valid,
    const int32_t total_deal_n, const int32_t max_deal_n,
    const int32_t num_heads, const int32_t channels, const int32_t num_levels,
    const int32_t num_points, const int32_t input_stride_2,
    const int32_t input_stride_3) {
  int32_t loop_num = (total_deal_n + max_deal_n - 1) / max_deal_n;
  int32_t num_levels_points = num_levels * num_points;
  int32_t sram_offset = 0;
  int32_t sram_dst_stride = total_deal_n * num_levels_points * sizeof(T);
  for (int i = 0; i < loop_num; i++) {
    int32_t deal_n = std::min(total_deal_n - i * max_deal_n, max_deal_n);
    int32_t deal_point_num = deal_n * num_levels_points;
    int32_t copy_size = deal_point_num * sizeof(T);
    __memcpy(loc_nram, sampling_loc_gdram + i * max_deal_n * input_stride_3 * 2,
             input_stride_2 * 2 * sizeof(T), GDRAM2NRAM,
             input_stride_2 * 2 * sizeof(T), input_stride_3 * 2 * sizeof(T),
             deal_n - 1);
    computePolationWeightOffsetCond(
        data_offset_nram, weight_polation_nram, cond_point_polation_nram,
        cond_point_valid_nram, loc_nram, mask_x_nram, mask_y_nram,
        spatial_offset_bd_nram, spatial_w_bd_nram, spatial_h_bd_nram,
        (T*)delata_xy_nram, buf_nram, cache_delta_xy, deal_n, num_levels,
        num_points, num_heads, channels);
    __memcpy(data_offset_sram + sram_offset, data_offset_nram, copy_size,
             NRAM2SRAM, sram_dst_stride, copy_size, 3);
    __memcpy(weight_polation_sram + sram_offset, weight_polation_nram,
             copy_size, NRAM2SRAM, sram_dst_stride, copy_size, 3);
    __memcpy(cond_point_polation_sram + sram_offset, cond_point_polation_nram,
             copy_size, NRAM2SRAM, sram_dst_stride, copy_size, 3);
    if (cache_point_valid) {
      __memcpy(cond_point_polation_sram + 4 * total_deal_n * num_levels_points +
                   sram_offset,
               cond_point_valid_nram, copy_size, NRAM2SRAM);
    }
    if (cache_delta_xy) {
      __memcpy((T*)delta_xy_sram + sram_offset, delata_xy_nram, copy_size,
               NRAM2SRAM, sram_dst_stride, copy_size, 3);
    }
    __memcpy(buf_nram, weight_attn_gdram + i * max_deal_n * input_stride_3,
             input_stride_2 * sizeof(T), GDRAM2NRAM, input_stride_2 * sizeof(T),
             input_stride_3 * sizeof(T), deal_n - 1);
    __bang_float2int32((int32_t*)cond_point_valid_nram, cond_point_valid_nram,
                       deal_point_num, 0);
    __bang_mul_scalar((int32_t*)cond_point_valid_nram,
                      (int32_t*)cond_point_valid_nram, (int32_t)0xffffffff,
                      deal_point_num);
    __bang_band((int8_t*)buf_nram, (int8_t*)buf_nram,
                (int8_t*)cond_point_valid_nram,
                deal_n * num_levels * num_points * sizeof(T));
    __memcpy(weight_attn_sram + sram_offset, buf_nram, copy_size, NRAM2SRAM);
    sram_offset += deal_point_num;
  }
  __sync_io_move_compute();
}
#endif

#if (__BANG_ARCH__ == 592)
__mlu_func__ void gatherAsync(void* dst, void* src, unsigned int* offset,
                              void* mask, int transfer_size,
                              mluMemcpyDirection_t dir, int dst_stride,
                              int transfer_num) {
  __gather_async(dst, src, offset, mask, transfer_size, dir, dst_stride,
                 transfer_num);
}

__mlu_func__ void gatherSync(void* dst, void* src, unsigned int* offset,
                             void* mask, int transfer_size,
                             mluMemcpyDirection_t dir, int dst_stride,
                             int transfer_num) {
  __gather(dst, src, offset, mask, transfer_size, dir, dst_stride,
           transfer_num);
}
#endif

#endif
