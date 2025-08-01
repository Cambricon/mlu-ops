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
#ifndef KERNELS_NMS_ROTATED_NMS_UTILS_H_
#define KERNELS_NMS_ROTATED_NMS_UTILS_H_

#include "kernels/debug.h"
#include "kernels/kernel.h"
#include "kernels/utils/common.h"

#define NMS_SIZE 64
#define NMS_UP(x, y) (((x) / (y) + (int)((x) % (y) > 0)) * (y))
#define NMS_DOWN(x, y) (((x) / (y)) * (y))
#define SIZE_NRAM_BUF (MAX_NRAM_SIZE)
#define SIZE_SRAM_BUF (MAX_SRAM_SIZE)
// score, x1, y1, x2, y2, max_index (reserve 2 num for half-type input)
#define REDUCE_NUM (7)

#define TABLE_LENGTH 64
__nram__ int table_float[TABLE_LENGTH] = {0, static_cast<int>(0xffffffff)};
__nram__ int16_t table_half[TABLE_LENGTH] = {0, static_cast<int16_t>(0xffff)};

// each box data contains 5 number: x, y, w, h, a
#define SINGLE_BOX_DIM 5
__nram__ int8_t nram_buffer[MAX_NRAM_SIZE];

template <typename IN_DT>
__mlu_func__ IN_DT loadGpr(IN_DT *p_value) {
  bool is_dram = __is_dram(p_value);
  if (std::is_same<IN_DT, half>::value) {
    if (is_dram) {
      return __load_gdram((half *)p_value);
    } else {
      return __load_sram((half *)p_value);
    }
  } else {
    if (is_dram) {
      return __load_gdram((float *)p_value);
    } else {
      return __load_sram((float *)p_value);
    }
  }
}

template <typename IN_DT>
__mlu_func__ void storeGpr(IN_DT *dst, IN_DT p_value) {
  bool b = __is_dram(dst);
  if (std::is_same<IN_DT, half>::value) {
    if (b) {
      __store_gdram((half *)dst, (half)p_value);
    } else {
      __store_sram((half *)dst, (half)p_value);
    }
  } else {
    if (b) {
      __store_gdram((float *)dst, (float)p_value);
    } else {
      __store_sram((float *)dst, (float)p_value);
    }
  }
}

template <typename IN_DT>
__mlu_func__ void findCoreMaxBox(
    IN_DT *input_score_ptr, IN_DT *score, IN_DT *temp, IN_DT *max_box,
    const IN_DT *input_x1_ptr, const IN_DT *input_y1_ptr,
    const IN_DT *input_x2_ptr, const IN_DT *input_y2_ptr,
    const mluMemcpyDirection_t load_dir, const int input_offset,
    const int repeat, const int remain, const int remain_pad,
    const int max_seg_pad, int &max_index) {
  if (__is_mpu()) {
    return;
  }
  for (int i = 0; i <= repeat; i++) {
    if (i == repeat && remain == 0) {
      break;
    }
    int seg_len = 0;  // the length every nms compute
    int cpy_len = 0;  // the length every nms memcpy
    i == repeat ? seg_len = remain_pad : seg_len = max_seg_pad;
    // check seg_len exceeds the limit of fp16 or not.
    // 65536 is the largest num that fp16 could express.
    if (std::is_same<IN_DT, half>::value && seg_len >= 65536) {
      MLULOG("seg length exceed the max num for fp16 datatype!");
      return;
    }
    i == repeat ? cpy_len = remain : cpy_len = max_seg_pad;
    /******NMS LOAD START******/
    __bang_write_value(score, seg_len, IN_DT(-INFINITY));
    __memcpy(score, input_score_ptr + input_offset + i * max_seg_pad,
             cpy_len * sizeof(IN_DT), load_dir);

    /******NMS LOAD END******/

    __bang_argmax(temp, score, seg_len);
    if (temp[0] > max_box[0]) {
      max_box[0] = temp[0];
      if (std::is_same<IN_DT, half>::value) {
        max_index = ((uint16_t *)temp)[1] + input_offset +
                    i * max_seg_pad;  // offset start from head of input_data
      } else if (std::is_same<IN_DT, float>::value) {
        max_index = ((uint32_t *)temp)[1] + input_offset +
                    i * max_seg_pad;  // offset start from head of input_data
      }
    }
  }  // for repeat
  // the max box's x1, y1, x2, y2 on every core
  max_box[1] = loadGpr(input_x1_ptr + max_index);
  max_box[2] = loadGpr(input_y1_ptr + max_index);
  max_box[3] = loadGpr(input_x2_ptr + max_index);
  max_box[4] = loadGpr(input_y2_ptr + max_index);
  ((uint32_t *)(max_box + 5))[0] = max_index;
}

template <typename IN_DT>
__mlu_func__ void findClusterMaxBox(IN_DT *sram, IN_DT *max_box, IN_DT *temp,
                                    IN_DT *input_data_score,
                                    const int core_limit) {
  // find the max with sram
  // copy every core's box info to sram, form: score---x1---y1---x2---y2---
  __memcpy(sram + REDUCE_NUM * coreId, max_box, REDUCE_NUM * sizeof(IN_DT),
           NRAM2SRAM);  // int32_t datatype
  __sync_cluster();

  // copy score from sram to nram and find the max
  __bang_write_value(temp, 64, IN_DT(-INFINITY));
  __memcpy(temp, sram, sizeof(IN_DT), SRAM2NRAM, sizeof(IN_DT),
           REDUCE_NUM * sizeof(IN_DT), coreDim - 1);
  __bang_argmax(max_box, temp, 64);
  int max_core = (std::is_same<IN_DT, half>::value) ? ((uint16_t *)max_box)[1]
                                                    : ((uint32_t *)max_box)[1];
  // copy the max box to max_box
  __memcpy(max_box, sram + max_core * REDUCE_NUM, REDUCE_NUM * sizeof(IN_DT),
           SRAM2NRAM);
}

template <typename T>
__mlu_func__ void BoxesTranpose(const T *boxes, T *boxes_trans,
                                const int32_t box_num, const int32_t box_dim) {
  int32_t task_per_core = box_num / taskDim;
  int32_t task_rem = box_num % taskDim;
  int32_t offset =
      task_per_core * taskId + (taskId < task_rem ? taskId : task_rem);
  task_per_core += taskId < task_rem ? 1 : 0;
  int32_t limit = MAX_NRAM_SIZE / sizeof(T) / 2;
#if __BANG_ARCH__ > 300
  int32_t deal_once = limit / box_dim;
  int32_t limit_aligned = deal_once * box_dim;
  int32_t repeat = task_per_core / deal_once;
  int32_t rem = task_per_core % deal_once;
  T *nram_box = (T *)nram_buffer;
  T *nram_box_trans = nram_box + limit_aligned;
  for (int32_t i = 0; i < repeat; i++) {
    __memcpy(nram_box, boxes + (offset + i * deal_once) * box_dim,
             limit_aligned * sizeof(T), GDRAM2NRAM);
    __bang_transpose(nram_box_trans, nram_box, deal_once, box_dim);
    __memcpy(boxes_trans + offset + i * deal_once, nram_box_trans,
             deal_once * sizeof(T), NRAM2GDRAM, box_num * sizeof(T),
             deal_once * sizeof(T), box_dim - 1);
  }
  if (rem != 0) {
    __memcpy(nram_box, boxes + (offset + repeat * deal_once) * box_dim,
             rem * box_dim * sizeof(T), GDRAM2NRAM);
    __bang_transpose(nram_box_trans, nram_box, rem, box_dim);
    __memcpy(boxes_trans + offset + repeat * deal_once, nram_box_trans,
             rem * sizeof(T), NRAM2GDRAM, box_num * sizeof(T), rem * sizeof(T),
             box_dim - 1);
  }
#else
  // height/width * sizeof(T) must be divisible by 64 on 2xx
  int32_t box_dim_aligned = PAD_UP(box_dim, 32);
  int32_t deal_once_aligned = PAD_DOWN(limit / box_dim_aligned, 32);
  int32_t limit_aligned = deal_once_aligned * box_dim_aligned;
  int32_t repeat = task_per_core / deal_once_aligned;
  int32_t rem = task_per_core % deal_once_aligned;
  int32_t rem_aligned = PAD_UP(rem, 32);
  T *nram_box = (T *)nram_buffer;
  T *nram_box_trans = nram_box + limit_aligned;
  for (int32_t i = 0; i < repeat; i++) {
    __memcpy(nram_box, boxes + (offset + i * deal_once_aligned) * box_dim,
             deal_once_aligned * box_dim * sizeof(T), GDRAM2NRAM);
    __memcpy(nram_box_trans, nram_box, box_dim * sizeof(T), NRAM2NRAM,
             box_dim_aligned * sizeof(T), box_dim * sizeof(T),
             deal_once_aligned - 1);
    __bang_transpose(nram_box, nram_box_trans, deal_once_aligned,
                     box_dim_aligned);
    __memcpy(boxes_trans + offset + i * deal_once_aligned, nram_box,
             deal_once_aligned * sizeof(T), NRAM2GDRAM, box_num * sizeof(T),
             deal_once_aligned * sizeof(T), box_dim - 1);
  }
  if (rem != 0) {
    __memcpy(nram_box, boxes + (offset + repeat * deal_once_aligned) * box_dim,
             rem * box_dim * sizeof(T), GDRAM2NRAM);
    __memcpy(nram_box_trans, nram_box, box_dim * sizeof(T), NRAM2NRAM,
             box_dim_aligned * sizeof(T), box_dim * sizeof(T), rem - 1);
    __bang_transpose(nram_box, nram_box_trans, rem_aligned, box_dim_aligned);
    __memcpy(boxes_trans + offset + repeat * deal_once_aligned, nram_box,
             rem * sizeof(T), NRAM2GDRAM, box_num * sizeof(T),
             rem_aligned * sizeof(T), box_dim - 1);
  }
#endif
}

// cross2d<T>(A, B) = A.x * B.y - A.y * B.x;
template <typename T>
inline __mlu_func__ void cross2d(T *result, const T *p1_x, const T *p1_y,
                                 const T *p2_x, const T *p2_y, const int &lenth,
                                 T *temp_ram) {
  __bang_mul((T *)temp_ram, (T *)p1_x, (T *)p2_y, lenth);
  __bang_mul((T *)result, (T *)p1_y, (T *)p2_x, lenth);
  __bang_sub((T *)result, (T *)temp_ram, (T *)result, lenth);
}

// dot2d<T>(A, B) =  A.x * B.x + A.y * B.y
template <typename T>
inline __mlu_func__ void dot2d(T *result, const T *p1_x, const T *p1_y,
                               const T *p2_x, const T *p2_y, const int &lenth,
                               T *temp_ram) {
  __bang_mul((T *)temp_ram, (T *)p1_x, (T *)p2_x, lenth);
  __bang_mul((T *)result, (T *)p1_y, (T *)p2_y, lenth);
  __bang_add((T *)result, (T *)temp_ram, (T *)result, lenth);
}

template <typename T>
__mlu_func__ void getRotatedVertices(T *pts_x, T *pts_y, T *box, T *temp1,
                                     T *temp2, T *temp3, T *temp4,
                                     const uint32_t &actual_compute_box_num) {
  // T cosTheta2 = (T)cos(theta) * 0.5f; -- temp1
  // T sinTheta2 = (T)sin(theta) * 0.5f; -- temp2
  // theta is the box's 5th data: a, rotated radian;
  __bang_cos((float *)temp1, ((float *)box) + 4 * actual_compute_box_num,
             actual_compute_box_num);
  __bang_sin((float *)temp2, ((float *)box) + 4 * actual_compute_box_num,
             actual_compute_box_num);

  __bang_mul_scalar((T *)temp1, (T *)temp1, (T)0.5, actual_compute_box_num);
  __bang_mul_scalar((T *)temp2, (T *)temp2, (T)0.5, actual_compute_box_num);

  // Temp3 = sinTheta2 * box.h;
  // Temp4 = cosTheta2 * box.w;
  __bang_mul((T *)temp3, (T *)temp2, ((T *)box) + 3 * actual_compute_box_num,
             actual_compute_box_num);
  __bang_mul((T *)temp4, (T *)temp1, ((T *)box) + 2 * actual_compute_box_num,
             actual_compute_box_num);
  // pts[0].x = box.x_ctr - sinTheta2 * box.h - cosTheta2 * box.w;
  // pts[1].x = box.x_ctr + sinTheta2 * box.h - cosTheta2 * box.w;
  __bang_sub((T *)pts_x, (T *)box, (T *)temp3, actual_compute_box_num);
  __bang_sub((T *)pts_x, (T *)pts_x, (T *)temp4, actual_compute_box_num);
  __bang_add((T *)pts_x + 1 * actual_compute_box_num, (T *)box, (T *)temp3,
             actual_compute_box_num);
  __bang_sub((T *)pts_x + 1 * actual_compute_box_num,
             (T *)pts_x + 1 * actual_compute_box_num, (T *)temp4,
             actual_compute_box_num);
  // Temp3 = cosTheta2 * box.h;
  // Temp4 = sinTheta2 * box.w;
  __bang_mul((T *)temp3, (T *)temp1, box + 3 * actual_compute_box_num,
             actual_compute_box_num);
  __bang_mul((T *)temp4, (T *)temp2, box + 2 * actual_compute_box_num,
             actual_compute_box_num);
  // pts[0].y = box.y_ctr + cosTheta2 * box.h - sinTheta2 * box.w;
  // pts[1].y = box.y_ctr - cosTheta2 * box.h - sinTheta2 * box.w;
  __bang_add((T *)pts_y, (T *)box + 1 * actual_compute_box_num, (T *)temp3,
             actual_compute_box_num);
  __bang_sub((T *)pts_y, (T *)pts_y, (T *)temp4, actual_compute_box_num);
  __bang_sub((T *)pts_y + 1 * actual_compute_box_num,
             (T *)box + 1 * actual_compute_box_num, (T *)temp3,
             actual_compute_box_num);
  __bang_sub((T *)pts_y + 1 * actual_compute_box_num,
             (T *)pts_y + 1 * actual_compute_box_num, (T *)temp4,
             actual_compute_box_num);
  // pts[2].x = 2 * box.x_ctr - pts[0].x;
  // pts[3].x = 2 * box.x_ctr - pts[1].x;
  __bang_mul_scalar((T *)pts_x + 2 * actual_compute_box_num, (T *)box, (T)2,
                    actual_compute_box_num);
  __bang_sub((T *)pts_x + 2 * actual_compute_box_num,
             (T *)pts_x + 2 * actual_compute_box_num, (T *)pts_x,
             actual_compute_box_num);
  __bang_mul_scalar((T *)pts_x + 3 * actual_compute_box_num, (T *)box, (T)2,
                    actual_compute_box_num);
  __bang_sub((T *)pts_x + 3 * actual_compute_box_num,
             (T *)pts_x + 3 * actual_compute_box_num,
             (T *)pts_x + 1 * actual_compute_box_num, actual_compute_box_num);
  // pts[2].y = 2 * box.y_ctr - pts[0].y;
  // pts[3].y = 2 * box.y_ctr - pts[1].y;
  __bang_mul_scalar((T *)pts_y + 2 * actual_compute_box_num,
                    (T *)box + 1 * actual_compute_box_num, (T)2,
                    actual_compute_box_num);
  __bang_sub((T *)pts_y + 2 * actual_compute_box_num,
             (T *)pts_y + 2 * actual_compute_box_num, (T *)pts_y,
             actual_compute_box_num);
  __bang_mul_scalar((T *)pts_y + 3 * actual_compute_box_num,
                    (T *)box + 1 * actual_compute_box_num, (T)2,
                    actual_compute_box_num);
  __bang_sub((T *)pts_y + 3 * actual_compute_box_num,
             (T *)pts_y + 3 * actual_compute_box_num,
             (T *)pts_y + 1 * actual_compute_box_num, actual_compute_box_num);
}

template <typename T>
__mlu_func__ void getIntersectionPoints(
    T *rotated_pts1_x, T *rotated_pts1_y, T *rotated_pts2_x, T *rotated_pts2_y,
    T *vec1_x, T *vec1_y, T *vec2_x, T *vec2_y, T *intersect_pts_x,
    T *intersect_pts_y, T *valid_pts, T *nums_in_ram, T *temp1_ram,
    T *temp2_ram, T *temp3_ram, T *temp4_ram, T *temp5_ram, T *temp6_ram,
    T *temp7_ram, T *temp8_ram, const uint32_t &actual_compute_box_num) {
  // Line vector, from p1 to p2 is: p1+(p2-p1)*t, t=[0,1]
  // for i = 0~3, vec[i] = pts[(i+1)%4] - pts[i]
  __bang_sub((T *)vec1_x, (T *)rotated_pts1_x + actual_compute_box_num,
             (T *)rotated_pts1_x, 3 * actual_compute_box_num);
  __bang_sub((T *)vec1_x + 3 * actual_compute_box_num, (T *)rotated_pts1_x,
             (T *)rotated_pts1_x + 3 * actual_compute_box_num,
             actual_compute_box_num);
  __bang_sub((T *)vec1_y, (T *)rotated_pts1_y + actual_compute_box_num,
             (T *)rotated_pts1_y, 3 * actual_compute_box_num);
  __bang_sub((T *)vec1_y + 3 * actual_compute_box_num, (T *)rotated_pts1_y,
             (T *)rotated_pts1_y + 3 * actual_compute_box_num,
             actual_compute_box_num);

  __bang_sub((T *)vec2_x, (T *)rotated_pts2_x + actual_compute_box_num,
             (T *)rotated_pts2_x, 3 * actual_compute_box_num);
  __bang_sub((T *)vec2_x + 3 * actual_compute_box_num, (T *)rotated_pts2_x,
             (T *)rotated_pts2_x + 3 * actual_compute_box_num,
             actual_compute_box_num);
  __bang_sub((T *)vec2_y, (T *)rotated_pts2_y + actual_compute_box_num,
             (T *)rotated_pts2_y, 3 * actual_compute_box_num);
  __bang_sub((T *)vec2_y + 3 * actual_compute_box_num, (T *)rotated_pts2_y,
             (T *)rotated_pts2_y + 3 * actual_compute_box_num,
             actual_compute_box_num);

  // First, line test - test all line combos for intersection, 4x4 possible
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      // T det = cross2d<T>(vec2[j], vec1[i]) -- temp2
      cross2d<T>((T *)temp2_ram, (T *)vec2_x + j * actual_compute_box_num,
                 (T *)vec2_y + j * actual_compute_box_num,
                 (T *)vec1_x + i * actual_compute_box_num,
                 (T *)vec1_y + i * actual_compute_box_num,
                 actual_compute_box_num, (T *)temp1_ram);
      // deal with parallel lines, temp2 = fabs(det), temp1 = temp2 <= 1e-14
      __bang_abs((T *)temp3_ram, (T *)temp2_ram, actual_compute_box_num);
      __bang_le_scalar((T *)temp1_ram, (T *)temp3_ram, (T)1e-14,
                       actual_compute_box_num);
      __bang_not((T *)temp1_ram, (T *)temp1_ram, actual_compute_box_num);

      // auto vec12 = pts2[j] - pts1[i], (temp4, temp5) = (vec12.x, vec12.y)
      __bang_sub((T *)temp4_ram,
                 (T *)rotated_pts2_x + j * actual_compute_box_num,
                 (T *)rotated_pts1_x + i * actual_compute_box_num,
                 actual_compute_box_num);
      __bang_sub((T *)temp5_ram,
                 (T *)rotated_pts2_y + j * actual_compute_box_num,
                 (T *)rotated_pts1_y + i * actual_compute_box_num,
                 actual_compute_box_num);

      // T t1 = cross2d<T>(vec2[j], vec12) / det  -- temp6
      cross2d<T>((T *)temp6_ram, (T *)vec2_x + j * actual_compute_box_num,
                 (T *)vec2_y + j * actual_compute_box_num, (T *)temp4_ram,
                 (T *)temp5_ram, actual_compute_box_num, (T *)temp7_ram);
      // T t2 = cross2d<T>(vec1[i], vec12) / det  -- temp7
      // NOTE: temp6(t1) is used after, reuse temp5(p2_y) as cross2d temp ram
      cross2d<T>((T *)temp7_ram, (T *)vec1_x + i * actual_compute_box_num,
                 (T *)vec1_y + i * actual_compute_box_num, (T *)temp4_ram,
                 (T *)temp5_ram, actual_compute_box_num, (T *)temp8_ram);

#if __BANG_ARCH__ == 372
      // Where temp1 = false, set recip input to 1, avoiding recip(0), cause inf
      __bang_not((T *)temp4_ram, (T *)temp1_ram, actual_compute_box_num);
      __bang_mul((T *)temp2_ram, (T *)temp2_ram, (T *)temp1_ram,
                 actual_compute_box_num);
      __bang_add((T *)temp2_ram, (T *)temp2_ram, (T *)temp4_ram,
                 actual_compute_box_num);
      // bang_recip only support float data type, others should use cast to
      // float first
      __bang_recip((float *)temp2_ram, (float *)temp2_ram,
                   actual_compute_box_num);
      __bang_mul((T *)temp6_ram, (T *)temp6_ram, (T *)temp2_ram,
                 actual_compute_box_num);
      __bang_mul((T *)temp7_ram, (T *)temp7_ram, (T *)temp2_ram,
                 actual_compute_box_num);
#else
      __bang_div((T *)temp6_ram, (T *)temp6_ram, (T *)temp2_ram,
                 actual_compute_box_num);
      __bang_div((T *)temp7_ram, (T *)temp7_ram, (T *)temp2_ram,
                 actual_compute_box_num);
#endif
      // temp1 &= (t1 >= 0.0f && t1 <= 1.0f)  -- temp7
      __bang_ge_scalar((T *)temp5_ram, (T *)temp6_ram, (T)0,
                       actual_compute_box_num);
      __bang_and((T *)temp1_ram, (T *)temp1_ram, (T *)temp5_ram,
                 actual_compute_box_num);
      __bang_le_scalar((T *)temp5_ram, (T *)temp6_ram, (T)1,
                       actual_compute_box_num);
      __bang_and((T *)temp1_ram, (T *)temp1_ram, (T *)temp5_ram,
                 actual_compute_box_num);

      // temp1 &= (t2 >= 0.0f && t2 <= 1.0f)  -- temp7
      __bang_ge_scalar((T *)temp5_ram, (T *)temp7_ram, (T)0,
                       actual_compute_box_num);
      __bang_and((T *)temp1_ram, (T *)temp1_ram, (T *)temp5_ram,
                 actual_compute_box_num);
      __bang_le_scalar((T *)temp5_ram, (T *)temp7_ram, (T)1,
                       actual_compute_box_num);
      __bang_and((T *)temp1_ram, (T *)temp1_ram, (T *)temp5_ram,
                 actual_compute_box_num);

      // NOTE: after there temp2 temp4 temp5 temp7 can be reused as temp ram

      // intersections = (pts1[i] + vec1[i] * t1) * temp1
      __bang_mul((T *)temp7_ram, (T *)vec1_x + i * actual_compute_box_num,
                 (T *)temp6_ram, actual_compute_box_num);
      __bang_add((T *)temp7_ram,
                 (T *)rotated_pts1_x + i * actual_compute_box_num,
                 (T *)temp7_ram, actual_compute_box_num);

      if (sizeof(T) == sizeof(float)) {
        __bang_float2int32((int32_t *)temp2_ram, (float *)temp1_ram,
                           actual_compute_box_num, 0);
        __bang_lut((int32_t *)temp2_ram, (uint32_t *)temp2_ram,
                   (int32_t *)table_float, actual_compute_box_num,
                   TABLE_LENGTH);
      } else {
        __bang_half2int16_rd((int16_t *)temp2_ram, (half *)temp2_ram,
                             actual_compute_box_num, 0);
        __bang_lut((int16_t *)temp2_ram, (uint16_t *)temp2_ram,
                   (int16_t *)table_half, actual_compute_box_num, TABLE_LENGTH);
      }
      __bang_band((int8_t *)((T *)intersect_pts_x +
                             (4 * i + j) * actual_compute_box_num),
                  (int8_t *)temp7_ram, (int8_t *)temp2_ram,
                  actual_compute_box_num * sizeof(T));

      __bang_mul((T *)temp7_ram, (T *)vec1_y + i * actual_compute_box_num,
                 (T *)temp6_ram, actual_compute_box_num);
      __bang_add((T *)temp7_ram,
                 (T *)rotated_pts1_y + i * actual_compute_box_num,
                 (T *)temp7_ram, actual_compute_box_num);

      __bang_band((int8_t *)((T *)intersect_pts_y +
                             (4 * i + j) * actual_compute_box_num),
                  (int8_t *)temp7_ram, (int8_t *)temp2_ram,
                  actual_compute_box_num * sizeof(T));

      // Assign `valid_pts` bit and accumulate `nums_in` of valid points of each
      // box pair
      __bang_or((T *)valid_pts + (4 * i + j) * actual_compute_box_num,
                (T *)valid_pts + (4 * i + j) * actual_compute_box_num,
                (T *)temp1_ram, actual_compute_box_num);
      __bang_add((T *)nums_in_ram, (T *)nums_in_ram, (T *)temp1_ram,
                 actual_compute_box_num);
    }
  }

  // Check for vertices of rect1 inside rect2
  // temp3 = ABdotAB
  dot2d<T>((T *)temp3_ram, (T *)vec2_x, (T *)vec2_y, (T *)vec2_x, (T *)vec2_y,
           actual_compute_box_num, (T *)temp7_ram);
  // temp4 = ADdotAD
  dot2d<T>((T *)temp4_ram, (T *)vec2_x + 3 * actual_compute_box_num,
           (T *)vec2_y + 3 * actual_compute_box_num,
           (T *)vec2_x + 3 * actual_compute_box_num,
           (T *)vec2_y + 3 * actual_compute_box_num, actual_compute_box_num,
           (T *)temp7_ram);
  // assume ABCD is the rectangle, and P is the point to be judged
  // P is inside ABCD iff. P's projection on AB lines within AB
  // and P's projection on AD lies within AD
  for (int i = 0; i < 4; i++) {
    // AP = pts1[i] - pts2[0] = (temp5, temp6)
    __bang_sub((T *)temp5_ram, (T *)rotated_pts1_x + i * actual_compute_box_num,
               (T *)rotated_pts2_x, actual_compute_box_num);
    __bang_sub((T *)temp6_ram, (T *)rotated_pts1_y + i * actual_compute_box_num,
               (T *)rotated_pts2_y, actual_compute_box_num);

    // temp7 = APdotAB = dot2d<T>(AP, AB)
    dot2d<T>((T *)temp7_ram, (T *)temp5_ram, (T *)temp6_ram, (T *)vec2_x,
             (T *)vec2_y, actual_compute_box_num, (T *)temp2_ram);
    // temp8 = APdotAD = -dot2d<T>(AP, DA)
    dot2d<T>((T *)temp8_ram, (T *)temp5_ram, (T *)temp6_ram,
             (T *)vec2_x + 3 * actual_compute_box_num,
             (T *)vec2_y + 3 * actual_compute_box_num, actual_compute_box_num,
             (T *)temp2_ram);
    __bang_mul_scalar((T *)temp8_ram, (T *)temp8_ram, (T)-1,
                      actual_compute_box_num);

    // ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <=
    // ADdotAD))
    __bang_ge_scalar((T *)temp1_ram, (T *)temp7_ram, (T)0,
                     actual_compute_box_num);
    __bang_ge_scalar((T *)temp2_ram, (T *)temp8_ram, (T)0,
                     actual_compute_box_num);
    __bang_and((T *)temp1_ram, (T *)temp1_ram, (T *)temp2_ram,
               actual_compute_box_num);
    __bang_le((T *)temp2_ram, (T *)temp7_ram, (T *)temp3_ram,
              actual_compute_box_num);
    __bang_and((T *)temp1_ram, (T *)temp1_ram, (T *)temp2_ram,
               actual_compute_box_num);
    __bang_le((T *)temp2_ram, (T *)temp8_ram, (T *)temp4_ram,
              actual_compute_box_num);
    __bang_and((T *)temp1_ram, (T *)temp1_ram, (T *)temp2_ram,
               actual_compute_box_num);

    // 16 means the 4x4 possible intersection points above
    if (sizeof(T) == sizeof(float)) {
      __bang_float2int32((int32_t *)temp2_ram, (float *)temp1_ram,
                         actual_compute_box_num, 0);
      __bang_lut((int32_t *)temp2_ram, (uint32_t *)temp2_ram,
                 (int32_t *)table_float, actual_compute_box_num, TABLE_LENGTH);
    } else {
      __bang_half2int16_rd((int16_t *)temp2_ram, (half *)temp1_ram,
                           actual_compute_box_num, 0);
      __bang_lut((int16_t *)temp2_ram, (uint16_t *)temp2_ram,
                 (int16_t *)table_half, actual_compute_box_num, TABLE_LENGTH);
    }
    __bang_band(
        (int8_t *)((T *)intersect_pts_x + (16 + i) * actual_compute_box_num),
        (int8_t *)((T *)rotated_pts1_x + i * actual_compute_box_num),
        (int8_t *)temp2_ram, actual_compute_box_num * sizeof(T));
    __bang_band(
        (int8_t *)((T *)intersect_pts_y + (16 + i) * actual_compute_box_num),
        (int8_t *)((T *)rotated_pts1_y + i * actual_compute_box_num),
        (int8_t *)temp2_ram, actual_compute_box_num * sizeof(T));

    // assign valid_pts bit and accumulate nums of valid points of each box pair
    __bang_or((T *)valid_pts + (16 + i) * actual_compute_box_num,
              (T *)valid_pts + (16 + i) * actual_compute_box_num,
              (T *)temp1_ram, actual_compute_box_num);
    __bang_add((T *)nums_in_ram, (T *)nums_in_ram, (T *)temp1_ram,
               actual_compute_box_num);
  }

  // Reverse the check - check for vertices of rect2 inside rect1
  // temp3 = ABdotAB
  dot2d<T>((T *)temp3_ram, (T *)vec1_x, (T *)vec1_y, (T *)vec1_x, (T *)vec1_y,
           actual_compute_box_num, (T *)temp7_ram);
  // temp4 = ADdotAD
  dot2d<T>((T *)temp4_ram, (T *)vec1_x + 3 * actual_compute_box_num,
           (T *)vec1_y + 3 * actual_compute_box_num,
           (T *)vec1_x + 3 * actual_compute_box_num,
           (T *)vec1_y + 3 * actual_compute_box_num, actual_compute_box_num,
           (T *)temp7_ram);
  for (int i = 0; i < 4; i++) {
    // AP = pts2[i] - pts1[0] = (temp5, temp6)
    __bang_sub((T *)temp5_ram, (T *)rotated_pts2_x + i * actual_compute_box_num,
               (T *)rotated_pts1_x, actual_compute_box_num);
    __bang_sub((T *)temp6_ram, (T *)rotated_pts2_y + i * actual_compute_box_num,
               (T *)rotated_pts1_y, actual_compute_box_num);

    // temp7 = APdotAB = dot2d<T>(AP, AB)
    dot2d<T>((T *)temp7_ram, (T *)temp5_ram, (T *)temp6_ram, (T *)vec1_x,
             (T *)vec1_y, actual_compute_box_num, (T *)temp2_ram);
    // temp8 = APdotAD = -dot2d<T>(AP, DA)
    dot2d<T>((T *)temp8_ram, (T *)temp5_ram, (T *)temp6_ram,
             (T *)vec1_x + 3 * actual_compute_box_num,
             (T *)vec1_y + 3 * actual_compute_box_num, actual_compute_box_num,
             (T *)temp2_ram);
    __bang_mul_scalar((T *)temp8_ram, (T *)temp8_ram, (T)-1,
                      actual_compute_box_num);

    // ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <=
    // ADdotAD))
    __bang_ge_scalar((T *)temp1_ram, (T *)temp7_ram, (T)0,
                     actual_compute_box_num);
    __bang_ge_scalar((T *)temp2_ram, (T *)temp8_ram, (T)0,
                     actual_compute_box_num);
    __bang_and((T *)temp1_ram, (T *)temp1_ram, (T *)temp2_ram,
               actual_compute_box_num);
    __bang_le((T *)temp2_ram, (T *)temp7_ram, (T *)temp3_ram,
              actual_compute_box_num);
    __bang_and((T *)temp1_ram, (T *)temp1_ram, (T *)temp2_ram,
               actual_compute_box_num);
    __bang_le((T *)temp2_ram, (T *)temp8_ram, (T *)temp4_ram,
              actual_compute_box_num);
    __bang_and((T *)temp1_ram, (T *)temp1_ram, (T *)temp2_ram,
               actual_compute_box_num);

    // 20 means the (4x4+4) possible intersection points above
    if (sizeof(T) == sizeof(float)) {
      __bang_float2int32((int32_t *)temp2_ram, (float *)temp1_ram,
                         actual_compute_box_num, 0);
      __bang_lut((int32_t *)temp2_ram, (uint32_t *)temp2_ram,
                 (int32_t *)table_float, actual_compute_box_num, TABLE_LENGTH);
    } else {
      __bang_half2int16_rd((int16_t *)temp2_ram, (half *)temp1_ram,
                           actual_compute_box_num, 0);
      __bang_lut((int16_t *)temp2_ram, (uint16_t *)temp2_ram,
                 (int16_t *)table_half, actual_compute_box_num, TABLE_LENGTH);
    }
    __bang_band(
        (int8_t *)((T *)intersect_pts_x + (20 + i) * actual_compute_box_num),
        (int8_t *)((T *)rotated_pts2_x + i * actual_compute_box_num),
        (int8_t *)temp2_ram, actual_compute_box_num * sizeof(T));
    __bang_band(
        (int8_t *)((T *)intersect_pts_y + (20 + i) * actual_compute_box_num),
        (int8_t *)((T *)rotated_pts2_y + i * actual_compute_box_num),
        (int8_t *)temp2_ram, actual_compute_box_num * sizeof(T));

    // assign valid_pts bit and accumulate nums of valid points of each box pair
    __bang_or((T *)valid_pts + (20 + i) * actual_compute_box_num,
              (T *)valid_pts + (20 + i) * actual_compute_box_num,
              (T *)temp1_ram, actual_compute_box_num);
    __bang_add((T *)nums_in_ram, (T *)nums_in_ram, (T *)temp1_ram,
               actual_compute_box_num);
  }
}

template <typename T>
__mlu_func__ void convexHullGraham(
    T *intersect_pts_x, T *intersect_pts_y, T *ordered_pts_x, T *ordered_pts_y,
    T *dist_ram, T *valid_box, T *valid_pts, T *nums_in_ram, T *temp1_ram,
    T *temp2_ram, T *temp3_ram, T *temp_long_1, T *temp_long_2, T *temp_long_3,
    const uint32_t &actual_box_num, const uint32_t &actual_compute_box_num) {
    if (__is_mpu()) {
      return;
    }
  // Step1. Find the point with minimum y, if more than 1 points have the same
  // minimum y, pick the one with the minimum x. set p[i].y to max_y_value if
  // not valid_pts, to avoid invalid result 24 means all possible intersection
  // points
  __bang_argmax((T *)temp2_ram, (T *)intersect_pts_y,
                24 * actual_compute_box_num);
  __bang_not((T *)temp_long_1, (T *)valid_pts, 24 * actual_compute_box_num);
  __bang_mul_scalar((T *)temp_long_1, (T *)temp_long_1,
                    (T)(((T *)temp2_ram)[0]), 24 * actual_compute_box_num);
  __bang_mul((T *)temp_long_2, (T *)intersect_pts_y, (T *)valid_pts,
             24 * actual_compute_box_num);
  __bang_add((T *)temp_long_2, (T *)temp_long_2, (T *)temp_long_1,
             24 * actual_compute_box_num);
  // temp2 = min_y_value(temp_long_2), use min_pool, channel=box_num, h=1, w=24
  __bang_minpool((T *)temp2_ram, (T *)temp_long_2, actual_compute_box_num, 1,
                 24, 1, 24, 1, 24);

  __bang_mul((T *)temp2_ram, (T *)temp2_ram, (T *)valid_box,
             actual_compute_box_num);

  // set p[i].x to max_x_value if not min_y point
  __bang_argmax((T *)temp1_ram, (T *)intersect_pts_x,
                24 * actual_compute_box_num);
  __bang_cycle_eq((T *)temp_long_1, (T *)temp_long_2, (T *)temp2_ram,
                  24 * actual_compute_box_num, actual_compute_box_num);
  __bang_and((T *)temp_long_1, (T *)temp_long_1, (T *)valid_pts,
             24 * actual_compute_box_num);
  __bang_not((T *)temp_long_3, (T *)temp_long_1, 24 * actual_compute_box_num);
  __bang_mul_scalar((T *)temp_long_3, (T *)temp_long_3, (T)((T *)temp1_ram)[0],
                    24 * actual_compute_box_num);
  __bang_mul((T *)temp_long_1, (T *)intersect_pts_x, (T *)temp_long_1,
             24 * actual_compute_box_num);
  __bang_add((T *)temp_long_1, (T *)temp_long_1, (T *)temp_long_3,
             24 * actual_compute_box_num);
  // temp3 = min_x_value(temp_long_1), use min_pool, channel=box_num, h=1, w=24
  __bang_minpool((T *)temp3_ram, (T *)temp_long_1, actual_compute_box_num, 1,
                 24, 1, 24, 1, 24);

  __bang_mul((T *)temp3_ram, (T *)temp3_ram, (T *)valid_box,
             actual_compute_box_num);

  // Step2. All points subtract starting-point (for sorting in the next step)
  __bang_cycle_sub((T *)ordered_pts_x, (T *)intersect_pts_x, (T *)temp3_ram,
                   24 * actual_compute_box_num, actual_compute_box_num);
  __bang_cycle_sub((T *)ordered_pts_y, (T *)intersect_pts_y, (T *)temp2_ram,
                   24 * actual_compute_box_num, actual_compute_box_num);
  __bang_mul((T *)ordered_pts_x, (T *)ordered_pts_x, (T *)valid_pts,
             24 * actual_compute_box_num);
  __bang_mul((T *)ordered_pts_y, (T *)ordered_pts_y, (T *)valid_pts,
             24 * actual_compute_box_num);

  // Step3. Sort every intersection point according to their relative
  //        cross-product values (essentially sorting according to angles)
  //        If the angles are the same, sort according to distance to origin
  dot2d<T>((T *)dist_ram, (T *)ordered_pts_x, (T *)ordered_pts_y,
           (T *)ordered_pts_x, (T *)ordered_pts_y, 24 * actual_compute_box_num,
           (T *)temp_long_3);

  T temp, temp_nums_in, temp_dist_1, temp_dist_2;
  T temp1_x, temp1_y;
  T temp2_x, temp2_y;
  for (int i = 0; i < actual_box_num; i++) {
    if (((T *)valid_box)[i]) {
      // make sure all nums_in[i] points are at the front
      for (int ii = 0; ii < 23; ii++) {
        for (int jj = ii + 1; jj < 24; jj++) {
          int ii_index = ii * actual_compute_box_num + i;
          int jj_index = jj * actual_compute_box_num + i;
          // ii point is not valid and jj point is valid, swap jj for ii
          if ((!((T *)valid_pts)[ii_index]) && ((T *)valid_pts)[jj_index]) {
            ((T *)ordered_pts_x)[ii_index] = ((T *)ordered_pts_x)[jj_index];
            ((T *)ordered_pts_y)[ii_index] = ((T *)ordered_pts_y)[jj_index];
            ((T *)dist_ram)[ii_index] = ((T *)dist_ram)[jj_index];
            ((T *)valid_pts)[ii_index] = true;
            ((T *)ordered_pts_x)[jj_index] = 0;
            ((T *)ordered_pts_y)[jj_index] = 0;
            ((T *)dist_ram)[jj_index] = 0;
            ((T *)valid_pts)[jj_index] = false;
            break;
          }
        }
      }
      temp_nums_in = ((T *)nums_in_ram)[i];
      // make original q[0] = min_x, min_y before sort
      for (int ii = 1; ii < temp_nums_in; ii++) {
        int ii_index = ii * actual_compute_box_num + i;
        if (((T *)dist_ram)[ii_index] == 0) {
          // swap q[ii_index] and q[0]
          ((T *)ordered_pts_x)[ii_index] = ((T *)ordered_pts_x)[i];
          ((T *)ordered_pts_y)[ii_index] = ((T *)ordered_pts_y)[i];
          ((T *)dist_ram)[ii_index] = ((T *)dist_ram)[i];
          ((T *)ordered_pts_x)[i] = 0;
          ((T *)ordered_pts_y)[i] = 0;
          ((T *)dist_ram)[i] = 0;
          break;
        }
      }
      for (int ii = 1; ii < temp_nums_in - 1; ii++) {
        for (int jj = ii + 1; jj < temp_nums_in; jj++) {
          int ii_index = ii * actual_compute_box_num + i;
          int jj_index = jj * actual_compute_box_num + i;
          temp1_x = ((T *)ordered_pts_x)[ii_index];
          temp1_y = ((T *)ordered_pts_y)[ii_index];
          temp2_x = ((T *)ordered_pts_x)[jj_index];
          temp2_y = ((T *)ordered_pts_y)[jj_index];
          // calculate cross product and sort q (ordered_pts)
          temp = (temp1_x * temp2_y) - (temp1_y * temp2_x);
          temp_dist_1 = ((T *)dist_ram)[ii_index];
          temp_dist_2 = ((T *)dist_ram)[jj_index];
          if ((temp < (T)-1e-6) ||
              ((fabs(temp) < (T)1e-6) && (temp_dist_1 > temp_dist_2))) {
            ((T *)ordered_pts_x)[ii_index] = temp2_x;
            ((T *)ordered_pts_y)[ii_index] = temp2_y;
            ((T *)ordered_pts_x)[jj_index] = temp1_x;
            ((T *)ordered_pts_y)[jj_index] = temp1_y;
            ((T *)dist_ram)[ii_index] = temp_dist_2;
            ((T *)dist_ram)[jj_index] = temp_dist_1;
          }
        }
      }

      // Step4:
      // Make sure there are at least 2 points(that don't overlap with each
      // other) in the stack
      int k;  // index of the non-overlapped second point
      for (k = 1; k < temp_nums_in; k++) {
        if (((T *)dist_ram)[k * actual_compute_box_num + i] > (T)1e-8) {
          break;
        }
      }
      if (k == temp_nums_in) {
        // We reach the end, which means the convex hull is just one point
        // set valid_box = 0, to get ious = 0
        ((T *)valid_box)[i] = 0;
        continue;
      }
      // q[1] = q[k];
      ((T *)ordered_pts_x)[actual_compute_box_num + i] =
          ((T *)ordered_pts_x)[k * actual_compute_box_num + i];
      ((T *)ordered_pts_y)[actual_compute_box_num + i] =
          ((T *)ordered_pts_y)[k * actual_compute_box_num + i];

      // Step 5:
      // Finally we can start the scanning process.
      // When a non-convex relationship between the 3 points is found
      // (either concave shape or duplicated points),
      // we pop the previous point from the stack
      // until the 3-point relationship is convex again, or
      // until the stack only contains two points
      int m = 2;  // 2 points in the stack
      for (int j = k + 1; j < temp_nums_in; j++) {
        // while (m > 1 && cross2d<T>(q[j] - q[m - 2], q[m - 1] - q[m - 2]) >=
        // 0) {
        //   m--;
        // }
        temp1_x = ((T *)ordered_pts_x)[j * actual_compute_box_num + i] -
                  ((T *)ordered_pts_x)[(m - 2) * actual_compute_box_num + i];
        temp1_y = ((T *)ordered_pts_y)[j * actual_compute_box_num + i] -
                  ((T *)ordered_pts_y)[(m - 2) * actual_compute_box_num + i];
        temp2_x = ((T *)ordered_pts_x)[(m - 1) * actual_compute_box_num + i] -
                  ((T *)ordered_pts_x)[(m - 2) * actual_compute_box_num + i];
        temp2_y = ((T *)ordered_pts_y)[(m - 1) * actual_compute_box_num + i] -
                  ((T *)ordered_pts_y)[(m - 2) * actual_compute_box_num + i];
        temp = (temp1_x * temp2_y) - (temp1_y * temp2_x);
        while ((m > 1) && (temp >= 0)) {
          m--;
          if (m > 1) {
            temp1_x =
                ((T *)ordered_pts_x)[j * actual_compute_box_num + i] -
                ((T *)ordered_pts_x)[(m - 2) * actual_compute_box_num + i];
            temp1_y =
                ((T *)ordered_pts_y)[j * actual_compute_box_num + i] -
                ((T *)ordered_pts_y)[(m - 2) * actual_compute_box_num + i];
            temp2_x =
                ((T *)ordered_pts_x)[(m - 1) * actual_compute_box_num + i] -
                ((T *)ordered_pts_x)[(m - 2) * actual_compute_box_num + i];
            temp2_y =
                ((T *)ordered_pts_y)[(m - 1) * actual_compute_box_num + i] -
                ((T *)ordered_pts_y)[(m - 2) * actual_compute_box_num + i];
            temp = (temp1_x * temp2_y) - (temp1_y * temp2_x);
          }
        }
        // q[m++] = q[j];
        ((T *)ordered_pts_x)[m * actual_compute_box_num + i] =
            ((T *)ordered_pts_x)[j * actual_compute_box_num + i];
        ((T *)ordered_pts_y)[m * actual_compute_box_num + i] =
            ((T *)ordered_pts_y)[j * actual_compute_box_num + i];
        m++;
      }
      // set last(24-m) valid_pts to false, to erase invalid q in polygon area
      for (int j = m; j < temp_nums_in; j++) {
        ((T *)valid_pts)[j * actual_compute_box_num + i] = 0;
      }
      ((T *)nums_in_ram)[i] = m;
    }
  }
}

template <typename T>
__mlu_func__ void polygonArea(T *ordered_pts_x, T *ordered_pts_y, T *valid_box,
                              T *valid_pts, T *nums_in_ram, T *temp1_ram,
                              T *temp2_ram, T *temp3_ram, T *temp4_ram,
                              T *temp5_ram, T *temp6_ram, T *temp7_ram,
                              T *temp8_ram,
                              const uint32_t &actual_compute_box_num) {
  // Set where nums_in <= 2, valid_box = false
  __bang_le_scalar((T *)temp1_ram, (T *)nums_in_ram, (T)2,
                   actual_compute_box_num);
  __bang_not((T *)temp1_ram, (T *)temp1_ram, actual_compute_box_num);
  __bang_and((T *)valid_box, (T *)valid_box, (T *)temp1_ram,
             actual_compute_box_num);

  // temp1 = area, initialize with all 0
  __bang_write_value((T *)temp1_ram, actual_compute_box_num, (T)0);
  __bang_argmax((T *)temp6_ram, (T *)nums_in_ram, actual_compute_box_num);

  // temp_nums_in = max(nums_in)
  T temp_nums_in = ((T *)temp6_ram)[0];
  for (int i = 1; i < temp_nums_in - 1; i++) {
    // q[i] - q[0]: (temp5, temp6)
    __bang_sub((T *)temp5_ram, (T *)ordered_pts_x + i * actual_compute_box_num,
               (T *)ordered_pts_x, actual_compute_box_num);
    __bang_sub((T *)temp6_ram, (T *)ordered_pts_y + i * actual_compute_box_num,
               (T *)ordered_pts_y, actual_compute_box_num);
    __bang_mul((T *)temp5_ram, (T *)temp5_ram,
               (T *)valid_pts + (i + 1) * actual_compute_box_num,
               actual_compute_box_num);
    __bang_mul((T *)temp6_ram, (T *)temp6_ram,
               (T *)valid_pts + (i + 1) * actual_compute_box_num,
               actual_compute_box_num);
    // q[i + 1] - q[0]: (temp7, temp8)
    __bang_sub((T *)temp7_ram,
               (T *)ordered_pts_x + (i + 1) * actual_compute_box_num,
               (T *)ordered_pts_x, actual_compute_box_num);
    __bang_sub((T *)temp8_ram,
               (T *)ordered_pts_y + (i + 1) * actual_compute_box_num,
               (T *)ordered_pts_y, actual_compute_box_num);
    __bang_mul((T *)temp7_ram, (T *)temp7_ram,
               (T *)valid_pts + (i + 1) * actual_compute_box_num,
               actual_compute_box_num);
    __bang_mul((T *)temp8_ram, (T *)temp8_ram,
               (T *)valid_pts + (i + 1) * actual_compute_box_num,
               actual_compute_box_num);
    // area += fabs(cross2d<T>(q[i] - q[0], q[i + 1] - q[0]));
    __bang_mul((T *)temp3_ram, (T *)temp5_ram, (T *)temp8_ram,
               actual_compute_box_num);
    __bang_mul((T *)temp4_ram, (T *)temp6_ram, (T *)temp7_ram,
               actual_compute_box_num);
    __bang_sub((T *)temp2_ram, (T *)temp3_ram, (T *)temp4_ram,
               actual_compute_box_num);
    __bang_abs((T *)temp2_ram, (T *)temp2_ram, actual_compute_box_num);
    __bang_add((T *)temp1_ram, (T *)temp1_ram, (T *)temp2_ram,
               actual_compute_box_num);
  }
  //  Set where valid_box = false, intersection = 0
  __bang_mul((T *)temp1_ram, (T *)temp1_ram, (T *)valid_box,
             actual_compute_box_num);
  //  area = area / 2.0
  __bang_mul_scalar((T *)temp1_ram, (T *)temp1_ram, (T)0.5,
                    actual_compute_box_num);
}

template <typename T>
__mlu_func__ void calIntersectIou(T *ious_ram, T *area1_ram, T *area2_ram,
                                  T *intersect_area_ram, T *temp1_ram,
                                  const int &mode,
                                  const uint32_t &actual_compute_box_num) {
  if (mode == 0) {
    // Iou = intersection / (area1 + area2 - intersection)
    __bang_add((T *)area1_ram, (T *)area1_ram, (T *)area2_ram,
               actual_compute_box_num);
    __bang_sub((T *)area1_ram, (T *)area1_ram, (T *)intersect_area_ram,
               actual_compute_box_num);
  } else {
    // Iof = intersection / (area1)
  }

#if __BANG_ARCH__ == 372
  // where area1 = 0, set recip input to 1, to avoid recip(0), which will cause
  // inf and its valid_box = 0, intersection_area = 0, ious = 0
  __bang_eq_scalar((T *)temp1_ram, (T *)area1_ram, (T)0,
                   actual_compute_box_num);
  __bang_add((T *)area1_ram, (T *)area1_ram, (T *)temp1_ram,
             actual_compute_box_num);
  // bang_recip only support float data type, others should use cast to float
  // first
  __bang_recip((float *)area1_ram, (float *)area1_ram, actual_compute_box_num);
  __bang_mul((T *)ious_ram, (T *)intersect_area_ram, (T *)area1_ram,
             actual_compute_box_num);
#else
  __bang_div((T *)ious_ram, (T *)intersect_area_ram, (T *)area1_ram,
             actual_compute_box_num);
#endif
}

#endif  // KERNELS_NMS_ROTATED_NMS_UTILS_H_
