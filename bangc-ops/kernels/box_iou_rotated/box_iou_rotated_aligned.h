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
#ifndef KERNELS_BOX_IOU_ROTATED_BOX_IOU_ROTATED_ALIGNED_H_
#define KERNELS_BOX_IOU_ROTATED_BOX_IOU_ROTATED_ALIGNED_H_

#include "kernels/box_iou_rotated/box_iou_rotated_utils.h"
#include "kernels/utils/common.h"

// macros defined in kernel.h
// #define COMPUTE_COUNT_ALIGN 64   // elem_count must be divisible by 64
// #define CEIL_ALIGN(x, align) (((x) + (align) - 1) / (align) * (align))
// #define FLOOR_ALIGN(x, align) ((x) / (align) * (align))

/* NRAM buffer
 * aligned = true
                      Total: 260 copies of max_box_pair ram
------------------------------------------------------------------------------
|  final data  | box1_trans | box2_trans |  box1_onchip  |    box2_onchip    |
|   each size  |    each ram size        |         each ram size             |
|    4xN       |        2x64 x N         |            2x64 x N               |
|            _/ 1~24 xN   2x4*2 vec|     |-----24*2xN----|--4xN-|---24*2xN---|
| valid_box |  valid_pts    vec1/2 |     |  intersectPts | Temp  \ orderedPts|
|   area1   |---------------------/      | & --2x4*2xN-- |  1~3   |   4~52   |
|   area2   |  25~48 xN     49~120 xN    |pts1.x/y 2.x/y |num_in:4|  Temp4/5 |
|   ious    | dist of q    temp_long 1~3 |---------------|--------|-- 53~54 -|
-----------------------------------------|               |Temp 6~10:  55~59 -|
                                         -------------------------------------
*/

// The addresses have already add offsets for each ipu core in union.mlu
template <typename T>
__mlu_func__ void MLUUnion1BoxIouRotatedAligned(const T *box1, const T *box2,
                                                T *ious, const int32_t num_box,
                                                const int32_t mode) {
  // If current ipu core don't need to work
  if (num_box <= 0) {
    return;
  }
  // NRAM divide by (4+4*COMPUTE_COUNT_ALIGN) copies of NRAM, counted by bytes
  const uint32_t copies_of_nram = 260 * sizeof(T);
  // Num of once max dealing box pair, also need align
  const uint32_t max_box_pair =
      FLOOR_ALIGN(MAX_NRAM_SIZE / copies_of_nram, COMPUTE_COUNT_ALIGN);
  // First, initialize ram with all 0, or could cause nan/inf unexcepted results
  __bang_write_zero((unsigned char *)nram_buffer,
                    copies_of_nram * max_box_pair);

  void *box1_trans = nram_buffer + 4 * max_box_pair * sizeof(T);
  void *box2_trans =
      nram_buffer + (1 * COMPUTE_COUNT_ALIGN + 4) * max_box_pair * sizeof(T);
  void *box1_onchip =
      nram_buffer + (2 * COMPUTE_COUNT_ALIGN + 4) * max_box_pair * sizeof(T);
  void *box2_onchip =
      nram_buffer + (3 * COMPUTE_COUNT_ALIGN + 4) * max_box_pair * sizeof(T);

  // load offchip current data, for loop
  int repeat = num_box / max_box_pair;
  int remainder = num_box % max_box_pair;
  repeat += int(remainder > 0);

  // Only consider loop offset in one ipu core
  size_t current_box_offset = 0;
  size_t current_ious_offset = 0;
  for (int loop_box_i = 0; loop_box_i < repeat; loop_box_i++) {
    int actual_box_num;
    if (remainder != 0) {
      actual_box_num = (loop_box_i == repeat - 1) ? remainder : max_box_pair;
    } else {
      actual_box_num = max_box_pair;
    }

    __memcpy((T *)box1_onchip, box1 + current_box_offset,
             actual_box_num * SINGLE_BOX_DIM * sizeof(T), GDRAM2NRAM);
    __memcpy((T *)box2_onchip, box2 + current_box_offset,
             actual_box_num * SINGLE_BOX_DIM * sizeof(T), GDRAM2NRAM);

    current_box_offset += actual_box_num * SINGLE_BOX_DIM;
    uint32_t actual_compute_box_num =
        CEIL_ALIGN(actual_box_num, COMPUTE_COUNT_ALIGN);

    // Trans: Nx5 -> 5xN, Mx5 -> 5xM
#if __BANG_ARCH__ >= 300
    // Transpose no need to align
    __bang_transpose((T *)box1_trans, (T *)box1_onchip, actual_compute_box_num,
                     SINGLE_BOX_DIM);
    __bang_transpose((T *)box2_trans, (T *)box2_onchip, actual_compute_box_num,
                     SINGLE_BOX_DIM);
#else
    // Transpose need onchip memcpy_str, align 5->COMPUTE_COUNT_ALIGN
    // Use box2_trans/box1_onchip as temp space
    __memcpy((T *)box2_trans, (T *)box1_onchip, SINGLE_BOX_DIM * sizeof(T),
             NRAM2NRAM, COMPUTE_COUNT_ALIGN * sizeof(T),
             SINGLE_BOX_DIM * sizeof(T), actual_box_num);
    __memcpy((T *)box1_onchip, (T *)box2_onchip, SINGLE_BOX_DIM * sizeof(T),
             NRAM2NRAM, COMPUTE_COUNT_ALIGN * sizeof(T),
             SINGLE_BOX_DIM * sizeof(T), actual_box_num);
    __bang_transpose((T *)box1_trans, (T *)box2_trans, actual_compute_box_num,
                     COMPUTE_COUNT_ALIGN);
    __bang_transpose((T *)box2_trans, (T *)box1_onchip, actual_compute_box_num,
                     COMPUTE_COUNT_ALIGN);
#endif  // BANG_ARCH if
    // After transpose, box1/2_onchip data can be over-written
    void *temp1_ram = (char *)box2_onchip;
    void *temp2_ram = ((char *)box2_onchip) + 1 * max_box_pair * sizeof(T);
    void *temp3_ram = ((char *)box2_onchip) + 2 * max_box_pair * sizeof(T);

    void *valid_box = nram_buffer;
    void *area1_ram = nram_buffer + 1 * max_box_pair * sizeof(T);
    void *area2_ram = nram_buffer + 2 * max_box_pair * sizeof(T);
    void *ious_ram = nram_buffer + 3 * max_box_pair * sizeof(T);

    // Initialize valid_box, set actual_box_num boxes to 1, else set to 0
    __bang_write_value((T *)valid_box, actual_compute_box_num, (T)1);
    if (actual_box_num < actual_compute_box_num) {
      for (int i = actual_box_num; i < actual_compute_box_num; i++) {
        ((T *)valid_box)[i] = 0;
      }
    }

    // Each box data: x, y, w, h, a
    // area1 = box1.h * box1.w;
    // area2 = box2.h * box2.w;
    __bang_mul((T *)area1_ram, ((T *)box1_trans) + 2 * actual_compute_box_num,
               ((T *)box1_trans) + 3 * actual_compute_box_num,
               actual_compute_box_num);
    __bang_mul((T *)area2_ram, ((T *)box2_trans) + 2 * actual_compute_box_num,
               ((T *)box2_trans) + 3 * actual_compute_box_num,
               actual_compute_box_num);

    // Where area < 1e-14(@float), valid_box set to 0
    __bang_write_value((T *)temp1_ram, COMPUTE_COUNT_ALIGN, (T)1e-14);
    __bang_cycle_ge((T *)temp2_ram, (T *)area1_ram, (T *)temp1_ram,
                    actual_compute_box_num, COMPUTE_COUNT_ALIGN);
    __bang_and((T *)valid_box, (T *)valid_box, (T *)temp2_ram,
               actual_compute_box_num);
    __bang_cycle_ge((T *)temp2_ram, (T *)area2_ram, (T *)temp1_ram,
                    actual_compute_box_num, COMPUTE_COUNT_ALIGN);
    __bang_and((T *)valid_box, (T *)valid_box, (T *)temp2_ram,
               actual_compute_box_num);

    // 1. Calculate new points
    // center_shift_x = (box1_raw.x_ctr + box2_raw.x_ctr) / 2.0;  ----temp1
    // center_shift_y = (box1_raw.y_ctr + box2_raw.y_ctr) / 2.0;  ----temp2
    __bang_add((T *)temp1_ram, (T *)box1_trans, (T *)box2_trans,
               actual_compute_box_num);
    __bang_add((T *)temp2_ram, ((T *)box1_trans) + 1 * actual_compute_box_num,
               ((T *)box2_trans) + 1 * actual_compute_box_num,
               actual_compute_box_num);
    __bang_mul_scalar((T *)temp1_ram, (T *)temp1_ram, (T)0.5,
                      actual_compute_box_num);
    __bang_mul_scalar((T *)temp2_ram, (T *)temp2_ram, (T)0.5,
                      actual_compute_box_num);
    // box1.x_ctr = box1_raw.x_ctr - center_shift_x;
    // box1.y_ctr = box1_raw.y_ctr - center_shift_y;
    // box2.x_ctr = box2_raw.x_ctr - center_shift_x;
    // box2.y_ctr = box2_raw.y_ctr - center_shift_y;
    __bang_sub((T *)box1_trans, (T *)box1_trans, (T *)temp1_ram,
               actual_compute_box_num);
    __bang_sub((T *)box1_trans + 1 * actual_compute_box_num,
               (T *)box1_trans + 1 * actual_compute_box_num, (T *)temp2_ram,
               actual_compute_box_num);
    __bang_sub((T *)box2_trans, (T *)box2_trans, (T *)temp1_ram,
               actual_compute_box_num);
    __bang_sub((T *)box2_trans + 1 * actual_compute_box_num,
               (T *)box2_trans + 1 * actual_compute_box_num, (T *)temp2_ram,
               actual_compute_box_num);

    // 2. Calculate rotated vertices
    // Rotated vertices, each box has 4 vertices, each point has (x, y)
    void *rotated_pts1_x =
        ((char *)box1_onchip) + 48 * max_box_pair * sizeof(T);
    void *rotated_pts1_y =
        ((char *)box1_onchip) + 52 * max_box_pair * sizeof(T);
    void *rotated_pts2_x =
        ((char *)box1_onchip) + 56 * max_box_pair * sizeof(T);
    void *rotated_pts2_y =
        ((char *)box1_onchip) + 60 * max_box_pair * sizeof(T);

    void *temp4_ram = ((char *)box2_onchip) + 52 * max_box_pair * sizeof(T);
    void *temp5_ram = ((char *)box2_onchip) + 53 * max_box_pair * sizeof(T);

    getRotatedVertices((T *)rotated_pts1_x, (T *)rotated_pts1_y,
                       (T *)box1_trans, (T *)temp1_ram, (T *)temp2_ram,
                       (T *)temp4_ram, (T *)temp5_ram, actual_compute_box_num);
    getRotatedVertices((T *)rotated_pts2_x, (T *)rotated_pts2_y,
                       (T *)box2_trans, (T *)temp1_ram, (T *)temp2_ram,
                       (T *)temp4_ram, (T *)temp5_ram, actual_compute_box_num);

    // After calculating rotated vertices, box1/2_trans data can be over-written
    // Intersect points = [24xN] points, each point has (x, y)
    void *intersect_pts_x = (char *)box1_onchip;
    void *intersect_pts_y =
        ((char *)box1_onchip) + 24 * max_box_pair * sizeof(T);
    // Record whether this position of intersect points is valid or not
    void *valid_pts = (char *)box1_trans;
    // Record each box pair has how many valid intersect points
    void *nums_in_ram = ((char *)box2_onchip) + 3 * max_box_pair * sizeof(T);
    // initialize valid_pts, nums_in
    __bang_write_zero((T *)valid_pts, 24 * actual_compute_box_num);
    __bang_write_zero((T *)nums_in_ram, actual_compute_box_num);

    // 3. Get all intersection points
    // Line vector, from p1 to p2 is: p1+(p2-p1)*t, t=[0,1]
    void *vec1_x = ((char *)box1_trans) + 24 * max_box_pair * sizeof(T);
    void *vec1_y = ((char *)box1_trans) + 28 * max_box_pair * sizeof(T);
    void *vec2_x = ((char *)box1_trans) + 32 * max_box_pair * sizeof(T);
    void *vec2_y = ((char *)box1_trans) + 36 * max_box_pair * sizeof(T);

    void *temp6_ram = ((char *)box2_onchip) + 54 * max_box_pair * sizeof(T);
    void *temp7_ram = ((char *)box2_onchip) + 55 * max_box_pair * sizeof(T);
    void *temp8_ram = ((char *)box2_onchip) + 56 * max_box_pair * sizeof(T);
    void *temp9_ram = ((char *)box2_onchip) + 57 * max_box_pair * sizeof(T);
    void *temp10_ram = ((char *)box2_onchip) + 58 * max_box_pair * sizeof(T);

    getIntersectPts((T *)rotated_pts1_x, (T *)rotated_pts1_y,
                    (T *)rotated_pts2_x, (T *)rotated_pts2_y, (T *)vec1_x,
                    (T *)vec1_y, (T *)vec2_x, (T *)vec2_y, (T *)intersect_pts_x,
                    (T *)intersect_pts_y, (T *)valid_pts, (T *)nums_in_ram,
                    (T *)temp1_ram, (T *)temp2_ram, (T *)temp3_ram,
                    (T *)temp4_ram, (T *)temp5_ram, (T *)temp6_ram,
                    (T *)temp7_ram, (T *)temp8_ram, (T *)temp9_ram,
                    (T *)temp10_ram, actual_compute_box_num);

    // Where nums_in <= 2, set valid_box to false
    __bang_write_value((T *)temp9_ram, COMPUTE_COUNT_ALIGN, (T)2);
    __bang_cycle_gt((T *)temp1_ram, (T *)nums_in_ram, (T *)temp9_ram,
                    actual_compute_box_num, COMPUTE_COUNT_ALIGN);
    __bang_and((T *)valid_box, (T *)valid_box, (T *)temp1_ram,
               actual_compute_box_num);
    __bang_cycle_and((T *)valid_pts, (T *)valid_pts, (T *)valid_box,
                     24 * actual_compute_box_num, actual_compute_box_num);

    // 4. Convex-hull-graham to order the intersection points in clockwise order
    // and find the contour area

    // Ordered points = [24xN] points, each point has (x, y)
    void *ordered_pts_x = ((char *)box2_onchip) + 4 * max_box_pair * sizeof(T);
    void *ordered_pts_y = ((char *)box2_onchip) + 28 * max_box_pair * sizeof(T);

    void *dist_ram = ((char *)box1_trans) + 24 * max_box_pair * sizeof(T);
    void *temp_long_1 = ((char *)box1_trans) + 48 * max_box_pair * sizeof(T);
    void *temp_long_2 = ((char *)box1_trans) + 72 * max_box_pair * sizeof(T);
    void *temp_long_3 = ((char *)box1_trans) + 96 * max_box_pair * sizeof(T);

    convexHullGraham((T *)intersect_pts_x, (T *)intersect_pts_y,
                     (T *)ordered_pts_x, (T *)ordered_pts_y, (T *)dist_ram,
                     (T *)valid_box, (T *)valid_pts, (T *)nums_in_ram,
                     (T *)temp7_ram, (T *)temp8_ram, (T *)temp9_ram,
                     (T *)temp_long_1, (T *)temp_long_2, (T *)temp_long_3,
                     actual_box_num, actual_compute_box_num);

    // 5. Calculate polygon area
    // set temp1 = intersection part area
    polygonArea((T *)ordered_pts_x, (T *)ordered_pts_y, (T *)valid_box,
                (T *)valid_pts, (T *)nums_in_ram, (T *)temp1_ram,
                (T *)temp2_ram, (T *)temp3_ram, (T *)temp4_ram, (T *)temp5_ram,
                (T *)temp6_ram, (T *)temp7_ram, (T *)temp8_ram, (T *)temp9_ram,
                actual_compute_box_num);

    // calculate finally ious according to mode
    calIntersectIou((T *)ious_ram, (T *)area1_ram, (T *)area2_ram,
                    (T *)temp1_ram, (T *)temp4_ram, (T *)temp9_ram, mode,
                    actual_compute_box_num);

    __memcpy(ious + current_ious_offset, (T *)ious_ram,
             actual_box_num * sizeof(T), NRAM2GDRAM);
    current_ious_offset += actual_box_num;
  }
}

#endif  // KERNELS_BOX_IOU_ROTATED_BOX_IOU_ROTATED_ALIGNED_H_
