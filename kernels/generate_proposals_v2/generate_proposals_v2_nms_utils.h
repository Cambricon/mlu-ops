/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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

#define ALIGN_NUM NFU_ALIGN_SIZE / sizeof(float)
#define PROPOSAL_NRAM_SIZE MAX_NRAM_SIZE
#define CALC_AREA_NRAM_FLT_CAP CALC_AREA_NRAM_SIZE / sizeof(float)

__nram__ char nram_buffer[PROPOSAL_NRAM_SIZE];
__mlu_shared__ char sram_buffer[MAX_SRAM_SIZE];

#define FLOAT_MIN_GPV2 (-(float)FLT_MAX)

template <typename T>
__mlu_func__ void calcExp(T *output, T *input, int length) {
#define LOG_2_E (1.44269504088f)
  __bang_mul_scalar(output, input, (float)LOG_2_E, length);
  __bang_pow2(output, output, length);
}

template <typename T>
__mlu_func__ void findCoreMaxBox(T *input_score_ptr, T *score, T *inter_x1,
                                 T *inter_y1, T *inter_x2, T *max_box,
                                 const T *input_x1_ptr, const T *input_y1_ptr,
                                 const T *input_x2_ptr, const T *input_y2_ptr,
                                 const int input_offset, const int repeat,
                                 const int remain, const int max_seg_num,
                                 int &local_max_index) {
  if (__is_mpu()) {
    return;
  }
  T local_max_score = FLOAT_MIN_GPV2;
  for (int i = 0; i <= repeat; i++) {
    if (i == repeat && remain == 0) {
      break;
    }
    const int actual_num = (i == repeat) ? remain : max_seg_num;
    const int actual_num_align = CEIL_ALIGN(actual_num, ALIGN_NUM);
    const int seg_offset = input_offset + i * max_seg_num;
    /******nms load start******/
    __bang_write_value(score, actual_num_align, FLOAT_MIN_GPV2);
    __memcpy(score, input_score_ptr + seg_offset, actual_num * sizeof(T),
             GDRAM2NRAM);
    /******nms load end******/
    __bang_argmax(inter_x1, score, actual_num_align);
    if (inter_x1[0] >= local_max_score) {
      if (inter_x1[0] == FLOAT_MIN_GPV2) {
        local_max_score = inter_x1[0];
        local_max_index = ((int *)inter_x1)[1] + seg_offset;
        continue;
      }

      /****** deal multi equal score ******/
      __bang_write_value(inter_y1, actual_num_align, (T)inter_x1[0]);
      __bang_eq(inter_x2, score, inter_y1, actual_num_align);
      const int tmp_index = __bang_findfirst1(inter_x2, actual_num_align);

      local_max_score = inter_x1[0];
      local_max_index = tmp_index + seg_offset;
    }
  }
  max_box[0] = local_max_score;
  max_box[1] = input_x1_ptr[local_max_index];
  max_box[2] = input_y1_ptr[local_max_index];
  max_box[3] = input_x2_ptr[local_max_index];
  max_box[4] = input_y2_ptr[local_max_index];
  ((uint32_t *)(max_box + 5))[0] = local_max_index;
}

template <typename T>
__mlu_func__ void findClusterMaxBox(T *sram, T *max_box, T *inter_x1,
                                    T *input_data_score) {
  // find the max with sram. copy every core's box info to sram.
  /******* 6: score,x1,y1,x2,y2,score_index ******/
  __memcpy(sram + 6 * coreId, max_box, 6 * sizeof(T), NRAM2SRAM);
  __sync_all_ipu();

  /***************************************************
   * copy score from sram to nram and find the max.
   * src: score1, x1, y1, x2, y2, index,
          score2, x1, y1, x2, y2, index,
          score3, x1, y1, x2, y2, index,
          score4, x1, y1, x2, y2, index
   * dst: score1, score2, score3, score4
   * src_stride: 6 * sizeof(T)
   * dst_stride: sizeof(T)
   * size: sizeof(T)
   * seg_num: 3
  ***************************************************/
  __bang_write_value(inter_x1, 64, FLOAT_MIN_GPV2);
  __memcpy(inter_x1, sram, sizeof(T), SRAM2NRAM, sizeof(T), 6 * sizeof(T),
           coreDim - 1);
  __bang_argmax(max_box, inter_x1, 64);
  int max_core = ((int *)max_box)[1];
  __memcpy(max_box, sram + max_core * 6, 6 * sizeof(T), SRAM2NRAM);
}

template <typename T>
__mlu_func__ void calMaxArea(T *max_box, const bool pixel_offset, T *max_area) {
  T max_box_x1 = max_box[1];
  T max_box_y1 = max_box[2];
  T max_box_x2 = max_box[3];
  T max_box_y2 = max_box[4];

  if (pixel_offset) {
    *max_area =
        (max_box_x2 - max_box_x1 + T(1)) * (max_box_y2 - max_box_y1 + T(1));
  } else {
    *max_area = (max_box_x2 - max_box_x1) * (max_box_y2 - max_box_y1);
  }
}

template <typename T>
__mlu_func__ void storeResult(T *max_box, T *nram_save, T *&output_boxes_tmp,
                              T *&output_scores_tmp, const int keep,
                              const int nram_save_limit_count,
                              const int nms_num, const float thresh_score,
                              int &nram_save_count, int &output_box_num) {
  /*****NMS STORE START*****/
  // store to nram
  if (max_box[0] > FLOAT_MIN_GPV2) {
    if (clusterId == 0 && coreId == 0) {
      __memcpy(nram_save + nram_save_count * 5, max_box, 5 * sizeof(T),
               NRAM2NRAM, 5 * sizeof(T), 5 * sizeof(T), 0);
      nram_save_count++;
      output_box_num++;
    }
  }
  // store to sram/gdram
  if (output_box_num != 0) {
    if ((nram_save_count == nram_save_limit_count) ||
        (float(max_box[0]) <= FLOAT_MIN_GPV2) || keep == nms_num - 1) {
      if (nram_save_count != 0) {
        if (clusterId == 0 && coreId == 0) {
          pvLock();
          // x1, y1, x2, y2
          __memcpy(output_boxes_tmp, nram_save + 1, 4 * sizeof(T), NRAM2GDRAM,
                   4 * sizeof(T), 5 * sizeof(T), nram_save_count - 1);
          // score
          __memcpy(output_scores_tmp, nram_save, sizeof(T), NRAM2GDRAM,
                   sizeof(T), 5 * sizeof(T), nram_save_count - 1);
          pvUnlock();
          output_boxes_tmp += nram_save_count * 4;
          output_scores_tmp += nram_save_count;
          nram_save_count = 0;
        }
      }
    }
  }  // if move data nram->sram/gdram
}

template <typename T>
__mlu_func__ void scoreUpdate(
    T *input_score_ptr, const T *input_boxes_ptr, const T *input_x1_ptr,
    const T *input_y1_ptr, const T *input_x2_ptr, const T *input_y2_ptr,
    T *scores, T *boxes, T *inter_x1, T *inter_y1, T *inter_x2, T *inter_y2,
    T *max_box, const float max_box_x1, const float max_box_y1,
    const float max_box_x2, const float max_box_y2, T *nram_save, int repeat,
    int remain, int max_seg_num, const float nms_thresh, const int input_offset,
    const bool pixel_offset, const float max_area, const int input_num_boxes,
    const int nms_id) {
  for (int i = 0; i <= repeat; i++) {
    if (i == repeat && remain == 0) {
      break;
    }
    const int actual_num = (i == repeat) ? remain : max_seg_num;
    const int actual_num_align = CEIL_ALIGN(actual_num, ALIGN_NUM);
    const int seg_offset = input_offset + i * max_seg_num;
    /*****NMS LOAD START*****/
    __memcpy(scores, input_score_ptr + seg_offset, actual_num * sizeof(T),
             GDRAM2NRAM, actual_num * sizeof(T), actual_num * sizeof(T), 0);
    __memcpy(boxes, input_boxes_ptr + seg_offset, actual_num * sizeof(T),
             GDRAM2NRAM, max_seg_num * sizeof(T), input_num_boxes * sizeof(T),
             4);
    T *x1 = boxes;
    T *y1 = x1 + max_seg_num;
    T *x2 = y1 + max_seg_num;
    T *y2 = x2 + max_seg_num;
    __bang_write_value(inter_y1, actual_num_align, (T)max_box_x1);
    __bang_maxequal(inter_x1, x1, inter_y1, actual_num_align);
    // max_x2
    __bang_write_value(inter_y2, actual_num_align, (T)max_box_x2);
    __bang_minequal(inter_x2, x2, inter_y2, actual_num_align);
    __bang_sub(inter_x1, inter_x2, inter_x1, actual_num_align);
    if (pixel_offset) {
      __bang_add_scalar(inter_x1, inter_x1, (T)1.0, actual_num_align);
    }
    // max_y1
    __bang_write_value(inter_x2, actual_num_align, (T)max_box_y1);
    __bang_maxequal(inter_y1, y1, inter_x2, actual_num_align);
    // max_y2
    __bang_write_value(inter_x2, actual_num_align, (T)max_box_y2);
    __bang_minequal(inter_y2, y2, inter_x2, actual_num_align);
    __bang_sub(inter_y1, inter_y2, inter_y1, actual_num_align);
    if (pixel_offset) {
      __bang_add_scalar(inter_y1, inter_y1, (T)1.0, actual_num_align);
    }

    __bang_write_value(inter_y2, actual_num_align, (T)0.0);
    __bang_maxequal(inter_x1, inter_x1, inter_y2, actual_num_align);
    __bang_maxequal(inter_y1, inter_y1, inter_y2, actual_num_align);
    // get the area_i
    __bang_mul(inter_x1, inter_x1, inter_y1, actual_num_align);
    // get the area of input_box (y2-y1) * (x2-x1)
#if __BANG_ARCH__ > 300
    if (pixel_offset) {
      __bang_fusion(FUSION_FSA, inter_y1, x2, x1, (T)1.0, actual_num_align,
                    actual_num_align);
      __bang_fusion(FUSION_FSA, inter_y2, y2, y1, (T)1.0, actual_num_align,
                    actual_num_align);
      __bang_mul(inter_x2, inter_y1, inter_y2, actual_num_align);
    } else {
      __bang_sub(inter_y1, x2, x1, actual_num_align);
      __bang_fusion(FUSION_FSM, inter_x2, y2, y1, inter_y1, actual_num_align,
                    actual_num_align);
    }
    // get the area_u  max_area + area - area_i
    __bang_fusion(FUSION_FAS, inter_x2, inter_x2, max_area, inter_x1,
                  actual_num_align, actual_num_align);
#else
    // get the area of input_box (y2-y1) * (x2-x1)
    __bang_sub(inter_y1, x2, x1, actual_num_align);
    __bang_sub(inter_y2, y2, y1, actual_num_align);
    if (pixel_offset) {
      __bang_add_scalar(inter_y1, inter_y1, (T)1.0, actual_num_align);
      __bang_add_scalar(inter_y2, inter_y2, (T)1.0, actual_num_align);
    }
    __bang_mul(inter_x2, inter_y1, inter_y2, actual_num_align);
    // get the area_u  max_area + area - area_i
    __bang_add_scalar(inter_x2, inter_x2, max_area, actual_num_align);
    __bang_sub(inter_x2, inter_x2, inter_x1, actual_num_align);
#endif
    // 2. select the box
    __bang_mul_scalar(inter_x2, inter_x2, nms_thresh, actual_num_align);
    __bang_le(inter_y1, inter_x1, inter_x2, actual_num_align);
    __bang_gt(inter_y2, inter_x1, inter_x2, actual_num_align);
    __bang_mul(inter_y1, scores, inter_y1, actual_num_align);
    __bang_mul_scalar(inter_y2, inter_y2, FLOAT_MIN_GPV2, actual_num_align);
    __bang_add(scores, inter_y1, inter_y2, actual_num_align);
    /*****NMS COMPUTE END*****/
    __memcpy(input_score_ptr + seg_offset, scores, actual_num * sizeof(T),
             NRAM2GDRAM);
  }
}

__mlu_func__ void getComputeParams(const int input_num, const int limit,
                                   const int memory_block,
                                   const int data_type_size, int *max_seg_num,
                                   int *repeat, int *remain_num, int *core_num,
                                   int *core_offset) {
  int avg_core_num = 0;
  int rem_core_num = 0;
  int len_core_num = 0;

  if (clusterDim == 0) {
    avg_core_num = input_num / taskDim;
    rem_core_num = input_num % taskDim;
    len_core_num = avg_core_num + (taskId < rem_core_num);
    *core_offset =
        avg_core_num * taskId + (taskId < rem_core_num ? taskId : rem_core_num);
  } else {
    const int avg_cluster_num = input_num / clusterDim;
    const int rem_cluster_num = input_num % clusterDim;
    const int len_cluster_num = avg_cluster_num + (clusterId < rem_cluster_num);
    const int cluster_offset_num =
        avg_cluster_num * clusterId +
        (clusterId < rem_cluster_num ? clusterId : rem_cluster_num);

    avg_core_num = len_cluster_num / coreDim;
    rem_core_num = len_cluster_num % coreDim;
    len_core_num = avg_core_num + (coreId < rem_core_num);
    *core_offset = cluster_offset_num + avg_core_num * coreId +
                   (coreId < rem_core_num ? coreId : rem_core_num);
  }

  *max_seg_num = FLOOR_ALIGN(limit / data_type_size, ALIGN_NUM);
  *repeat = len_core_num / *max_seg_num;
  *remain_num = len_core_num % *max_seg_num;
  *core_num = len_core_num;
}

template <typename T>
__mlu_func__ void nonMaximumSuppress(
    T *output_boxes_ptr, T *output_scores_ptr, int *output_num,
    T *input_scores_ptr, const T *input_boxes_ptr, T *workspace,
    const float nms_thresh, const int max_output_num, const int scores_num,
    const bool pixel_offset, const int box_stride) {
  // nram 13 * N, N = max_seg_num
  // | output_boxes | scores | boxes | inter_x1|
  // |   4 * N      |     N  | 4*N   |   N     |

  // inter_y1| inter_x2 | inter_y2 |
  // |   N   |   N      |   N      |
  int32_t *loop_end_flag = (int32_t *)(sram_buffer + 28);
  loop_end_flag[0] = 0;
  // scores, boxes, x1, y1, x2, y2, inter_x1, inter_y1, inter_x2, inter_y2
  const int memory_block = 13;
  const int nram_save_limit_count = 256;

  // input data gdram ptr
  const T *input_x1_ptr = input_boxes_ptr;
  const T *input_y1_ptr = input_x1_ptr + box_stride;
  const T *input_x2_ptr = input_y1_ptr + box_stride;
  const T *input_y2_ptr = input_x2_ptr + box_stride;

  int limit = 0;
  int max_seg_num = 0;
  int repeat = 0;
  int remain_num = 0;
  int core_offset = 0;
  int core_num = 0;
  int nram_save_count = 0;

  limit =
      (MAX_NRAM_SIZE - NFU_ALIGN_SIZE - nram_save_limit_count * sizeof(T) * 5) /
      memory_block;
  getComputeParams(scores_num, limit, memory_block, sizeof(T), &max_seg_num,
                   &repeat, &remain_num, &core_num, &core_offset);
  // init nram ptr
  T *scores = (T *)nram_buffer;
  T *x1 = scores + max_seg_num;
  T *y1 = x1 + max_seg_num;
  T *x2 = y1 + max_seg_num;
  T *y2 = x2 + max_seg_num;
  T *boxes = y2 + max_seg_num;
  T *inter_x1 = boxes + 4 * max_seg_num;
  T *inter_y1 = inter_x1 + max_seg_num;
  T *inter_x2 = inter_y1 + max_seg_num;
  T *inter_y2 = inter_x2 + max_seg_num;
  T *max_box = inter_y2 + max_seg_num;
  T *nram_save = max_box + NFU_ALIGN_SIZE;

  const int nms_num = scores_num > max_output_num ? max_output_num : scores_num;

  for (int nms_id = 0; nms_id < nms_num; ++nms_id) {
    if (taskDim != 1) {
      __sync_all_ipu();
    }
    /*****Find MaxBox Box*****/
    int max_index = 0;            // the max score index.
    int global_max_index = 0;     // for u1.
    T max_area = 0;               // the max score area.
    max_box[0] = FLOAT_MIN_GPV2;  // init 0.
    findCoreMaxBox(input_scores_ptr, scores, inter_x1, inter_y1, inter_x2,
                   max_box, input_x1_ptr, input_y1_ptr, input_x2_ptr,
                   input_y2_ptr, core_offset, repeat, remain_num, max_seg_num,
                   max_index);
    if (taskDim == 1) {
      calMaxArea(max_box, pixel_offset, &max_area);
      input_scores_ptr[max_index] = FLOAT_MIN_GPV2;
      global_max_index = max_index;
    } else if (taskDim == 4) {
      __sync_all_ipu();
      findClusterMaxBox((T *)sram_buffer, max_box, inter_x1, input_scores_ptr);
      calMaxArea(max_box, pixel_offset, &max_area);
      global_max_index = ((int *)(max_box + 5))[0];
    }
    /******by now, we get: max_score|max_index|max_box|max_area******/
    storeResult(max_box, nram_save, output_boxes_ptr, output_scores_ptr, nms_id,
                nram_save_limit_count, nms_num, nms_thresh, nram_save_count,
                *output_num);
    if (taskDim == 1) {
      if (float(max_box[0]) <= FLOAT_MIN_GPV2 || (nms_id == nms_num - 1)) {
        break;
      }
    } else {
      if (float(max_box[0]) <= FLOAT_MIN_GPV2 || (nms_id == nms_num - 1)) {
        if (coreId == 0) {
          loop_end_flag[0] = 1;
        }
      }
      __sync_all_ipu();
      if (loop_end_flag[0] == 1) {
        break;
      }
    }
    /*** NMS STORE END***/
    scoreUpdate(input_scores_ptr, input_boxes_ptr, input_x1_ptr, input_y1_ptr,
                input_x2_ptr, input_y2_ptr, scores, boxes, inter_x1, inter_y1,
                inter_x2, inter_y2, max_box, max_box[1], max_box[2], max_box[3],
                max_box[4], nram_save, repeat, remain_num, max_seg_num,
                nms_thresh, core_offset, pixel_offset, max_area, box_stride,
                nms_id);
  }
}  // for nms_id < nms_num
