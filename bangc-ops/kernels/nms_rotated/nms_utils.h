/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
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

#include "kernels/utils/common.h"
#include "kernels/kernel.h"
#include "kernels/debug.h"

#define NMS_SIZE 64
#define NMS_UP(x, y) (x / y + (int)(x % y > 0)) * y
#define NMS_DOWN(x, y) (x / y) * y
#define SIZE_NRAM_BUF (MAX_NRAM_SIZE)
#define SIZE_SRAM_BUF (MAX_SRAM_SIZE)
// score, x1, y1, x2, y2, max_index (reserve 2 num for half-type input)
#define REDUCE_NUM (7)

template <typename IN_DT>
__mlu_func__ void findCoreMaxBox(IN_DT *input_score_ptr,
                                 IN_DT *score,
                                 IN_DT *temp,
                                 IN_DT *max_box,
                                 const IN_DT *input_x1_ptr,
                                 const IN_DT *input_y1_ptr,
                                 const IN_DT *input_x2_ptr,
                                 const IN_DT *input_y2_ptr,
                                 const mluMemcpyDirection_t load_dir,
                                 const int input_offset,
                                 const int repeat,
                                 const int remain,
                                 const int remain_pad,
                                 const int max_seg_pad,
                                 int &max_index) {
  if (coreId != 0x80) {
    for (int i = 0; i <= repeat; i++) {
      if (i == repeat && remain == 0) {
        break;
      }
      int seg_len           = 0;  // the length every nms compute
      int cpy_len           = 0;  // the length every nms memcpy
      i == repeat ? seg_len = remain_pad : seg_len = max_seg_pad;
      // check seg_len exceeds the limit of fp16 or not.
      // 65536 is the largest num that fp16 could express.
      if (std::is_same<IN_DT, half>::value && seg_len >= 65536) {
        MLULOG("seg length exceed the max num for fp16 datatype!");
        return;
      }
      i == repeat ? cpy_len = remain : cpy_len = max_seg_pad;
      /******NMS LOAD START******/
      __bang_write_zero(score, seg_len);
      __memcpy(score, input_score_ptr + input_offset + i * max_seg_pad,
               cpy_len * sizeof(IN_DT), load_dir);

      /******NMS LOAD END******/

      __bang_max(temp, score, seg_len);
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
    max_box[1]                     = input_x1_ptr[max_index];
    max_box[2]                     = input_y1_ptr[max_index];
    max_box[3]                     = input_x2_ptr[max_index];
    max_box[4]                     = input_y2_ptr[max_index];
    ((uint32_t *)(max_box + 5))[0] = max_index;
  }
}

template <typename IN_DT>
__mlu_func__ void findClusterMaxBox(IN_DT *sram,
                                    IN_DT *max_box,
                                    IN_DT *temp,
                                    IN_DT *input_data_score,
                                    const int core_limit) {
  // find the max with sram
  // copy every core's box info to sram, form: score---x1---y1---x2---y2---
  __memcpy(sram + REDUCE_NUM * coreId, max_box, REDUCE_NUM * sizeof(IN_DT),
           NRAM2SRAM);  // int32_t datatype
  __sync_cluster();

  // copy score from sram to nram and find the max
  __bang_write_zero(temp, 64);
  __memcpy(temp, sram, sizeof(IN_DT), SRAM2NRAM, sizeof(IN_DT),
           REDUCE_NUM * sizeof(IN_DT), coreDim - 1);
  __bang_max(max_box, temp, 64);
  int max_core =
      (std::is_same<IN_DT, half>::value) ? ((uint16_t *)max_box)[1] :
                                            ((uint32_t *)max_box)[1];
  // copy the max box to max_box
  __memcpy(max_box, sram + max_core * REDUCE_NUM,
           REDUCE_NUM * sizeof(IN_DT), SRAM2NRAM);
}

template<typename T>
__mlu_func__ void BoxesTranpose(const T *boxes,
                                T *boxes_trans,
                                const int32_t box_num,
                                const int32_t box_dim) {
  int32_t task_per_core = box_num / taskDim;
  int32_t task_rem = box_num % taskDim;
  int32_t offset = task_per_core * taskId + (taskId < task_rem ? taskId : task_rem);
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
             rem * sizeof(T), NRAM2GDRAM, box_num * sizeof(T),
             rem * sizeof(T), box_dim - 1);
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
              box_dim_aligned * sizeof(T), box_dim * sizeof(T),
              rem - 1);
    __bang_transpose(nram_box, nram_box_trans, rem_aligned,
                  box_dim_aligned);
    __memcpy(boxes_trans + offset + repeat * deal_once_aligned, nram_box,
             rem * sizeof(T), NRAM2GDRAM, box_num * sizeof(T),
             rem_aligned * sizeof(T), box_dim - 1);
  }
#endif
}

#endif  // KERNELS_NMS_ROTATED_NMS_UTILS_H_
