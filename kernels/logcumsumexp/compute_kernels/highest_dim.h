/*******************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modif y, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS for A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE for ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *******************************************************************************/
#pragma once

extern __nram__ char nram_buffer[MAX_NRAM_SIZE];
extern __mlu_shared__ char sram_buffer[MAX_SRAM_SIZE];

// LCSE execution for rows
template <typename T>
__mlu_func__ void
rowsKernel(T *output, const T *source, int32_t deal_length,
           int32_t rounds, int32_t address_offset) {
  T *nram_src0 = (T *)nram_buffer;
  T *nram_src1 = nram_src0 + deal_length;
  T *nram_out = nram_src0 + ((MAX_NRAM_SIZE/sizeof(T)) >> 1);
  int32_t part_length = deal_length / 4;
  int32_t length_offset = deal_length % 4;
  int32_t copy_offset = taskId * part_length + __mluop_min(taskId, length_offset);
  if (taskId < length_offset)part_length += 1;

  T *thisNram = nram_src1;
  T *lastNram = nram_src0;
  __memcpy(lastNram, source + copy_offset,
           part_length * sizeof(T), GDRAM2NRAM);
  __memcpy(output + copy_offset, lastNram,
           part_length * sizeof(T), NRAM2GDRAM);
  __mluop_exp(lastNram, lastNram, nullptr, 0, part_length);

  for (int i = 1; i < rounds; i++) {
    thisNram = (i % 2 == 0) ? nram_src0 : nram_src1;
    lastNram = (i % 2 == 0) ? nram_src1 : nram_src0;
    __memcpy(thisNram,
             source + i * address_offset + copy_offset,
             part_length * sizeof(T),
             GDRAM2NRAM);
    __mluop_exp(thisNram, thisNram, nullptr, 0, part_length);
    __bang_add(thisNram, thisNram, lastNram, part_length);
    __mluop_log(nram_out, thisNram, nullptr, 0, part_length);
    __memcpy(output + i * address_offset + copy_offset,
             nram_out,
             part_length * sizeof(T),
             NRAM2GDRAM);
    __sync_cluster();
  }
}

// LCSE execution for one part
template <typename T>
__mlu_func__ void
onePartKernel(const T *source, T *output, int32_t width,
              int32_t height, int32_t part_size) {
    T *offset_cores = (T *)sram_buffer;
    T *nram_src = (T *)nram_buffer;
    T *offset_area = nram_src + ((MAX_NRAM_SIZE/sizeof(T)) >> 1);
    int32_t data_size = width * height;
    __memcpy(nram_src, source + part_size * taskId,
             data_size * sizeof(T), GDRAM2NRAM);
    __mluop_exp(nram_src, nram_src, nullptr, 0, data_size);
    for (int i = 1; i < height; i++) {
        __bang_add(nram_src + width * i,
                   nram_src + width * i,
                   nram_src + width * (i - 1),
                   width);
    }
    // offset between cores
    __memcpy(offset_cores + width * taskId,
             nram_src + (height - 1) * width,
             width * sizeof(T),
             NRAM2SRAM);
    __sync_cluster();
    __memcpy(offset_area,
             offset_cores,
             width * sizeof(T) * 5,
             SRAM2NRAM);
    __bang_write_zero(offset_area + width * 5, width);
    // [0],[1],[2],[3] are 4 offsets for 4 cores;
    // the [4] records and saves the cumulative offset for next round;
    // the [5] is to support the offset calculate;
    for (int i = 5; i > 0; i--) {
        __bang_move(offset_area + width * i,
                    offset_area + width * (i - 1),
                    width * sizeof(T));
    }
    __bang_move(offset_area,
                offset_area + width * 5,
                width * sizeof(T));
    for (int i = 1; i < 5; i++) {
        __bang_add(offset_area + width * i,
                   offset_area + width * i,
                   offset_area + width * (i - 1),
                   width);
    }
    __bang_cycle_add(nram_src,
                     nram_src,
                     offset_area + width * taskId,
                     data_size,
                     width);
    if (taskId == 0) {
        __memcpy(offset_cores + width * 4,
                 offset_area + width * 4,
                 width * sizeof(T),
                 NRAM2SRAM);
    }

    __mluop_log(nram_src, nram_src, nullptr, 0, data_size);
    __memcpy(output + part_size * taskId, nram_src,
             data_size * sizeof(T), NRAM2GDRAM);
}

// lowest dimension executing kernel=========================================
template <typename T>
__mlu_func__ void
highestDimKernel(const T *input,
                 T *output,
                 int32_t axis_size,
                 int32_t lower_size) {
    int32_t data_size = axis_size * lower_size;
    int32_t nram_size = CoreCapacity / sizeof(T);
    int32_t part_width = lower_size;
    int32_t part_height = nram_size / part_width;
    int32_t part_size = part_height * part_width;
    int32_t cluster_size = part_size << 2;

    // data has a too large width to be deal as parts
    if (part_height < 6) {
        int32_t cluster_capacity = nram_size >> 1;
        int32_t batches_num
          = (lower_size + cluster_capacity - 1) / cluster_capacity;
        for (int i = 0; i < batches_num; i++) {
          int32_t deal_length;
          if (i < batches_num - 1)
            deal_length = cluster_capacity;
          else
            deal_length = lower_size - (batches_num - 1) * cluster_capacity;
          rowsKernel(output + i * cluster_capacity,
                     input + i * cluster_capacity,
                     deal_length, axis_size, lower_size);
        }
    } else {  // deal as parts
        T *offset_cores = (T *)sram_buffer;
        for (int i = 0; i < part_width; i++)
          offset_cores[part_width * 4 + i] = 0;
        int32_t parts_num = (axis_size + part_height - 1) / part_height;
        int32_t rounds_num = (parts_num + 3) / 4;
        int32_t round = 0;
        __sync_cluster();
        while (round < rounds_num - 1) {
            onePartKernel(input + cluster_size * round,
                          output + cluster_size * round,
                          part_width, part_height, part_size);
            round++;
        }
        int32_t last_round_parts = parts_num - (rounds_num - 1) * 4;
        int32_t lastRoundCores = last_round_parts;
        int32_t last_part_size
          = data_size - part_size * ((rounds_num - 1)
          * 4 + last_round_parts - 1);
        int32_t last_part_height = last_part_size / part_width;

        if (taskId < lastRoundCores - 1) {
          onePartKernel(input + cluster_size * round,
                        output + cluster_size * round,
                        part_width, part_height, part_size);
        } else if (taskId == lastRoundCores - 1) {
          onePartKernel(input + cluster_size * round,
                        output + cluster_size * round,
                        part_width, last_part_height, part_size);
        } else {
          __sync_cluster();
        }
    }
}
