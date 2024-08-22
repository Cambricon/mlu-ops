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

#include <type_traits>

// highest dimension executing kernel=========================================
template <typename T>
__mlu_global__ void
highestDimKernel(const T *input,
                 T *output,
                 int32_t axis_size,
                 int32_t lower_size) {
    const int32_t core_size = sizeof(T) == 4 ?
      CoreCapacity / sizeof(T) / 2 : CoreCapacity / sizeof(T) / 4;
    const int32_t total_width = lower_size;
    const int32_t total_height = axis_size;
    int32_t cores_working = total_width / (N_ALIGN / sizeof(T));
    if (cores_working > taskDim) {
      cores_working = taskDim;
    }
    const int32_t core_width = total_width % cores_working == 0 ?
      total_width / cores_working : total_width / (cores_working - 1);

    const int32_t last_core_width = total_width % cores_working == 0 ?
                core_width : total_width - core_width * (cores_working - 1);
    const int32_t core_height = core_size / core_width < total_height ?
                            core_size / core_width : total_height;

    T *nram_src = (T *)nram_buffer;
    T *nram_offset = nram_src + core_size;
    T *add_buffer;
    if (std::is_same<T, half>::value) {
      add_buffer = nram_offset + core_size;
    }

    if (taskId < cores_working) {
      for (int i = 0; i < total_height; i += core_height) {
        int deal_height = i + core_height <= total_height ?
                            core_height : total_height % core_height;
        int deal_width = taskId == cores_working - 1 ?
                            last_core_width : core_width;

        __memcpy(nram_src, input + i * total_width + taskId * core_width,
                 deal_width * sizeof(T), GDRAM2NRAM, deal_width * sizeof(T),
                 total_width * sizeof(T), deal_height - 1);
        if (std::is_same<T, float>::value) {
          __mluop_exp(nram_src, nram_src, nullptr, 0,
                      deal_width * deal_height);
        } else {
          __mluop_exp(nram_src, nram_src, add_buffer, 0,
                      deal_width * deal_height);
        }

        if (i != 0) {
          __bang_add(nram_src, nram_src, nram_offset, deal_width);
        }
        for (int j = 1; j < deal_height; j++) {
          __bang_add(nram_src + j * deal_width,
                    nram_src + j * deal_width,
                    nram_src + (j - 1) * deal_width,
                    deal_width);
        }
        __bang_move(nram_offset,
                    nram_src + (deal_height - 1) * deal_width,
                    deal_width * sizeof(T));
        if (std::is_same<T, float>::value) {
          __mluop_log(nram_src, nram_src, nullptr, 0,
                      deal_width * deal_height);
        } else {
          __mluop_log(nram_src, nram_src, add_buffer, 0,
                      deal_width * deal_height);
        }
        __memcpy(output + i * total_width + taskId * core_width, nram_src,
                 deal_width * sizeof(T), NRAM2GDRAM, total_width * sizeof(T),
                 deal_width * sizeof(T), deal_height - 1);
      }
    }
}

