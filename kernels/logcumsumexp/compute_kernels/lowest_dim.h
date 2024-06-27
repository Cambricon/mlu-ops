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

// LCSE execution for small part
template <typename T>
__mlu_func__ void smallPartLCSE(T *output,
                                const T *source,
                                int32_t data_size,
                                int32_t axis_size,
                                int32_t parts) {
    T *nram_src0 = (T *)nram_buffer;
    T *nram_src1 = nram_src0 + ((MAX_NRAM_SIZE/sizeof(T)) >> 1);
    int32_t part_width = axis_size;
    int32_t part_height = N_ALIGN / sizeof(T);
    int32_t part_size = part_width * part_height;

    __memcpy(nram_src0, source, data_size * sizeof(T), GDRAM2NRAM);
    __mluop_exp(nram_src0, nram_src0, nullptr, 0, data_size);
    for (int i = 0; i < parts; i++) {
        __bang_transpose(nram_src1 + part_size * i,
                         nram_src0 + part_size * i,
                         part_height, part_width);
    }
    for (int i = 0; i < parts; i++) {
        for (int j = 1; j < part_width; j++) {
            __bang_add(nram_src1 + part_size * i + part_height * j,
                       nram_src1 + part_size * i + part_height * j,
                       nram_src1 + part_size * i + part_height * (j - 1),
                       part_height);
        }
    }
    for (int i = 0; i < parts; i++) {
        __bang_transpose(nram_src0 + part_size * i,
                         nram_src1 + part_size * i,
                         part_width, part_height);
    }
    __mluop_log(nram_src0, nram_src0, nullptr, 0, data_size);
    __memcpy(output, nram_src0, data_size * sizeof(T), NRAM2GDRAM);
}

// highest dimension executing kernel====================================
template <typename T>
__mlu_global__ void
lowestDimKernel(const T *input,
                T *output,
                int32_t axis_size,
                int32_t higher_size) {
    // if nram_size > part_size,
    // there will be several parts on one nram every round;
    // if nram_size < part_size, call dimOneKernel for batches.
    int32_t data_size = axis_size * higher_size;
    int32_t nram_size = CoreCapacity / sizeof(T);
    int32_t nram_height = N_ALIGN / sizeof(T);
    int32_t nram_width = nram_size / nram_height;
    int32_t part_height = nram_height;
    int32_t part_width = axis_size;
    int32_t parts_per_core = nram_width / part_width;

    int32_t deal_size = parts_per_core * part_width * part_height;
    int32_t round_size = deal_size * taskDim;
    int32_t round = 0;
    int32_t deal_rounds = (data_size + round_size - 1) / round_size;
    while (round < deal_rounds - 1) {
        smallPartLCSE(output + round * round_size + taskId * deal_size,
                      input + round * round_size + taskId * deal_size,
                      deal_size, axis_size, parts_per_core);
        round++;
    }
    // last round
    int32_t last_round_size = data_size - round_size * (deal_rounds - 1);
    int32_t last_round_height = last_round_size / part_width;
    int32_t last_round_parts
      = (last_round_height + part_height - 1) / part_height;
    int32_t last_round_cores
      = (last_round_parts + parts_per_core - 1)/ parts_per_core;
    int32_t last_core_size
      = data_size - (deal_rounds - 1) * round_size
      - (last_round_cores - 1) * deal_size;

    if (taskId < last_round_cores - 1) {
      smallPartLCSE(output + round * round_size + taskId * deal_size,
                    input + round * round_size + taskId * deal_size,
                    deal_size, axis_size, parts_per_core);
    } else if (taskId == last_round_cores - 1) {
        T *nram_src = (T *)nram_buffer;
        __bang_write_zero(nram_src, deal_size);
        smallPartLCSE(output + round * round_size + taskId * deal_size,
                      input + round * round_size + taskId * deal_size,
                      last_core_size, axis_size, parts_per_core);
    }
}