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

template <typename T>
__mlu_func__ void
batchKernel(const T *source,
            T *result,
            int32_t batches_num,
            int32_t width,
            int32_t height) {
    T *nram_src = (T *)nram_buffer;
    int32_t batch_size = width * height;
    int32_t data_size = batch_size * batches_num;
    __memcpy(nram_src, source, data_size * sizeof(T), GDRAM2NRAM);
    __mluop_exp(nram_src, nram_src, nullptr, 0, data_size);

    for (int i = 0; i < batches_num; i++) {
        for (int j = 1; j < height; j++) {
            __bang_add(nram_src + i * batch_size + j * width,
                       nram_src + i * batch_size + j * width,
                       nram_src + i * batch_size + (j - 1) * width,
                       width);
        }
    }
    __mluop_log(nram_src, nram_src, nullptr, 0, data_size);
    __memcpy(result, nram_src, data_size * sizeof(T), NRAM2GDRAM);
}

// mid dimension executing kernel============================================
template <typename T>
__mlu_func__ void
midDimKernel_unino8(const T *input,
                    T *result,
                    int32_t axis_size,
                    int32_t lower_size,
                    int32_t higher_size) {
  int32_t nram_size = CoreCapacity / sizeof(T);
  int32_t batches_num = higher_size;
  int32_t batch_size = axis_size * lower_size;
  int32_t batches_per_core = nram_size / batch_size;
  int32_t core_size = batch_size * batches_per_core;
  int32_t rounds_num
    = (batches_num + (batches_per_core * taskDim) - 1)
    / (batches_per_core * taskDim);
  if (batches_per_core > 0) {
    for (int i = 0; i < rounds_num - 1; i++) {
        batchKernel(input + i * core_size * taskDim + core_size * taskId,
                    result + i * core_size * taskDim + core_size * taskId,
                    batches_per_core, lower_size, axis_size);
    }
    int32_t last_round_batches
      = batches_num - batches_per_core * taskDim * (rounds_num - 1);
    int32_t lastRoundCores
      = (last_round_batches + batches_per_core - 1) / batches_per_core;
    int32_t last_core_batches
      = last_round_batches - batches_per_core * (lastRoundCores - 1);
    if (taskId < lastRoundCores - 1) {
      batchKernel(
        input + (rounds_num - 1) * core_size * taskDim + core_size * taskId,
        result + (rounds_num - 1) * core_size * taskDim + core_size * taskId,
        batches_per_core, lower_size, axis_size);
    }
    if (taskId == lastRoundCores - 1) {
      batchKernel(
        input + (rounds_num - 1) * core_size * taskDim + core_size * taskId,
        result + (rounds_num - 1) * core_size * taskDim + core_size * taskId,
        last_core_batches, lower_size, axis_size);
    }
  } else {
    if (clusterId == 0) {
      for (int i = 0; i < batches_num; i++) {
        lowestDimKernel_unino1(input + i * batch_size,
                                result + i * batch_size,
                                axis_size, lower_size);
        __sync_cluster();
      }
    }
  }
}