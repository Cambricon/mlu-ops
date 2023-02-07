/*******************************************************************************
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
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *******************************************************************************/
#ifndef KERNELS_INDICE_CONVOLUTION_BACKWARD_DATA_INDICE_CONVOLUTION_BACKWARD_DATA_H_  // NOLINT
#define KERNELS_INDICE_CONVOLUTION_BACKWARD_DATA_INDICE_CONVOLUTION_BACKWARD_DATA_H_  // NOLINT

#include <vector>

#include "mlu_op.h"
#include "core/tensor.h"

inline int getMaxNumInArray(const int64_t arr[], const int num) {
  int max_num = (int)(arr[0]);
  for (int i = 1; i < num; ++i) {
    max_num = max_num > (int)(arr[i]) ? max_num : (int)(arr[i]);
  }
  return max_num;
}

#endif  // KERNELS_INDICE_CONVOLUTION_BACKWARD_DATA_INDICE_CONVOLUTION_BACKWARD_DATA_H_ // NOLINT
