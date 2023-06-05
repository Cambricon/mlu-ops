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
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include "gtest/gtest.h"
#include "tools.h"
#include "core/logging.h"
#include "stats_test.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define KL_EPSILON (1e-10)

template <typename T>
double diff_kl(T *data1, T *data2, size_t elem_num) {
  double data1_sum = 0.0;
  double data2_sum = 0.0;
#pragma omp parallel for reduction(+ : data1_sum, data2_sum) schedule(guided)
  for (size_t i = 0; i < elem_num; i++) {
    data1_sum += std::max(std::abs((double)data1[i]), (double)KL_EPSILON);
    data2_sum += std::max(std::abs((double)data2[i]), (double)KL_EPSILON);
  }  // end omp parallel block
  double entropy = 0.0;
#pragma omp parallel for reduction(+ : entropy) schedule(guided)
  for (size_t i = 0; i < elem_num; i++) {
    double data1_prob =
        std::max(std::abs((double)data1[i]), (double)KL_EPSILON) / data1_sum;
    double data2_prob =
        std::max(std::abs((double)data2[i]), (double)KL_EPSILON) / data2_sum;
    entropy += 0.5 * data1_prob * log(data1_prob / data2_prob) +
               0.5 * data2_prob * log(data2_prob / data1_prob);
  }  // end omp parallel block
  return entropy;
}

template double diff_kl<float>(float *data1, float *data2, size_t elem_num);
template double diff_kl<double>(double *data1, double *data2, size_t elem_num);
