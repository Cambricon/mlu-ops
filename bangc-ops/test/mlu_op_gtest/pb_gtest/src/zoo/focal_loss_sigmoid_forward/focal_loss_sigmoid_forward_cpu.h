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
#ifndef a_a
#define a_a

#include <climits>
#include <cfloat>
#include <vector>
#include <cmath>

#include "mlu_op.h"
#include "executor.h"
#include "core/type.h"
#include "core/logging.h"

using mluoptest::ComputationPreference;
using mluoptest::LossReduction;

void focalLossSigmoidForwardCpuFast(const float *input, const size_t input_num,
                                    const float *target,
                                    const size_t target_num,
                                    const float *weight,
                                    const size_t weight_num, const float alpha,
                                    const float gamma, float *output) {
  size_t N = target_num;
  size_t C = input_num / target_num;

  if (weight_num == 0) {
    for (int i = 0; i < input_num; ++i) {
      int32_t row_num = i / C;
      int32_t col_num = i % C;
      int32_t t = target[row_num];
      float p = 1. / (1. + exp(-input[i]));
      float temp_p = pow(1. - p, gamma) * log(fmax(p, FLT_MIN));
      float temp_n = pow(p, gamma) * log(fmax(1. - p, FLT_MIN));
      if (t == col_num) {
        output[i] = -alpha * temp_p;
      } else {
        output[i] = -(1 - alpha) * temp_n;
      }
    }

  } else {
    for (int i = 0; i < input_num; ++i) {
      int32_t row_num = i / C;
      int32_t col_num = i % C;
      int32_t t = target[row_num];
      float p = 1. / (1. + exp(-input[i]));
      float temp_p = pow(1. - p, gamma) * log(fmax(p, FLT_MIN));
      float temp_n = pow(p, gamma) * log(fmax(1. - p, FLT_MIN));
      if (t == col_num) {
        output[i] = -alpha * temp_p;
      } else {
        output[i] = -(1 - alpha) * temp_n;
      }
      output[i] *= weight[t];
    }
  }
}

void focalLossSigmoidForwardCpuHighPrecisionDoub(
    const float *input, const size_t input_num, const float *target,
    const size_t target_num, const float *weight, const size_t weight_num,
    const float alpha, const float gamma, float *output) {
  double focal_loss_temp = 0.0;
  size_t N = target_num;
  size_t C = input_num / target_num;
  double alpha_double = double(alpha);
  double gamma_double = double(gamma);

  if (weight_num == 0) {
    for (int i = 0; i < input_num; ++i) {
      int32_t row_num = i / C;
      int32_t col_num = i % C;
      int32_t t = target[row_num];
      double x = double(input[i]);
      double p = 1. / (1. + exp(-x));
      double temp_p = pow(1. - p, gamma_double) * log(fmax(p, DBL_MIN));
      double temp_n = pow(p, gamma_double) * log(fmax(1. - p, DBL_MIN));
      if (t == col_num) {
        focal_loss_temp = -alpha_double * temp_p;
      } else {
        focal_loss_temp = -(1 - alpha_double) * temp_n;
      }
      output[i] = float(focal_loss_temp);
    }

  } else {
    for (int i = 0; i < input_num; ++i) {
      int32_t row_num = i / C;
      int32_t col_num = i % C;
      int32_t t = target[row_num];
      double x = double(input[i]);
      double p = 1. / (1. + exp(-x));
      double temp_p = pow(1. - p, gamma_double) * log(fmax(p, DBL_MIN));
      double temp_n = pow(p, gamma_double) * log(fmax(1. - p, DBL_MIN));
      if (t == col_num) {
        focal_loss_temp = -alpha_double * temp_p;
      } else {
        focal_loss_temp = -(1 - alpha_double) * temp_n;
      }
      output[i] = float(focal_loss_temp);
      output[i] *= weight[t];

      double pow_temp = pow(1. - p, gamma);
      double log_temp = log(fmax(p, DBL_MIN));
    }
  }
}

void focalLossSigmoidForwardCpu(const ComputationPreference prefer,
                                const LossReduction reduction,
                                const float *input, const size_t input_num,
                                const float *target, const size_t target_num,
                                const float *weight, const size_t weight_num,
                                const float alpha, const float gamma,
                                float *output) {
  float *focal_loss = (float *)malloc(input_num * sizeof(float));

  switch (prefer) {
    case ComputationPreference::COMPUTATION_FAST: {
      VLOG(5) << "[focalLossSigmoidForwardCpu] call Fast.";
      focalLossSigmoidForwardCpuFast(input, input_num, target, target_num,
                                     weight, weight_num, alpha, gamma,
                                     focal_loss);
    }; break;
    case ComputationPreference::COMPUTATION_HIGH_PRECISION: {
      VLOG(5) << "[focalLossSigmoidForwardCpu] call HighPrecision.";
      focalLossSigmoidForwardCpuHighPrecisionDoub(
          input, input_num, target, target_num, weight, weight_num, alpha,
          gamma, focal_loss);
    }; break;
    default:
      LOG(ERROR) << "[focalLossSigmoidForwardCpu] not Implemented.";
  }

  switch (reduction) {
    case LossReduction::LOSS_REDUCTION_NONE: {
      for (int i = 0; i < input_num; ++i) {
        output[i] = focal_loss[i];
      }
    } break;
    case LossReduction::LOSS_REDUCTION_SUM:
    case LossReduction::LOSS_REDUCTION_MEAN:
    default:
      LOG(ERROR) << "[focalLossSigmoidForwardCpu] Not support reduction:["
                 << reduction << "].";
  }
  free(focal_loss);
}

#endif  // a_a
