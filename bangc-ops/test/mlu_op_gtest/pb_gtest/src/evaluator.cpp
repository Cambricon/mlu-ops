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
#ifdef __AVX__
#include <immintrin.h>
#include <limits>
#endif
#include <algorithm>
#include <utility>
#include <vector>
#include <string>
#include "evaluator.h"

namespace mluoptest {
const double EPSILON = 1e-9;
const double EPSILON_FLOAT = 1e-6;
const double EPSILON_HALF = 1e-3;

// found inf or nan, return true.
#ifdef __AVX__
bool hasNanOrInf(float *data, size_t count) {
  const __m256 exp_bit = _mm256_set1_ps(std::numeric_limits<float>::infinity());

  size_t stride = 256 / (sizeof(float) * 8);  // 1 __m256 saved 8 *
                                              // (sizeof(float) * 8 bit)
  size_t repeat = count / stride * stride;

  __m256 m_data;
  for (size_t i = 0; i < repeat; i += stride) {
    m_data = _mm256_load_ps(data + i);
    m_data = _mm256_and_ps(exp_bit, m_data);
    m_data = _mm256_cmp_ps(m_data, exp_bit, _CMP_EQ_OQ);
    if (_mm256_movemask_ps(m_data) != 0) {
      return true;
    }
  }

  for (size_t i = repeat; i < count - repeat; ++i) {
    if (std::isnan(data[i]) || std::isinf(data[i])) {
      return true;
    }
  }
  return false;
}
#else
bool hasNanOrInf(float *data, size_t count) {
  for (int i = 0; i < count; ++i) {
    if (std::isinf(data[i]) || std::isnan(data[i])) {
      return true;
    }
  }
  return false;
}
#endif

void resetNanOrInfAsZero(float *a, float *b, size_t count) {
  bool has_nan = false;
  bool has_inf = false;
  for (size_t i = 0; i < count; ++i) {
    if (unlikely(std::isnan(a[i]) && std::isnan(b[i]))) {
      a[i] = 0.0f;
      b[i] = 0.0f;
      has_nan = true;
    } else if (unlikely(std::isinf(a[i]) && std::isinf(b[i]) && a[i] == b[i])) {
      // if a is inf, b is -inf, don't deal here.
      // when check hasNanOrInf will set diff as DBL_MAX (instead infinity).
      a[i] = 0.0f;
      b[i] = 0.0f;
      has_inf = true;
    }
  }
  if (has_nan) {
    VLOG(4) << "Found result of baseline and mlu are both NaN, set them as "
               "0, and go on.";
  }
  if (has_inf) {
    VLOG(4) << "Found result of baseline and mlu are both Inf, set them as "
               "0, and go on.";
  }
}

void skipNanOrInfAsZero(float *a, float *b, size_t count) {
  bool has_nan = false;
  bool has_inf = false;
  for (size_t i = 0; i < count; ++i) {
    int tmp = *(int *)&a[i];
    if (unlikely(std::isnan(a[i]))) {
      a[i] = 0.0f;
      b[i] = 0.0f;
      has_nan = true;
    } else if (unlikely(std::isinf(a[i]))) {
      a[i] = 0.0f;
      b[i] = 0.0f;
      has_inf = true;
    }
  }
  if (has_nan) {
    VLOG(4) << "Found result of baseline is NaN,"
            << " set baseline and mlu as 0, and go on.";
  }
  if (has_inf) {
    VLOG(4) << "Found result of baseline is Inf,"
            << " set baseline and mlu as 0, and go on.";
  }
}

double Evaluator::computeDiff1(float *cpu_result, float *mlu_result,
                               size_t count) {
  if (hasNanOrInf(cpu_result, count) || hasNanOrInf(mlu_result, count)) {
    LOG(ERROR) << "Found NaN or Inf when compute diff, return DBL_MAX "
                  "instead.";
    return DBL_MAX;
  }

  double numerator_sum = 0.0;
  double denominator_sum = 0.0;
  for (int i = 0; i < count; i++) {
    numerator_sum += fabs(cpu_result[i] - mlu_result[i]);
    denominator_sum += fabs(cpu_result[i]);
  }

  return numerator_sum / (denominator_sum + EPSILON);
}

double Evaluator::computeDiff2(float *cpu_result, float *mlu_result,
                               size_t count) {
  if (hasNanOrInf(cpu_result, count) || hasNanOrInf(mlu_result, count)) {
    LOG(ERROR) << "Found NaN or Inf when compute diff, return DBL_MAX "
                  "instead.";
    return DBL_MAX;
  }

  double numerator_sum = 0.0;
  double denominator_sum = 0.0;
  for (int i = 0; i < count; i++) {
    float delta = fabs(cpu_result[i] - mlu_result[i]);
    numerator_sum += pow(delta, 2);
    denominator_sum += pow(fabs(cpu_result[i]), 2);
  }

  return sqrt(numerator_sum / (denominator_sum + EPSILON));
}

double Evaluator::computeDiff3_2(float *baseline_result, float *mlu_result,
                                 size_t count) {
  if (hasNanOrInf(baseline_result, count) || hasNanOrInf(mlu_result, count)) {
    LOG(ERROR) << "Found NaN or Inf when compute diff, return DBL_MAX "
                  "instead.";
    return DBL_MAX;
  }
  double max_value = 0.0;
  for (int i = 0; i < count; ++i) {
    double ratio = fabs(mlu_result[i] - baseline_result[i]);
    max_value = (ratio > max_value) ? ratio : max_value;
  }
  return max_value;
}

// aka maxape
double Evaluator::computeDiff3(float *baseline_result, float *mlu_result,
                               size_t count, mluOpDataType_t dtype) {
  if (hasNanOrInf(baseline_result, count) || hasNanOrInf(mlu_result, count)) {
    LOG(ERROR) << "Found NaN or Inf when compute diff, return DBL_MAX "
                  "instead.";
    return DBL_MAX;
  }
  double max_value = 0.0;
  for (int i = 0; i < count; ++i) {
    float numerator = fabs(mlu_result[i] - baseline_result[i]);
    double ratio = 0;
    if (((MLUOP_DTYPE_HALF == dtype) &&
         (fabs(baseline_result[i]) < EPSILON_HALF)) ||
        ((MLUOP_DTYPE_FLOAT == dtype) &&
         (fabs(baseline_result[i]) < EPSILON_FLOAT))) {
      ratio = numerator;
    } else {
      ratio = numerator / (fabs(baseline_result[i]) + EPSILON);
    }
    max_value = (ratio > max_value) ? ratio : max_value;
  }
  return max_value;
}

double Evaluator::computeDiff4(float *baseline_result, float *mlu_result,
                               size_t count) {
  if (hasNanOrInf(baseline_result, count) || hasNanOrInf(mlu_result, count)) {
    LOG(ERROR) << "Found NaN or Inf when compute diff, return DBL_MAX "
                  "instead.";
    return DBL_MAX;
  }

  double max_value = 0.0;
  int max_count = 0;
  int num_count = 0;
  for (int i = 0; i < count; ++i) {
    max_count += mlu_result[i] < baseline_result[i];
    num_count += mlu_result[i] != baseline_result[i];
  }

  max_value = (num_count < 100) ? 0 : max_count / (num_count + EPSILON);
  return max_value;
}

double Evaluator::computeError(float *baseline_result, float *mlu_result,
                               size_t count, const Criterion &criterion,
                               const std::string &name,
                               const mluOpDataType_t dtype,
                               bool skip_nan_n_inf) {
  double error = -1;
  auto func = criterion.formula;

  if (skip_nan_n_inf) {
    // if one of mlu and baseline is nan/inf, set them zero
    skipNanOrInfAsZero(baseline_result, mlu_result, count);
  } else {
    // if both mlu and baseline is nan/inf, set them zero
    resetNanOrInfAsZero(baseline_result, mlu_result, count);
  }

  switch (func) {
    case DIFF1: {
      error = computeDiff1(baseline_result, mlu_result, count);
      break;
    }
    case DIFF2: {
      error = computeDiff2(baseline_result, mlu_result, count);
      break;
    }
    case DIFF3: {
      error = computeDiff3(baseline_result, mlu_result, count, dtype);
      break;
    }
    case DIFF3_2: {
      error = computeDiff3_2(baseline_result, mlu_result, count);
      break;
    }
    case DIFF4: {
      error = computeDiff4(baseline_result, mlu_result, count);
      break;
    }
    default:
      GTEST_CHECK(false,
                  "Evaluator: found unsupported criterion when compute "
                  "result error.");
  }

  error_vec_.push_back(ErrorWrap(name, criterion, error));
  return error;
}

bool Evaluator::isPassed() {
  if (error_vec_.empty()) {
    LOG(WARNING) << "The result error is empty, it means output shape is 0 "
                    "in pb, and skip compute "
                    "result error.";
  }

  for (int i = 0; i < error_vec_.size(); ++i) {
    if (error_vec_[i].criterion.enable == false) {
      continue;
    }
    auto func = error_vec_[i].criterion.formula;
    auto threshold = error_vec_[i].criterion.threshold;
    auto error = error_vec_[i].error;
    if (std::isnan(error) || std::isinf(error)) {
      return false;
    } else if (Formula::DIFF4 == func) {
      if (error > threshold || ((error < 1 - threshold) && (error != 0)) ||
          error < 0) {
        return false;
      }
    } else if (error > threshold || error < 0) {
      return false;
    } else {
      // pass, and next
    }
  }

  return true;
}

// only when failed, call this func. to get error reason
std::vector<std::string> Evaluator::what() {
  std::vector<std::string> res;
  for (int i = 0; i < error_vec_.size(); ++i) {
    if (error_vec_[i].criterion.enable == false) {
      continue;
    }
    auto func = error_vec_[i].criterion.formula;
    auto threshold = error_vec_[i].criterion.threshold;
    auto error = error_vec_[i].error;
    auto name = error_vec_[i].name;
    if (std::isnan(error) || std::isinf(error)) {
      std::ostringstream oss;
      oss.setf(std::ios::scientific);
      oss << error;
      res.emplace_back("The error " + oss.str() + " of [" + name +
                       "] is NOT digit.");
    } else if (Formula::DIFF4 == func) {
      if (error > threshold || ((error < 1 - threshold) && (error != 0)) ||
          error < 0) {
        std::ostringstream oss;
        oss.setf(std::ios::scientific);
        oss << error;
        res.emplace_back("The error " + oss.str() + " of [" + name +
                         "] is over " + showFormula(func) + " threshold " +
                         " [" + std::to_string(1 - threshold) + " , " +
                         std::to_string(threshold) + "]");
      }
    } else if (error > threshold || error < 0) {
      std::ostringstream oss_error, oss_threshold;
      oss_error.setf(std::ios::scientific);
      oss_threshold.setf(std::ios::scientific);
      oss_error << error;
      oss_threshold << threshold;
      res.emplace_back("The error " + oss_error.str() + " of [" + name +
                       "] is over " + showFormula(func) + " threshold " +
                       oss_threshold.str());
    } else {
      // pass
    }
  }

  return res;
}

std::string Evaluator::showFormula(Formula f) {
  switch (f) {
    case DIFF1:
      return "DIFF1";
    case DIFF2:
      return "DIFF2";
    case DIFF3:
      return "DIFF3";
    case DIFF3_2:
      return "DIFF3_2";
    case DIFF4:
      return "DIFF4";
    default:
      GTEST_CHECK(false, "Evaluator: got an unsupported criterion formula.");
  }
}

double Evaluator::computeEfficiency(double num, double latency, double den) {
  if (num < 0 || latency <= 0 || den <= 0) {
    // if didn't set these values
    return -1;
  }
  return num / (latency * den);
}

void Evaluator::copy(const Evaluator *e) {
  error_vec_ = e->error_vec_;
  criterion_vec_ = e->criterion_vec_;
}

}  // namespace mluoptest
