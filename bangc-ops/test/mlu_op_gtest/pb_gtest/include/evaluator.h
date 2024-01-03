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
#ifndef TEST_MLU_OP_GTEST_INCLUDE_EVALUATOR_H_
#define TEST_MLU_OP_GTEST_INCLUDE_EVALUATOR_H_

#ifdef __AVX__
#include <immintrin.h>
#include <limits>
#endif
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <list>
#include <sstream>
#include <vector>
#include <tuple>
#include <string>
#include <utility>
#include <map>
#include <set>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/logging.h"
#include "tools.h"
#include "cpu_dtype.h"
#include "internal_perf.h"
#include "variable.h"
#include "perf_test.h"
#include "accuracy_test.h"

namespace mluoptest {

// determine whether the test case passes.
// this class receives:
// 1.output data --> error
// 2.latency of mlu and gpu
// 3.io size/io bandwidth  --> io efficiency
// 4.ops/peak force  --> compute efficiency
// 5.interface time --> print
// 6.workspace --> print

const double EPSILON = 1e-9;
const double EPSILON_FLOAT = 1e-6;
const double EPSILON_HALF = 1e-4;

#ifdef __AVX__
// found inf or nan, return true.
template <typename T>
static bool hasNanOrInf(T *data, size_t count);

template <>
bool hasNanOrInf<float>(float *data, size_t count) {
  const __m256 exp_bit = _mm256_set1_ps(std::numeric_limits<float>::infinity());
  size_t stride =
      256 / (sizeof(float) * 8);  // 1 __m256 saved 8 * (sizeof(float) * 8 bit)
  size_t deal_count = (count / stride) * stride;
  __m256 m_data;
  for (size_t i = 0; i < deal_count; i += stride) {
    m_data = _mm256_load_ps(data + i);
    m_data = _mm256_and_ps(exp_bit, m_data);
    m_data = _mm256_cmp_ps(m_data, exp_bit, _CMP_EQ_OQ);
    if (_mm256_movemask_ps(m_data) != 0) {
      return true;
    }
  }
  for (size_t i = deal_count; i < count; ++i) {
    if (std::isnan(data[i]) || std::isinf(data[i])) {
      return true;
    }
  }
  return false;
}

template <>
bool hasNanOrInf<double>(double *data, size_t count) {
  const __m256d exp_bit =
      _mm256_set1_pd(std::numeric_limits<double>::infinity());
  size_t stride = 256 / (sizeof(double) *
                         8);  // 1 __m256 saved 4 * (sizeof(double) * 8 bit)
  size_t deal_count = (count / stride) * stride;
  __m256d m_data;
  for (size_t i = 0; i < deal_count; i += stride) {
    m_data = _mm256_load_pd(data + i);
    m_data = _mm256_and_pd(exp_bit, m_data);
    m_data = _mm256_cmp_pd(m_data, exp_bit, _CMP_EQ_OQ);
    if (_mm256_movemask_pd(m_data) != 0) {
      return true;
    }
  }
  for (size_t i = deal_count; i < count; ++i) {
    if (std::isnan(data[i]) || std::isinf(data[i])) {
      return true;
    }
  }
  return false;
}

template <typename T>
static bool twoBytesNanInf(T *data, size_t count, int first_mask,
                           int second_mask) {
  float *data_float = reinterpret_cast<float *>(data);
  __m256 first_half_inf_bit = _mm256_set1_ps(*(float *)&first_mask);
  __m256 second_half_inf_bit = _mm256_set1_ps(*(float *)&second_mask);

  size_t stride = 256 / (sizeof(float) * 8);
  size_t deal_count = (count / 2 / stride) * stride;

  __m256 m_data;
  for (size_t i = 0; i < deal_count; i += stride) {
    m_data = _mm256_loadu_ps(data_float + i);
    m_data = _mm256_and_ps(first_half_inf_bit, m_data);
    m_data = _mm256_cmp_ps(m_data, first_half_inf_bit, _CMP_EQ_OQ);
    if (_mm256_movemask_ps(m_data) != 0) {
      return true;
    }
  }
  for (size_t i = 0; i < deal_count; i += stride) {
    m_data = _mm256_loadu_ps(data_float + i);
    m_data = _mm256_and_ps(second_half_inf_bit, m_data);
    m_data = _mm256_cmp_ps(m_data, second_half_inf_bit, _CMP_EQ_OQ);
    if (_mm256_movemask_ps(m_data) != 0) {
      return true;
    }
  }
  for (size_t i = deal_count * 2; i < count; ++i) {
    if (std::isnan(data[i]) || std::isinf(data[i])) {
      return true;
    }
  }
  return false;
}

template <>
bool hasNanOrInf<half>(half *data, size_t count) {
  return twoBytesNanInf(data, count, 0x7c000000, 0x00007c00);
}

// TODO(None): test itself
template <>
bool hasNanOrInf<bfloat16>(bfloat16 *data, size_t count) {
  return twoBytesNanInf(data, count, 0x7f800000, 0x00007f80);
}

#else  // __AVX__
template <typename T>
bool hasNanOrInf(T *data, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (std::isinf(data[i]) || std::isnan(data[i])) {
      return true;
    }
  }
  return false;
}
#endif

template <typename T>
bool isNanOrInf(T data) {
  if (std::isinf(data) || std::isnan(data)) {
    return true;
  }
  return false;
}

template <typename T>
static void resetNanOrInfAsZero(T *a, T *b, size_t count) {
  bool has_nan = false;
  bool has_inf = false;

  for (size_t i = 0; i < count; ++i) {
    if (unlikely(std::isnan(a[i]) && std::isnan(b[i]))) {
      a[i] = (T)0;
      b[i] = (T)0;
      has_nan = true;
    } else if (unlikely(std::isinf(a[i]) && std::isinf(b[i]) && a[i] == b[i])) {
      // if a is inf, b is -inf, don't deal here.
      // when check hasNanOrInf will set diff as DBL_MAX (instead infinity).
      a[i] = (T)0;
      b[i] = (T)0;
      has_inf = true;
    }
  }
  if (has_nan) {
    VLOG(4) << "Found result of baseline and mlu are both NaN, set them as 0, "
               "and go on.";
  }
  if (has_inf) {
    VLOG(4) << "Found result of baseline and mlu are both Inf, set them as 0, "
               "and go on.";
  }
}

enum StorageDtype {
  FLOAT = 0,
  VOID = 1,
};

class Evaluator {
 public:
  Evaluator() {}
  virtual ~Evaluator() {}
  void copy(const Evaluator *e);
  void setStorageDtype(StorageDtype storage_dtype);

  enum Formula { DIFF1, DIFF2, DIFF3, DIFF3_2, DIFF4, DIFF_KL };

  static std::string Formula2str(Formula f) {
    static std::map<Formula, std::string> f_names = {
        {Formula::DIFF1, "DIFF1"}, {Formula::DIFF2, "DIFF2"},
        {Formula::DIFF3, "DIFF3"}, {Formula::DIFF3_2, "DIFF3_2"},
        {Formula::DIFF4, "DIFF4"}, {Formula::DIFF_KL, "DIFF_KL"}};
    return f_names[f];
  }

  struct Criterion {
    Criterion() {}
    Criterion(Formula f, double t, bool e = true)
        : formula(f), threshold(t), enable(e) {}
    Criterion(Formula f, double t, double ti, bool e = true)
        : formula(f), threshold(t), threshold_imag(ti), enable(e) {}
    Criterion &operator=(const Criterion &c) {
      this->formula = c.formula;
      this->threshold = c.threshold;
      this->threshold_imag = c.threshold_imag;
      this->enable = c.enable;
      return *this;
    }
    Formula formula;
    double threshold =
        0;  // threshold for real numbers or the real part of complex numbers
    double threshold_imag =
        0;  // threshold for the imaginary part of complex numbers
    bool enable =
        true;  // if false, only compute it, but won't mark case failed.

    bool operator<(const struct Criterion &c) const {
      if (formula == c.formula) {
        return false;  // for deduplication
      } else {
        return formula < c.formula;
      }
    }
  };

  // pack 1 tensor's name/ criterion(func/threshold)/ diff together.
  struct ErrorWrap {
    ErrorWrap(std::string n, Criterion c, double e)
        : name(n), criterion(c), error(e) {}
    ErrorWrap(std::string n, Criterion c, double e, double ei,
              mluOpDataType_t dt)
        : name(n), criterion(c), error(e), error_imag(ei), dtype(dt) {}
    std::string name = "";  // tensor's name
    Criterion criterion;    // criterion
    double error =
        0;  // the error of this criterion, also for real part of complex number
    double error_imag =
        0;  // the error of the imaginary part, only used for complex number
    mluOpDataType_t dtype;
  };

  // compute error between baseline and mlu results by given criterion
  void computeDiff(void *baseline_result, void *mlu_result, const size_t count,
                   const std::set<Criterion> criterions,
                   const std::string &name, const mluOpDataType_t dtype);

  // compute efficiency by formula:
  // theory_ops / latency / peak_compute_force
  // theory_io / latency / io_bandwidth
  double computeEfficiency(double num, double latency, double den);

  bool isPassed();
  std::vector<std::string> what();
  // get name + criterion + error
  const std::vector<ErrorWrap> &errors() { return error_vec_; }

  // TODO(None): delete this api. but now the refactor plan is not ready,
  // and all op called this api, so just keep it.
  void setMluWorkspaceSize(size_t size) { workspace_size_ = size; }
  double getMluWorkspaceSize() { return workspace_size_; }

 private:
  // distinguish_nan_inf
  template <typename T>
  bool compareNanInfStrict(T mlu_out[], T baseline_out[], size_t index) {
    T mlu = mlu_out[index];
    T baseline = baseline_out[index];
    if (!((std::isnan(mlu) && std::isnan(baseline)) || (mlu == baseline))) {
      return false;
    }
    return true;
  }

  // not distinguish_nan_inf
  template <typename T>
  bool compareNanInfLoose(T mlu_out[], T baseline_out[], size_t index) {
    T mlu = mlu_out[index];
    T baseline = baseline_out[index];
    if (std::isnan(mlu)) {
      if (std::isnan(baseline)) {
        return true;
      } else if (std::isinf(baseline)) {
        nan_inf_neq_pos_.emplace_back(index);
        return true;
      } else {
        return false;
      }
    }  // mlu is nan
    if (std::isinf(mlu)) {
      if (mlu == baseline) {
        return true;
      } else if (std::isnan(baseline)) {
        nan_inf_neq_pos_.emplace_back(index);
        return true;
      } else {
        return false;
      }
    }
    return true;
  }

  // if forst level accuracy(wiki:52441150), only compare nan/inf, skip other
  // position
  template <typename T>
  void dealNanInf() {
    skip_compute_diff_ = false;
    bool has_nan_inf = false;
    T *a = reinterpret_cast<T *>(base_array_);
    T *b = reinterpret_cast<T *>(mlu_array_);
    if (threshold_l1_) {
      resetNanOrInfAsZero(a, b, count_total_);
      nanInfRemain<T>();
      return;
    }
    // TODO(None): have vector.push_back() in for loop, open it after
    // modify code logic #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < count_total_; ++i) {
      if (isNanOrInf(a[i]) || isNanOrInf(b[i])) {
        skip_compute_diff_ = true;
        has_nan_inf = true;
        bool res = false;
        if (true == global_var.loose_check_nan_inf_) {
          res = compareNanInfLoose(a, b, i);
        } else {
          res = compareNanInfStrict(a, b, i);
        }
        if (false == res) {
          nan_inf_wrong_pos_.emplace_back(i);
          nan_inf_pass_ = false;
        }
      }
    }
    if (!nan_inf_wrong_pos_.empty()) {
      auto first_pos = *std::min_element(std::begin(nan_inf_wrong_pos_),
                                         std::end(nan_inf_wrong_pos_));
      LOG(ERROR) << "Found NaN or Inf, but mlu is not equal to baseline,"
                 << " return DBL_MAX instead."
                 << "The first wrong position is " << first_pos;
      return;
    }
    if (has_nan_inf) {
      nan_inf_pass_ = true;
      if (!nan_inf_neq_pos_.empty()) {
        auto first_pos = *std::min_element(std::begin(nan_inf_neq_pos_),
                                           std::end(nan_inf_neq_pos_));
        LOG(WARNING) << "The results of baseline and mlu are not equal, "
                     << "one of them is nan and the other is inf, "
                     << "the current mode will not distinguish nan and inf, "
                     << "this case will pass. The first position is "
                     << first_pos;
      }
    }
  }

  template <typename T>
  void compute_diff() {
    if (skip_compute_diff_) {
      return;
    }
    switch (func_) {
      case Evaluator::Formula::DIFF1: {
        computeDiff1<T>();
      } break;
      case Evaluator::Formula::DIFF2: {
        computeDiff2<T>();
      } break;
      case Evaluator::Formula::DIFF3: {
        computeDiff3<T>();
      } break;
      case Evaluator::Formula::DIFF3_2: {
        computeDiff3_2<T>();
      } break;
      case Evaluator::Formula::DIFF4: {
        computeDiff4<T>();
      } break;
      case Evaluator::Formula::DIFF_KL: {
        computeDiffKl<T>();
      } break;
      default: {
        GTEST_CHECK(false,
                    "Evaluator: found unsupported criterion when compute "
                    "result error.");
      }
    }
  }

  template <typename T>
  void computeDiff1() {
    double numerator_sum = 0.0;
    double numerator_sum_imag = 0.0;
    double denominator_sum = 0.0;
    double denominator_sum_imag = 0.0;
    T *base_array = reinterpret_cast<T *>(base_array_);
    T *mlu_array = reinterpret_cast<T *>(mlu_array_);
#pragma omp parallel for reduction \
        (+:numerator_sum, denominator_sum, numerator_sum_imag, \
         denominator_sum_imag) schedule(guided)
    for (size_t i = 0; i < count_total_; i += stride_) {
      numerator_sum += std::abs(double(mlu_array[i]) - double(base_array[i]));
      denominator_sum += std::abs(double(base_array[i]));
      if (is_complex_) {
        numerator_sum_imag +=
            std::abs(double(mlu_array[i + 1]) - double(base_array[i + 1]));
        denominator_sum_imag += std::abs(double(base_array[i + 1]));
      }
    }  // end omp parallel block
    error_ = numerator_sum / (denominator_sum + EPSILON);
    if (is_complex_) {
      error_imag_ = numerator_sum_imag / (denominator_sum_imag + EPSILON);
    }
  }

  template <typename T>
  void computeDiff2() {
    double numerator_sum = 0.0;
    double numerator_sum_imag = 0.0;
    double denominator_sum = 0.0;
    double denominator_sum_imag = 0.0;
    T *base_array = reinterpret_cast<T *>(base_array_);
    T *mlu_array = reinterpret_cast<T *>(mlu_array_);
#pragma omp parallel for reduction \
        (+:numerator_sum, denominator_sum, numerator_sum_imag, \
        denominator_sum_imag) schedule(guided)
    for (size_t i = 0; i < count_total_; i += stride_) {
      numerator_sum +=
          std::pow(double(mlu_array[i]) - double(base_array[i]), 2);
      denominator_sum += std::pow(double(base_array[i]), 2);
      if (is_complex_) {
        numerator_sum_imag +=
            std::pow(double(mlu_array[i + 1]) - double(base_array[i + 1]), 2);
        denominator_sum_imag += std::pow(double(base_array[i + 1]), 2);
      }
    }  // end omp parallel block
    error_ = std::sqrt(numerator_sum / (denominator_sum + EPSILON));
    if (is_complex_) {
      error_imag_ =
          std::sqrt(numerator_sum_imag / (denominator_sum_imag + EPSILON));
    }
  }

  template <typename T>
  void computeDiff3() {
    T *base_array = reinterpret_cast<T *>(base_array_);
    T *mlu_array = reinterpret_cast<T *>(mlu_array_);
    double max_ratio = 0;
    double max_ratio_imag = 0;
    double EPS = 0;
    if (dtype_ == MLUOP_DTYPE_HALF || dtype_ == MLUOP_DTYPE_COMPLEX_HALF) {
      EPS = EPSILON_HALF;
    } else if (dtype_ == MLUOP_DTYPE_FLOAT ||
               dtype_ == MLUOP_DTYPE_COMPLEX_FLOAT) {
      EPS = EPSILON_FLOAT;
    }
#pragma omp parallel for reduction(max                          \
                                   : max_ratio, max_ratio_imag) \
    schedule(guided)
    for (size_t i = 0; i < count_total_; i += stride_) {
      double numerator = std::abs(double(mlu_array[i]) - double(base_array[i]));
      double denominator = std::abs(double(base_array[i]));
      double ratio =
          (denominator < EPS) ? numerator : numerator / (denominator + EPSILON);
      max_ratio = (ratio > max_ratio) ? ratio : max_ratio;
      if (is_complex_) {
        numerator =
            std::abs(double(mlu_array[i + 1]) - double(base_array[i + 1]));
        denominator = std::abs(double(base_array[i + 1]));
        ratio = (denominator < EPS) ? numerator
                                    : numerator / (denominator + EPSILON);
        max_ratio_imag = (ratio > max_ratio_imag) ? ratio : max_ratio_imag;
      }
    }  // end omp parallel block
    error_ = max_ratio;
    if (is_complex_) {
      error_imag_ = max_ratio_imag;
    }
  }

  template <typename T>
  void computeDiff3_2() {
    T *base_array = reinterpret_cast<T *>(base_array_);
    T *mlu_array = reinterpret_cast<T *>(mlu_array_);
    double max_ratio = 0;
    double max_ratio_imag = 0;
#pragma omp parallel for reduction(max                          \
                                   : max_ratio, max_ratio_imag) \
    schedule(guided)
    for (size_t i = 0; i < count_total_; i += stride_) {
      double ratio = std::abs(double(mlu_array[i]) - double(base_array[i]));
      max_ratio = (ratio > max_ratio) ? ratio : max_ratio;
      if (is_complex_) {
        ratio = std::abs(double(mlu_array[i + 1]) - double(base_array[i + 1]));
        max_ratio_imag = (ratio > max_ratio_imag) ? ratio : max_ratio_imag;
      }
    }  // end omp parallel block
    error_ = max_ratio;
    if (is_complex_) {
      error_imag_ = max_ratio_imag;
    }
  }

  template <typename T>
  struct TupleHash {
    size_t operator()(const std::tuple<T, T> &t) const {
      size_t seed;
      hash_combine(seed, (T)std::get<0>(t));
      hash_combine(seed, (T)std::get<1>(t));
      return seed;
    }
  };

#ifndef KL_EPSILON
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
#endif

  template <typename T>
  void computeDiff4() {
    T *base_array = reinterpret_cast<T *>(base_array_);
    T *mlu_array = reinterpret_cast<T *>(mlu_array_);
    double max_count = 0;
    double num_count = 0;
    double max_count_imag = 0;
    double num_count_imag = 0;
    std::unordered_set<std::tuple<T, T>, TupleHash<T>> unrepeat_res;
    std::unordered_set<std::tuple<T, T>, TupleHash<T>> unrepeat_res_imag;
    for (size_t i = 0; i < count_total_; i += stride_) {
      if (is_complex_) {
        if (mlu_array[i + 1] != base_array[i + 1]) {
          auto unrepeat_tuple =
              std::make_tuple(mlu_array[i + 1], base_array[i + 1]);
          if (unrepeat_res_imag.end() ==
              unrepeat_res_imag.find(unrepeat_tuple)) {
            max_count_imag += mlu_array[i + 1] < base_array[i + 1];
            num_count_imag++;
            unrepeat_res_imag.emplace(unrepeat_tuple);
          }
        }
      }
      if (mlu_array[i] != base_array[i]) {
        auto unrepeat_tuple = std::make_tuple(mlu_array[i], base_array[i]);
        if (unrepeat_res.end() == unrepeat_res.find(unrepeat_tuple)) {
          max_count += mlu_array[i] < base_array[i];
          num_count++;
          unrepeat_res.emplace(unrepeat_tuple);
        }
      }
    }
    error_ = (num_count < 100) ? -1 : max_count / num_count;
    if (is_complex_) {
      error_imag_ =
          (num_count_imag < 100) ? -1 : max_count_imag / num_count_imag;
    }
  }

  template <typename T>
  void computeDiffKl() {
    T *base_array = reinterpret_cast<T *>(base_array_);
    T *mlu_array = reinterpret_cast<T *>(mlu_array_);
    // diff_kl needs count_total_ <= INT64_MAX because of the length in python
    // func is int64 diff_kl skips the data with too less quantity
    if (count_total_ < (size_t)1000 || count_total_ > (size_t)INT64_MAX) {
      error_ = -1;
    } else {
      error_ = diff_kl(base_array, mlu_array, count_total_);
    }
  }

  template <typename T>
  void check_nan_inf() {
    skip_compute_diff_ = false;
    // if diff is LEVEL 1, return. Else only compute where is nan/inf.
    dealNanInf<T>();
    if (skip_compute_diff_) {
      setErrorWrap();
    }
  }

  template <typename T>
  void nanInfRemain() {
    T *base_array = reinterpret_cast<T *>(base_array_);
    T *mlu_array = reinterpret_cast<T *>(mlu_array_);
    if (hasNanOrInf(base_array, count_total_) ||
        hasNanOrInf(mlu_array, count_total_)) {
      LOG(ERROR)
          << "Found NaN or Inf when compute diff, return DBL_MAX instead.";
      skip_compute_diff_ = true;
      nan_inf_pass_ = false;
    }
  }

  void init(void *baseline_result, void *mlu_result, const size_t count,
            const std::set<Criterion> criterions, const std::string &name,
            const mluOpDataType_t dtype);

  void computeDiffForOneCriterion();
  void computeDiffFloatAndDouble();
  void computeDiffByDtype();
  void thresholdLevel1();
  void setErrorWrap();
  void checkNanInfFloatAndDouble();
  void checkNanInfByDtype();
  inline std::string showFormula(Formula f);

  std::function<void(Evaluator *)> checkNanInfFunc = nullptr;
  std::function<void(Evaluator *)> computeDiffFunc = nullptr;

  inline void selectFuncPtr() {
    if (VOID == storage_dtype_) {
      checkNanInfFunc = &Evaluator::checkNanInfByDtype;
      computeDiffFunc = &Evaluator::computeDiffByDtype;
    } else if (FLOAT == storage_dtype_) {
      checkNanInfFunc = &Evaluator::checkNanInfFloatAndDouble;
      computeDiffFunc = &Evaluator::computeDiffFloatAndDouble;
    }
  }

  void *base_array_ = nullptr;
  void *mlu_array_ = nullptr;
  mluOpDataType_t dtype_;
  int stride_ = -1;
  size_t count_ = -1;
  // count with complex
  size_t count_total_ = -1;
  double error_ = -1;
  double error_imag_ = -1;
  bool skip_compute_diff_ = false;
  bool is_complex_ = false;
  bool threshold_l1_ = false;
  bool nan_inf_pass_ = false;
  bool criterion_matching_ = true;
  Criterion cur_criterion_;
  StorageDtype storage_dtype_ = FLOAT;
  Formula func_;
  std::string name_ = "";
  std::set<Criterion> criterions_;
  std::vector<Criterion>
      criterion_vec_;  // vector of (diff1+thresdhold) /(diff2 + threshold)
  std::vector<ErrorWrap> error_vec_;  // vetor output's error
  // record the position where nan/inf output is wrong
  std::vector<size_t> nan_inf_wrong_pos_;
  // used for LOOSE_CHECK_NAN_INF mode, record the position where one is inf the
  // other is nan
  std::vector<size_t> nan_inf_neq_pos_;

  double workspace_size_ = -1;  // for -1
};

// ref: pageid 38651906
// ref: pageid 42177509
struct PerfInfo {  // perf info for certain device (mlu or gpu)
  // hardware time baseline (mlu only
  double hardware_time_base = -1;
  // interface time (mlu only
  double interface_time = -1;
  // memcpy host to device time (mlu only
  double h2d_time = -1;
  double d2h_time = -1;
  // hardware time of mlu or gpu
  double hardware_time = -1;  // us
  // mlu hardware time coefficient of variantion
  double hardware_time_cv = -1;
  double hardware_time_layer = -1;  // us
  // compute efficiency of mlu or gpu
  double compute_efficiency = -1;
  // io efficiency of mlu or gpu
  double io_efficiency = -1;
  // workspace size of mlu or gpu
  double workspace_size = -1;

  // kernel invoked by mlu or gpu
  std::vector<std::string> kernel_name_lists;
  bool kernel_tracing_enabled = false;
  // time point and hardware time for monitor
  std::list<std::tuple<size_t, float>> raw_hwtime_list;

  // theory ops/io/peak force/bandwidth for efficiency
  double theory_ops = -1;     // op
  double theory_io = -1;      // bytes
  double compute_force = -1;  // op/s
  double io_bandwidth = -1;   // GB/s
};

struct EvaluateResult {
  // id
  std::string op_name = "";
  std::string case_path = "";
  // perf info
  PerfInfo gpu;
  PerfInfo mlu;
  // errors
  std::vector<Evaluator::ErrorWrap> errors;
  // result
  bool is_passed = false;
  // whether compute completely or enter foolproof
  bool compute_completed = false;
  std::vector<std::string> what;
  // gtest internal time perf
  GtestInternal gtest;
};

}  // namespace mluoptest

#endif  // TEST_MLU_OP_GTEST_INCLUDE_EVALUATOR_H_
