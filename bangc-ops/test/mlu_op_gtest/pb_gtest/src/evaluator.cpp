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
#include <set>
#include <vector>
#include <string>
#include "evaluator.h"

namespace mluoptest {

void Evaluator::init(void *baseline_result, void *mlu_result,
                     const size_t count, const std::set<Criterion> criterions,
                     const std::string &name, const mluOpDataType_t dtype,
                     const bool skip_nan_n_inf) {
  base_array_ = baseline_result;
  mlu_array_ = mlu_result;
  count_ = count;
  count_total_ = is_complex_ ? count_ * 2 : count_;
  criterions_ = criterions;
  name_ = name;
  dtype_ = dtype;
  skip_nan_n_inf_ = skip_nan_n_inf;
  if (dtype_ == MLUOP_DTYPE_COMPLEX_HALF ||
      dtype_ == MLUOP_DTYPE_COMPLEX_FLOAT) {
    is_complex_ = true;
  } else {
    is_complex_ = false;
  }
  stride_ = is_complex_ ? 2 : 1;
  thresholdLevel1();
}

void Evaluator::computeErrorForOneCriterion() {
  if (skip_compute_diff_) {
    return;
  }
  func_ = cur_criterion_.formula;

  switch (dtype_) {
    case MLUOP_DTYPE_DOUBLE: {
      compute_diff<double>();
    } break;
    default: {
      compute_diff<float>();
    }
  }
  error_vec_.push_back(
      ErrorWrap(name_, cur_criterion_, error_, error_imag_, dtype_));
}

void Evaluator::setErrorWrap() {
  for (auto &it : criterions_) {
    if (nan_inf_pass_) {
      if (it.formula == DIFF4) {
        error_vec_.push_back(ErrorWrap(name_, it, -1, -1, dtype_));
      } else {  // !DIFF4
        error_vec_.push_back(ErrorWrap(name_, it, 0, 0, dtype_));
      }
    } else {  // !nan_inf_pass
      auto error_max = std::numeric_limits<double>::max();
      error_vec_.push_back(ErrorWrap(name_, it, error_max, error_max, dtype_));
    }
  }
}

void Evaluator::thresholdLevel1() {
  threshold_l1_ = true;
  // if one op only need compute diff4 in the future, fix here.
  for (auto &it : criterions_) {
    if (it.formula == DIFF4) {
      continue;
    } else {
      if (!(it.threshold == 0 && it.threshold_imag == 0)) {
        threshold_l1_ = false;
        return;
      }
    }
  }
}

void Evaluator::computeError(void *baseline_result, void *mlu_result,
                             const size_t count,
                             const std::set<Criterion> criterions,
                             const std::string &name,
                             const mluOpDataType_t dtype,
                             const bool skip_nan_n_inf) {
  if (0 == criterions.size()) {
    criterion_matching_ = false;
    LOG(ERROR) << "Error func in mluop_gtest and pb/pt may mismatch,"
               << " now no error func is used, plese check.";
    return;
  }
  init(baseline_result, mlu_result, count, criterions, name, dtype,
       skip_nan_n_inf);
  switch (dtype_) {
    case MLUOP_DTYPE_DOUBLE: {
      check_nan_inf<double>();
    } break;
    default: {
      check_nan_inf<float>();
    }
  }
  for (auto &it : criterions_) {
    cur_criterion_ = it;
    computeErrorForOneCriterion();
  }
}

bool Evaluator::isPassed() {
  if (!criterion_matching_) {
    return false;
  }
  if (error_vec_.empty()) {
    LOG(WARNING) << "The result error is empty, it means output shape is 0 "
                    "in pb, "
                    "and skip compute result error.";
  }
  for (size_t i = 0; i < error_vec_.size(); ++i) {
    if (error_vec_[i].criterion.enable == false) {
      continue;
    }
    auto func = error_vec_[i].criterion.formula;
    auto threshold = error_vec_[i].criterion.threshold;
    auto error = error_vec_[i].error;
    auto threshold_imag = error_vec_[i].criterion.threshold_imag;
    auto error_imag = error_vec_[i].error_imag;
    auto dtype = error_vec_[i].dtype;

// for diff4, error < 0 means it is not used because number of data points are
// less than 100. also, for diff4, thred < 0 means gpu_diff4 is 0.0 or 1.0, in
// this case, mlu should always pass
#define CHECK_ERROR(err, thred)             \
  if (std::isnan(err) || std::isinf(err)) { \
    return false;                           \
  } else if (Formula::DIFF4 == func) {      \
    if (err >= 0 && thred >= 0) {           \
      if (err == 0.0 || (err == 1.0)) {     \
        return false;                       \
      }                                     \
    }                                       \
  } else if (err > thred || err < 0) {      \
    return false;                           \
  }

    switch (dtype) {
      case MLUOP_DTYPE_COMPLEX_HALF:
      case MLUOP_DTYPE_COMPLEX_FLOAT: {
        CHECK_ERROR(error, threshold);
        CHECK_ERROR(error_imag, threshold_imag);
      } break;
      default: {
        CHECK_ERROR(error, threshold);
      }
    }
#undef CHECK_ERROR
  }
  return true;
}

// only when failed, call this func. to get error reason
std::vector<std::string> Evaluator::what() {
  std::vector<std::string> res;
  for (size_t i = 0; i < error_vec_.size(); ++i) {
    if (error_vec_[i].criterion.enable == false) {
      continue;
    }
    auto func = error_vec_[i].criterion.formula;
    auto threshold = error_vec_[i].criterion.threshold;
    auto error = error_vec_[i].error;
    auto name = error_vec_[i].name;
    auto threshold_imag = error_vec_[i].criterion.threshold_imag;
    auto error_imag = error_vec_[i].error_imag;
    auto dtype = error_vec_[i].dtype;

#define GET_ERROR_INFO(err, thred, err_name)                                   \
  if (std::isnan(err) || std::isinf(err)) {                                    \
    std::ostringstream oss;                                                    \
    oss.setf(std::ios::scientific);                                            \
    oss << err;                                                                \
    res.emplace_back("The " + err_name + " " + oss.str() + " of [" + name +    \
                     "] is NOT digit.");                                       \
  } else if (Formula::DIFF4 == func) {                                         \
    if (err >= 0 && thred >= 0) {                                              \
      if (err == 0.0 || (err == 1.0)) {                                        \
        std::ostringstream oss;                                                \
        oss.setf(std::ios::scientific);                                        \
        oss << err;                                                            \
        res.emplace_back("The " + err_name + " " + oss.str() + " of [" +       \
                         name + "] is over " + showFormula(func) +             \
                         " threshold (0, 1) ");                                \
      }                                                                        \
    }                                                                          \
  } else if (err > thred || err < 0) {                                         \
    std::ostringstream oss_err, oss_thred;                                     \
    oss_err.setf(std::ios::scientific);                                        \
    oss_thred.setf(std::ios::scientific);                                      \
    oss_err << err;                                                            \
    oss_thred << thred;                                                        \
    res.emplace_back("The " + err_name + " " + oss_err.str() + " of [" +       \
                     name + "] is over " + showFormula(func) + " threshold " + \
                     oss_thred.str());                                         \
  }

    switch (dtype) {
      case MLUOP_DTYPE_COMPLEX_HALF:
      case MLUOP_DTYPE_COMPLEX_FLOAT: {
        GET_ERROR_INFO(error, threshold, std::string("error_real"));
        GET_ERROR_INFO(error_imag, threshold_imag, std::string("error_imag"));
      } break;
      default: {
        GET_ERROR_INFO(error, threshold, std::string("error"));
      }
    }
#undef CHECK_ERROR
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
