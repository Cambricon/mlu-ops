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
#ifndef TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_EVALUATOR_H_
#define TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_EVALUATOR_H_

#include <algorithm>
#include <iostream>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "core/logging.h"
#include "pb_test_tools.h"
#include "perf_test.h"

namespace mluoptest {

// determine whether the test case passes.
// this class receives:
// 1.output data --> error
// 2.latency of mlu
// 3.io size/io bandwidth  --> io efficiency
// 4.ops/peak force  --> compute efficiency
// 5.interface time --> print
// 6.workspace --> print
class Evaluator {
 public:
  Evaluator() {}
  virtual ~Evaluator() {}
  void copy(const Evaluator *e);

  enum Formula { DIFF1, DIFF2, DIFF3, DIFF3_2, DIFF4 };

  struct Criterion {
    Criterion(Formula f, double t, bool e = true)
        : formula(f), threshold(t), enable(e) {}
    Formula formula;
    double threshold = 0;
    // if false, only compute it, but won't mark case failed.
    bool enable = true;
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
    std::string name = "";  // tensor's name
    Criterion criterion;    // criterion
    double error = 0;       // the error of this criterion
  };

  // compute error between A and B, by given criterion
  double computeError(float *a, float *b, size_t count,
                      const Criterion &criterion, const std::string &name,
                      const mluOpDataType_t dtype, bool skip_nan_n_inf = false);

  // compute efficiency by formula:
  // theory_ops / latency / peak_compute_force
  // theory_io / latency / io_bandwidth
  double computeEfficiency(double num, double latency, double den);

  bool isPassed();
  std::vector<std::string> what();
  // get name + criterion + error
  const std::vector<ErrorWrap> &errors() { return error_vec_; }

  // and all op called this api, so just keep it.
  void setMluWorkspaceSize(size_t size) { workspace_size_ = size; }
  double getMluWorkspaceSize() { return workspace_size_; }

 private:
  double computeDiff1(float *cpu_result, float *mlu_result, size_t count);
  double computeDiff2(float *cpu_result, float *mlu_result, size_t count);
  double computeDiff3(float *baseline_result, float *mlu_result, size_t count,
                      mluOpDataType_t dtype);
  double computeDiff3_2(float *baseline_result, float *mlu_result,
                        size_t count);
  double computeDiff4(float *baseline_result, float *mlu_result, size_t count);
  inline std::string showFormula(Formula f);
  // vector of (diff1+thresdhold) /(diff2 + threshold)
  std::vector<Criterion> criterion_vec_;
  std::vector<ErrorWrap> error_vec_;  // vetor output's error

  double workspace_size_ = -1;  // for -1
};

struct PerfInfo {  // perf info for certain device (mlu)
  // hardware time baseline (mlu only
  double hardware_time_base = -1;
  // interface time (mlu only
  double interface_time = -1;
  // memcpy host to device time (mlu only
  double h2d_time = -1;
  double d2h_time = -1;
  // hardware time of mlu
  double hardware_time = -1;  // us
  // compute efficiency of mlu
  double compute_efficiency = -1;
  // io efficiency of mlu
  double io_efficiency = -1;
  // workspace size of mlu
  double workspace_size = -1;

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
  PerfInfo mlu;
  // errors
  std::vector<Evaluator::ErrorWrap> errors;
  // result
  bool is_passed = false;
  std::vector<std::string> what;
};

}  // namespace mluoptest

#endif  // TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_EVALUATOR_H_
