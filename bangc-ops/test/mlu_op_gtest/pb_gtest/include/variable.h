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
#ifndef TEST_MLU_OP_GTEST_INCLUDE_VARIABLE_H_
#define TEST_MLU_OP_GTEST_INCLUDE_VARIABLE_H_
#include <list>
#include <string>
#include <cctype>

#include "tools.h"
#include "math_half.h"
#include "internal_perf.h"

namespace mluoptest {

struct TestSummary {
  size_t case_count = 0;
  size_t suite_count = 0;
  std::list<std::string> failed_list;
};

class GlobalVar {
 public:
  std::string cases_dir_ = "";
  std::string cases_list_ = "";
  std::string case_path_ = "";
  std::string get_vmpeak_ = "";
  TestSummary summary_;
  TestInternalInfo internal_info_;

  int dev_id_ =
      -1;  // the picked device id, make sure gtest run on the picked device.
  int rand_n_ = -1;  // pick n * random case, -1 for uninitialized
  int repeat_ = 1;   // perf-repeat repeat * kernel enqueue cnrtQueue_t, and get
                     // ave hw_time
  int thread_num_ = 1;    // thread num
  bool shuffle_ = false;  // shuffle cases.
  unsigned int half2float_algo_ = getEnvInt(
      "MLUOP_GTEST_EXPERIMENT_HALF2FLOAT_ALGO",
      AlgoHalfToFloat::CPU_INTRINSIC);  // half2float algorithm selection
  bool mlu_only_ =
      false;  // the mlu-only mode, skip cpu compute() and other func().
  bool zero_input_ = false;  // the zero-input mode, set input gdram zero.
  bool use_default_queue_ =
      false;  // will use cnrt default queue (set nullptr in mluOpSetQueue)
  bool test_llc_ = false;  // eliminate influence of llc (wiki:303333445)
  bool unaligned_mlu_address_random_ = false;
  int unaligned_mlu_address_set_ = 0;  // mlu address offset
  bool enable_gtest_internal_perf =
      false;                // will record gtest internal perf info
  bool exclusive_ = false;  // compute mode set to exclusive, only one process
                            // can run on picked device at a time.
  bool compatible_test_ = false;  // test old version api
  std::string kernel_trace_policy_ =
      "hook";  // trace kernel launch, use 'hook' method by default
  bool enable_cnpapi_ = false;  // trace kernel by `cnpapiInit` (which conflicts
                                // with host perf / cnperf)
  bool run_on_jenkins_ = false;  // whether run on jenkins
  int test_algo_ =
      -2147483648;  // test the i-th algo of the operator with multiple algos
  bool random_mlu_address_ = false;
  bool monitor_mlu_hardware_ = false;
  // TODO(None): once all op bugs are fixed, force const dram check and
  // remove this arg
  bool enable_const_dram_ = false;
  bool auto_tuning_ = false;
  bool loose_check_nan_inf_ = false;  // one of the mlu and baseline is nan and
                                      // the other is inf will pass.

  /**
   * match 'key=val' pattern, and extract val from str
   */
  std::string getParam(const std::string &str, std::string key);

  void init(int *argc, char **argv);

  void print();

 private:
  void validate();
  bool current_arg_valid = false;
  bool paramDefinedMatch(std::string &, std::string);
  void checkUnsupportedTest() const;
};

extern GlobalVar global_var;

}  // namespace mluoptest

#endif  // TEST_MLU_OP_GTEST_INCLUDE_VARIABLE_H_
