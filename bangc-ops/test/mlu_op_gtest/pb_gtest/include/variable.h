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
#ifndef TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_VARIABLE_H_
#define TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_VARIABLE_H_
#include <list>
#include <string>
#include <cctype>

#include "internal_perf.h"
#include "pb_test_tools.h"

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

  // the picked device id, make sure gtest run on the picked device.
  int dev_id_ = 0;
  int rand_n_ = -1;  // pick n * random case, -1 for uninitialized
  int repeat_ = 1;   // perf-repeat repeat * kernel enqueue cnrtQueue_t, and get
                     // ave hw_time
  int thread_num_ = 1;    // thread num
  bool shuffle_ = false;  // shuffle cases.
  bool mlu_only_ = false;
  bool zero_input_ = false;
  bool use_default_queue_ = false;
  bool test_llc_ = false;
  bool unaligned_mlu_address_random_ = false;
  int unaligned_mlu_address_set_ = 0;
  bool enable_gtest_internal_perf = false;

  std::string getParam(const std::string &str, std::string key);

  void init(int argc, char **argv);

  void print();
};

extern GlobalVar global_var;

}  // namespace mluoptest

#endif  // TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_VARIABLE_H_
