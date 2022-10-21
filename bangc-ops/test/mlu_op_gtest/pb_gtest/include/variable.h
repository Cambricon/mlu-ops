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

  // the picked device id, make sure gtest run on the picked device.
  int dev_id_ = 0;
  int rand_n_ = -1;  // pick n * random case, -1 for uninitialized
  int repeat_ = 1;   // perf-repeat repeat * kernel enqueue cnrtQueue_t, and get
                     // ave hw_time
  int thread_num_ = 1;    // thread num
  bool shuffle_ = false;  // shuffle cases.

  std::string getParam(const std::string &str, std::string key) {
    key = key + "=";
    auto npos = str.find(key);
    if (npos == std::string::npos) {
      return "";
    } else {
      return str.substr(npos + key.length());
    }
  }

  void init(int argc, char **argv) {
    auto to_int = [](std::string s, std::string opt) -> int {
      if (std::count_if(s.begin(), s.end(),
                        [](unsigned char c) { return std::isdigit(c); }) ==
              s.size() &&
          !s.empty()) {
        return std::atoi(s.c_str());
      } else {
        return -1;
      }
    };
    for (int i = 0; i < argc; i++) {
      std::string arg = argv[i];
      cases_dir_ =
          cases_dir_.empty() ? getParam(arg, "--cases_dir") : cases_dir_;
      cases_list_ =
          cases_list_.empty() ? getParam(arg, "--cases_list") : cases_list_;
      case_path_ =
          case_path_.empty() ? getParam(arg, "--case_path") : case_path_;
      get_vmpeak_ =
          get_vmpeak_.empty() ? getParam(arg, "--get_vmpeak") : get_vmpeak_;
      rand_n_ = (rand_n_ == -1) ? to_int(getParam(arg, "--rand_n"), "--rand_n")
                                : rand_n_;
      repeat_ = getParam(arg, "--perf_repeat").empty()
                    ? repeat_
                    : to_int(getParam(arg, "--perf_repeat"), "--perf_repeat");
      thread_num_ = getParam(arg, "--thread").empty()
                        ? thread_num_
                        : to_int(getParam(arg, "--thread"), "--thread");

      shuffle_ = (shuffle_ == false)
                     ? (arg.find("--gtest_shuffle") != std::string::npos)
                     : shuffle_;
    }
    // print();
  }

  void print() {
    std::cout << "cases_dir is " << cases_dir_ << std::endl;
    std::cout << "cases_list is " << cases_list_ << std::endl;
    std::cout << "cases_path is " << case_path_ << std::endl;
    std::cout << "get_vmpeak is " << get_vmpeak_ << std::endl;
    std::cout << "rand_n is " << rand_n_ << std::endl;
    std::cout << "repeat is " << repeat_ << std::endl;
    std::cout << "thread is " << thread_num_ << std::endl;
    std::cout << "shuffle is " << shuffle_ << std::endl;
  }
};

}  // namespace mluoptest

#endif  // TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_VARIABLE_H_
