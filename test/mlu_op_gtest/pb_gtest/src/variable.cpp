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
#include <ostream>
#include "variable.h"

using namespace mluoptest;  // NOLINT

GlobalVar mluoptest::global_var;

std::string GlobalVar::getParam(const std::string &str, std::string key) {
  key = key + "=";
  auto npos = str.find(key);
  if (npos == std::string::npos) {
    return "";
  } else {
    return str.substr(npos + key.length());
  }
}

void GlobalVar::init(int argc, char **argv) {
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
    cases_dir_ = cases_dir_.empty() ? getParam(arg, "--cases_dir") : cases_dir_;
    cases_list_ =
        cases_list_.empty() ? getParam(arg, "--cases_list") : cases_list_;
    case_path_ = case_path_.empty() ? getParam(arg, "--case_path") : case_path_;
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
    mlu_only_ = (mlu_only_ == false)
                    ? (arg.find("--mlu_only") != std::string::npos)
                    : mlu_only_;
    test_llc_ = (test_llc_ == false)
                    ? (arg.find("--test_llc") != std::string::npos)
                    : test_llc_;
    use_default_queue_ =
        (use_default_queue_ == false)
            ? ((arg.find("--use_default_queue") != std::string::npos) ||
               getEnv("MLUOP_GTEST_USE_DEFAULT_QUEUE", false))
            : use_default_queue_;
    unaligned_mlu_address_random_ =
        (unaligned_mlu_address_random_ == false)
            ? ((arg.find("--unaligned_mlu_address_random") !=
                std::string::npos) ||
               getEnv("MLUOP_GTEST_UNALIGNED_ADDRESS_RANDOM", false))
            : unaligned_mlu_address_random_;
    unaligned_mlu_address_set_ =
        (getParam(arg, "--unaligned_mlu_address_set").empty())
            ? getEnvInt("MLUOP_GTEST_UNALIGNED_ADDRESS_SET", 0)
            : to_int(getParam(arg, "--unaligned_mlu_address_set"),
                     "--unaligned_mlu_address_set");
    enable_gtest_internal_perf =
        (enable_gtest_internal_perf == false)
            ? ((arg.find("--internal_perf") != std::string::npos) ||
               getEnv("MLUOP_GTEST_INTERNAL_PERF", false))
            : enable_gtest_internal_perf;
    zero_input_ = (zero_input_ == false)
                      ? (arg.find("--zero_input") != std::string::npos)
                      : zero_input_;
  }
  // print();
}
void GlobalVar::print() {
  std::cout << "cases_dir is " << cases_dir_ << std::endl;
  std::cout << "cases_list is " << cases_list_ << std::endl;
  std::cout << "cases_path is " << case_path_ << std::endl;
  std::cout << "get_vmpeak is " << get_vmpeak_ << std::endl;
  std::cout << "rand_n is " << rand_n_ << std::endl;
  std::cout << "repeat is " << repeat_ << std::endl;
  std::cout << "thread is " << thread_num_ << std::endl;
  std::cout << "shuffle is " << shuffle_ << std::endl;
  std::cout << "mlu_only is " << mlu_only_ << std::endl;
  std::cout << "use_default_queue is " << use_default_queue_ << std::endl;
  std::cout << "unaligned_mlu_address_random is "
            << unaligned_mlu_address_random_ << std::endl;
  std::cout << "unaligned_mlu_address_set is " << unaligned_mlu_address_set_
            << std::endl;
  std::cout << "gtest_internal_perf is " << enable_gtest_internal_perf
            << std::endl;
  std::cout << "zero_input is " << zero_input_ << std::endl;
}
