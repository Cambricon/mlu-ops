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
#include <string>  //  std::string
#include <stdexcept>
#include "variable.h"

#include "kernel_tracing.h"

mluoptest::GlobalVar mluoptest::global_var;
// XXX(zhaolianshui): --gtest* should be reserved for gtest, somehow
// --gtest_shuffle is also used
//                    in mluop_gtest
#define ORG_GTEST_PREFIX "--gtest"
#define IS_GTEST_ARG(cmd_arg) \
  (cmd_arg.rfind(ORG_GTEST_PREFIX, 0) != std::string::npos)

namespace mluoptest {

std::string GlobalVar::getParam(const std::string &str, std::string key) {
  key = key + "=";
  auto npos = str.rfind(key, 0);
  if (npos == std::string::npos) {
    return "";
  } else {
    current_arg_valid = IS_GTEST_ARG(str) ? false : true;
    return str.substr(npos + key.length());
  }
}

bool GlobalVar::paramDefinedMatch(std::string &arg, std::string valid_param) {
  if (arg == valid_param) {
    current_arg_valid = IS_GTEST_ARG(arg) ? false : true;
    return true;
  } else {
    return false;
  }
}

void GlobalVar::init(int *argc, char **argv) {
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
  for (int i = 1; i < *argc; i++) {
    std::string arg = argv[i];
    cases_dir_ = getParam(arg, "--cases_dir").empty()
                     ? cases_dir_
                     : getParam(arg, "--cases_dir");
    cases_list_ = getParam(arg, "--cases_list").empty()
                      ? cases_list_
                      : getParam(arg, "--cases_list");
    case_path_ = getParam(arg, "--case_path").empty()
                     ? case_path_
                     : getParam(arg, "--case_path");
    get_vmpeak_ = getParam(arg, "--get_vmpeak").empty()
                      ? get_vmpeak_
                      : getParam(arg, "--get_vmpeak");
    rand_n_ = getParam(arg, "--rand_n").empty()
                  ? rand_n_
                  : to_int(getParam(arg, "--rand_n"), "--rand_n");
    repeat_ = getParam(arg, "--perf_repeat").empty()
                  ? repeat_
                  : to_int(getParam(arg, "--perf_repeat"), "--perf_repeat");
    thread_num_ = getParam(arg, "--thread").empty()
                      ? thread_num_
                      : to_int(getParam(arg, "--thread"), "--thread");
    half2float_algo_ =
        getParam(arg, "--half2float_algo").empty()
            ? half2float_algo_
            : (unsigned int)to_int(getParam(arg, "--half2float_algo"),
                                   "--half2float_algo");
    unaligned_mlu_address_set_ =
        getParam(arg, "--unaligned_mlu_address_set").empty()
            ? unaligned_mlu_address_set_
            : to_int(getParam(arg, "--unaligned_mlu_address_set"),
                     "--unaligned_mlu_address_set");
    test_algo_ = getParam(arg, "--test_algo").empty()
                     ? test_algo_
                     : to_int(getParam(arg, "--test_algo"), "--test_algo");
    auto_tuning_ =
        paramDefinedMatch(arg, "--auto_tuning") ? true : auto_tuning_;
    shuffle_ = paramDefinedMatch(arg, "--gtest_shuffle") ? true : shuffle_;
    mlu_only_ = paramDefinedMatch(arg, "--mlu_only") ? true : mlu_only_;
    test_llc_ = paramDefinedMatch(arg, "--test_llc") ? true : test_llc_;
    use_default_queue_ = paramDefinedMatch(arg, "--use_default_queue")
                             ? true
                             : use_default_queue_;
    loose_check_nan_inf_ = paramDefinedMatch(arg, "--loose_check_nan_inf")
                               ? true
                               : loose_check_nan_inf_;
    unaligned_mlu_address_random_ =
        paramDefinedMatch(arg, "--unaligned_mlu_address_random")
            ? true
            : unaligned_mlu_address_random_;
    enable_gtest_internal_perf = paramDefinedMatch(arg, "--internal_perf")
                                     ? true
                                     : enable_gtest_internal_perf;
    zero_input_ = paramDefinedMatch(arg, "--zero_input") ? true : zero_input_;
    exclusive_ = paramDefinedMatch(arg, "--exclusive") ? true : exclusive_;
    enable_cnpapi_ =
        paramDefinedMatch(arg, "--enable_cnpapi") ? true : enable_cnpapi_;
    // BUG(zhaolianshui): separate cmd and env args
    compatible_test_ = (compatible_test_ == false)
                           ? (paramDefinedMatch(arg, "--compatible_test") ||
                              getEnv("MLUOP_GTEST_COMPATIBLE_TEST", false))
                           : compatible_test_;
    random_mlu_address_ = paramDefinedMatch(arg, "--random_mlu_address")
                              ? true
                              : random_mlu_address_;
    monitor_mlu_hardware_ = paramDefinedMatch(arg, "--monitor_mlu_hardware")
                                ? true
                                : monitor_mlu_hardware_;
    // TODO(None): once all op bugs are fixed, force const dram check
    // and remove this arg
    enable_const_dram_ = paramDefinedMatch(arg, "--enable_const_dram")
                             ? true
                             : enable_const_dram_;

    if (current_arg_valid) {
      // make sure the last element of argv, i.e. argv[argc], is always null
      for (int j = i; j < *argc; ++j) {
        argv[j] = argv[j + 1];
      }
      --(*argc);
      --i;
    }
    current_arg_valid = false;
  }
  // get args from env
  use_default_queue_ = (use_default_queue_ == false)
                           ? getEnv("MLUOP_GTEST_USE_DEFAULT_QUEUE", false)
                           : use_default_queue_;
  loose_check_nan_inf_ = (loose_check_nan_inf_ == false)
                             ? getEnv("MLUOP_GTEST_LOOSE_CHECK_NAN_INF", false)
                             : loose_check_nan_inf_;
  unaligned_mlu_address_random_ =
      (unaligned_mlu_address_random_ == false)
          ? getEnv("MLUOP_GTEST_UNALIGNED_ADDRESS_RANDOM", false)
          : unaligned_mlu_address_random_;
  unaligned_mlu_address_set_ =
      (unaligned_mlu_address_set_ == 0)
          ? getEnvInt("MLUOP_GTEST_UNALIGNED_ADDRESS_SET", 0)
          : unaligned_mlu_address_set_;
  enable_gtest_internal_perf = (enable_gtest_internal_perf == false)
                                   ? getEnv("MLUOP_GTEST_INTERNAL_PERF", false)
                                   : enable_gtest_internal_perf;
  exclusive_ = (exclusive_ == false) ? getEnv("MLUOP_GTEST_EXCLUSIVE", false)
                                     : exclusive_;
  run_on_jenkins_ = (run_on_jenkins_ == false)
                        ? std::getenv("JENKINS_URL") != NULL
                        : run_on_jenkins_;
  enable_cnpapi_ = (enable_cnpapi_ == false)
                       ? getEnv("MLUOP_GTEST_ENABLE_CNPAPI", false)
                       : enable_cnpapi_;
  random_mlu_address_ = (random_mlu_address_ == false)
                            ? getEnv("MLUOP_GTEST_RANDOM_MLU_ADDRESS", false)
                            : random_mlu_address_;
  monitor_mlu_hardware_ =
      (monitor_mlu_hardware_ == false)
          ? getEnv("MLUOP_GTEST_MONITOR_MLU_HARDWARE", false)
          : monitor_mlu_hardware_;
  // TODO(None): once all op bugs are fixed, force const dram check and
  // remove this arg
  enable_const_dram_ = (enable_const_dram_ == false)
                           ? getEnv("MLUOP_GTEST_ENABLE_CONST_DRAM", false)
                           : enable_const_dram_;

  // validate();
  // print();

  checkUnsupportedTest();
}
void GlobalVar::validate() {
#if MLUOP_GTEST_DISABLE_CNRT_HOOK
  if (!run_on_jenkins_) {
    kernel_trace_policy_ = "disabled";
  }
#endif
  if (enable_cnpapi_) {
    kernel_trace_policy_ = "cnpapi";
  }
  if (!kernelTracingCtx::is_method_available(kernel_trace_policy_)) {
    LOG(FATAL) << "INVALID kernel tracing policy: " << kernel_trace_policy_
               << ". Supported: "
               << kernelTracingCtx::getSupportedMethodNames();
    throw std::runtime_error(
        "invalid kernel tracing policy, please check you input");
  }
}
void GlobalVar::print() {
#define ENDL "\n"
  std::cout << "cases_dir is " << cases_dir_ << ENDL;
  std::cout << "cases_list is " << cases_list_ << ENDL;
  std::cout << "cases_path is " << case_path_ << ENDL;
  std::cout << "get_vmpeak is " << get_vmpeak_ << ENDL;
  std::cout << "rand_n is " << rand_n_ << ENDL;
  std::cout << "repeat is " << repeat_ << ENDL;
  std::cout << "thread is " << thread_num_ << ENDL;
  std::cout << "half2float_algo is " << half2float_algo_ << ENDL;
  std::cout << "shuffle is " << shuffle_ << ENDL;
  std::cout << "mlu_only is " << mlu_only_ << ENDL;
  std::cout << "use_default_queue is " << use_default_queue_ << ENDL;
  std::cout << "unaligned_mlu_address_random is "
            << unaligned_mlu_address_random_ << ENDL;
  std::cout << "unaligned_mlu_address_set is " << unaligned_mlu_address_set_
            << ENDL;
  std::cout << "gtest_internal_perf is " << enable_gtest_internal_perf << ENDL;
  std::cout << "zero_input is " << zero_input_ << ENDL;
  std::cout << "exclusive is " << exclusive_ << ENDL;
  std::cout << "compatible_test is " << compatible_test_ << ENDL;
  std::cout << "run_on_jenkins is " << run_on_jenkins_ << ENDL;
  std::cout << "enable_cnpapi is " << enable_cnpapi_ << std::endl;
  std::cout << "random_mlu_address is " << random_mlu_address_ << ENDL;
#undef ENDL
}

void GlobalVar::checkUnsupportedTest() const {
  // random_mlu_address use MLU memory pool, which is not mutex guarded, so
  // don't use it in multi-thread mode
  if (thread_num_ > 1) {
    if (random_mlu_address_) {
      LOG(ERROR) << "Does not support random_mlu_address in multi-thread mode.";
      exit(EXIT_FAILURE_MLUOP);
    }
    if (monitor_mlu_hardware_) {
      LOG(ERROR)
          << "Does not support monitor_mlu_hardware in multi-thread mode.";
      exit(EXIT_FAILURE_MLUOP);
    }
  }
}

}  // namespace mluoptest
