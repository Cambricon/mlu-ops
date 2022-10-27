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
#ifndef TEST_MLU_OP_GTEST_SRC_GTEST_MLU_OP_GTEST_H_
#define TEST_MLU_OP_GTEST_SRC_GTEST_MLU_OP_GTEST_H_

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <list>
#include <tuple>
#include <memory>
#include <thread>  // NOLINT
#include <mutex>   // NOLINT
#include "gtest/gtest.h"
#include "test_env.h"
#include "executor.h"
#include "evaluator.h"
#include "case_collector.h"
#include "thread_pool.h"
#include "pb_test_tools.h"

using namespace ::testing;  // NOLINT

// string is op name
// size_t is case_id
// for multi thread: get case_list by op_name, and case_id is useless.
// for single thread: get case_list by op_name, and get case_path by case_list +
// case_id
class TestSuite : public TestWithParam<std::tuple<std::string, size_t>> {
 public:
  TestSuite() {}
  virtual ~TestSuite() {}

  // setup for 1 test suite
  static void SetUpTestCase();
  static void TearDownTestCase();

  // setup for 1 test case
  void SetUp() {}
  void TearDown();
  void Run();

  static std::string op_name_;
  static std::vector<std::string> case_path_vec_;
  static std::shared_ptr<mluoptest::ExecuteContext> ectx_;
  static std::shared_ptr<mluoptest::ExecuteConfig> ecfg_;

 private:
  void Thread1();
  void ThreadX();

  std::list<mluoptest::EvaluateResult> res_;

  void print(mluoptest::EvaluateResult eva, bool average = false);
  void report(mluoptest::EvaluateResult eva);
  void recordXml(mluoptest::EvaluateResult eva);
};

#endif  // TEST_MLU_OP_GTEST_SRC_GTEST_MLU_OP_GTEST_H_
