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
#ifndef TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_CASE_COLLECTOR_H_
#define TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_CASE_COLLECTOR_H_

#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <list>
#include <string>
#include <iostream>
#include <vector>
#include "tools.h"
#include "variable.h"
#include "gtest/gtest.h"

class Collector {
 public:
  explicit Collector(const std::string &name);
  virtual ~Collector() {}
  std::vector<std::string> list();
  size_t num();  // return gtest repeat num NOT case number.
 private:
  std::string op_name_ = "";
  std::vector<std::string> read_case(std::string file_name);
  void grep_case(std::string dir, std::vector<std::string> &res);
  void grep_dir(std::string dir, std::vector<std::string> &op_dirs);
  std::string current_dir();

  std::vector<std::string> list_by_case_list(std::string);
  std::vector<std::string> list_by_case_dir(std::string);
  std::vector<std::string> list_by_case_path(std::string);
};

#endif  // TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_CASE_COLLECTOR_H_
