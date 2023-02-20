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
#ifndef TEST_MLUOP_GTEST_PB_GTEST_SRC_ZOO_ADD_N_ADD_N_H_
#define TEST_MLUOP_GTEST_PB_GTEST_SRC_ZOO_ADD_N_ADD_N_H_
#include <vector>
#include "executor.h"
#include "core/type.h"

namespace mluoptest {

class AddNExecutor : public Executor {
 public:
  size_t workspace_size = 0;
  AddNExecutor() {}
  ~AddNExecutor() {}

  void paramCheck();
  void compute();
  void cpuCompute();
  void workspaceMalloc();
  void workspaceFree();
  size_t get_size_of_data_type(mluOpDataType_t dtype);
  int expand_num_after_first(int num);
  void expand_compute_cpu(std::vector<int> shape_a,
                          std::vector<int> shape_b,
                          float *input,
                          float *output);
  bool canBroadCast(std::vector<int> shape0, std::vector<int> shape1);
  int64_t getTheoryOps() override;
};

}  // namespace mluoptest
#endif  //  TEST_MLUOP_GTEST_PB_GTEST_SRC_ZOO_ADD_N_ADD_N_H_
