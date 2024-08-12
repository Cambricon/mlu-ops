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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_CHOLESKY__
#define TEST_MLU_OP_GTEST_SRC_ZOO_CHOLESKY_
#include "executor.h"
namespace mluoptest {
class CholeskyExecutor : public Executor {
 private:
  size_t size_workspace_ = 0;
  int stride_ = 0;
  mluOpDataType_t type_ = MLUOP_DTYPE_FLOAT;
  bool result_mul = false;
  int type_size_ = 4;
  bool trans_ = true;
  bool upper_ = false;
  int ldda_ = 0;
  int n_ = 0;
  int64_t batch_size_ = 1;

 public:
  CholeskyExecutor() {}
  ~CholeskyExecutor() {}
  void paramCheck();
  void compute();
  void cpuCompute();
  void prepareComputeParam();

  int64_t getTheoryOps() override;
};
}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_CHOLESKY_
