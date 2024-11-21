/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_FFT_FFT_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_FFT_FFT_H_
#include <set>
#include <vector>
#include "executor.h"

namespace mluoptest {

class FftExecutor : public Executor {
 public:
  FftExecutor() {}
  ~FftExecutor() {}

  void paramCheck() override;
  void workspaceMalloc() override;
  void compute() override;
  void workspaceFree() override;
  int64_t getTheoryOps() override;
  int64_t getTheoryIoSize() override;
  std::set<Evaluator::Formula> getCriterionsUse() const override;

 private:
  mluOpFFTPlan_t fft_plan_;
  size_t reservespace_size_ = 0, workspace_size_ = 0;
  void *reservespace_addr_ = nullptr;
  void *workspace_addr_ = nullptr;
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_FFT_FFT_H_
