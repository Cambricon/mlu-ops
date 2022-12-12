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
#ifndef TEST_MLUOP_GTEST_SRC_ZOO_PSAMASK_FORWARD_PSAMASK_FORWARD_H_
#define TEST_MLUOP_GTEST_SRC_ZOO_PSAMASK_FORWARD_PSAMASK_FORWARD_H_
#include <vector>
#include "core/type.h"
#include "executor.h"

namespace mluoptest {
class PsamaskForwardExecutor : public Executor {
 public:
  PsamaskForwardExecutor() {}
  ~PsamaskForwardExecutor() {}

  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;

 private:
  template <typename T>
  void psamaskCollectForwardCPU(const T *const input, T *output, const int num_,
                                const int h_feature, const int w_feature,
                                const int h_mask, const int w_mask,
                                const int half_h_mask, const int half_w_mask);
  template <typename T>
  void psamaskDistributeForwardCPU(const T *const input, T *output,
                                   const int num_, const int h_feature,
                                   const int w_feature, const int h_mask,
                                   const int w_mask, const int half_h_mask,
                                   const int half_w_mask);
};

}  // namespace mluoptest
#endif  // TEST_MLUOP_GTEST_SRC_ZOO_PSAMASK_FORWARD_PSAMASK_FORWARD_H_
