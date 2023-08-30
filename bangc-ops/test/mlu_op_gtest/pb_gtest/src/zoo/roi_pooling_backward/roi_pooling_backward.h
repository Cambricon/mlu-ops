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
 * OR IMPLIED, INCLUDING BUT NOKType LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHKType HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef TEST_MLUOP_GTEST_SRC_ZOO_ROI_POOLING_BACKWARD_ROI_POOLING_BACKWARD_H_
#define TEST_MLUOP_GTEST_SRC_ZOO_ROI_POOLING_BACKWARD_ROI_POOLING_BACKWARD_H_

#include <set>
#include "executor.h"

namespace mluoptest {

class RoiPoolingBackwardExecutor : public Executor {
 public:
  RoiPoolingBackwardExecutor() {}
  ~RoiPoolingBackwardExecutor() {}

  void paramCheck();
  void compute();
  void cpuCompute();
  int64_t getTheoryOps() override;
  std::set<Evaluator::Formula> getCriterionsUse() const override;
};

}  // namespace mluoptest
#endif  // TEST_MLUOP_GTEST_SRC_ZOO_ROI_POOLING_BACKWARD_ROI_POOLING_BACKWARD_H_

