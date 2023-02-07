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
#ifndef TEST_MLU_OP_GTEST_PB_GTEST_SRC_ZOO_BBOX_OVERLAPS_BBOX_OVERLAPS_H_
#define TEST_MLU_OP_GTEST_PB_GTEST_SRC_ZOO_BBOX_OVERLAPS_BBOX_OVERLAPS_H_
#include <string.h>
#include "executor.h"

namespace mluoptest {

class BboxOverlapsExecutor : public Executor {
 public:
  BboxOverlapsExecutor() {}
  ~BboxOverlapsExecutor() {}

  void paramCheck();
  void compute();
  void cpuCompute();
  int64_t getTheoryOps() override;

 private:
  template <typename T>
  void cpuBboxOverlaps(const T *, const T *, T *, const int, const int,
                       const int, const bool, const int);
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_PB_GTEST_SRC_ZOO_BBOX_OVERLAPS_BBOX_OVERLAPS_H_
