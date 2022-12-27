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
#ifndef TEST_MLUOP_GTEST_SRC_ZOO_ROTATED_FEATURE_\
ALIGN_FORWARD_ROTATED_FEATURE_ALIGN_FORWARD_H_
#define TEST_MLUOP_GTEST_SRC_ZOO_ROTATED_FEATURE_\
ALIGN_FORWARD_ROTATED_FEATURE_ALIGN_FORWARD_H_
#include "executor.h"

namespace mluoptest {
class RotatedFeatureAlignForwardExecutor : public Executor {
 public:
  RotatedFeatureAlignForwardExecutor() {}
  ~RotatedFeatureAlignForwardExecutor() {}
  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;

 private:
  void initData();
  float bilinear_interpolate(const float *input, const int height,
                             const int width, const int channels, float y,
                             float x, int c, const int index);
  int points_ = 1;
  float spatial_scale_ = 1.0;
  int64_t theory_ops_ = 0;
};
}  // namespace mluoptest
#endif  // TEST_MLUOP_GTEST_SRC_ZOO_ROTATED_FEATURE_\
ALIGN_FORWARD_ROTATED_FEATURE_ALIGN_FORWARD_H_
