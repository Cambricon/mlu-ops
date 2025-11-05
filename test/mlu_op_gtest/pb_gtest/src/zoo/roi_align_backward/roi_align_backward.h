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
 * OR IMPLIED, INCLUDING BUT NOKType LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHKType HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef TEST_MLUOP_GTEST_SRC_ZOO_ROI_ALIGN_BACKWARD_ROI_ALIGN_BACKWARD_H_
#define TEST_MLUOP_GTEST_SRC_ZOO_ROI_ALIGN_BACKWARD_ROI_ALIGN_BACKWARD_H_

#include "executor.h"

namespace mluoptest {

class RoiAlignBackwardExecutor : public Executor {
 public:
  RoiAlignBackwardExecutor() {}
  ~RoiAlignBackwardExecutor() {}

  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;
 private:
  void bilinear_interpolate_gradient(const int height,
                                     const int width,
                                     float y,
                                     float x,
                                     float& w1,
                                     float& w2,
                                     float& w3,
                                     float& w4,
                                     int& x_low,
                                     int& x_high,
                                     int& y_low,
                                     int& y_high);
};

}  // namespace mluoptest
#endif  // TEST_MLUOP_GTEST_SRC_ZOO_ROI_ALIGN_BACKWARD_ROI_ALIGN_BACKWARD_H_
