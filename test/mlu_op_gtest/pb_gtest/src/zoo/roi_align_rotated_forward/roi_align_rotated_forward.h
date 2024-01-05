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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_ROIALIGNROTATED_FORWARD_\
ROIALIGNROTATED_FORWARD_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_ROIALIGNROTATED_FORWARD_\
ROIALIGNROTATED_FORWARD_H_

#include <vector>

#include "executor.h"

#define ROI_OFFSET 6

struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  float w1;
  float w2;
  float w3;
  float w4;
};

namespace mluoptest {
class RoiAlignRotatedForwardExecutor : public Executor {
 public:
  RoiAlignRotatedForwardExecutor() {}
  ~RoiAlignRotatedForwardExecutor() {}

  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;

 private:
  void preCalcForBilinearInterpolate(
      const int height, const int width, const int channel,
      const int pooled_height, const int pooled_width, const int roi_bin_grid_h,
      const int roi_bin_grid_w, const float roi_start_x,
      const float roi_start_y, const float bin_size_h, const float bin_size_w,
      const float roi_center_x, const float roi_center_y, const float cos_theta,
      const float sin_theta, std::vector<PreCalc> &pre_calc);
  int64_t theory_ops_ = 0;
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_ROIALIGNROTATED_FORWARD_\
ROIALIGNROTATED_FORWARD_H_
