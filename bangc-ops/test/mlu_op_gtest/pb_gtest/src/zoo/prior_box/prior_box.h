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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_PRIOR_BOX_PRIOR_BOX_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_PRIOR_BOX_PRIOR_BOX_H_
#include "executor.h"

namespace mluoptest {
class PriorBoxExecutor : public Executor {
 public:
  PriorBoxExecutor() {}
  ~PriorBoxExecutor() {}
  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;

 private:
  void initData();
  void priorBox_Cpu_Kernel(
      float* min_sizes, const int min_sizes_num, float* new_aspect_ratios,
      const int new_aspect_ratios_num, float* variances,
      const int variances_num, float* max_sizes, const int max_sizes_num,
      const int height, const int width, const int im_height,
      const int im_width, const float step_h, const float step_w,
      const float offset, const bool clip,
      const bool min_max_aspect_ratios_order, float* output,
      const int output_size, float* var, const int var_size);

  int height_;
  int width_;
  int im_height_;
  int im_width_;
  float step_w_;
  float step_h_;
  float offset_;
  bool clip_;
  bool min_max_aspect_ratios_order_;
  int64_t theory_op_size_;

  mluOpTensorDescriptor_t min_sizes_desc_;
  mluOpTensorDescriptor_t aspect_ratios_desc_;
  mluOpTensorDescriptor_t variances_desc_;
  mluOpTensorDescriptor_t max_sizes_desc_;
  mluOpTensorDescriptor_t output_desc_;
  mluOpTensorDescriptor_t var_desc_;
};
}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_PRIOR_BOX_PRIOR_BOX_H_
