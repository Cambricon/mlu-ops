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
#ifndef TEST_MLUOP_GTEST_SRC_ZOO_MASKED_IM2COL_FORWARD_MASKED_IM2COL_FORWARD_H_
#define TEST_MLUOP_GTEST_SRC_ZOO_MASKED_IM2COL_FORWARD_MASKED_IM2COL_FORWARD_H_
#include "executor.h"

namespace mluoptest {

class MaskedIm2colForwardExecutor : public Executor {
 public:
  MaskedIm2colForwardExecutor() {}
  ~MaskedIm2colForwardExecutor() {}

  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  void workspaceMalloc() override;
  void workspaceFree() override;

 private:
  void init();
  void printDataInfo();
  int batchs_ = 0;
  int height_ = 0;
  int width_ = 0;
  int channels_ = 0;
  int kernel_h = 0;
  int kernel_w = 0;
  int pad_h = 0;
  int pad_w = 0;
  int mask_cnt_ = 0;
  size_t workspace_size_ = 0;
  void *workspace_ = nullptr;
};

}  // namespace mluoptest
#endif  // TEST_MLUOP_GTEST_SRC_ZOO_MASKED_IM2COL_FORWARD_MASKED_IM2COL_FORWARD_H_
        // //NOLINT
