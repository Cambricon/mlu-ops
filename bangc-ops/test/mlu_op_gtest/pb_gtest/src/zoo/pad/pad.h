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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_PAD_PAD_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_PAD_PAD_H_
#include "executor.h"

namespace mluoptest {

class PadExecutor : public Executor {
 private:
  int origin_[20];
  int current_[20];
  uint64_t host_padding_value_ = 0x0;
  float fp32_padding_value_    = 0.0;

 public:
  PadExecutor() {}
  ~PadExecutor() {}

  void paramCheck();
  void compute();
  void cpuCompute();
  void padCpu(float *input, float *output, int64_t *paddings,
              int64_t *shape, int dims, float const_value);
  void padPerLayer(float *input,
                   int64_t in_idx,
                   float *output,
                   int64_t out_idx,
                   int64_t *paddings,
                   int64_t *shape,
                   int dims,
                   int64_t cnt);
  int64_t getTheoryOps() override;
  int64_t getTheoryIoSize() override;
};

}  // namespace mluoptest
#endif  // TEST_MLUOP_GTEST_SRC_ZOO_PAD_PAD_H_
