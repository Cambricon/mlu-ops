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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_MOE_DISPATCH_FORWARD_MOE_DISPATCH_FORWARD_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_MOE_DISPATCH_FORWARD_MOE_DISPATCH_FORWARD_H_
#include "executor.h"

namespace mluoptest {

class MoeDispatchForwardExecutor : public Executor {
 public:
  MoeDispatchForwardExecutor() {}
  ~MoeDispatchForwardExecutor() {}

  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;
  int64_t getTheoryIoSize() override;

 private:
  void initData();
  int samples_ = 0;
  int capacity_ = 0;
  int hidden_ = 0;
  int num_experts_ = 0;
  mluOpTensorDescriptor_t desc_gates_ = NULL;
  mluOpTensorDescriptor_t desc_indices_ = NULL;
  mluOpTensorDescriptor_t desc_locations_ = NULL;
  mluOpTensorDescriptor_t desc_input_ = NULL;
  mluOpTensorDescriptor_t desc_dispatch_ = NULL;
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_MOE_DISPATCH_FORWARD_\
MOE_DISPATCH_FORWARD_H_
