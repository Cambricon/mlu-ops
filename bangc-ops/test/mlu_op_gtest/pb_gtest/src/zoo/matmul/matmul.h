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

#ifndef TEST_MLUOP_GTEST_SRC_ZOO_MATMUL_MATMUL_H_
#define TEST_MLUOP_GTEST_SRC_ZOO_MATMUL_MATMUL_H_
#include "executor.h"
#define USE_MLUOP_MATMUL_V2 1
namespace mluoptest {
class MatmulExecutor : public Executor {
 public:
  MatmulExecutor() {}
  ~MatmulExecutor() {}
  void paramCheck() override;
#if USE_MLUOP_MATMUL_V2
  void workspaceMalloc() override;
  void workspaceFree() override;
#endif
  void compute() override;
  void cpuCompute() override;
  void castIn() override;
  void setQuantizedParam() override;
  int64_t getTheoryOps() override;
 private:
  mluOpQuantizeMode_t quant_mode_;
  mluOpAtomicsMode_t atomics_mode_ = MLUOP_ATOMICS_NOT_ALLOWED;
#if USE_MLUOP_MATMUL_V2
  size_t workspace_size_                        = 0;
  void *workspace_                              = nullptr;
  mluOpMatMulDescriptor_t matmul_desc_           = nullptr;
  mluOpMatMulAlgo_t algo_                        = nullptr;
  mluOpMatMulHeuristicResult_t heuristic_result_ = nullptr;
#endif
};
}  // namespace mluoptest
#endif  // TEST_MLUOP_GTEST_SRC_ZOO_MATMUL_MATMUL_H_
