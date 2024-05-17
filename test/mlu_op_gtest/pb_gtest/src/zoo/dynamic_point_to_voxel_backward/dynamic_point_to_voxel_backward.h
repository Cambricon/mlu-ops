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
#ifndef TEST_MLUOP_GTEST_SRC_ZOO_DYNAMIC_POINT_TO_VOXEL_BACKWARD_\
DYNAMIC_POINT_TO_VOXEL_BACKWARD_H_
#define TEST_MLUOP_GTEST_SRC_ZOO_DYNAMIC_POINT_TO_VOXEL_BACKWARD_\
DYNAMIC_POINT_TO_VOXEL_BACKWARD_H_
#include <vector>

#include "executor.h"
#include "mlu_op.h"

namespace mluoptest {
class DynamicPointToVoxelBackwardExecutor : public Executor {
 public:
  DynamicPointToVoxelBackwardExecutor() {}
  ~DynamicPointToVoxelBackwardExecutor() { workspaceFree(); }
  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;
  void workspaceMalloc() override;
  void workspaceFree() override;

 private:
  size_t workspace_size_ = 0;
  int64_t theory_ops_ = 0;
};
}  // namespace mluoptest
#endif  // TEST_MLUOP_GTEST_SRC_ZOO_DYNAMIC_POINT_TO_VOXEL_BACKWARD_
        // DYNAMIC_POINT_TO_VOXEL_BACKWARD_H_
