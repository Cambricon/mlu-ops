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
#ifndef MLU_OP_GTEST_SRC_ZOO_VOXEL_POOLING_FORWARD_VOXEL_POOLING_FORWARD_H_
#define MLU_OP_GTEST_SRC_ZOO_VOXEL_POOLING_FORWARD_VOXEL_POOLING_FORWARD_H_
#include "executor.h"

namespace mluoptest {
class VoxelPoolingForwardExecutor : public Executor {
 public:
  VoxelPoolingForwardExecutor() {}
  ~VoxelPoolingForwardExecutor() {}
  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;
  int64_t getTheoryIoSize() override;

 private:
  int batch_size_;
  int num_points_;
  int num_channels_;
  int num_voxel_x_;
  int num_voxel_y_;
  int num_voxel_z_;
  int64_t theory_io_size_;
  void initData();
  void printDataInfo();
  void voxelPoolingForwardCpuKernel(const int batch_size, const int num_points,
                                    const int num_channels,
                                    const int num_voxel_x,
                                    const int num_voxel_y,
                                    const int num_voxel_z, const int *geom_xyz,
                                    const float *input_features,
                                    float *output_features, int *pos_memo);
};
}  // namespace mluoptest
#endif  // MLU_OP_GTEST_SRC_ZOO_VOXEL_POOLING_FORWARD_VOXEL_POOLING_FORWARD_H_
