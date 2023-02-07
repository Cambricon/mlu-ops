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
#ifndef GTEST_SRC_ZOO_ROIAWARE_POOL3D_FORWARD_ROIAWARE_POOL3D_FORWARD_H_
#define GTEST_SRC_ZOO_ROIAWARE_POOL3D_FORWARD_ROIAWARE_POOL3D_FORWARD_H_

#include "executor.h"

namespace mluoptest {

class RoiawarePool3dForwardExecutor : public Executor {
 public:
  RoiawarePool3dForwardExecutor() {}
  ~RoiawarePool3dForwardExecutor() { workspaceFree(); }
  void paramCheck() override;
  void workspaceMalloc() override;
  void workspaceFree() override;
  void compute() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;

 private:
  void initData();
  void printDataInfo();
  int pool_method_;
  int boxes_num_;
  int pts_num_;
  int channels_;
  int max_pts_each_voxel_;
  int out_x_;
  int out_y_;
  int out_z_;
  void *dev_rois_ = NULL;
  void *dev_pts_ = NULL;
  void *dev_pts_feature_ = NULL;
  void *dev_argmax_ = NULL;
  void *dev_pts_idx_of_voxels_ = NULL;
  void *dev_pooled_features_ = NULL;
  mluOpTensorDescriptor_t desc_rois_ = NULL;
  mluOpTensorDescriptor_t desc_pts_ = NULL;
  mluOpTensorDescriptor_t desc_pts_feature_ = NULL;
  mluOpTensorDescriptor_t desc_argmax_ = NULL;
  mluOpTensorDescriptor_t desc_pts_idx_of_voxels_ = NULL;
  mluOpTensorDescriptor_t desc_pooled_features_ = NULL;
  size_t workspace_size_;
};
}  // namespace mluoptest

#endif  // GTEST_SRC_ZOO_ROIAWARE_POOL3D_FORWARD_ROIAWARE_POOL3D_FORWARD_H_
