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

#include "dynamic_point_to_voxel_backward.h"

#include <algorithm>

namespace mluoptest {
void DynamicPointToVoxelBackwardExecutor::paramCheck() {
  VLOG(4) << "[DynamicPointToVoxelBackwardExecutor] Param check.";
  GTEST_CHECK(parser_->getInputNum() == 6,
              "[DynamicPointToVoxelBackwardExecutor] Input number is wrong.");
  GTEST_CHECK(parser_->getOutputNum() == 1,
              "[DynamicPointToVoxelBackwardExecutor] Output number is wrong.");
  GTEST_CHECK(
      parser_->getProtoNode()->has_dynamic_point_to_voxel_backward_param(),
      "[DynamicPointToVoxelBackwardExecutor] Missing param.");
}

void DynamicPointToVoxelBackwardExecutor::workspaceMalloc() {
  VLOG(4)
      << "[DynamicPointToVoxelBackwardExecutor] call workspaceMalloc() Begin.";
  mluOpReduceMode_t reduce_type = (mluOpReduceMode_t)parser_->getProtoNode()
                                      ->dynamic_point_to_voxel_backward_param()
                                      .reduce_type();
  auto feats_desc = tensor_desc_[1].tensor;
  void *workspace_ptr = nullptr;
  MLUOP_CHECK(mluOpGetDynamicPointToVoxelBackwardWorkspaceSize(
      handle_, reduce_type, feats_desc, &workspace_size_));
  if (workspace_size_) {
    workspace_ptr = mlu_runtime_.allocate(workspace_size_);
  }
  workspace_.push_back(workspace_ptr);
  eva_->setMluWorkspaceSize(workspace_size_);
  VLOG(4) << "[DynamicPointToVoxelBackwardExecutor] call workspaceMalloc() End.";
}

void DynamicPointToVoxelBackwardExecutor::compute() {
  VLOG(4) << "[DynamicPointToVoxelBackwardExecutor] call compute() Begin.";
  // get params
  mluOpReduceMode_t reduce_type = (mluOpReduceMode_t)parser_->getProtoNode()
                                      ->dynamic_point_to_voxel_backward_param()
                                      .reduce_type();
  auto grad_voxel_feats_desc = tensor_desc_[0].tensor;
  auto feats_desc = tensor_desc_[1].tensor;
  auto voxel_feats_desc = tensor_desc_[2].tensor;
  auto point2voxel_map_desc = tensor_desc_[3].tensor;
  auto voxel_points_count_desc = tensor_desc_[4].tensor;
  auto voxel_num_desc = tensor_desc_[5].tensor;
  auto gard_feats_desc = tensor_desc_[6].tensor;

  auto grad_voxel_feats = data_vector_[0].device_ptr;
  auto feats = data_vector_[1].device_ptr;
  auto voxel_feats = data_vector_[2].device_ptr;
  auto point2voxel_map = data_vector_[3].device_ptr;
  auto voxel_points_count = data_vector_[4].device_ptr;
  auto voxel_num = data_vector_[5].device_ptr;
  auto gard_feats = data_vector_[6].device_ptr;

  interface_timer_.start();
  MLUOP_CHECK(mluOpDynamicPointToVoxelBackward(
      handle_, reduce_type, grad_voxel_feats_desc, grad_voxel_feats,
      feats_desc, feats, voxel_feats_desc, voxel_feats,
      point2voxel_map_desc, point2voxel_map,
      voxel_points_count_desc, voxel_points_count,
      voxel_num_desc, voxel_num, workspace_[0],
      workspace_size_, gard_feats_desc, gard_feats));
  interface_timer_.stop();
  VLOG(4) << "[DynamicPointToVoxelBackwardExecutor] call compute() End.";
}

void DynamicPointToVoxelBackwardExecutor::workspaceFree() {
  if (workspace_[0]) {
    VLOG(4)
        << "[DynamicPointToVoxelBackwardExecutor] Free device workspace space.";
    mlu_runtime_.deallocate(workspace_[0]);
    workspace_[0] = nullptr;
  }
}

void DynamicPointToVoxelBackwardExecutor::cpuCompute() {
  VLOG(4) << "[DynamicPointToVoxelBackwardExecutor] call cpuCompute() Begin.";
  // get params
  VLOG(4) << "[DynamicPointToVoxelBackwardExecutor] call cpuCompute() End.";
}

int64_t DynamicPointToVoxelBackwardExecutor::getTheoryOps() {
  if (parser_->device() != CPU) {
    return -1;
  }
  VLOG(4) << "getTheoryOps: " << theory_ops_ << " ops";
  return theory_ops_;
}

}  // namespace mluoptest