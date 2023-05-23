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

#include "dynamic_point_to_voxel_backward.h"

#include <algorithm>  // std::min
#include <cstring>    // memset

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
  void* workspace_ptr = nullptr;
  MLUOP_CHECK(mluOpGetDynamicPointToVoxelBackwardWorkspaceSize(
      handle_, reduce_type, feats_desc, &workspace_size_));
  if (workspace_size_) {
    workspace_ptr = mlu_runtime_.allocate(workspace_size_);
  }
  workspace_.push_back(workspace_ptr);
  eva_->setMluWorkspaceSize(workspace_size_);
  VLOG(4)
      << "[DynamicPointToVoxelBackwardExecutor] call workspaceMalloc() End.";
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
      handle_, reduce_type, grad_voxel_feats_desc, grad_voxel_feats, feats_desc,
      feats, voxel_feats_desc, voxel_feats, point2voxel_map_desc,
      point2voxel_map, voxel_points_count_desc, voxel_points_count,
      voxel_num_desc, voxel_num, workspace_[0], workspace_size_,
      gard_feats_desc, gard_feats));
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
  // reduce_mode == REDUCE_MODE_MAX
  auto grad_voxel_feats = cpu_fp32_input_[0];
  auto feats = cpu_fp32_input_[1];
  auto voxel_feats = cpu_fp32_input_[2];
  auto point2voxel_map = cpu_fp32_input_[3];
  auto voxel_points_count = cpu_fp32_input_[4];
  auto voxel_num = cpu_fp32_input_[5];
  auto grad_feats = cpu_fp32_output_[0];

  auto feats_desc = tensor_desc_[1].tensor;
  int M = voxel_num[0];
  int C = feats_desc->dims[1];
  int N = feats_desc->dims[0];
  VLOG(5) << "M=" << M;
  VLOG(5) << "C=" << C;
  VLOG(5) << "N=" << N;

  memset(grad_feats, 0, N * C);
  if (parser_->getInputDataCount(0) == 0 ||
      parser_->getInputDataCount(1) == 0 || M == 0) {
    return;
  }
  std::vector<int32_t> voxel_from_vec(M * C, N);
  int32_t* voxel_from = voxel_from_vec.data();

  // kernel1
  for (int32_t x = 0; x < N; x++) {
    int32_t point_to = point2voxel_map[x];
    const int input_offset = x * C;
    const float* feats_offset = feats + input_offset;

    if (point_to == -1) {
      continue;
    }

    const int reduced_offset = point_to * C;
    const float* reduced_feats_offset = voxel_feats + reduced_offset;
    int32_t* reduce_from_offset = voxel_from + reduced_offset;

    for (int32_t i = 0; i < C; i++) {
      if (feats_offset[i] == reduced_feats_offset[i]) {
        reduce_from_offset[i] = std::min(reduce_from_offset[i], x);
      }
    }
  }

  // kernel2
  for (int32_t x = 0; x < M; x++) {
    const int reduced_offset = x * C;
    const int32_t* scatter_to_offset = voxel_from + reduced_offset;
    const float* grad_reduced_feats_offset = grad_voxel_feats + reduced_offset;

    for (int32_t i = 0; i < C; i++) {
      grad_feats[scatter_to_offset[i] * C + i] = grad_reduced_feats_offset[i];
    }
  }

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
