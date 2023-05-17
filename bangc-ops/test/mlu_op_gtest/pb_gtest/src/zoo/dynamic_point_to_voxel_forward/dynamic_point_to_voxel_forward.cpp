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

#include "dynamic_point_to_voxel_forward.h"

#include <algorithm>

namespace mluoptest {
void DynamicPointToVoxelForwardExecutor::paramCheck() {
  VLOG(4) << "[DynamicPointToVoxelForwardExecutor] Param check.";
  GTEST_CHECK(parser_->getInputNum() == 2,
              "[DynamicPointToVoxelForwardExecutor] Input number is wrong.");
  GTEST_CHECK(parser_->getOutputNum() == 5,
              "[DynamicPointToVoxelForwardExecutor] Output number is wrong.");
  GTEST_CHECK(
      parser_->getProtoNode()->has_dynamic_point_to_voxel_forward_param(),
      "[DynamicPointToVoxelForwardExecutor] Missing param.");
}

void DynamicPointToVoxelForwardExecutor::workspaceMalloc() {
  VLOG(4)
      << "[DynamicPointToVoxelForwardExecutor] call workspaceMalloc() Begin.";
  auto feats_desc = tensor_desc_[0].tensor;
  auto coors_desc = tensor_desc_[1].tensor;
  void *workspace_ptr = nullptr;
  MLUOP_CHECK(mluOpGetDynamicPointToVoxelForwardWorkspaceSize(
      handle_, feats_desc, coors_desc, &workspace_size_));
  if (workspace_size_) {
    workspace_ptr = mlu_runtime_.allocate(workspace_size_);
  }
  workspace_.push_back(workspace_ptr);
  eva_->setMluWorkspaceSize(workspace_size_);
  VLOG(4) << "[DynamicPointToVoxelForwardExecutor] call workspaceMalloc() End.";
}

void DynamicPointToVoxelForwardExecutor::compute() {
  VLOG(4) << "[DynamicPointToVoxelForwardExecutor] call compute() Begin.";
  // get params
  mluOpReduceMode_t reduce_type;
  ReduceMode reduce_mode = parser_->getProtoNode()
                               ->dynamic_point_to_voxel_forward_param()
                               .reduce_type();
  if (reduce_mode == REDUCE_MODE_MAX) {
    reduce_type = MLUOP_REDUCE_DMAX;
  } else if (reduce_mode == REDUCE_MODE_MEAN) {
    reduce_type = MLUOP_REDUCE_DMEAN;
  } else {
    reduce_type = MLUOP_REDUCE_DSUM;
  }

  auto feats_desc = tensor_desc_[0].tensor;
  auto coors_desc = tensor_desc_[1].tensor;
  auto voxel_feats_desc = tensor_desc_[2].tensor;
  auto voxel_coors_desc = tensor_desc_[3].tensor;
  auto point2voxel_map_desc = tensor_desc_[4].tensor;
  auto voxel_points_count_desc = tensor_desc_[5].tensor;
  auto voxel_num_desc = tensor_desc_[6].tensor;

  auto feats = data_vector_[0].device_ptr;
  auto coors = data_vector_[1].device_ptr;
  auto voxel_feats = data_vector_[2].device_ptr;
  auto voxel_coors = data_vector_[3].device_ptr;
  auto point2voxel_map = data_vector_[4].device_ptr;
  auto voxel_points_count = data_vector_[5].device_ptr;
  auto voxel_num = data_vector_[6].device_ptr;

  interface_timer_.start();
  MLUOP_CHECK(mluOpDynamicPointToVoxelForward(
      handle_, reduce_type, feats_desc, feats, coors_desc, coors, workspace_[0],
      workspace_size_, voxel_feats_desc, voxel_feats, voxel_coors_desc,
      voxel_coors, point2voxel_map_desc, point2voxel_map,
      voxel_points_count_desc, voxel_points_count, voxel_num_desc, voxel_num));
  interface_timer_.stop();
  VLOG(4) << "[DynamicPointToVoxelForwardExecutor] call compute() End.";
}

void DynamicPointToVoxelForwardExecutor::workspaceFree() {
  if (workspace_[0]) {
    VLOG(4)
        << "[DynamicPointToVoxelForwardExecutor] Free device workspace space.";
    mlu_runtime_.deallocate(workspace_[0]);
    workspace_[0] = nullptr;
  }
}

void DynamicPointToVoxelForwardExecutor::cpuCompute() {
  VLOG(4) << "[DynamicPointToVoxelForwardExecutor] call cpuCompute() Begin.";
  // get params
  ReduceMode reduce_mode = parser_->getProtoNode()
                               ->dynamic_point_to_voxel_forward_param()
                               .reduce_type();

  auto feats = cpu_fp32_input_[0];
  auto coors = cpu_fp32_input_[1];
  auto voxel_feats = cpu_fp32_output_[0];
  auto voxel_coors = cpu_fp32_output_[1];
  auto point2voxel_map = cpu_fp32_output_[2];
  auto voxel_points_count = cpu_fp32_output_[3];
  auto voxel_num = cpu_fp32_output_[4];

  auto coors_desc = tensor_desc_[1].tensor;
  auto feats_desc = tensor_desc_[0].tensor;
  int N = coors_desc->dims[0];
  int num_features = feats_desc->dims[1];
  auto voxel_feats_desc = tensor_desc_[2].tensor;
  int out_n = voxel_feats_desc->dims[0];

  if (parser_->getInputDataCount(0) == 0 ||
      parser_->getInputDataCount(1) == 0) {
    return;
  }

  // coors fill mask
  for (int i = 0; i < N; i++) {
    if (coors[i * 3] < 0 || coors[i * 3 + 1] < 0 || coors[i * 3 + 2] < 0) {
      coors[i * 3] = -1;
      coors[i * 3 + 1] = -1;
      coors[i * 3 + 2] = -1;
    }
  }

  // op unique
  // 1) generator index
  int32_t input_len = parser_->getInputDataCount(1);
  int32_t flat_num = input_len / N;
  std::vector<int> index(N);
  std::vector<int> index_sorted(N);
  std::generate(index.begin(), index.end(), [n = 0]() mutable { return n++; });

  // 2) sort
  auto compareInput = [=](int32_t i, int32_t j) {
    for (int32_t idx = 0; idx < flat_num; ++idx) {
      int32_t lhs = coors[idx + i * flat_num];
      int32_t rhs = coors[idx + j * flat_num];
      if (lhs < rhs) {
        return true;
      } else if (lhs > rhs) {
        return false;
      }
    }
    return false;
  };
  std::sort(index.begin(), index.end(), compareInput);
  index_sorted = index;
  // 3) unique
  auto diffInput = [=](int32_t i, int32_t j) {
    for (int32_t idx = 0; idx < flat_num; ++idx) {
      int32_t lhs = coors[idx + i * flat_num];
      int32_t rhs = coors[idx + j * flat_num];
      if (lhs != rhs) {
        return false;
      }
    }
    return true;
  };
  std::vector<int>::iterator it;
  it = std::unique(index.begin(), index.end(), diffInput);
  index.resize(std::distance(index.begin(), it));

  // 4) get output
  // 4.1) voxel_num
  int32_t unique_dim = index.size();
  voxel_num[0] = unique_dim;

  // 4.2) voxel_coors
  bool flag = false;
  if (coors[index[0] * flat_num] == -1) {
    flag = true;
    voxel_num[0] -= 1;
  }
  for (int32_t i = 0; i < voxel_num[0]; i++) {
    for (int32_t j = 0; j < flat_num; j++) {
      if (flag) {
        voxel_coors[i * flat_num + j] = coors[index[i + 1] * flat_num + j];
      } else {
        voxel_coors[i * flat_num + j] = coors[index[i] * flat_num + j];
      }
    }
  }

  // mask
  std::vector<int> mask(N, 0);
  for (int32_t i = 1; i < N; i++) {
    if (diffInput(index_sorted[i], index_sorted[i - 1])) {
      mask[i] = 0;
    } else {
      mask[i] = 1;
    }
  }
  for (int32_t i = 1; i < N; i++) {
    mask[i] = mask[i - 1] + mask[i];
  }
  // 4.3) point2voxel_map and voxel_points_count
  for (int32_t i = 1; i < N; i++) {
    point2voxel_map[index_sorted[i]] = mask[i];
  }
  for (int32_t i = 0; i < unique_dim; i++) {
    voxel_points_count[i] = 1;
    for (int32_t j = i; j < N; j++) {
      if (index[i] == index_sorted[j]) {
        if (i + 1 == unique_dim) {
          voxel_points_count[i] = N - j;
        } else {
          for (int32_t k = j + 1; k < N; k++) {
            if (index[i + 1] != index_sorted[k]) {
              voxel_points_count[i]++;
            } else {
              break;
            }
          }
        }
      }
      break;
    }
  }

  // op reduce feats
  float fill_value = 0x0;
  if (reduce_mode == REDUCE_MODE_MAX) {
    fill_value = -1.17549e038;
  }
  for (int i = 0; i < out_n * num_features; i++) {
    voxel_feats[i] = fill_value;
  }

  if (flag) {
    for (int i = 0; i < voxel_num[0]; i++) {
      voxel_points_count[i] = voxel_points_count[i + 1];
    }
  }
  for (int32_t i = 0; i < N; i++) {
    if (flag) {
      point2voxel_map[i] -= 1;
    }
    int32_t reduce_to = point2voxel_map[i];
    if (reduce_to == -1) continue;
    int32_t reduce_count = voxel_points_count[reduce_to];
    const float *feats_offset = feats + i * num_features;
    float *voxel_feats_offset = voxel_feats + reduce_to * num_features;
    if (reduce_mode == REDUCE_MODE_MAX) {
      for (int32_t j = 0; j < num_features; j++) {
        float old_value = feats_offset[j];
        float new_value = voxel_feats_offset[j];
        if (old_value >= new_value) {
          voxel_feats_offset[j] = old_value;
          theory_ops_++;
        }
      }
    } else if (reduce_mode == REDUCE_MODE_MEAN) {
      for (int32_t j = 0; j < num_features; j++) {
        float old_value = feats_offset[j];
        float new_value = old_value / reduce_count;
        voxel_feats_offset[j] += new_value;
        theory_ops_++;
      }
    }
  }
  VLOG(4) << "[DynamicPointToVoxelForwardExecutor] call cpuCompute() End.";
}

int64_t DynamicPointToVoxelForwardExecutor::getTheoryOps() {
  if (parser_->device() != CPU) {
    return -1;
  }
  VLOG(4) << "getTheoryOps: " << theory_ops_ << " ops";
  return theory_ops_;
}

}  // namespace mluoptest
