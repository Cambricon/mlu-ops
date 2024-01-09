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
  if (parser_->getInputDataCount(0) == 0 ||
      parser_->getInputDataCount(1) == 0) {
    return;
  }

  // Get input
  ReduceMode reduce_mode = parser_->getProtoNode()
                               ->dynamic_point_to_voxel_forward_param()
                               .reduce_type();
  auto feats = cpu_fp32_input_[0];
  auto coors = cpu_fp32_input_[1];
  auto feats_desc = tensor_desc_[0].tensor;
  auto coors_desc = tensor_desc_[1].tensor;
  const int32_t N = coors_desc->dims[0];
  const int32_t num_coors = coors_desc->dims[1];
  const int32_t num_features = feats_desc->dims[1];

  // Get output
  auto voxel_feats = cpu_fp32_output_[0];
  auto voxel_coors = cpu_fp32_output_[1];
  auto point2voxel_map = cpu_fp32_output_[2];
  auto voxel_points_count = cpu_fp32_output_[3];
  auto voxel_num = cpu_fp32_output_[4];
  auto voxel_feats_desc = tensor_desc_[2].tensor;

  // step 0
  // If coors[i,:] contains negative numbers, coors[i,:] fill with -1
  for (int32_t i = 0; i < N; ++i) {
    if (coors[i * 3] < 0 || coors[i * 3 + 1] < 0 || coors[i * 3 + 2] < 0) {
      coors[i * 3] = -1;
      coors[i * 3 + 1] = -1;
      coors[i * 3 + 2] = -1;
    }
  }

  // step 1
  // 1.1 Get index
  std::vector<int> index(N);
  std::generate(index.begin(), index.end(), [n = 0]() mutable { return n++; });

  // 1.2 Sort by comparing coordinates
  auto compareInput = [=](int32_t i, int32_t j) {
    for (int32_t idx = 0; idx < num_coors; ++idx) {
      const int32_t lhs = coors[idx + i * num_coors];
      const int32_t rhs = coors[idx + j * num_coors];
      if (lhs < rhs) {
        return true;
      } else if (lhs > rhs) {
        return false;
      }
    }
    return false;
  };
  std::sort(index.begin(), index.end(), compareInput);
  std::vector<int> index_sorted = index;

  // 1.3 Unique
  auto diffInput = [=](int32_t i, int32_t j) {
    for (int32_t idx = 0; idx < num_coors; ++idx) {
      const int32_t lhs = coors[idx + i * num_coors];
      const int32_t rhs = coors[idx + j * num_coors];
      if (lhs != rhs) {
        return false;
      }
    }
    return true;
  };
  std::vector<int>::iterator it =
      std::unique(index.begin(), index.end(), diffInput);
  index.resize(std::distance(index.begin(), it));

  // 2. Calculate voxel_num
  const bool flag = coors[index[0] * num_coors] == -1;
  voxel_num[0] = index.size() - static_cast<int32_t>(flag);
  for (int32_t i = 0; i < voxel_num[0]; ++i) {
    for (int32_t j = 0; j < num_coors; ++j) {
      if (flag) {
        voxel_coors[i * num_coors + j] = coors[index[i + 1] * num_coors + j];
      } else {
        voxel_coors[i * num_coors + j] = coors[index[i] * num_coors + j];
      }
    }
  }

  // 3. Calculate point2voxel_map
  std::vector<int> mask(N, 0);
  for (int32_t i = 1; i < N; ++i) {
    if (!diffInput(index_sorted[i], index_sorted[i - 1])) {
      mask[i] = 1;
    }
    mask[i] += mask[i - 1];
  }

  for (int32_t i = 0; i < N; ++i) {
    point2voxel_map[index_sorted[i]] = mask[i] - static_cast<int32_t>(flag);
  }

  // 4. Calculate voxel_points_count
  int32_t voxel_points_count_idx = 0;
  for (int32_t i = 0; i < index_sorted.size();) {
    int32_t count = 1;
    for (int32_t j = i + 1; j < N; ++j) {
      if (diffInput(index_sorted[i], index_sorted[j])) {
        count++;
      } else {
        break;
      }
    }
    i += count;

    if (flag && i == count) {
      continue;
    }

    voxel_points_count[voxel_points_count_idx] = count;
    voxel_points_count_idx++;
  }

  // 5. Calculate voxel_feats
  const float fill_value = reduce_mode == REDUCE_MODE_MAX ? -1.17549e038 : 0x0;
  for (int32_t i = 0; i < voxel_feats_desc->dims[0] * num_features; ++i) {
    voxel_feats[i] = fill_value;
  }

  for (int32_t i = 0; i < N; ++i) {
    const int32_t point2voxel_idx = point2voxel_map[i];
    if (point2voxel_idx == -1) {
      continue;
    }

    const int32_t reduce_count = voxel_points_count[point2voxel_idx];
    const float *feats_offset = feats + i * num_features;
    float *voxel_feats_offset = voxel_feats + point2voxel_idx * num_features;
    if (reduce_mode == REDUCE_MODE_MAX) {
      for (int32_t j = 0; j < num_features; ++j) {
        if (feats_offset[j] >= voxel_feats_offset[j]) {
          voxel_feats_offset[j] = feats_offset[j];
          theory_ops_++;
        }
      }
    } else if (reduce_mode == REDUCE_MODE_MEAN) {
      for (int32_t j = 0; j < num_features; ++j) {
        voxel_feats_offset[j] += feats_offset[j] / reduce_count;
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
