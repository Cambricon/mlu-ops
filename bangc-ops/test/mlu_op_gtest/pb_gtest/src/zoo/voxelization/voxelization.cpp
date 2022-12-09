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
#include "voxelization.h"

#include "kernels/kernel.h"
#include "mlu_op.h"

namespace mluoptest {
void VoxelizationExecutor::paramCheck() {
  GTEST_CHECK(parser_->getInputNum() == 3);
  GTEST_CHECK(parser_->getOutputNum() == 4);
  if (!parser_->getProtoNode()->has_voxelization_param()) {
    LOG(ERROR) << "Missing voxelization param.";
  }
}

void VoxelizationExecutor::workspaceMalloc() {
  VLOG(4) << "[VoxelizationExecutor] call workspaceMalloc() Begin.";
  auto tensor_points = tensor_desc_[0].tensor;
  auto tensor_voxel_size = tensor_desc_[1].tensor;
  auto tensor_coors_range = tensor_desc_[2].tensor;
  auto tensor_voxels = tensor_desc_[3].tensor;
  auto tensor_coors = tensor_desc_[4].tensor;
  auto tensor_num_points_per_voxel = tensor_desc_[5].tensor;
  auto tensor_voxel_num = tensor_desc_[6].tensor;

  int32_t max_points =
      parser_->getProtoNode()->voxelization_param().max_points();
  int32_t max_voxels =
      parser_->getProtoNode()->voxelization_param().max_voxels();
  int32_t NDim = parser_->getProtoNode()->voxelization_param().ndim();
  bool deterministic =
      parser_->getProtoNode()->voxelization_param().deterministic();

  void *workspace_ptr = nullptr;
  MLUOP_CHECK(mluOpGetVoxelizationWorkspaceSize(
      handle_, tensor_points, tensor_voxel_size, tensor_coors_range, max_points,
      max_voxels, NDim, deterministic, tensor_voxels, tensor_coors,
      tensor_num_points_per_voxel, tensor_voxel_num, &workspace_size_));
  if (workspace_size_) {
    workspace_ptr = mlu_runtime_.allocate(workspace_size_);
  }
  workspace_.push_back(workspace_ptr);
  eva_->setMluWorkspaceSize(workspace_size_);
  VLOG(4) << "[VoxelizationExecutor] call workspaceMalloc() End.";
}

void VoxelizationExecutor::compute() {
  VLOG(4) << "[VoxelizationExecutor] call compute() Begin.";
  // get params
  int32_t max_points =
      parser_->getProtoNode()->voxelization_param().max_points();
  int32_t max_voxels =
      parser_->getProtoNode()->voxelization_param().max_voxels();
  int32_t NDim = parser_->getProtoNode()->voxelization_param().ndim();
  bool deterministic =
      parser_->getProtoNode()->voxelization_param().deterministic();

  auto tensor_points = tensor_desc_[0].tensor;
  auto tensor_voxel_size = tensor_desc_[1].tensor;
  auto tensor_coors_range = tensor_desc_[2].tensor;
  auto tensor_voxels = tensor_desc_[3].tensor;
  auto tensor_coors = tensor_desc_[4].tensor;
  auto tensor_num_points_per_voxel = tensor_desc_[5].tensor;
  auto tensor_voxel_num = tensor_desc_[6].tensor;

  auto dev_points = data_vector_[0].device_ptr;
  auto dev_voxel_size = data_vector_[1].device_ptr;
  auto dev_coors_range = data_vector_[2].device_ptr;
  auto dev_voxels = data_vector_[3].device_ptr;
  auto dev_coors = data_vector_[4].device_ptr;
  auto dev_num_points_per_voxel = data_vector_[5].device_ptr;
  auto dev_voxel_num = data_vector_[6].device_ptr;

  interface_timer_.start();
  MLUOP_CHECK(mluOpVoxelization(
      handle_, tensor_points, dev_points, tensor_voxel_size, dev_voxel_size,
      tensor_coors_range, dev_coors_range, max_points, max_voxels, NDim,
      deterministic, workspace_[0], workspace_size_, tensor_voxels, dev_voxels,
      tensor_coors, dev_coors, tensor_num_points_per_voxel,
      dev_num_points_per_voxel, tensor_voxel_num, dev_voxel_num));
  interface_timer_.stop();
  VLOG(4) << "[VoxelizationExecutor] call compute() End.";
}

void VoxelizationExecutor::workspaceFree() {
  if (workspace_[0]) {
    VLOG(4) << "[VoxelizationExecutor] Free device workspace space.";
    mlu_runtime_.deallocate(workspace_[0]);
    workspace_[0] = nullptr;
  }
}

void dynamicVoxelize(const float *points, int32_t *coors, const float voxel_x,
                     const float voxel_y, const float voxel_z,
                     const float coors_x_min, const float coors_y_min,
                     const float coors_z_min, const float coors_x_max,
                     const float coors_y_max, const float coors_z_max,
                     const int32_t grid_x, const int32_t grid_y,
                     const int32_t grid_z, const size_t num_points,
                     const size_t num_features, const size_t NDim) {
  for (size_t index = 0; index < num_points; ++index) {
    const float *points_offset = points + index * num_features;
    int32_t *coors_offset = coors + index * NDim;
    int32_t c_x = floorf((points_offset[0] - coors_x_min) / voxel_x);
    if (c_x < 0 || c_x >= grid_x) {
      coors_offset[0] = -1;
      continue;
    }

    int32_t c_y = floorf((points_offset[1] - coors_y_min) / voxel_y);
    if (c_y < 0 || c_y >= grid_y) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      continue;
    }

    int32_t c_z = floorf((points_offset[2] - coors_z_min) / voxel_z);
    if (c_z < 0 || c_z >= grid_z) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      coors_offset[2] = -1;
    } else {
      coors_offset[0] = c_z;
      coors_offset[1] = c_y;
      coors_offset[2] = c_x;
    }
  }
}

void pointToVoxelidx(const int32_t *coor, int32_t *point_to_voxelidx,
                     int32_t *point_to_pointidx, const int32_t max_points,
                     const int32_t max_voxels, const size_t num_points,
                     const size_t NDim) {
  for (size_t index = 0; index < num_points; ++index) {
    const int32_t *coor_offset = coor + index * NDim;
    if (coor_offset[0] == -1) {
      point_to_pointidx[index] = -1;
      point_to_voxelidx[index] = -1;
      continue;
    }

    int32_t num = 0;
    int32_t coor_x = coor_offset[0];
    int32_t coor_y = coor_offset[1];
    int32_t coor_z = coor_offset[2];

    for (int i = 0; i < index; ++i) {
      auto prev_coor = coor + i * NDim;
      if (prev_coor[0] == -1) {
        continue;
      }

      if ((prev_coor[0] == coor_x) && (prev_coor[1] == coor_y) &&
          (prev_coor[2] == coor_z)) {
        num++;
        if (num == 1) {
          point_to_pointidx[index] = i;
        } else if (num >= max_points) {
          break;
        }
      }
    }
    if (num == 0) {
      point_to_pointidx[index] = index;
    }
    if (num < max_points) {
      point_to_voxelidx[index] = num;
    } else {
      point_to_voxelidx[index] = -1;
    }
  }
}

void determinVoxelNum(float *num_points_per_voxel, int32_t *point_to_voxelidx,
                      int32_t *point_to_pointidx, int32_t *coor_to_voxelidx,
                      float *voxel_num, const int32_t max_points,
                      const int32_t max_voxels, const size_t num_points) {
  for (size_t i = 0; i < num_points; ++i) {
    int point_pos_in_voxel = point_to_voxelidx[i];
    coor_to_voxelidx[i] = -1;

    if (point_pos_in_voxel == -1) {
      continue;
    } else if (point_pos_in_voxel == 0) {
      int voxelidx = voxel_num[0];
      if (voxel_num[0] >= max_voxels) continue;
      voxel_num[0] += 1;
      coor_to_voxelidx[i] = voxelidx;
      num_points_per_voxel[voxelidx] = 1;
    } else {
      int point_idx = point_to_pointidx[i];
      int voxelidx = coor_to_voxelidx[point_idx];
      if (voxelidx != -1) {
        coor_to_voxelidx[i] = voxelidx;
        num_points_per_voxel[voxelidx] += 1;
      }
    }
  }
}

void assignPointToVoxel(const float *points, int32_t *point_to_voxelidx,
                        int32_t *coor_to_voxelidx, float *voxels,
                        const int32_t max_points, const size_t num_features,
                        const size_t num_points, const size_t NDim) {
  for (size_t thread_idx = 0; thread_idx < num_points * num_features;
       ++thread_idx) {
    int32_t index = thread_idx / num_features;
    int32_t num = point_to_voxelidx[index];
    int32_t voxelidx = coor_to_voxelidx[index];
    if (num > -1 && voxelidx > -1) {
      float *voxels_offset =
          voxels + voxelidx * max_points * num_features + num * num_features;

      int32_t k = thread_idx % num_features;
      voxels_offset[k] = points[thread_idx];
    }
  }
}

void assignVoxelCoors(int32_t *temp_coors, int32_t *point_to_voxelidx,
                      int32_t *coor_to_voxelidx, float *coors,
                      const size_t num_points, const size_t NDim) {
  for (size_t thread_idx = 0; thread_idx < num_points * NDim; ++thread_idx) {
    int32_t index = thread_idx / NDim;
    int32_t num = point_to_voxelidx[index];
    int32_t voxelidx = coor_to_voxelidx[index];
    if (num == 0 && voxelidx > -1) {
      float *coors_offset = coors + voxelidx * NDim;
      int32_t k = thread_idx % NDim;
      coors_offset[k] = temp_coors[thread_idx];
    }
  }
}

void VoxelizationExecutor::deterministic_hard_voxelize(
    const float *points, const float *voxel_size, const float *coors_range,
    const int32_t num_points, const int32_t num_features,
    const int32_t max_points, const int32_t max_voxels, const int32_t NDim,
    float *voxels, float *coors, float *num_points_per_voxel,
    float *voxel_num) {
  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int32_t grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int32_t grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int32_t grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  int32_t count = num_points * NDim;
  int32_t *temp_coors =
      (int32_t *)cpu_runtime_.allocate(count * sizeof(int32_t));

  dynamicVoxelize(points, temp_coors, voxel_x, voxel_y, voxel_z, coors_x_min,
                  coors_y_min, coors_z_min, coors_x_max, coors_y_max,
                  coors_z_max, grid_x, grid_y, grid_z, num_points, num_features,
                  NDim);

  count = num_points;
  int32_t *point_to_pointidx =
      (int32_t *)cpu_runtime_.allocate(count * sizeof(int32_t));
  count = num_points;
  int32_t *point_to_voxelidx =
      (int32_t *)cpu_runtime_.allocate(count * sizeof(int32_t));

  pointToVoxelidx(temp_coors, point_to_voxelidx, point_to_pointidx, max_points,
                  max_voxels, num_points, NDim);

  count = num_points;
  int32_t *coor_to_voxelidx =
      (int32_t *)cpu_runtime_.allocate(count * sizeof(int32_t));

  *voxel_num = 0;
  determinVoxelNum(num_points_per_voxel, point_to_voxelidx, point_to_pointidx,
                   coor_to_voxelidx, voxel_num, max_points, max_voxels,
                   num_points);

  assignPointToVoxel(points, point_to_voxelidx, coor_to_voxelidx, voxels,
                     max_points, num_features, num_points, NDim);

  assignVoxelCoors(temp_coors, point_to_voxelidx, coor_to_voxelidx, coors,
                   num_points, NDim);

  cpu_runtime_.deallocate(temp_coors);
  temp_coors = nullptr;
  cpu_runtime_.deallocate(point_to_pointidx);
  point_to_pointidx = nullptr;
  cpu_runtime_.deallocate(point_to_voxelidx);
  point_to_voxelidx = nullptr;
  cpu_runtime_.deallocate(coor_to_voxelidx);
  coor_to_voxelidx = nullptr;
}

void VoxelizationExecutor::cpuCompute() {
  VLOG(4) << "[VoxelizationExecutor] call cpuCompute() Begin.";
  // get params
  int32_t max_points =
      parser_->getProtoNode()->voxelization_param().max_points();
  int32_t max_voxels =
      parser_->getProtoNode()->voxelization_param().max_voxels();
  size_t NDim = parser_->getProtoNode()->voxelization_param().ndim();
  bool deterministic =
      parser_->getProtoNode()->voxelization_param().deterministic();

  auto tensor_points = tensor_desc_[0].tensor;
  size_t num_points = tensor_points->dims[0];
  size_t num_features = tensor_points->dims[1];

  float *points = cpu_fp32_input_[0];
  float *voxel_size = cpu_fp32_input_[1];
  float *coors_range = cpu_fp32_input_[2];
  float *voxels = cpu_fp32_output_[0];
  float *coors = cpu_fp32_output_[1];
  float *num_points_per_voxel = cpu_fp32_output_[2];
  float *voxel_num = cpu_fp32_output_[3];

  if (deterministic == true) {
    deterministic_hard_voxelize(points, voxel_size, coors_range, num_points,
                                num_features, max_points, max_voxels, NDim,
                                voxels, coors, num_points_per_voxel, voxel_num);
  } else {
    VLOG(4) << "[VoxelizationExecutor] non-deterministic not supported!";
  }

  VLOG(4) << "[VoxelizationExecutor] call cpuCompute() End.";
}

int64_t VoxelizationExecutor::getTheoryIoSize() {
  // get params
  int32_t max_points =
      parser_->getProtoNode()->voxelization_param().max_points();
  int32_t max_voxels =
      parser_->getProtoNode()->voxelization_param().max_voxels();

  auto tensor_points = tensor_desc_[0].tensor;
  size_t num_points = tensor_points->dims[0];
  size_t num_features = tensor_points->dims[1];

  int64_t total_size = 0;
  // mluOpUnionKernelDynamicVoxelize
  total_size += num_points * 3 * sizeof(float) + 3 * sizeof(float) +
                6 * sizeof(float) + num_points * 3 * sizeof(int32_t);

  // mluOpUnionKernelPoint2Voxel
  total_size += num_points * 3 * sizeof(int32_t) +
                num_points * sizeof(int32_t) + num_points * sizeof(int32_t);

  // mluOpUnionKernelCalcPointsPerVoxel
  total_size += num_points * 3 * sizeof(int32_t) +
                max_voxels * sizeof(int32_t) + sizeof(int32_t);

  // mluOpUnionKernelAssignVoxelsCoors
  total_size += num_points * num_features * sizeof(float) +
                num_points * 3 * sizeof(int32_t) +
                num_points * 2 * sizeof(int32_t) +
                max_voxels * max_points * num_features * sizeof(float) +
                max_voxels * 3 * sizeof(int32_t);

  return total_size;
}

int64_t VoxelizationExecutor::getTheoryOps() {
  auto tensor_points = tensor_desc_[0].tensor;
  size_t num_points = tensor_points->dims[0];

  int64_t theory_ops = 0;
  int32_t cp_count = 31;
  // mluOpUnionKernelDynamicVoxelize
  int32_t split_num = 9;
  int32_t max_nram_size = 384 * 1024;
  int32_t deal_num =
      FLOOR_ALIGN(max_nram_size / split_num / sizeof(int32_t), NFU_ALIGN_SIZE);
  int32_t repeat = (num_points + deal_num - 1) / deal_num;
  theory_ops += repeat * cp_count;

  // mluOpUnionKernelPoint2Voxel
  cp_count = 7;
  for (int32_t point_idx = 0; point_idx < num_points; ++point_idx) {
    repeat = (point_idx + deal_num - 1) / deal_num;
    theory_ops += repeat * cp_count;
  }

  return theory_ops;
}

}  // namespace mluoptest
