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
#include "voxel_pooling_forward.h"

#include "mlu_op.h"

namespace mluoptest {
void VoxelPoolingForwardExecutor::paramCheck() {
  VLOG(4) << "VoxelPoolingForwardExecutor call paramCheck begin.";
  if (!parser_->getProtoNode()->has_voxel_pooling_forward_param()) {
    LOG(ERROR)
        << "VoxelPoolingForwardExecutor Lose voxel_pooling_forward_param.";
  }
  GTEST_CHECK(parser_->getInputNum() == 2);
  GTEST_CHECK(parser_->getOutputNum() == 2);
  for (int i = 0; i < parser_->getInputNum() - 1; i++) {
    GTEST_CHECK(!parser_->inputIsNull(i))
  }
  for (int i = 0; i < parser_->getOutputNum(); i++) {
    GTEST_CHECK(!parser_->outputIsNull(i))
  }

  VLOG(4) << "VoxelPoolingForwardExecutor call paramCheck end.";
}

void VoxelPoolingForwardExecutor::initData() {
  VLOG(4) << "VoxelPoolingForwardExecutor call initData begin.";
  auto voxel_pooling_forward_proto_param =
      parser_->getProtoNode()->voxel_pooling_forward_param();
  batch_size_ = voxel_pooling_forward_proto_param.batch_size();
  num_points_ = voxel_pooling_forward_proto_param.num_points();
  num_channels_ = voxel_pooling_forward_proto_param.num_channels();
  num_voxel_x_ = voxel_pooling_forward_proto_param.num_voxel_x();
  num_voxel_y_ = voxel_pooling_forward_proto_param.num_voxel_y();
  num_voxel_z_ = voxel_pooling_forward_proto_param.num_voxel_z();
  theory_io_size_ = 0;
  VLOG(4) << "VoxelPoolingForwardExecutor call initData end.";
}

void VoxelPoolingForwardExecutor::printDataInfo() {
  VLOG(4) << "################### printfDataInfo() Begin ##";
  VLOG(4) << "# batch_size:   " << batch_size_;
  VLOG(4) << "# num_points:   " << num_points_;
  VLOG(4) << "# num_channels: " << num_channels_;
  VLOG(4) << "# num_voxel_x:  " << num_voxel_x_;
  VLOG(4) << "# num_voxel_y:  " << num_voxel_y_;
  VLOG(4) << "# num_voxel_z:  " << num_voxel_z_;
  VLOG(4) << "################### printfDataInfo() End ##";
}

void VoxelPoolingForwardExecutor::compute() {
  VLOG(4) << "VoxelPoolingForwardExecutor call compute begin.";
  initData();
  printDataInfo();
  auto geom_xyz_desc = tensor_desc_[0].tensor;
  auto input_features_desc = tensor_desc_[1].tensor;
  auto output_features_desc = tensor_desc_[2].tensor;
  auto pos_memo_desc = tensor_desc_[3].tensor;

  auto geom_xyz_data_ptr = data_vector_[0].device_ptr;
  auto input_features_data_ptr = data_vector_[1].device_ptr;
  auto output_features_data_ptr = data_vector_[2].device_ptr;
  auto pos_memo_data_ptr = data_vector_[3].device_ptr;
  VLOG(4) << "VoxelPoolingForwardExecutor call mluOpFill().";
  float output_features_init_vaule = 0;
  MLUOP_CHECK(mluOpFill_v3(handle_, MLUOP_POINTER_MODE_HOST,
                        &output_features_init_vaule, output_features_desc,
                        output_features_data_ptr));
  int pos_memo_init_vaule = -1;
  MLUOP_CHECK(mluOpFill_v3(handle_, MLUOP_POINTER_MODE_HOST,
                           &pos_memo_init_vaule,
                           pos_memo_desc, pos_memo_data_ptr));

  VLOG(4) << "VoxelPoolingForwardExecutor call mluOpVoxelPoolingForward().";

  interface_timer_.start();
  MLUOP_CHECK(mluOpVoxelPoolingForward(
      handle_, batch_size_, num_points_, num_channels_, num_voxel_x_,
      num_voxel_y_, num_voxel_z_, geom_xyz_desc, geom_xyz_data_ptr,
      input_features_desc, input_features_data_ptr, output_features_desc,
      output_features_data_ptr, pos_memo_desc, pos_memo_data_ptr));
  interface_timer_.stop();
  VLOG(4) << "VoxelPoolingForwardExecutor call compute end.";
}

void VoxelPoolingForwardExecutor::voxelPoolingForwardCpuKernel(
    const int batch_size, const int num_points, const int num_channels,
    const int num_voxel_x, const int num_voxel_y, const int num_voxel_z,
    const int *geom_xyz, const float *input_features, float *output_features,
    int *pos_memo) {
  theory_io_size_ += batch_size * num_points * 3 * sizeof(int);

  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    for (int pt_idx = 0; pt_idx < num_points; ++pt_idx) {
      int x = geom_xyz[batch_idx * num_points * 3 + pt_idx * 3];
      int y = geom_xyz[batch_idx * num_points * 3 + pt_idx * 3 + 1];
      int z = geom_xyz[batch_idx * num_points * 3 + pt_idx * 3 + 2];

      // skip if coord of current voxel is out of boundary.
      if (x < 0 || x >= num_voxel_x || y < 0 || y >= num_voxel_y || z < 0 ||
          z >= num_voxel_z) {
        continue;
      }
      pos_memo[batch_idx * num_points * 3 + pt_idx * 3] = batch_idx;
      pos_memo[batch_idx * num_points * 3 + pt_idx * 3 + 1] = y;
      pos_memo[batch_idx * num_points * 3 + pt_idx * 3 + 2] = x;

      theory_io_size_ += 3 * sizeof(int);

      int in_offset = (batch_idx * num_points + pt_idx) * num_channels;
      int out_offset =
          (batch_idx * num_voxel_y * num_voxel_x + y * num_voxel_x + x) *
          num_channels;

      for (int c_idx = 0; c_idx < num_channels; ++c_idx) {
        output_features[out_offset + c_idx] +=
            input_features[in_offset + c_idx];
      }

      theory_io_size_ += num_channels * sizeof(float) * 3;
    }
  }
}

void VoxelPoolingForwardExecutor::cpuCompute() {
  VLOG(4) << "VoxelPoolingForwardExecutor call cpuCompute begin.";
  float *geom_xyz_fp = cpu_fp32_input_[0];
  float *input_features = cpu_fp32_input_[1];
  float *output_features = cpu_fp32_output_[0];
  float *pos_memo_fp = cpu_fp32_output_[1];

  int *geom_xyz = (int *)geom_xyz_fp;
  int *pos_memo = (int *)pos_memo_fp;

  int geom_xyz_size = batch_size_ * num_points_ * 3;
  int input_features_size = batch_size_ * num_points_ * num_channels_;
  int output_features_size =
      batch_size_ * num_voxel_y_ * num_voxel_x_ * num_channels_;
  int pos_memo_size = batch_size_ * num_points_ * 3;

  // gtest tensor value need to be float, so float to int
  for (int i = 0; i < pos_memo_size; ++i) {
    geom_xyz[i] = (int)(geom_xyz_fp[i]);
  }

  // initial output value
  int output_features_initial_value = 0;
  std::memset(output_features, output_features_initial_value,
              output_features_size * sizeof(float));
  int pos_memo_initial_value = -1;
  std::memset(pos_memo, pos_memo_initial_value, pos_memo_size * sizeof(int));

  voxelPoolingForwardCpuKernel(
      batch_size_, num_points_, num_channels_, num_voxel_x_, num_voxel_y_,
      num_voxel_z_, geom_xyz, input_features, output_features, pos_memo);

  // gtest tensor value need to be float, so int to float
  for (int i = 0; i < pos_memo_size; ++i) {
    pos_memo_fp[i] = (float)(pos_memo[i]);
  }

  VLOG(4) << "VoxelPoolingForwardExecutor call cpuCompute end.";
}

int64_t VoxelPoolingForwardExecutor::getTheoryOps() {
  int64_t theory_ops =
      parser_->getOutputDataCount(0) * 2 + parser_->getOutputDataCount(1);
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

int64_t VoxelPoolingForwardExecutor::getTheoryIoSize() {
  if (parser_->device() != CPU) {
    return -1;
  }
  VLOG(4) << "getTheoryIOs: " << theory_io_size_ << " Bytes";
  return theory_io_size_;
}
}  // namespace mluoptest
