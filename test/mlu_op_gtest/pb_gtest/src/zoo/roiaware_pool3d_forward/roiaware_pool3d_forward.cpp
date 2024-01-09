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
#include "roiaware_pool3d_forward.h"

#include <algorithm>
#include <string>

#include "mlu_op.h"

namespace mluoptest {

static void lidarToLocalCoords(const float shift_x, const float shift_y,
                               const float rz, float &local_x, float &local_y) {
  float cosa = cos(-rz), sina = sin(-rz);
  local_x = shift_x * cosa + shift_y * (-sina);
  local_y = shift_x * sina + shift_y * cosa;
}

static int checkPtInBox3d(const float *pt, const float *box3d, float &local_x,
                          float &local_y) {
  // pt: [x, y, z]
  // box3d: [cx, cy, cz, dx, dy, dz, rz] in LiDAR coordinate
  // cz in the bottom center
  float x = pt[0], y = pt[1], z = pt[2];
  float cx = box3d[0], cy = box3d[1], cz = box3d[2];
  float dx = box3d[3], dy = box3d[4], dz = box3d[5], rz = box3d[6];
  // shift to the center since cz in box3d is the bottom center
  cz += dz / 2.0;

  if (fabsf(z - cz) > dz / 2.0) return 0;

  lidarToLocalCoords(x - cx, y - cy, rz, local_x, local_y);
  int in_flag = (local_x > -dx / 2.0) & (local_x < dx / 2.0) &
                (local_y > -dy / 2.0) & (local_y < dy / 2.0);
  return in_flag;
}

static void collectInsidePtsForBox3d(const int boxes_num, const int pts_num,
                                     const int max_pts_each_voxel,
                                     const int out_x, const int out_y,
                                     const int out_z, const float *rois,
                                     const float *pts, int *pts_idx_of_voxels) {
  // rois: (boxes_num, 7) [x, y, z, x_size, y_size, z_size, rz]
  // pts: (pts_num, 3) [x, y, z]
  // pts_idx_of_voxels: (boxes_num, out_x, out_y, out_z, max_pts_each_voxel)

  const int max_num_pts = max_pts_each_voxel - 1;  // index 0 is the counter
  for (int box_idx = 0; box_idx < boxes_num; box_idx++) {
    const float *rois_cur_box = rois + box_idx * 7;
    int *pts_idx_of_voxels_cur_box = pts_idx_of_voxels + box_idx * out_x *
                                                             out_y * out_z *
                                                             max_pts_each_voxel;
    for (int pt_idx = 0; pt_idx < pts_num; pt_idx++) {
      const float *pts_cur_pts = pts + pt_idx * 3;
      float local_x = 0, local_y = 0;
      int cur_in_flag =
          checkPtInBox3d(pts_cur_pts, rois_cur_box, local_x, local_y);
      if (cur_in_flag > 0) {
        // cz=rois[2] in the bottom center
        float local_z = pts_cur_pts[2] - rois_cur_box[2];
        float x_size = rois_cur_box[3], y_size = rois_cur_box[4],
              z_size = rois_cur_box[5];

        float x_res = x_size / out_x;
        float y_res = y_size / out_y;
        float z_res = z_size / out_z;

        int x_idx = int((local_x + x_size / 2) / x_res);
        int y_idx = int((local_y + y_size / 2) / y_res);
        int z_idx = int(local_z / z_res);

        x_idx = std::min(std::max(x_idx, 0), out_x - 1);
        y_idx = std::min(std::max(y_idx, 0), out_y - 1);
        z_idx = std::min(std::max(z_idx, 0), out_z - 1);

        int base_offset = x_idx * out_y * out_z * max_pts_each_voxel +
                          y_idx * out_z * max_pts_each_voxel +
                          z_idx * max_pts_each_voxel;
        int cnt = pts_idx_of_voxels_cur_box[base_offset];
        if (cnt < max_num_pts) {
          pts_idx_of_voxels_cur_box[base_offset + cnt + 1] = pt_idx;
          pts_idx_of_voxels_cur_box[base_offset]++;
        }
      }
    }
  }
}

static void roiawareMaxPool3d(const int boxes_num, const int pts_num,
                              const int channels, const int max_pts_each_voxel,
                              const int out_x, const int out_y, const int out_z,
                              const float *pts_feature,
                              const int *pts_idx_of_voxels,
                              float *pooled_features, int *argmax) {
  // pts_feature: (pts_num, channels)
  // pts_idx_of_voxels: (boxes_num, out_x, out_y, out_z, max_pts_each_voxel)
  // pooled_features: (boxes_num, out_x, out_y, out_z, channels)
  // argmax: (boxes_num, out_x, out_y, out_z, channels)
  for (int box_idx = 0; box_idx < boxes_num; box_idx++) {
    const int *pts_idx_of_voxels_cur_box =
        pts_idx_of_voxels +
        box_idx * out_x * out_y * out_z * max_pts_each_voxel;
    float *pooled_features_cur_box =
        pooled_features + box_idx * out_x * out_y * out_z * channels;
    int *argmax_cur_box = argmax + box_idx * out_x * out_y * out_z * channels;
    for (int voxel_idx = 0; voxel_idx < out_x * out_y * out_z; voxel_idx++) {
      const int *pts_idx_of_voxels_cur_voxel =
          pts_idx_of_voxels_cur_box + voxel_idx * max_pts_each_voxel;
      float *pooled_features_cur_voxel =
          pooled_features_cur_box + voxel_idx * channels;
      int *argmax_cur_voxel = argmax_cur_box + voxel_idx * channels;
      for (int channel_idx = 0; channel_idx < channels; channel_idx++) {
        int argmax_idx = -1;
        float max_val = -1e50;
        int total_pts = pts_idx_of_voxels_cur_voxel[0];
        float pts_feature_cur_channel;
        for (int k = 1; k <= total_pts; k++) {
          pts_feature_cur_channel =
              pts_feature[pts_idx_of_voxels_cur_voxel[k] * channels +
                          channel_idx];
          if (pts_feature_cur_channel > max_val) {
            max_val = pts_feature_cur_channel;
            argmax_idx = pts_idx_of_voxels_cur_voxel[k];
          }
        }
        if (argmax_idx != -1) {
          pooled_features_cur_voxel[channel_idx] = max_val;
          argmax_cur_voxel[channel_idx] = argmax_idx;
        }
      }
    }
  }
}

static void roiawareAvgPool3d(const int boxes_num, const int pts_num,
                              const int channels, const int max_pts_each_voxel,
                              const int out_x, const int out_y, const int out_z,
                              const float *pts_feature,
                              const int *pts_idx_of_voxels,
                              float *pooled_features) {
  // pts_feature: (pts_num, channels)
  // pts_idx_of_voxels: (boxes_num, out_x, out_y, out_z, max_pts_each_voxel)
  // pooled_features: (boxes_num, out_x, out_y, out_z, channels)
  // argmax: (boxes_num, out_x, out_y, out_z, channels)
  for (int box_idx = 0; box_idx < boxes_num; box_idx++) {
    const int *pts_idx_of_voxels_cur_box =
        pts_idx_of_voxels +
        box_idx * out_x * out_y * out_z * max_pts_each_voxel;
    float *pooled_features_cur_box =
        pooled_features + box_idx * out_x * out_y * out_z * channels;
    for (int voxel_idx = 0; voxel_idx < out_x * out_y * out_z; voxel_idx++) {
      const int *pts_idx_of_voxels_cur_voxel =
          pts_idx_of_voxels_cur_box + voxel_idx * max_pts_each_voxel;
      float *pooled_features_cur_voxel =
          pooled_features_cur_box + voxel_idx * channels;
      for (int channel_idx = 0; channel_idx < channels; channel_idx++) {
        float sum_val = 0;
        int total_pts = pts_idx_of_voxels_cur_voxel[0];
        for (int k = 1; k <= total_pts; k++) {
          sum_val += pts_feature[pts_idx_of_voxels_cur_voxel[k] * channels +
                                 channel_idx];
        }
        if (total_pts > 0) {
          pooled_features_cur_voxel[channel_idx] = sum_val / total_pts;
        }
      }
    }
  }
}

void cpuRoiawarePool3dForward(const int boxes_num, const int pts_num,
                              const int channels, const int max_pts_each_voxel,
                              const int out_x, const int out_y, const int out_z,
                              const float *rois, const float *pts,
                              const float *pts_feature, int *argmax,
                              int *pts_idx_of_voxels, float *pooled_features,
                              const int pool_method) {
  VLOG(4)
      << "[GTEST_ROIAWARE_POOL3D_FORWARD] collectInsidePtsForBox3d() Begin.";
  collectInsidePtsForBox3d(boxes_num, pts_num, max_pts_each_voxel, out_x, out_y,
                           out_z, rois, pts, pts_idx_of_voxels);
  if (pool_method == 0) {
    VLOG(4) << "[GTEST_ROIAWARE_POOL3D_FORWARD] roiawareMaxPool3d() Begin.";
    roiawareMaxPool3d(boxes_num, pts_num, channels, max_pts_each_voxel, out_x,
                      out_y, out_z, pts_feature, pts_idx_of_voxels,
                      pooled_features, argmax);
  } else if (pool_method == 1) {
    VLOG(4) << "[GTEST_ROIAWARE_POOL3D_FORWARD] roiawareAvgPool3d() Begin.";
    roiawareAvgPool3d(boxes_num, pts_num, channels, max_pts_each_voxel, out_x,
                      out_y, out_z, pts_feature, pts_idx_of_voxels,
                      pooled_features);
  }
  return;
}

void RoiawarePool3dForwardExecutor::printDataInfo() {
  VLOG(4) << "############################### printfDataInfo() Begin ##";
  VLOG(4) << "# pool_method:        " << pool_method_;
  VLOG(4) << "# boxes_num:          " << boxes_num_;
  VLOG(4) << "# pts_num:            " << pts_num_;
  VLOG(4) << "# channels:           " << channels_;
  VLOG(4) << "# max_pts_each_voxel: " << max_pts_each_voxel_;
  VLOG(4) << "# out_x:              " << out_x_;
  VLOG(4) << "# out_y:              " << out_y_;
  VLOG(4) << "# out_z:              " << out_z_;
  VLOG(4) << "############################### printfDataInfo() End ##";
}

void RoiawarePool3dForwardExecutor::initData() {
  VLOG(4) << "RoiawarePool3dForwardExecutor::initData() Begin.";
  // get params
  desc_rois_ = tensor_desc_[0].tensor;
  desc_pts_ = tensor_desc_[1].tensor;
  desc_pts_feature_ = tensor_desc_[2].tensor;
  desc_argmax_ = tensor_desc_[3].tensor;
  desc_pts_idx_of_voxels_ = tensor_desc_[4].tensor;
  desc_pooled_features_ = tensor_desc_[5].tensor;

  dev_rois_ = data_vector_[0].device_ptr;
  dev_pts_ = data_vector_[1].device_ptr;
  dev_pts_feature_ = data_vector_[2].device_ptr;
  dev_argmax_ = data_vector_[3].device_ptr;
  dev_pts_idx_of_voxels_ = data_vector_[4].device_ptr;
  dev_pooled_features_ = data_vector_[5].device_ptr;

  auto roiaware_pool3d_forward_proto_desc =
      parser_->getProtoNode()->roiaware_pool3d_forward_param();
  pool_method_ = roiaware_pool3d_forward_proto_desc.pool_method();

  boxes_num_ = roiaware_pool3d_forward_proto_desc.boxes_num();
  pts_num_ = roiaware_pool3d_forward_proto_desc.pts_num();
  channels_ = roiaware_pool3d_forward_proto_desc.channels();
  max_pts_each_voxel_ = roiaware_pool3d_forward_proto_desc.max_pts_each_voxel();
  out_x_ = roiaware_pool3d_forward_proto_desc.out_x();
  out_y_ = roiaware_pool3d_forward_proto_desc.out_y();
  out_z_ = roiaware_pool3d_forward_proto_desc.out_z();
  VLOG(4) << "RoiawarePool3dForwardExecutor::initData() End.";
}

void RoiawarePool3dForwardExecutor::paramCheck() {
  VLOG(4) << "RoiawarePool3dForwardExecutor::paramCheck() Begin.";
  GTEST_CHECK(parser_->getInputNum() == 3);
  GTEST_CHECK(parser_->getOutputNum() == 3);
  GTEST_CHECK(parser_->getProtoNode()->has_roiaware_pool3d_forward_param(),
              "RoiawarePool3dForwardExecutor::paramCheck() "
              "roiaware_pool3d_forward param not found!");
  VLOG(4) << "RoiawarePool3dForwardExecutor::paramCheck() End.";
}

void RoiawarePool3dForwardExecutor::workspaceMalloc() {
  void *tmp = nullptr;
  // allocate extra space when broadcast
  MLUOP_CHECK(mluOpGetRoiawarePool3dForwardWorkspaceSize(
      handle_, tensor_desc_[0].tensor, tensor_desc_[1].tensor,
      tensor_desc_[2].tensor, &workspace_size_));
  if (workspace_size_ > 0) {
    VLOG(4) << "[GTEST_ROIAWARE_POOL3D_FORWARD] Need malloc workspace space.";
    tmp = mlu_runtime_.allocate(workspace_size_);
    VLOG(4) << "[GTEST_ROIAWARE_POOL3D_FORWARD] Malloc addr: " << tmp
            << ", size: " << workspace_size_ << " bytes";
  } else {
    VLOG(4) << "[GTEST_ROIAWARE_POOL3D_FORWARD] Don't need to Malloc workspace "
               "space.";
  }
  workspace_.push_back(tmp);
  eva_->setMluWorkspaceSize(workspace_size_);
}

void RoiawarePool3dForwardExecutor::workspaceFree() {
  if (workspace_[0]) {
    VLOG(4) << "[GTEST_ROIAWARE_POOL3D_FORWARD] Free device workspace space.";
    mlu_runtime_.deallocate(workspace_[0]);
    workspace_[0] = nullptr;
  }
}

void RoiawarePool3dForwardExecutor::compute() {
  VLOG(4) << "RoiawarePool3dForwardExecutor::compute() Begin.";
  initData();
  printDataInfo();

  interface_timer_.start();
  MLUOP_CHECK(mluOpRoiawarePool3dForward(
      handle_, pool_method_, boxes_num_, pts_num_, channels_, desc_rois_,
      dev_rois_, desc_pts_, dev_pts_, desc_pts_feature_, dev_pts_feature_,
      workspace_[0], workspace_size_, max_pts_each_voxel_, out_x_, out_y_,
      out_z_, desc_argmax_, dev_argmax_, desc_pts_idx_of_voxels_,
      dev_pts_idx_of_voxels_, desc_pooled_features_, dev_pooled_features_));
  interface_timer_.stop();
  VLOG(4) << "RoiawarePool3dForwardExecutor::compute() End.";
}

void RoiawarePool3dForwardExecutor::cpuCompute() {
  VLOG(4) << "RoiawarePool3dForwardExecutor::cpuCompute() Begin.";
  float *rois = cpu_fp32_input_[0];
  float *pts = cpu_fp32_input_[1];
  float *pts_feature = cpu_fp32_input_[2];
  float *argmax = cpu_fp32_output_[0];
  float *pts_idx_of_voxels = cpu_fp32_output_[1];
  float *pooled_features = cpu_fp32_output_[2];
  int pts_idx_initial_value = 0;
  int pooled_features_initial_value = 0;
  int argmax_initial_value = (pool_method_ == 0) ? -1 : 0;
  std::memset((void *)pts_idx_of_voxels, pts_idx_initial_value,
              boxes_num_ * out_x_ * out_y_ * out_z_ * max_pts_each_voxel_ *
                  sizeof(int));
  std::memset(
      (void *)pooled_features, pooled_features_initial_value,
      boxes_num_ * out_x_ * out_y_ * out_z_ * channels_ * sizeof(float));
  std::memset((void *)argmax, argmax_initial_value,
              boxes_num_ * out_x_ * out_y_ * out_z_ * channels_ * sizeof(int));

  cpuRoiawarePool3dForward(boxes_num_, pts_num_, channels_, max_pts_each_voxel_,
                           out_x_, out_y_, out_z_, rois, pts, pts_feature,
                           (int *)argmax, (int *)pts_idx_of_voxels,
                           pooled_features, pool_method_);

  // gtest ouput value need to be float, so int 2 float
  for (int i = 0; i < boxes_num_ * out_x_ * out_y_ * out_z_ * channels_; i++) {
    argmax[i] = (float)(((int *)argmax)[i]);
  }
  for (int i = 0;
       i < boxes_num_ * out_x_ * out_y_ * out_z_ * max_pts_each_voxel_; i++) {
    pts_idx_of_voxels[i] = (float)(((int *)pts_idx_of_voxels)[i]);
  }
  VLOG(4) << "RoiawarePool3dForwardExecutor::cpuCompute() End.";
}

int64_t RoiawarePool3dForwardExecutor::getTheoryOps() {
  int64_t theory_ops =
      boxes_num_ * 7 * pts_num_ * 3 +
      boxes_num_ * out_x_ * out_y_ * out_z_ * max_pts_each_voxel_ * channels_;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
