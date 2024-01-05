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
#include "roipoint_pool3d.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace mluoptest {

void lidarToLocalCoords(float shift_x, float shift_y, float rz, float &local_x,
                        float &local_y) {
  float cosa = std::cos(-rz), sina = std::sin(-rz);
  local_x = shift_x * cosa + shift_y * (-sina);
  local_y = shift_x * sina + shift_y * cosa;
}

int checkPointInBox3d(const float *pt, const float *box3d, float &local_x,
                      float &local_y) {
  // param pt: (x, y, z)
  // param box3d: (cx, cy, cz, dx, dy, dz, rz) in LiDAR coordinate, cz in the
  // bottom center
  float x = pt[0], y = pt[1], z = pt[2];
  float cx = box3d[0], cy = box3d[1], cz = box3d[2];
  float dx = box3d[3], dy = box3d[4], dz = box3d[5], rz = box3d[6];
  // shift to the center since cz in box3d is the bottom center
  cz += dz / 2.0;

  if ((z - cz) > (dz / 2.0) || (z - cz) < -(dz / 2.0)) {
    return 0;
  }
  lidarToLocalCoords(x - cx, y - cy, rz, local_x, local_y);
  int in_flag = (local_x > -dx / 2.0) & (local_x < dx / 2.0) &
                (local_y > -dy / 2.0) & (local_y < dy / 2.0);
  return in_flag;
}

void assignPointsToBox3d(int batch_size, int pts_num, int boxes_num,
                         const float *xyz, const float *boxes3d,
                         int *pts_assign) {
  // params xyz: (B, N, 3)
  // params boxes3d: (B, M, 7)
  // params pts_assign: (B, N, M): idx of the corresponding box3d, -1 means
  // background points
  for (int bs_idx = 0; bs_idx < batch_size; bs_idx++) {
    for (int box_idx = 0; box_idx < boxes_num; box_idx++) {
      for (int pt_idx = 0; pt_idx < pts_num; pt_idx++) {
        int assign_idx =
            bs_idx * pts_num * boxes_num + pt_idx * boxes_num + box_idx;
        pts_assign[assign_idx] = 0;

        int box_offset = bs_idx * boxes_num * 7 + box_idx * 7;
        int pt_offset = bs_idx * pts_num * 3 + pt_idx * 3;
        float local_x = 0, local_y = 0;
        int cur_in_flag = checkPointInBox3d(
            xyz + pt_offset, boxes3d + box_offset, local_x, local_y);
        pts_assign[assign_idx] = cur_in_flag;
      }
    }
  }
}

void getPooledIdx(int batch_size, int pts_num, int boxes_num,
                  int sampled_pts_num, const int *pts_assign, int *pts_idx,
                  float *pooled_empty_flag) {
  // params xyz: (B, N, 3)
  // params pts_feature: (B, N, C)
  // params pts_assign: (B, N, M)
  // params pts_idx: (B, M, sampled_pts_num)
  // params pooled_empty_flag: (B, M)
  for (int bs_idx = 0; bs_idx < batch_size; bs_idx++) {
    for (int box_idx = 0; box_idx < boxes_num; box_idx++) {
      int cnt = 0;
      for (int pt_idx = 0; pt_idx < pts_num; pt_idx++) {
        if (pts_assign[bs_idx * pts_num * boxes_num + pt_idx * boxes_num +
                       box_idx]) {
          if (cnt < sampled_pts_num) {
            pts_idx[bs_idx * boxes_num * sampled_pts_num +
                    box_idx * sampled_pts_num + cnt] = pt_idx;
            cnt++;
          } else {
            break;
          }
        }
      }

      if (cnt == 0) {
        pooled_empty_flag[bs_idx * boxes_num + box_idx] = 1;
      } else if (cnt < sampled_pts_num) {
        // duplicate same points for sampling
        for (int pt_idx = cnt; pt_idx < sampled_pts_num; pt_idx++) {
          int duplicate_idx = pt_idx % cnt;
          int base_offset =
              bs_idx * boxes_num * sampled_pts_num + box_idx * sampled_pts_num;
          pts_idx[base_offset + pt_idx] = pts_idx[base_offset + duplicate_idx];
        }
      }
    }
  }
}

void roipointPool3dForward(int batch_size, int pts_num, int boxes_num,
                           int feature_in_len, int sampled_pts_num,
                           const float *xyz, const int *pts_idx,
                           const float *pts_feature, float *pooled_features,
                           float *pooled_empty_flag) {
  // params xyz: (B, N, 3)
  // params pts_idx: (B, M, sampled_pts_num)
  // params pts_feature: (B, N, C)
  // params pooled_features: (B, M, sampled_pts_num, 3+C)
  // params pooled_empty_flag: (B, M)
  for (int bs_idx = 0; bs_idx < batch_size; bs_idx++) {
    for (int box_idx = 0; box_idx < boxes_num; box_idx++) {
      if (pooled_empty_flag[bs_idx * boxes_num + box_idx] == 1) {
        continue;
      }

      for (int sample_pt_idx = 0; sample_pt_idx < sampled_pts_num;
           sample_pt_idx++) {
        int temp_idx = bs_idx * boxes_num * sampled_pts_num +
                       box_idx * sampled_pts_num + sample_pt_idx;
        int src_pt_idx = pts_idx[temp_idx];
        int dst_feature_offset = temp_idx * (3 + feature_in_len);

        for (int j = 0; j < 3; j++) {
          pooled_features[dst_feature_offset + j] =
              xyz[bs_idx * pts_num * 3 + src_pt_idx * 3 + j];
        }

        int src_feature_offset =
            bs_idx * pts_num * feature_in_len + src_pt_idx * feature_in_len;
        memcpy(pooled_features + dst_feature_offset + 3,
               pts_feature + src_feature_offset,
               feature_in_len * sizeof(float));
      }
    }
  }
}

void cpuRoiPointPool3d(int batch_size, int pts_num, int boxes_num,
                       int feature_len, int sampled_pts_num, float *points,
                       float *point_features, float *boxes3d, int *pts_assign,
                       int *pts_idx, float *pooled_features,
                       float *pooled_empty_flag) {
  assignPointsToBox3d(batch_size, pts_num, boxes_num, points, boxes3d,
                      pts_assign);
  getPooledIdx(batch_size, pts_num, boxes_num, sampled_pts_num, pts_assign,
               pts_idx, pooled_empty_flag);
  roipointPool3dForward(batch_size, pts_num, boxes_num, feature_len,
                        sampled_pts_num, points, pts_idx, point_features,
                        pooled_features, pooled_empty_flag);
  return;
}

void RoipointPool3dExecutor::paramCheck() {
  GTEST_CHECK(parser_->getInputNum() == 3);
  GTEST_CHECK(parser_->getOutputNum() == 2);
  if (!parser_->getProtoNode()->has_roipoint_pool3d_param()) {
    LOG(ERROR) << "[GTEST_ROIPOINT_POOL3D] Missing roipoint_pool3d param.";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
}

void RoipointPool3dExecutor::workspaceMalloc() {
  VLOG(4) << "RoipointPool3dExecutor::workspaceMalloc() Begin.";
  auto tensor_points = tensor_desc_[0].tensor;
  auto tensor_point_features = tensor_desc_[1].tensor;
  auto tensor_boxes3d = tensor_desc_[2].tensor;
  auto tensor_pooled_features = tensor_desc_[3].tensor;
  auto tensor_pooled_empty_flag = tensor_desc_[4].tensor;

  int batch_size = tensor_points->dims[0];
  int pts_num = tensor_points->dims[1];
  int boxes_num = tensor_boxes3d->dims[1];
  int feature_len = tensor_point_features->dims[2];
  int sampled_pts_num = tensor_pooled_features->dims[2];

  void *workspace_ptr = nullptr;
  MLUOP_CHECK(mluOpGetRoiPointPool3dWorkspaceSize(
      handle_, batch_size, pts_num, boxes_num, feature_len, sampled_pts_num,
      tensor_points, tensor_point_features, tensor_boxes3d,
      tensor_pooled_features, tensor_pooled_empty_flag, &workspace_size));
  if (workspace_size) {
    workspace_ptr = mlu_runtime_.allocate(workspace_size);
  }
  workspace_.push_back(workspace_ptr);

  eva_->setMluWorkspaceSize(workspace_size);
  VLOG(4) << "RoipointPool3dExecutor::workspaceMalloc() End.";
}

void RoipointPool3dExecutor::compute() {
  VLOG(4) << "RoipointPool3dExecutor::compute() Begin.";
  // get params
  auto param_proto_desc = parser_->getProtoNode()->roipoint_pool3d_param();
  int sampled_pts_num = param_proto_desc.num_sampled_points();

  auto tensor_points = tensor_desc_[0].tensor;
  auto tensor_point_features = tensor_desc_[1].tensor;
  auto tensor_boxes3d = tensor_desc_[2].tensor;
  auto tensor_pooled_features = tensor_desc_[3].tensor;
  auto tensor_pooled_empty_flag = tensor_desc_[4].tensor;

  int batch_size = tensor_points->dims[0];
  int pts_num = tensor_points->dims[1];
  int boxes_num = tensor_boxes3d->dims[1];
  int feature_len = tensor_point_features->dims[2];

  auto dev_points = data_vector_[0].device_ptr;
  auto dev_point_features = data_vector_[1].device_ptr;
  auto dev_boxes3d = data_vector_[2].device_ptr;
  auto dev_pooled_features = data_vector_[3].device_ptr;
  auto dev_pooled_empty_flag = data_vector_[4].device_ptr;

  interface_timer_.start();
  MLUOP_CHECK(mluOpRoiPointPool3d(
      handle_, batch_size, pts_num, boxes_num, feature_len, sampled_pts_num,
      tensor_points, dev_points, tensor_point_features, dev_point_features,
      tensor_boxes3d, dev_boxes3d, workspace_[0], workspace_size,
      tensor_pooled_features, dev_pooled_features, tensor_pooled_empty_flag,
      dev_pooled_empty_flag));
  interface_timer_.stop();
  VLOG(4) << "RoipointPool3dExecutor::compute() End.";
}

void RoipointPool3dExecutor::cpuCompute() {
  VLOG(4) << "RoipointPool3dExecutor::cpuCompute() Begin.";
  auto tensor_points = tensor_desc_[0].tensor;
  auto tensor_point_features = tensor_desc_[1].tensor;
  auto tensor_boxes3d = tensor_desc_[2].tensor;
  auto tensor_pooled_features = tensor_desc_[3].tensor;
  auto tensor_pooled_empty_flag = tensor_desc_[4].tensor;

  int batch_size = tensor_points->dims[0];
  int pts_num = tensor_points->dims[1];
  int boxes_num = tensor_boxes3d->dims[1];
  int feature_len = tensor_point_features->dims[2];
  int sampled_pts_num = tensor_pooled_features->dims[2];

  auto points = cpu_fp32_input_[0];
  auto point_features = cpu_fp32_input_[1];
  auto boxes3d = cpu_fp32_input_[2];
  auto pooled_features = cpu_fp32_output_[0];
  auto pooled_empty_flag = cpu_fp32_output_[1];

  // params pts_assign: (B, N, M)
  int count = batch_size * pts_num * boxes_num;
  int *pts_assign = (int *)cpu_runtime_.allocate(new int[count]);

  // params pts_idx: (B, M ,sampled_pts_num)
  count = batch_size * boxes_num * sampled_pts_num;
  int *pts_idx = (int *)cpu_runtime_.allocate(new int[count]);

  cpuRoiPointPool3d(batch_size, pts_num, boxes_num, feature_len,
                    sampled_pts_num, points, point_features, boxes3d,
                    pts_assign, pts_idx, pooled_features, pooled_empty_flag);

  cpu_runtime_.deallocate(pts_assign);
  pts_assign = nullptr;
  cpu_runtime_.deallocate(pts_idx);
  pts_idx = nullptr;
  VLOG(4) << "RoipointPool3dExecutor::cpuCompute() End.";
}

void RoipointPool3dExecutor::workspaceFree() {
  if (workspace_[0]) {
    VLOG(4) << "Free device workspace space.";
    mlu_runtime_.deallocate(workspace_[0]);
    workspace_[0] = nullptr;
  }
}

int64_t RoipointPool3dExecutor::getTheoryOps() {
  auto tensor_points = tensor_desc_[0].tensor;
  auto tensor_point_features = tensor_desc_[1].tensor;
  auto tensor_boxes3d = tensor_desc_[2].tensor;
  auto tensor_pooled_features = tensor_desc_[3].tensor;
  auto tensor_pooled_empty_flag = tensor_desc_[4].tensor;

  int64_t batch_size = tensor_points->dims[0];
  int64_t pts_num = tensor_points->dims[1];
  int64_t boxes_num = tensor_boxes3d->dims[1];
  int64_t feature_len = tensor_point_features->dims[2];
  int64_t sampled_pts_num = tensor_pooled_features->dims[2];

  int64_t count = 21 + feature_len;
  int64_t theory_ops = batch_size * pts_num * count * boxes_num;
  return theory_ops;
}

}  // namespace mluoptest
