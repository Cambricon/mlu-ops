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
#include "roiaware_pool3d_backward.h"

#include <algorithm>
#include <string>

#include "mlu_op.h"

namespace mluoptest {

static void roiawareMaxPool3dBackward(const int boxes_num, const int out_x,
                                      const int out_y, const int out_z,
                                      const int channels, const int *argmax,
                                      const float *grad_out, float *grad_in) {
  // argmax: (boxes_num, out_x, out_y, out_z, channels)
  // grad_out: (boxes_num, out_x, out_y, out_z, channels)
  // grad_in: (pts_num, channels)
  for (int box_idx = 0; box_idx < boxes_num; box_idx++) {
    const int *argmax_cur_box =
        argmax + box_idx * out_x * out_y * out_z * channels;
    const float *grad_out_cur_box =
        grad_out + box_idx * out_x * out_y * out_z * channels;
    for (int voxel_idx = 0; voxel_idx < out_x * out_y * out_z; voxel_idx++) {
      const int *argmax_cur_voxel = argmax_cur_box + voxel_idx * channels;
      const float *grad_out_cur_voxel = grad_out_cur_box + voxel_idx * channels;
      for (int channel_idx = 0; channel_idx < channels; channel_idx++) {
        const int *argmax_cur_channel = argmax_cur_voxel + channel_idx;
        const float *grad_out_cur_channel = grad_out_cur_voxel + channel_idx;
        if (argmax_cur_channel[0] != -1) {
          float *grad_in_cur_channel =
              grad_in + argmax_cur_channel[0] * channels + channel_idx;
          grad_in_cur_channel[0] += grad_out_cur_channel[0] * 1;
        }
      }
    }
  }
}

static void roiawareAvgPool3dBackward(const int boxes_num, const int out_x,
                                      const int out_y, const int out_z,
                                      const int channels,
                                      const int max_pts_each_voxel,
                                      const int *pts_idx_of_voxels,
                                      const float *grad_out, float *grad_in) {
  // pts_idx_of_voxels: (boxes_num, out_x, out_y, out_z, max_pts_each_voxel)
  // grad_out: (boxes_num, out_x, out_y, out_z, channels)
  // grad_in: (pts_num, channels)
  for (int box_idx = 0; box_idx < boxes_num; box_idx++) {
    const int *pts_idx_of_voxels_cur_box =
        pts_idx_of_voxels +
        box_idx * out_x * out_y * out_z * max_pts_each_voxel;
    const float *grad_out_cur_box =
        grad_out + box_idx * out_x * out_y * out_z * channels;
    for (int voxel_idx = 0; voxel_idx < out_x * out_y * out_z; voxel_idx++) {
      const int *pts_idx_of_voxels_cur_voxel =
          pts_idx_of_voxels_cur_box + voxel_idx * max_pts_each_voxel;
      const float *grad_out_cur_voxel = grad_out_cur_box + voxel_idx * channels;
      int total_pts = pts_idx_of_voxels_cur_voxel[0];
      if (total_pts <= 0) continue;
      float cur_grad = 1.0f / float(total_pts);
      for (int channel_idx = 0; channel_idx < channels; channel_idx++) {
        const float *grad_out_cur_channel = grad_out_cur_voxel + channel_idx;
        for (int k = 1; k <= total_pts; k++) {
          float *grad_in_cur_channel =
              grad_in + pts_idx_of_voxels_cur_voxel[k] * channels + channel_idx;
          grad_in_cur_channel[0] += grad_out_cur_channel[0] * cur_grad;
        }
      }
    }
  }
}

void cpuRoiawarePool3dBackward(const int pool_method, const int boxes_num,
                               const int out_x, const int out_y,
                               const int out_z, const int channels,
                               const int max_pts_each_voxel,
                               const int *pts_idx_of_voxels, const int *argmax,
                               const float *grad_out, float *grad_in) {
  if (pool_method == 0) {
    VLOG(4) << "[GTEST_ROIAWARE_POOL3D_BACKWARD] roiawareMaxPool3dBackward() "
               "Begin.";
    roiawareMaxPool3dBackward(boxes_num, out_x, out_y, out_z, channels, argmax,
                              grad_out, grad_in);
    VLOG(4)
        << "[GTEST_ROIAWARE_POOL3D_BACKWARD] roiawareMaxPool3dBackward() done.";
  } else if (pool_method == 1) {
    VLOG(4) << "[GTEST_ROIAWARE_POOL3D_BACKWARD] roiawareAvgPool3dBackward() "
               "Begin.";
    roiawareAvgPool3dBackward(boxes_num, out_x, out_y, out_z, channels,
                              max_pts_each_voxel, pts_idx_of_voxels, grad_out,
                              grad_in);
    VLOG(4)
        << "[GTEST_ROIAWARE_POOL3D_BACKWARD] roiawareAvgPool3dBackward() done.";
  }
  return;
}

void RoiawarePool3dBackwardExecutor::printDataInfo() {
  VLOG(4) << "############################### printfDataInfo() Begin ##";
  VLOG(4) << "# pool_method:        " << pool_method_;
  VLOG(4) << "# boxes_num:          " << boxes_num_;
  VLOG(4) << "# out_x:              " << out_x_;
  VLOG(4) << "# out_y:              " << out_y_;
  VLOG(4) << "# out_z:              " << out_z_;
  VLOG(4) << "# channels:           " << channels_;
  VLOG(4) << "# max_pts_each_voxel: " << max_pts_each_voxel_;
  VLOG(4) << "# pts_num:            " << pts_num_;
  VLOG(4) << "############################### printfDataInfo() End ##";
}

void RoiawarePool3dBackwardExecutor::initData() {
  VLOG(4) << "RoiawarePool3dBackwardExecutor::initData() Begin.";
  // get params
  desc_pts_idx_of_voxels_ = tensor_desc_[0].tensor;
  desc_argmax_ = tensor_desc_[1].tensor;
  desc_grad_out_ = tensor_desc_[2].tensor;
  desc_grad_in_ = tensor_desc_[3].tensor;

  dev_pts_idx_of_voxels_ = data_vector_[0].device_ptr;
  dev_argmax_ = data_vector_[1].device_ptr;
  dev_grad_out_ = data_vector_[2].device_ptr;
  dev_grad_in_ = data_vector_[3].device_ptr;

  auto roiaware_pool3d_backward_proto_desc =
      parser_->getProtoNode()->roiaware_pool3d_backward_param();

  pool_method_ = roiaware_pool3d_backward_proto_desc.pool_method();
  boxes_num_ = roiaware_pool3d_backward_proto_desc.boxes_num();
  out_x_ = roiaware_pool3d_backward_proto_desc.out_x();
  out_y_ = roiaware_pool3d_backward_proto_desc.out_y();
  out_z_ = roiaware_pool3d_backward_proto_desc.out_z();
  channels_ = roiaware_pool3d_backward_proto_desc.channels();
  max_pts_each_voxel_ =
      roiaware_pool3d_backward_proto_desc.max_pts_each_voxel();

  pts_num_ = desc_grad_in_->dims[0];

  VLOG(4) << "RoiawarePool3dBackwardExecutor::initData() End.";
}

void RoiawarePool3dBackwardExecutor::paramCheck() {
  VLOG(4) << "RoiawarePool3dBackwardExecutor::paramCheck() Begin.";
  GTEST_CHECK(parser_->getInputNum() == 3);
  GTEST_CHECK(parser_->getOutputNum() == 1);
  if (!parser_->getProtoNode()->has_roiaware_pool3d_backward_param()) {
    LOG(ERROR) << "[GTEST_ROIAWARE_POOL3D_BACKWARD] Missing "
                  "roiaware_pool3d_backward param.";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
  VLOG(4) << "RoiawarePool3dBackwardExecutor::paramCheck() End.";
}

void RoiawarePool3dBackwardExecutor::compute() {
  VLOG(4) << "RoiawarePool3dBackwardExecutor::compute() Begin.";
  initData();
  printDataInfo();

  interface_timer_.start();

  MLUOP_CHECK(mluOpRoiawarePool3dBackward(
      handle_, pool_method_, boxes_num_, out_x_, out_y_, out_z_, channels_,
      max_pts_each_voxel_, desc_pts_idx_of_voxels_, dev_pts_idx_of_voxels_,
      desc_argmax_, dev_argmax_, desc_grad_out_, dev_grad_out_, desc_grad_in_,
      dev_grad_in_));

  interface_timer_.stop();
  VLOG(4) << "RoiawarePool3dBackwardExecutor::compute() End.";
}

void RoiawarePool3dBackwardExecutor::cpuCompute() {
  VLOG(4) << "RoiawarePool3dBackwardExecutor::cpuCompute() Begin.";

  int *pts_idx_of_voxels = (int *)(cpu_fp32_input_[0]);
  int *argmax = (int *)(cpu_fp32_input_[1]);
  float *grad_out = cpu_fp32_input_[2];
  float *grad_in = cpu_fp32_output_[0];

  // gtest input value need to be float, so float 2 int
  for (int i = 0;
       i < boxes_num_ * out_x_ * out_y_ * out_z_ * max_pts_each_voxel_; i++) {
    pts_idx_of_voxels[i] = (int)(((float *)pts_idx_of_voxels)[i]);
  }
  for (int i = 0; i < boxes_num_ * out_x_ * out_y_ * out_z_ * channels_; i++) {
    argmax[i] = (int)(((float *)argmax)[i]);
  }

  int grad_in_initial_value = 0;
  std::memset((void *)grad_in, grad_in_initial_value,
              pts_num_ * channels_ * sizeof(float));

  cpuRoiawarePool3dBackward(pool_method_, boxes_num_, out_x_, out_y_, out_z_,
                            channels_, max_pts_each_voxel_, pts_idx_of_voxels,
                            argmax, grad_out, grad_in);

  VLOG(4) << "RoiawarePool3dBackwardExecutor::cpuCompute() End.";
}

int64_t RoiawarePool3dBackwardExecutor::getTheoryOps() {
  int64_t theory_ops = 0;
  if (pool_method_ == 0) {
    theory_ops = boxes_num_ * out_x_ * out_y_ * out_z_ * channels_;
  } else {
    theory_ops = boxes_num_ * out_x_ * out_y_ * out_z_ * max_pts_each_voxel_ *
                 channels_ * 2;
  }
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
