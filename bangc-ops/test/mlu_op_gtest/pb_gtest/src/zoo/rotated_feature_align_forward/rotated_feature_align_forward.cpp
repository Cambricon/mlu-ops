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
#include "rotated_feature_align_forward.h"

namespace mluoptest {
void RotatedFeatureAlignForwardExecutor::initData() {
  // get params
  spatial_scale_ = parser_->getProtoNode()
                       ->rotated_feature_align_forward_param()
                       .spatial_scale();
  points_ =
      parser_->getProtoNode()->rotated_feature_align_forward_param().points();
}

void RotatedFeatureAlignForwardExecutor::paramCheck() {
  VLOG(4) << "[RotatedFeatureAlignForwardExecutor] Param check.";
  GTEST_CHECK(parser_->getInputNum() == 2,
              "[RotatedFeatureAlignForwardExecutor] Input number is wrong.");
  GTEST_CHECK(parser_->getOutputNum() == 1,
              "[RotatedFeatureAlignForwardExecutor] Output number is wrong.");
  GTEST_CHECK(
      parser_->getProtoNode()->has_rotated_feature_align_forward_param(),
      "[RotatedFeatureAlignForwardExecutor] Missing param.");
}

void RotatedFeatureAlignForwardExecutor::compute() {
  // init param
  initData();

  auto input_desc = tensor_desc_[0].tensor;
  auto bboxes_desc = tensor_desc_[1].tensor;
  auto output_desc = tensor_desc_[2].tensor;
  const void *input_ptr = data_vector_[0].device_ptr;
  const void *bboxes_ptr = data_vector_[1].device_ptr;
  void *output_ptr = data_vector_[2].device_ptr;

  interface_timer_.start();
  MLUOP_CHECK(mluOpRotatedFeatureAlignForward(
      handle_, input_desc, input_ptr, bboxes_desc, bboxes_ptr, spatial_scale_,
      points_, output_desc, output_ptr));
  interface_timer_.stop();
}

// Modyfied from mmcv
float RotatedFeatureAlignForwardExecutor::bilinear_interpolate(
    const float *input, const int height, const int width, const int channels,
    float y, float x, int c, const int index) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (float)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (float)x_low;
  } else {
    x_high = x_low + 1;
  }

  float ly = y - y_low;
  float lx = x - x_low;
  float hy = 1. - ly, hx = 1. - lx;

  // do bilinear interpolation
  float v1 = input[y_low * width * channels + x_low * channels + c];
  float v2 = input[y_low * width * channels + x_high * channels + c];
  float v3 = input[y_high * width * channels + x_low * channels + c];
  float v4 = input[y_high * width * channels + x_high * channels + c];
  float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

void RotatedFeatureAlignForwardExecutor::cpuCompute() {
  const int output_size = parser_->getOutputDataCount(0);
  auto input_desc = tensor_desc_[0].tensor;
  auto bboxes_desc = tensor_desc_[1].tensor;

  const int batch = input_desc->dims[0];
  const int height = input_desc->dims[1];
  const int width = input_desc->dims[2];
  const int channels = input_desc->dims[3];
  const int bboxes_offset = bboxes_desc->dims[3];
  float px[5] = {0, 0, 0, 0, 0};
  float py[5] = {0, 0, 0, 0, 0};

  for (int index = 0; index < output_size; index++) {
    const int c = index % channels;
    const int pw = (index / channels) % width;
    const int ph = (index / channels / width) % height;
    const int n = index / channels / width / height;
    const int bboxes_nhw = n * width * height * 5 + ph * width * 5 + pw * 5;
    const int input_n = n * width * height * channels;
    const float *bboxes_offset = cpu_fp32_input_[1] + bboxes_nhw;
    const float *input_offset = cpu_fp32_input_[0] + input_n;

    float roi_y = bboxes_offset[0] * spatial_scale_;
    float roi_x = bboxes_offset[1] * spatial_scale_;
    px[0] = roi_x;
    py[0] = roi_y;

    if (points_ > 1) {
      float roi_w = bboxes_offset[2] * spatial_scale_;
      float roi_h = bboxes_offset[3] * spatial_scale_;
      float roi_a = bboxes_offset[4];

      float w_2 = roi_w / 2, h_2 = roi_h / 2;
      float cosa = cosf(roi_a), sina = sinf(roi_a);
      float wx = cosa * w_2, wy = sina * w_2;
      float hx = -sina * h_2, hy = cosa * h_2;

      px[1] = roi_x + wx + hx;
      py[1] = roi_y + wy + hy;
      px[2] = roi_x - wx + hx;
      py[2] = roi_y - wy + hy;
      px[3] = roi_x - wx - hx;
      py[3] = roi_y - wy - hy;
      px[4] = roi_x + wx - hx;
      py[4] = roi_y + wy - hy;
    }
    float output_temp = cpu_fp32_input_[0][index];
    for (int i = 0; i < points_; i++) {
      output_temp += bilinear_interpolate(input_offset, height, width, channels,
                                          py[i], px[i], c, i);
      theory_ops_++;
    }
    cpu_fp32_output_[0][index] = output_temp;
  }
}

int64_t RotatedFeatureAlignForwardExecutor::getTheoryOps() {
  VLOG(4) << "theory_ops_: " << theory_ops_ << " ops";
  return theory_ops_;
}

}  // namespace mluoptest
