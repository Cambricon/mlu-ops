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
#include "rotated_feature_align_backward.h"

namespace mluoptest {

void RotatedFeatureAlignBackwardExecutor::initData() {
  // get params
  spatial_scale_ = parser_->getProtoNode()
                       ->rotated_feature_align_backward_param()
                       .spatial_scale();
  points_ =
      parser_->getProtoNode()->rotated_feature_align_backward_param().points();
}

void RotatedFeatureAlignBackwardExecutor::paramCheck() {
  VLOG(4) << "[RotatedFeatureAlignBackwardExecutor] Param check.";
  GTEST_CHECK(parser_->getInputNum() == 2,
              "[RotatedFeatureAlignBackwardExecutor] Input number is wrong.");
  GTEST_CHECK(parser_->getOutputNum() == 1,
              "[RotatedFeatureAlignBackwardExecutor] Output number is wrong.");
  GTEST_CHECK(
      parser_->getProtoNode()->has_rotated_feature_align_backward_param(),
      "[RotatedFeatureAlignBackwardExecutor] Missing param.");
}

void RotatedFeatureAlignBackwardExecutor::compute() {
  // init param
  initData();

  auto top_output_desc = tensor_desc_[0].tensor;
  auto bboxes_desc = tensor_desc_[1].tensor;
  auto bottom_input_desc = tensor_desc_[2].tensor;
  const void *top_output_ptr = data_vector_[0].device_ptr;
  const void *bboxes_ptr = data_vector_[1].device_ptr;
  void *bottom_input_ptr = data_vector_[2].device_ptr;

  interface_timer_.start();
  MLUOP_CHECK(mluOpRotatedFeatureAlignBackward(
      handle_, top_output_desc, top_output_ptr, bboxes_desc, bboxes_ptr,
      spatial_scale_, points_, bottom_input_desc, bottom_input_ptr));
  interface_timer_.stop();
}

void RotatedFeatureAlignBackwardExecutor::bilinear_interpolate_gradient(
    const int height, const int width, float y, float x, float *w1, float *w2,
    float *w3, float *w4, int *x_low, int *x_high, int *y_low, int *y_high,
    const int index) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    *w1 = *w2 = *w3 = *w4 = 0.;
    *x_low = *x_high = *y_low = *y_high = -1;
    return;
  }
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;
  *y_low = (int)y;
  *x_low = (int)x;
  if ((*y_low) >= height - 1) {
    *y_high = *y_low = height - 1;
    y = (float)(*y_low);
  } else {
    *y_high = (*y_low) + 1;
  }
  if ((*x_low) >= width - 1) {
    *x_high = *x_low = width - 1;
    x = (float)(*x_low);
  } else {
    *x_high = (*x_low) + 1;
  }
  float ly = y - (*y_low);
  float lx = x - (*x_low);
  float hy = 1. - ly, hx = 1. - lx;
  *w1 = hy * hx;
  *w2 = hy * lx;
  *w3 = ly * hx;
  *w4 = ly * lx;
  return;
}

void RotatedFeatureAlignBackwardExecutor::cpuCompute() {
  const int output_size = parser_->getOutputDataCount(0);
  auto top_output_desc = tensor_desc_[0].tensor;
  auto bboxes_desc = tensor_desc_[1].tensor;
  const int batch = top_output_desc->dims[0];
  const int height = top_output_desc->dims[1];
  const int width = top_output_desc->dims[2];
  const int channels = top_output_desc->dims[3];
  const int bboxes_offset = bboxes_desc->dims[3];
  float px[5] = {0, 0, 0, 0, 0};
  float py[5] = {0, 0, 0, 0, 0};
  for (int index = 0; index < output_size; index++) {
    const int c = index % channels;
    const int pw = (index / channels) % width;
    const int ph = (index / channels / width) % height;
    const int n = index / channels / width / height;
    const int bboxes_nhw = n * width * height * 5 + ph * width * 5 + pw * 5;
    const int top_output_n = n * width * height * channels;
    const float *bboxes_offset = cpu_fp32_input_[1] + bboxes_nhw;
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
    float top_output_temp = cpu_fp32_input_[0][index];
    for (int i = 0; i < points_; i++) {
      float w1 = 0., w2 = 0., w3 = 0., w4 = 0.;
      int x_low = 0, x_high = 0, y_low = 0, y_high = 0;
      bilinear_interpolate_gradient(height, width, py[i], px[i], &w1, &w2, &w3,
                                    &w4, &x_low, &x_high, &y_low, &y_high, i);
      float g1 = top_output_temp * w1;
      float g2 = top_output_temp * w2;
      float g3 = top_output_temp * w3;
      float g4 = top_output_temp * w4;
      theory_ops_ += 4;
      if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
        int tl_offset =
            top_output_n + y_low * width * channels + x_low * channels + c;
        int tr_offset =
            top_output_n + y_low * width * channels + x_high * channels + c;
        int bl_offset =
            top_output_n + y_high * width * channels + x_low * channels + c;
        int br_offset =
            top_output_n + y_high * width * channels + x_high * channels + c;
        cpu_fp32_output_[0][tl_offset] += g1;
        cpu_fp32_output_[0][tr_offset] += g2;
        cpu_fp32_output_[0][bl_offset] += g3;
        cpu_fp32_output_[0][br_offset] += g4;
        theory_ops_ += 4;
      }
    }
    cpu_fp32_output_[0][index] += top_output_temp;
  }
}

int64_t RotatedFeatureAlignBackwardExecutor::getTheoryOps() {
  VLOG(4) << "theory_ops_: " << theory_ops_ << " ops";
  return theory_ops_;
}
}  // namespace mluoptest
