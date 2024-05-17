/*******************************************************************************
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
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *******************************************************************************/
#include "deform_roi_pool_backward.h"

#include <algorithm>
#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <bitset>

namespace mluoptest {

void DeformRoiPoolBackwardExecutor::printDataInfo() {
  VLOG(4) << "############################### printfDataInfo() Begin ##";
  VLOG(4) << "# batchs:         " << batchs;
  VLOG(4) << "# height:         " << height;
  VLOG(4) << "# width:          " << width;
  VLOG(4) << "# channels:       " << channels;
  VLOG(4) << "# pool_height:    " << pooled_height;
  VLOG(4) << "# pool_width:     " << pooled_width;
  VLOG(4) << "# rois_num:       " << rois_num;
  VLOG(4) << "# spatial_scale:  " << spatial_scale;
  VLOG(4) << "# sampling_ratio: " << sampling_ratio;
  VLOG(4) << "# gamma:          " << gamma;
  VLOG(4) << "############################### printfDataInfo() End ##";
}

void DeformRoiPoolBackwardExecutor::initData() {
  grad_output_desc = tensor_desc_[0].tensor;
  input_desc = tensor_desc_[1].tensor;
  rois_desc = tensor_desc_[2].tensor;
  if (parser_->getInputNum() == 4) {
    offset_desc = tensor_desc_[3].tensor;
    grad_input_desc = tensor_desc_[4].tensor;
    grad_offset_desc = tensor_desc_[5].tensor;
  } else {
    grad_input_desc = tensor_desc_[3].tensor;
  }

  batchs = input_desc->dims[0];
  height = input_desc->dims[1];
  width = input_desc->dims[2];
  channels = input_desc->dims[3];
  rois_num = rois_desc->dims[0];

  // get params
  auto deform_roi_pool_backward_proto_desc =
      parser_->getProtoNode()->deform_roi_pool_backward_param();
  sampling_ratio = deform_roi_pool_backward_proto_desc.sampling_ratio();
  gamma = deform_roi_pool_backward_proto_desc.gamma();
  spatial_scale = deform_roi_pool_backward_proto_desc.spatial_scale();
  pooled_height = deform_roi_pool_backward_proto_desc.pooled_height();
  pooled_width = deform_roi_pool_backward_proto_desc.pooled_width();
}
void DeformRoiPoolBackwardExecutor::paramCheck() {
  if (!(parser_->getInputNum() == 4 && parser_->getOutputNum() == 2) &&
      !(parser_->getInputNum() == 3 && parser_->getOutputNum() == 1)) {
    LOG(ERROR) << "DeformRoiPoolBackward param number is wrong. ";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
}

void DeformRoiPoolBackwardExecutor::compute() {
  initData();
  printDataInfo();

  auto grad_output_ptr = data_vector_[0].device_ptr;
  auto input_ptr = data_vector_[1].device_ptr;
  auto rois_ptr = data_vector_[2].device_ptr;
  void *offset_ptr = NULL;
  void *grad_input_ptr = NULL;
  void *grad_offset_ptr = NULL;

  if (parser_->getInputNum() == 4) {
    offset_ptr = data_vector_[3].device_ptr;
    grad_input_ptr = data_vector_[4].device_ptr;
    grad_offset_ptr = data_vector_[5].device_ptr;
  } else {
    grad_input_ptr = data_vector_[3].device_ptr;
  }

  interface_timer_.start();
  MLUOP_CHECK(mluOpDeformRoiPoolBackward(
      handle_, grad_output_desc, grad_output_ptr, input_desc, input_ptr,
      rois_desc, rois_ptr, offset_desc, offset_ptr, pooled_height, pooled_width,
      spatial_scale, sampling_ratio, gamma, grad_input_desc, grad_input_ptr,
      grad_offset_desc, grad_offset_ptr));
  interface_timer_.stop();
}

void bilinear_interpolate_gradient(const int height, const int width,
                                   const int channels, float y, float x, int c,
                                   float &w1, float &w2, float &w3, float &w4,
                                   int &x_low, int &x_high, int &y_low,
                                   int &y_high,
                                   const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int)y;
  x_low = (int)x;

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

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
}

void DeformRoiPoolBackwardExecutor::cpuCompute() {
  int grad_output_size = parser_->getInputDataCount(0);
  for (int index = 0; index < grad_output_size; index++) {
    // (n, ph, pw, c) is an element in the pooled output
    const int c = index % channels;
    const int pw = (index / channels) % pooled_width;
    const int ph = (index / channels / pooled_width) % pooled_height;
    const int n = index / channels / pooled_width / pooled_height;
    const float *offset_rois = cpu_fp32_input_[2] + n * 5;
    const int roi_batch_ind = offset_rois[0];
    const float *offset_input =
        cpu_fp32_input_[1] + roi_batch_ind * height * width * channels;
    float *offset_grad_input =
        cpu_fp32_output_[0] + roi_batch_ind * height * width * channels;
    // Do not using rounding; this implementation detail is critical
    float roi_start_w = offset_rois[1] * spatial_scale - 0.5;
    float roi_start_h = offset_rois[2] * spatial_scale - 0.5;
    float roi_end_w = offset_rois[3] * spatial_scale - 0.5;
    float roi_end_h = offset_rois[4] * spatial_scale - 0.5;
    float roi_width = roi_end_w - roi_start_w;
    float roi_height = roi_end_h - roi_start_h;
    float bin_size_h =
        static_cast<float>(roi_height) / static_cast<float>(pooled_height);
    float bin_size_w =
        static_cast<float>(roi_width) / static_cast<float>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h =
        (sampling_ratio > 0)
            ? sampling_ratio
            : static_cast<int>(ceilf(roi_height / pooled_height));
    int roi_bin_grid_w =
        (sampling_ratio > 0)
            ? sampling_ratio
            : static_cast<int>(ceilf(roi_width / pooled_width));

    // Compute roi offset
    if (offset_desc != NULL && cpu_fp32_input_[3] != NULL) {
      const float *offset_cur_w = cpu_fp32_input_[3] +
                                  n * pooled_width * pooled_height * 2 +
                                  ph * pooled_width + pw;
      float offset_roi_w = gamma * roi_width * offset_cur_w[0];
      float offset_roi_h =
          gamma * roi_height * offset_cur_w[pooled_width * pooled_height];
      roi_start_w += offset_roi_w;
      roi_start_h += offset_roi_h;
    }
    // We do average pooling inside a bin
    const float count = std::max(roi_bin_grid_h * roi_bin_grid_w, 1);
    float grad_output_this_bin = cpu_fp32_input_[0][index] / count;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      const float y = roi_start_h + ph * bin_size_h +
                      static_cast<float>(iy + .5f) * bin_size_h /
                          static_cast<float>(roi_bin_grid_h);
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const float x = roi_start_w + pw * bin_size_w +
                        static_cast<float>(ix + .5f) * bin_size_w /
                            static_cast<float>(roi_bin_grid_w);
        float w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;
        bilinear_interpolate_gradient(height, width, channels, y, x, c, w1, w2,
                                      w3, w4, x_low, x_high, y_low, y_high,
                                      index);
        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          offset_grad_input[y_low * width * channels + x_low * channels + c] +=
              grad_output_this_bin * w1;
          offset_grad_input[y_low * width * channels + x_high * channels + c] +=
              grad_output_this_bin * w2;
          offset_grad_input[y_high * width * channels + x_low * channels + c] +=
              grad_output_this_bin * w3;
          offset_grad_input[y_high * width * channels + x_high * channels +
                            c] += grad_output_this_bin * w4;
          if (offset_desc != NULL && cpu_fp32_input_[3] != NULL) {
            float input_00 =
                offset_input[y_low * width * channels + x_low * channels + c];
            float input_10 =
                offset_input[y_low * width * channels + x_high * channels + c];
            float input_01 =
                offset_input[y_high * width * channels + x_low * channels + c];
            float input_11 =
                offset_input[y_high * width * channels + x_high * channels + c];
            float ogx = gamma * roi_width * grad_output_this_bin *
                        (input_11 * (y - y_low) + input_10 * (y_high - y) +
                         input_01 * (y_low - y) + input_00 * (y - y_high));
            float ogy = gamma * roi_height * grad_output_this_bin *
                        (input_11 * (x - x_low) + input_01 * (x_high - x) +
                         input_10 * (x_low - x) + input_00 * (x - x_high));
            cpu_fp32_output_[1][n * pooled_width * pooled_height * 2 +
                                ph * pooled_width + pw] += ogx;
            cpu_fp32_output_[1][n * pooled_width * pooled_height * 2 +
                                pooled_width * pooled_height +
                                ph * pooled_width + pw] += ogy;
          }
        }
      }
    }
  }
}

}  // namespace mluoptest
