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
#include "deform_roi_pool_forward.h"

#include <algorithm>
#include <string>

namespace mluoptest {

void DeformRoiPoolForwardExecutor::printDataInfo() {
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

void DeformRoiPoolForwardExecutor::initData() {
  input_desc = tensor_desc_[0].tensor;
  rois_desc = tensor_desc_[1].tensor;
  if (parser_->getInputNum() == 3) {
    offset_desc = tensor_desc_[2].tensor;
    output_desc = tensor_desc_[3].tensor;
  } else {
    output_desc = tensor_desc_[2].tensor;
  }

  batchs = input_desc->dims[0];
  height = input_desc->dims[1];
  width = input_desc->dims[2];
  channels = input_desc->dims[3];
  rois_num = rois_desc->dims[0];
  pooled_height = output_desc->dims[1];
  pooled_width = output_desc->dims[2];
  // get params
  auto deform_roi_pool_forward_proto_desc =
      parser_->getProtoNode()->deform_roi_pool_forward_param();
  sampling_ratio = deform_roi_pool_forward_proto_desc.sampling_ratio();
  gamma = deform_roi_pool_forward_proto_desc.gamma();
  spatial_scale = deform_roi_pool_forward_proto_desc.spatial_scale();
  pooled_height = deform_roi_pool_forward_proto_desc.pooled_height();
  pooled_width = deform_roi_pool_forward_proto_desc.pooled_width();
}

void DeformRoiPoolForwardExecutor::paramCheck() {
  if (parser_->getInputNum() > 3 || parser_->getInputNum() < 2) {
    LOG(ERROR) << "DeformRoiPoolForward input number is wrong. ";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "DeformRoiPoolForward output number is wrong. ";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
}

void DeformRoiPoolForwardExecutor::compute() {
  initData();
  printDataInfo();

  auto input_ptr = data_vector_[0].device_ptr;
  auto rois_ptr = data_vector_[1].device_ptr;
  void *offset_ptr = NULL;
  void *output_ptr = NULL;

  if (parser_->getInputNum() == 3) {
    offset_ptr = data_vector_[2].device_ptr;
    output_ptr = data_vector_[3].device_ptr;
  } else {
    output_ptr = data_vector_[2].device_ptr;
  }

  interface_timer_.start();
  MLUOP_CHECK(mluOpDeformRoiPoolForward(
      handle_, input_desc, input_ptr, rois_desc, rois_ptr, offset_desc,
      offset_ptr, pooled_height, pooled_width, spatial_scale, sampling_ratio,
      gamma, output_desc, output_ptr));
  interface_timer_.stop();
}

// Modyfied from mmcv
float bilinear_interpolate(const float *input, const int height,
                           const int width, const int channels, float y,
                           float x, int c,
                           const int index /* index for debug only*/) {
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

  float v1 = input[y_low * width * channels + x_low * channels + c];
  float v2 = input[y_low * width * channels + x_high * channels + c];
  float v3 = input[y_high * width * channels + x_low * channels + c];
  float v4 = input[y_high * width * channels + x_high * channels + c];
  float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

void DeformRoiPoolForwardExecutor::cpuCompute() {
  int output_size = parser_->getOutputDataCount(0);
  for (int index = 0; index < output_size; index++) {
    // (n, ph, pw, c) is an element in the pooled output
    const int c = index % channels;
    const int pw = (index / channels) % pooled_width;
    const int ph = (index / channels / pooled_width) % pooled_height;
    const int n = index / channels / pooled_width / pooled_height;
    const float *offset_rois = cpu_fp32_input_[1] + n * 5;
    const int roi_batch_ind = offset_rois[0];

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
    const float *offset_input =
        cpu_fp32_input_[0] + roi_batch_ind * height * width * channels;

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
    if (offset_desc != NULL && cpu_fp32_input_[2] != NULL) {
      const float *offset_cur_w = cpu_fp32_input_[2] +
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
    float output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
      const float y = roi_start_h + ph * bin_size_h +
                      static_cast<float>(iy + .5f) * bin_size_h /
                          static_cast<float>(roi_bin_grid_h);
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const float x = roi_start_w + pw * bin_size_w +
                        static_cast<float>(ix + .5f) * bin_size_w /
                            static_cast<float>(roi_bin_grid_w);
        float val = bilinear_interpolate(offset_input, height, width, channels,
                                         y, x, c, index);
        output_val += val;
      }
    }
    cpu_fp32_output_[0][index] = output_val / count;
  }
}

}  // namespace mluoptest
