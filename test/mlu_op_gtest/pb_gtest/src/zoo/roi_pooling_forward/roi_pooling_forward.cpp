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
#include "roi_pooling_forward.h"

#include <algorithm>
#include <string>

#define getParam(ty, ctx) \
  (ty) parser_->getProtoNode()->roi_pooling_forward_param().ctx()
#define getTensorDims(x, y) tensor_desc_[x].tensor->dims[y]
#define getTensorDesc(x) tensor_desc_[x].tensor
#define getDevicePtr(x) data_vector_[x].device_ptr

namespace mluoptest {
static char pooling_mode_str[3][64] =
                          {"MLUOP_POOLING_MAX",
                           "MLUOP_POOLING_AVERAGE_COUNT_INCLUDE_PADDING",
                           "MLUOP_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING"};

void RoiPoolingForwardExecutor::cpuRoiPoolingForward(float *input_v,
                                                     float *rois,
                                                     int batch_v,
                                                     int height,
                                                     int width,
                                                     int channels,
                                                     int pool_height,
                                                     int pool_width,
                                                     int rois_num,
                                                     float spatial_scale,
                                                     float *output,
                                                     float *argmax) {
  int bin_num = rois_num * pool_height * pool_width * channels;
  theory_ops = 0;
  for (int index = 0; index < bin_num; index++) {
    int c = index % channels;
    int pw = (index / channels) % pool_width;
    int ph = (index / channels / pool_width) % pool_height;
    int n = index / channels / pool_width / pool_height;

    const float *offset_rois = rois + n * 5;
    int batch_id = (int)offset_rois[0];
    int roi_x1 = round(offset_rois[1] * spatial_scale);
    int roi_y1 = round(offset_rois[2] * spatial_scale);
    int roi_x2 = round(offset_rois[3] * spatial_scale);
    int roi_y2 = round(offset_rois[4] * spatial_scale);

    int roi_w = std::max(roi_x2 - roi_x1 + 1, 1);
    int roi_h = std::max(roi_y2 - roi_y1 + 1, 1);
    float bin_size_w = static_cast<float>(roi_w) /
                       static_cast<float>(pool_width);
    float bin_size_h = static_cast<float>(roi_h) /
                       static_cast<float>(pool_height);

    int bin_x1 = floor(static_cast<float>(pw) * bin_size_w);
    int bin_y1 = floor(static_cast<float>(ph) * bin_size_h);
    int bin_x2 = ceil(static_cast<float>(pw + 1) * bin_size_w);
    int bin_y2 = ceil(static_cast<float>(ph + 1) * bin_size_h);
    bin_x1 = std::min(std::max(bin_x1 + roi_x1, 0), width);
    bin_y1 = std::min(std::max(bin_y1 + roi_y1, 0), height);
    bin_x2 = std::min(std::max(bin_x2 + roi_x1, 0), width);
    bin_y2 = std::min(std::max(bin_y2 + roi_y1, 0), height);
    bool is_empty = (bin_y2 <= bin_y1) || (bin_x2 <= bin_x1);

    const float *offset_input = input_v + (batch_id * height *
                                width * channels + c);
    float max_v = is_empty ? 0 : -FLT_MAX;
    float max_idx = -1.0;
    for (int h = bin_y1; h < bin_y2; h++) {
      for (int w = bin_x1; w < bin_x2; w++) {
        int offset = (h * width + w) * channels;
        if (offset_input[offset] > max_v) {
          max_v = offset_input[offset];
          max_idx = (float)(offset / channels);
        }
        theory_ops++;
      }
    }
    output[index] = max_v;
    if (argmax != NULL) {
      argmax[index] = max_idx;
    }
  }
}

void RoiPoolingForwardExecutor::initData() {
  VLOG(4) << "############################### initData() Begin ##";
  batch_ = getTensorDims(0, 0);
  height_ = getTensorDims(0, 1);
  width_ = getTensorDims(0, 2);
  channels_ = getTensorDims(0, 3);
  pool_height_ = getTensorDims(2, 1);
  pool_width_ = getTensorDims(2, 2);
  rois_num_ = getTensorDims(2, 0);
  spatial_scale_ = getParam(float, spatial_scale);
  pooling_mode_ = getParam(mluOpPoolingMode_t, mode);
  /*input tensor*/
  input_mlu_ = getDevicePtr(0);
  rois_mlu_ = getDevicePtr(1);
  /*output tensor*/
  output_mlu_ = getDevicePtr(2);
  argmax_mlu_ = (int *)getDevicePtr(3);
  /*tensor desc*/
  input_desc_ = getTensorDesc(0);
  rois_desc_ = getTensorDesc(1);
  output_desc_ = getTensorDesc(2);
  VLOG(4) << "############################### initData() End ##";
}


void RoiPoolingForwardExecutor::compute() {
  VLOG(4) << "############################### compute() Begin ##";
  initData();

  interface_timer_.start();
  MLUOP_CHECK(mluOpRoiPoolingForward(handle_, pooling_mode_, input_desc_,
                   input_mlu_, rois_desc_, rois_mlu_, spatial_scale_,
                   output_desc_, output_mlu_, argmax_mlu_));
  interface_timer_.stop();
  VLOG(4) << "############################### compute() End ##";
}

void RoiPoolingForwardExecutor::cpuCompute() {
  VLOG(4) << "############################### cpuCompute() Begin ##";
  float *input_cpu = cpu_fp32_input_[0];
  float *rois_cpu = cpu_fp32_input_[1];
  float *output_cpu = cpu_fp32_output_[0];
  float *argmax_cpu = cpu_fp32_output_[1];

  cpuRoiPoolingForward(input_cpu, rois_cpu, batch_, height_, width_, channels_,
                       pool_height_, pool_width_, rois_num_, spatial_scale_,
                       output_cpu, argmax_cpu);
  VLOG(4) << "############################### cpuCompute() End ##";
}

int64_t RoiPoolingForwardExecutor::getTheoryOps() {
  return theory_ops * 2;
}

}  // namespace mluoptest
