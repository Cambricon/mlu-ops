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
#include "psroipool_backward.h"

#include <algorithm>

namespace mluoptest {
void PsroipoolBackwardExecutor::paramCheck() {
  VLOG(4) << "[PsroipoolBackwardExecutor] param check.";
  GTEST_CHECK(parser_->getInputNum() == 3,
              "[PsroipoolBackwardExecutor] Input number is wrong.");
  GTEST_CHECK(parser_->getOutputNum() == 1,
              "[PsroipoolBackwardExecutor] Output number is wrong.");
  GTEST_CHECK(parser_->getProtoNode()->has_psroipool_backward_param(),
              "[PsroipoolBackwardExecutor] Missing param");
}

void PsroipoolBackwardExecutor::initData() {
  output_dim_ =
      parser_->getProtoNode()->psroipool_backward_param().output_dim();
  pooled_height_ =
      parser_->getProtoNode()->psroipool_backward_param().pooled_height();
  pooled_width_ =
      parser_->getProtoNode()->psroipool_backward_param().pooled_width();
  spatial_scale_ =
      parser_->getProtoNode()->psroipool_backward_param().spatial_scale();
}

void PsroipoolBackwardExecutor::compute() {
  initData();
  auto top_input_desc = tensor_desc_[0].tensor;
  auto mapping_channel_desc = tensor_desc_[1].tensor;
  auto rois_desc = tensor_desc_[2].tensor;
  auto bottom_output_desc = tensor_desc_[3].tensor;

  const void* top_input = data_vector_[0].device_ptr;
  const void* mapping_channel = data_vector_[1].device_ptr;
  const void* rois = data_vector_[2].device_ptr;
  void* bottom_output = data_vector_[3].device_ptr;

  interface_timer_.start();
  MLUOP_CHECK(mluOpPsRoiPoolBackward(
      handle_, pooled_height_, pooled_width_, spatial_scale_, output_dim_,
      top_input_desc, top_input, rois_desc, rois, mapping_channel_desc,
      mapping_channel, bottom_output_desc, bottom_output));
  interface_timer_.stop();
}

void PsroipoolBackwardExecutor::cpuCompute() {
  auto top_input_desc = tensor_desc_[0].tensor;
  auto mapping_channel_desc = tensor_desc_[1].tensor;
  auto rois_desc = tensor_desc_[2].tensor;
  auto bottom_output_desc = tensor_desc_[3].tensor;

  auto top_input_cpu = cpu_fp32_input_[0];
  auto mapping_channel_cpu = cpu_fp32_input_[1];
  auto rois_cpu = cpu_fp32_input_[2];
  auto bottom_output_cpu = cpu_fp32_output_[0];

  const int bottom_n = bottom_output_desc->dims[0];
  const int bottom_h = bottom_output_desc->dims[1];
  const int bottom_w = bottom_output_desc->dims[2];
  const int bottom_c = bottom_output_desc->dims[3];

  const int rois_n = rois_desc->dims[0];
  const int rois_offset = rois_desc->dims[1];

  for (int roi_id = 0; roi_id < rois_n; roi_id++) {
    int top_batch_offset =
        roi_id * pooled_height_ * pooled_width_ * output_dim_;
    int roi_add = roi_id * rois_offset;
    int batch_i = rois_cpu[roi_add];
    int bottom_add = batch_i * bottom_h * bottom_w * bottom_c;

    float roi_start_w =
        static_cast<float>(round(rois_cpu[roi_add + 1])) * spatial_scale_;
    float roi_start_h =
        static_cast<float>(round(rois_cpu[roi_add + 2])) * spatial_scale_;
    float roi_end_w =
        static_cast<float>(round(rois_cpu[roi_add + 3]) + 1.) * spatial_scale_;
    float roi_end_h =
        static_cast<float>(round(rois_cpu[roi_add + 4]) + 1.) * spatial_scale_;

    float roi_width = std::max(roi_end_w - roi_start_w, (float)0.1);
    float roi_height = std::max(roi_end_h - roi_start_h, (float)0.1);
    float bin_size_h = (float)roi_height / (float)(pooled_height_);
    float bin_size_w = (float)roi_width / (float)(pooled_width_);

    for (int top_c = 0; top_c < output_dim_; top_c++) {
      for (int top_h = 0; top_h < pooled_height_; top_h++) {
        for (int top_w = 0; top_w < pooled_width_; top_w++) {
          int top_index = top_batch_offset +
                          top_h * pooled_width_ * output_dim_ +
                          top_w * output_dim_ + top_c;
          int hstart =
              floor(static_cast<float>(top_h) * bin_size_h + roi_start_h);
          int wstart =
              floor(static_cast<float>(top_w) * bin_size_w + roi_start_w);
          int hend =
              ceil(static_cast<float>(top_h + 1) * bin_size_h + roi_start_h);
          int wend =
              ceil(static_cast<float>(top_w + 1) * bin_size_w + roi_start_w);

          hstart = std::min(std::max(hstart, 0), bottom_h);
          hend = std::min(std::max(hend, 0), bottom_h);
          wstart = std::min(std::max(wstart, 0), bottom_w);
          wend = std::min(std::max(wend, 0), bottom_w);

          bool is_empty = (hend <= hstart) || (wend <= wstart);
          int c = mapping_channel_cpu[top_index];
          float bin_area = (hend - hstart) * (wend - wstart);
          float diff_val = is_empty ? 0. : top_input_cpu[top_index] / bin_area;

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              int bottom_index = h * bottom_w * bottom_c + w * bottom_c + c;
              bottom_output_cpu[bottom_index + bottom_add] += diff_val;
              theory_ops_ += 7;
            }
          }
        }
      }
    }
  }
}

int64_t PsroipoolBackwardExecutor::getTheoryOps() {
  VLOG(4) << "getTheoryOps: " << theory_ops_ << " ops";
  return theory_ops_;
}
}  // namespace mluoptest
