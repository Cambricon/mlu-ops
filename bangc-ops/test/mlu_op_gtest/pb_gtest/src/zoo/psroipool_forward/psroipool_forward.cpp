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
#include "psroipool_forward.h"

#include <algorithm>
#include <string>
#include <vector>

#include "mlu_op.h"

namespace mluoptest {
void PsroipoolForwardExecutor::paramCheck() {
  VLOG(4) << "[PsroipoolForwardExecutor] param check.";
  if (parser_->getInputNum() != 2) {
    LOG(ERROR)
        << "[PsroipoolForwardExecutor] input number is wrong. It should be 2, "
        << "but now is " << parser_->getInputNum();
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
  if (parser_->getOutputNum() != 2) {
    LOG(ERROR)
        << "[PsroipoolForwardExecutor] output number is wrong. It should be 2, "
        << "but now is" << parser_->getOutputNum();
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
  for (int i = 0; i < parser_->getInputNum(); i++) {
    if (i == 1 && parser_->inputIsNull(i)) {
      LOG(ERROR) << "[PsroipoolForwardExecutor] input [" << i
                 << "] is nullptr.";
      throw std::invalid_argument(std::string(__FILE__) + " +" +
                                  std::to_string(__LINE__));
    }
  }
}

void PsroipoolForwardExecutor::initData() {
  output_dim_ = parser_->getProtoNode()->psroipool_forward_param().output_dim();
  pooled_height_ =
      parser_->getProtoNode()->psroipool_forward_param().pooled_height();
  pooled_width_ =
      parser_->getProtoNode()->psroipool_forward_param().pooled_width();
  spatial_scale_ =
      parser_->getProtoNode()->psroipool_forward_param().spatial_scale();
  group_size_ = parser_->getProtoNode()->psroipool_forward_param().group_size();
}

void PsroipoolForwardExecutor::compute() {
  initData();
  auto input_desc = tensor_desc_[0].tensor;
  auto rois_desc = tensor_desc_[1].tensor;
  auto output_desc = tensor_desc_[2].tensor;
  auto mapping_channel_desc = tensor_desc_[3].tensor;

  const void* input = data_vector_[0].device_ptr;
  const void* rois = data_vector_[1].device_ptr;
  void* output = data_vector_[2].device_ptr;
  void* mapping_channel = data_vector_[3].device_ptr;

  interface_timer_.start();
  MLUOP_CHECK(mluOpPsRoiPoolForward(
      handle_, pooled_height_, pooled_width_, spatial_scale_, group_size_,
      output_dim_, input_desc, input, rois_desc, rois, output_desc, output,
      mapping_channel_desc, mapping_channel));
  interface_timer_.stop();
}

void PsroipoolForwardExecutor::cpuCompute() {
  auto input_desc = tensor_desc_[0].tensor;
  auto rois_desc = tensor_desc_[1].tensor;

  auto input_cpu = cpu_fp32_input_[0];
  auto rois_cpu = cpu_fp32_input_[1];
  auto output_cpu = cpu_fp32_output_[0];
  auto mapping_channel_cpu = cpu_fp32_output_[1];

  const int input_n = input_desc->dims[0];
  const int input_h = input_desc->dims[1];
  const int input_w = input_desc->dims[2];
  const int input_c = input_desc->dims[3];

  const int rois_n = rois_desc->dims[0];
  const int rois_offset = rois_desc->dims[1];

  for (int roi_id = 0; roi_id < rois_n; roi_id++) {
    int out_batch_offset =
        roi_id * output_dim_ * pooled_height_ * pooled_width_;
    int roi_add = roi_id * rois_offset;
    int batch_i = rois_cpu[roi_add];
    int input_add = batch_i * input_h * input_w * input_c;

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

    for (int out_c = 0; out_c < output_dim_; out_c++) {
      for (int out_h = 0; out_h < pooled_height_; out_h++) {
        for (int out_w = 0; out_w < pooled_width_; out_w++) {
          int out_index = out_batch_offset +
                          out_h * pooled_width_ * output_dim_ +
                          out_w * output_dim_ + out_c;
          int hstart =
              floor(static_cast<float>(out_h) * bin_size_h + roi_start_h);
          int wstart =
              floor(static_cast<float>(out_w) * bin_size_w + roi_start_w);
          int hend =
              ceil(static_cast<float>(out_h + 1) * bin_size_h + roi_start_h);
          int wend =
              ceil(static_cast<float>(out_w + 1) * bin_size_w + roi_start_w);

          hstart = std::min(std::max(hstart, 0), input_h);
          hend = std::min(std::max(hend, 0), input_h);
          wstart = std::min(std::max(wstart, 0), input_w);
          wend = std::min(std::max(wend, 0), input_w);

          bool is_empty = (hend <= hstart) || (wend <= wstart);
          int gw = out_w;
          int gh = out_h;
          int c = out_c * group_size_ * group_size_ + gh * group_size_ + gw;
          float out_sum = 0;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              int bottom_index = h * input_w * input_c + w * input_c + c;
              out_sum += input_cpu[bottom_index + input_add];
              theory_ops_ += 7;
            }
          }
          float bin_area = (hend - hstart) * (wend - wstart);
          if (is_empty) {
            output_cpu[out_index] = 0;
          } else {
            output_cpu[out_index] = out_sum / bin_area;
            theory_ops_ += 1;
          }
          mapping_channel_cpu[out_index] = c;
        }
      }
    }
  }
}

int64_t PsroipoolForwardExecutor::getTheoryOps() {
  if (parser_->device() != CPU) {
    return -1;
  }
  VLOG(4) << "getTheoryOps: " << theory_ops_ << " ops";
  return theory_ops_;
}
}  // namespace mluoptest
