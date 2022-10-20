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
#include <algorithm>
#include <string>
#include <vector>

#include "prior_box.h"

namespace mluoptest {
void PriorBoxExecutor::paramCheck() {
  VLOG(4) << "Prior_box param check";
  GTEST_CHECK(parser_->getInputNum() == 4);
  GTEST_CHECK(parser_->getOutputNum() == 2);
  for (int i = 0; i < parser_->getInputNum() - 1; i++) {
    GTEST_CHECK(!parser_->inputIsNull(i))
  }
  for (int i = 0; i < parser_->getOutputNum(); i++) {
    GTEST_CHECK(!parser_->outputIsNull(i))
  }
}

void PriorBoxExecutor::initData() {
  height_ = parser_->getProtoNode()->prior_box_param().height();
  width_ = parser_->getProtoNode()->prior_box_param().width();
  im_height_ = parser_->getProtoNode()->prior_box_param().im_height();
  im_width_ = parser_->getProtoNode()->prior_box_param().im_width();
  step_w_ = parser_->getProtoNode()->prior_box_param().step_w();
  step_h_ = parser_->getProtoNode()->prior_box_param().step_h();
  offset_ = parser_->getProtoNode()->prior_box_param().offset();
  clip_ = parser_->getProtoNode()->prior_box_param().clip();
  min_max_aspect_ratios_order_ =
      parser_->getProtoNode()->prior_box_param().min_max_aspect_ratios_order();
  min_sizes_desc_ = tensor_desc_[0].tensor;
  aspect_ratios_desc_ = tensor_desc_[1].tensor;
  variances_desc_ = tensor_desc_[2].tensor;
  max_sizes_desc_ = tensor_desc_[3].tensor;
  output_desc_ = tensor_desc_[4].tensor;
  var_desc_ = tensor_desc_[5].tensor;
  theory_op_size_ = 0;
}

void PriorBoxExecutor::compute() {
  initData();
  auto min_sizes = data_vector_[0].device_ptr;
  auto aspect_ratios = data_vector_[1].device_ptr;
  auto variances = data_vector_[2].device_ptr;
  auto max_sizes = data_vector_[3].device_ptr;
  auto output = data_vector_[4].device_ptr;
  auto var = data_vector_[5].device_ptr;
  interface_timer_.start();
  MLUOP_CHECK(mluOpPriorBox(
      handle_, min_sizes_desc_, min_sizes, aspect_ratios_desc_, aspect_ratios,
      variances_desc_, variances, max_sizes_desc_, max_sizes, height_, width_,
      im_height_, im_width_, step_h_, step_w_, offset_, clip_,
      min_max_aspect_ratios_order_, output_desc_, output, var_desc_, var));
  interface_timer_.stop();
}

void PriorBoxExecutor::priorBox_Cpu_Kernel(
    float* min_sizes, const int min_sizes_num, float* new_aspect_ratios,
    const int new_aspect_ratios_num, float* variances, const int variances_num,
    float* max_sizes, const int max_sizes_num, const int height,
    const int width, const int im_height, const int im_width,
    const float step_h, const float step_w, const float offset, const bool clip,
    const bool min_max_aspect_ratios_order, float* output,
    const int output_size, float* var, const int var_size) {
  auto img_width = im_width;
  auto img_height = im_height;

  auto feature_width = width;
  auto feature_height = height;
  int num_priors = new_aspect_ratios_num * min_sizes_num;
  if (max_sizes_num) {
    num_priors += max_sizes_num;
  }

  float* bt = output;
  for (int h = 0; h < feature_height; ++h) {
    for (int w = 0; w < feature_width; ++w) {
      float center_x = (w + offset) * step_w;
      float center_y = (h + offset) * step_h;
      float box_width, box_height;
      for (size_t s = 0; s < min_sizes_num; ++s) {
        auto min_size = min_sizes[s];
        if (min_max_aspect_ratios_order) {
          box_width = box_height = min_size / 2.;
          bt[0] = (center_x - box_width) / img_width;
          bt[1] = (center_y - box_height) / img_height;
          bt[2] = (center_x + box_width) / img_width;
          bt[3] = (center_y + box_height) / img_height;

          theory_op_size_ += 8;
          bt += 4;
          if (max_sizes_num > 0) {
            auto max_size = max_sizes[s];
            // square prior with size sqrt(minSize * maxSize)
            box_width = box_height = sqrt(min_size * max_size) / 2.;
            bt[0] = (center_x - box_width) / img_width;
            bt[1] = (center_y - box_height) / img_height;
            bt[2] = (center_x + box_width) / img_width;
            bt[3] = (center_y + box_height) / img_height;
            bt += 4;
            theory_op_size_ += 10;
          }

          // priors with different aspect ratios
          for (size_t r = 0; r < new_aspect_ratios_num; ++r) {
            float ar = new_aspect_ratios[r];
            if (fabs(ar - 1.) < 1e-6) {
              continue;
            }
            box_width = min_size * sqrt(ar) / 2.;
            box_height = min_size / sqrt(ar) / 2.;
            bt[0] = (center_x - box_width) / img_width;
            bt[1] = (center_y - box_height) / img_height;
            bt[2] = (center_x + box_width) / img_width;
            bt[3] = (center_y + box_height) / img_height;
            bt += 4;
            theory_op_size_ += 12;
          }
        } else {
          // priors with different aspect ratios
          for (size_t r = 0; r < new_aspect_ratios_num; ++r) {
            float ar = new_aspect_ratios[r];
            box_width = min_size * sqrt(ar) / 2.;
            box_height = min_size / sqrt(ar) / 2.;
            bt[0] = (center_x - box_width) / img_width;
            bt[1] = (center_y - box_height) / img_height;
            bt[2] = (center_x + box_width) / img_width;
            bt[3] = (center_y + box_height) / img_height;
            bt += 4;
            theory_op_size_ += 12;
          }
          if (max_sizes_num > 0) {
            auto max_size = max_sizes[s];
            // square prior with size sqrt(minSize * maxSize)
            box_width = box_height = sqrt(min_size * max_size) / 2.;
            bt[0] = (center_x - box_width) / img_width;
            bt[1] = (center_y - box_height) / img_height;
            bt[2] = (center_x + box_width) / img_width;
            bt[3] = (center_y + box_height) / img_height;
            bt += 4;
            theory_op_size_ += 10;
          }
        }
      }
    }
  }

  if (clip) {
    for (int d = 0; d < output_size; d++) {
      output[d] = std::min<float>(std::max<float>(output[d], 0.), 1.);
    }
  }

  const int box_nums = feature_height * feature_width * num_priors;
  for (int i = 0; i < box_nums; i++) {
    for (int j = 0; j < variances_num; j++) {
      var[i * variances_num + j] = variances[j];
    }
  }
}

void PriorBoxExecutor::cpuCompute() {
  const int height = height_;
  const int width = width_;
  const int image_height = im_height_;
  const int image_width = im_width_;

  float* min_sizes = cpu_fp32_input_[0];
  float* aspect_ratios = cpu_fp32_input_[1];
  float* variances = cpu_fp32_input_[2];
  float* max_sizes = cpu_fp32_input_[3];

  const int min_sizes_num = min_sizes_desc_->total_element_num;
  const int aspect_ratios_num = aspect_ratios_desc_->total_element_num;
  const int variances_num = variances_desc_->total_element_num;
  const int max_sizes_num = max_sizes_desc_->total_element_num;
  const int output_num = output_desc_->total_element_num;
  const int var_num = var_desc_->total_element_num;
  const float step_h = step_h_;
  const float step_w = step_w_;
  const float offset = offset_;
  const bool clip = clip_;
  const bool min_max_aspect_ratios_order = min_max_aspect_ratios_order_;
  float* output = cpu_fp32_output_[0];
  float* var = cpu_fp32_output_[1];

  priorBox_Cpu_Kernel(min_sizes, min_sizes_num, aspect_ratios,
                      aspect_ratios_num, variances, variances_num, max_sizes,
                      max_sizes_num, height, width, image_height, image_width,
                      step_h, step_w, offset, clip, min_max_aspect_ratios_order,
                      output, output_num, var, var_num);
}

int64_t PriorBoxExecutor::getTheoryOps() {
  if (parser_->device() != CPU) {
    return -1;
  }
  int64_t theory_ops = theory_op_size_;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}
}  // namespace mluoptest
