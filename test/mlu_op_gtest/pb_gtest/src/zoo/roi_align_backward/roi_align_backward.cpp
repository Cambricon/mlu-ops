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
#include "roi_align_backward.h"

#include <algorithm>
#include <iostream>
#include <vector>

namespace mluoptest {

void RoiAlignBackwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->getProtoNode()->has_roi_align_backward_param(),
              "mluOpRoiAlignBackward: lose param. ");

  GTEST_CHECK(parser_->getInputNum() == 2 || parser_->getInputNum() == 4,
              "mluOpRoiAlignBackward: tensor input number is wrong.");
  GTEST_CHECK(parser_->getOutputNum() == 1,
              "mluOpRoiAlignBackward: tensor output number is wrong.");
}

void RoiAlignBackwardExecutor::compute() {
  auto grads_desc = parser_->getMetaTensor(0).tensor;
  auto boxes_desc = parser_->getMetaTensor(1).tensor;

  auto grads_ptr = parser_->getMetaTensor(0).dev_ptr;
  auto boxes_ptr = parser_->getMetaTensor(1).dev_ptr;

  float spatial_scale =
      parser_->getProtoNode()->roi_align_backward_param().spatial_scale();
  int sampling_ratio =
      parser_->getProtoNode()->roi_align_backward_param().sampling_ratio();
  bool aligned = parser_->getProtoNode()->roi_align_backward_param().aligned();
  int pool_mode =
      parser_->getProtoNode()->roi_align_backward_param().pool_mode();
  int version = parser_->getProtoNode()->roi_align_backward_param().version();
  VLOG(4) << "call mluOp mluOpRoiAlignBackward()";
  interface_timer_.start();
  if (version == 0) {
    VLOG(4) << "call mluOp mluOpRoiAlignBackward";
    if (pool_mode == 1) {
      auto grads_image_desc = parser_->getMetaTensor(2).tensor;
      auto grads_image_ptr = parser_->getMetaTensor(2).dev_ptr;
      MLUOP_CHECK(mluOpRoiAlignBackward(
          handle_, spatial_scale, sampling_ratio, aligned, grads_desc,
          grads_ptr, boxes_desc, boxes_ptr, grads_image_desc, grads_image_ptr));
    } else if (pool_mode == 0) {
      VLOG(4) << "call mluOp mluOpRoiAlignBackward_v2 with max mode";
      auto argmax_x_desc = parser_->getMetaTensor(2).tensor;
      auto argmax_y_desc = parser_->getMetaTensor(3).tensor;
      auto grads_image_desc = parser_->getMetaTensor(4).tensor;

      auto argmax_x_ptr = parser_->getMetaTensor(2).dev_ptr;
      auto argmax_y_ptr = parser_->getMetaTensor(3).dev_ptr;
      auto grads_image_ptr = parser_->getMetaTensor(4).dev_ptr;
      MLUOP_CHECK(mluOpRoiAlignBackward_v2(
          handle_, grads_desc, grads_ptr, boxes_desc, boxes_ptr, argmax_x_desc,
          argmax_x_ptr, argmax_y_desc, argmax_y_ptr, spatial_scale,
          sampling_ratio, aligned, pool_mode, grads_image_desc,
          grads_image_ptr));
    }
  } else if (version == 1) {
    VLOG(4) << "call mluOp mluOpRoiAlignBackward_v2";
    if (pool_mode == 1) {
      VLOG(4) << "call mluOp mluOpRoiAlignBackward_v2 with average mode";
      mluOpTensorDescriptor_t argmax_x_desc = nullptr;
      mluOpTensorDescriptor_t argmax_y_desc = nullptr;
      void *argmax_x_ptr = NULL;
      void *argmax_y_ptr = NULL;
      auto grads_image_desc = parser_->getMetaTensor(2).tensor;
      auto grads_image_ptr = parser_->getMetaTensor(2).dev_ptr;

      MLUOP_CHECK(mluOpRoiAlignBackward_v2(
          handle_, grads_desc, grads_ptr, boxes_desc, boxes_ptr, argmax_x_desc,
          argmax_x_ptr, argmax_y_desc, argmax_y_ptr, spatial_scale,
          sampling_ratio, aligned, pool_mode, grads_image_desc,
          grads_image_ptr));
    } else {
      VLOG(4) << "call mluOp mluOpRoiAlignBackward_v2 with max mode";
      auto argmax_x_desc = parser_->getMetaTensor(2).tensor;
      auto argmax_y_desc = parser_->getMetaTensor(3).tensor;
      auto grads_image_desc = parser_->getMetaTensor(4).tensor;

      auto argmax_x_ptr = parser_->getMetaTensor(2).dev_ptr;
      auto argmax_y_ptr = parser_->getMetaTensor(3).dev_ptr;
      auto grads_image_ptr = parser_->getMetaTensor(4).dev_ptr;

      MLUOP_CHECK(mluOpRoiAlignBackward_v2(
          handle_, grads_desc, grads_ptr, boxes_desc, boxes_ptr, argmax_x_desc,
          argmax_x_ptr, argmax_y_desc, argmax_y_ptr, spatial_scale,
          sampling_ratio, aligned, pool_mode, grads_image_desc,
          grads_image_ptr));
    }
  }
  interface_timer_.stop();
}

void RoiAlignBackwardExecutor::bilinear_interpolate_gradient(
    const int height, const int width, float y, float x, float &w1, float &w2,
    float &w3, float &w4, int &x_low, int &x_high, int &y_low, int &y_high) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = w4 = 0.0;
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

  float ly = y - y_low, lx = x - x_low;
  float hy = 1. - ly, hx = 1. - lx;
  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
  return;
}

void RoiAlignBackwardExecutor::cpuCompute() {
  auto input = parser_->getMetaTensor(0).cpu_ptr;
  auto boxes = parser_->getMetaTensor(1).cpu_ptr;
  auto input_desc = parser_->getMetaTensor(0).tensor;
  float spatial_scale =
      parser_->getProtoNode()->roi_align_backward_param().spatial_scale();
  int sampling_ratio =
      parser_->getProtoNode()->roi_align_backward_param().sampling_ratio();
  bool aligned = parser_->getProtoNode()->roi_align_backward_param().aligned();
  int pool_mode =
      parser_->getProtoNode()->roi_align_backward_param().pool_mode();
  int version = parser_->getProtoNode()->roi_align_backward_param().version();

  size_t input_n = input_desc->getDimIndex(0);
  size_t input_h = input_desc->getDimIndex(1);
  size_t input_w = input_desc->getDimIndex(2);
  size_t input_c = input_desc->getDimIndex(3);
  size_t input_offset_n = input_h * input_w * input_c;
  size_t input_offset_h = input_w * input_c;
  auto output = parser_->getMetaTensor(2).cpu_ptr;
  auto output_desc = parser_->getMetaTensor(2).tensor;

  if (pool_mode == 1) {
    int output_n = output_desc->getDimIndex(0);
    int output_h = output_desc->getDimIndex(1);
    int output_w = output_desc->getDimIndex(2);
    int output_c = output_desc->getDimIndex(3);

    std::memset(output, 0.0, parser_->getMetaTensor(2).size_in_bytes);
    size_t output_offset_n = output_h * output_w * output_c;
    size_t output_offset_h = output_w * output_c;

    for (int idx_n = 0; idx_n < input_n; idx_n++) {
      // check whether box_idx is valid
      int32_t curr_idx = (int32_t)boxes[idx_n * 5];
      if (curr_idx < 0 || curr_idx >= output_n) {
        continue;
      }
      int output_offset = curr_idx * output_offset_n;

      float offset = aligned ? 0.5 : 0;
      float x1 = boxes[idx_n * 5 + 1] * spatial_scale - offset;
      float y1 = boxes[idx_n * 5 + 2] * spatial_scale - offset;
      float x2 = boxes[idx_n * 5 + 3] * spatial_scale - offset;
      float y2 = boxes[idx_n * 5 + 4] * spatial_scale - offset;
      float roi_width = x2 - x1;
      float roi_height = y2 - y1;
      if (!aligned) {
        roi_width = std::max(roi_width, (float)1.0);
        roi_height = std::max(roi_height, (float)1.0);
      }

      float bin_size_h = roi_height / input_h;
      float bin_size_w = roi_width / input_w;
      int roi_bin_grid_h =
          (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / input_h);
      int roi_bin_grid_w =
          (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / input_w);
      const float count = roi_bin_grid_h * roi_bin_grid_w;

      for (int ih = 0; ih < input_h; ++ih) {
        for (int iw = 0; iw < input_w; ++iw) {
          for (int ic = 0; ic < input_c; ++ic) {
            float input_this_bin =
                input[idx_n * input_offset_n + ih * input_offset_h +
                      iw * input_c + ic];
            for (int iy = 0; iy < roi_bin_grid_h; ++iy) {
              const float y = y1 + ih * bin_size_h +
                              (iy + .5) * bin_size_h / (float)roi_bin_grid_h;
              for (int ix = 0; ix < roi_bin_grid_w; ++ix) {
                const float x = x1 + iw * bin_size_w +
                                (ix + .5) * bin_size_w / (float)roi_bin_grid_w;

                float w1, w2, w3, w4;
                int x_low, x_high, y_low, y_high;
                bilinear_interpolate_gradient(output_h, output_w, y, x, w1, w2,
                                              w3, w4, x_low, x_high, y_low,
                                              y_high);
                float g1 = input_this_bin * w1 / count;
                float g2 = input_this_bin * w2 / count;
                float g3 = input_this_bin * w3 / count;
                float g4 = input_this_bin * w4 / count;

                if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
                  output[output_offset + y_low * output_offset_h +
                         x_low * output_c + ic] += g1;
                  output[output_offset + y_low * output_offset_h +
                         x_high * output_c + ic] += g2;
                  output[output_offset + y_high * output_offset_h +
                         x_low * output_c + ic] += g3;
                  output[output_offset + y_high * output_offset_h +
                         x_high * output_c + ic] += g4;
                }
              }  // for ix
            }    // for iy
          }      // for ic
        }        // for iw
      }          // for ih
    }            // for idx_n
  } else if (pool_mode == 0) {
    auto argmax_x = parser_->getMetaTensor(2).cpu_ptr;
    auto argmax_x_desc = parser_->getMetaTensor(2).tensor;
    auto argmax_y = parser_->getMetaTensor(3).cpu_ptr;
    auto argmax_y_desc = parser_->getMetaTensor(3).tensor;
    output = parser_->getMetaTensor(4).cpu_ptr;
    output_desc = parser_->getMetaTensor(4).tensor;

    size_t output_n = output_desc->getDimIndex(0);
    size_t output_h = output_desc->getDimIndex(1);
    size_t output_w = output_desc->getDimIndex(2);
    size_t output_c = output_desc->getDimIndex(3);

    size_t output_offset_n = output_h * output_w * output_c;
    size_t output_offset_h = output_w * output_c;

    // set zeros to all elements of output
    std::memset(output, 0.0, parser_->getMetaTensor(4).size_in_bytes);

    for (int idx_n = 0; idx_n < input_n; idx_n++) {
      // check whether box_idx is valid
      int curr_idx = (int)boxes[idx_n * 5];
      if (curr_idx < 0 || curr_idx >= output_n) {
        LOG(ERROR)
            << "mluOpRoiAlignBackward: boxes_id is out range of output_n.";
        continue;
      }
      int output_offset = curr_idx * output_offset_n;

      for (int ih = 0; ih < input_h; ++ih) {
        for (int iw = 0; iw < input_w; ++iw) {
          for (int ic = 0; ic < input_c; ++ic) {
            int index = idx_n * input_offset_n + ih * input_offset_h +
                        iw * input_c + ic;
            float input_this_bin = input[index];
            const float y = argmax_y[index];
            const float x = argmax_x[index];
            if (y != -1.f) {
              float w1, w2, w3, w4;
              int x_low, x_high, y_low, y_high;
              bilinear_interpolate_gradient(output_h, output_w, y, x, w1, w2,
                                            w3, w4, x_low, x_high, y_low,
                                            y_high);
              if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
                float g1 = input_this_bin * w1;
                float g2 = input_this_bin * w2;
                float g3 = input_this_bin * w3;
                float g4 = input_this_bin * w4;

                output[output_offset + y_low * output_offset_h +
                       x_low * output_c + ic] += g1;
                output[output_offset + y_low * output_offset_h +
                       x_high * output_c + ic] += g2;
                output[output_offset + y_high * output_offset_h +
                       x_low * output_c + ic] += g3;
                output[output_offset + y_high * output_offset_h +
                       x_high * output_c + ic] += g4;
              }  // if x_low, x_high, y_low, y_high
            }    // if y
          }      // for ic
        }        // for iw
      }          // for ih
    }            // for idx_n
  }              // if pool_mode
  return;
}  // cpuCompute()

int64_t RoiAlignBackwardExecutor::getTheoryOps() {
  int64_t theory_ops = 0;
  float *host_boxes = nullptr;
  Device device = parser_->device();
  if (device != Device::CPU) {
    auto boxes_desc = tensor_desc_[1].tensor;
    auto boxes_dtype = boxes_desc->getDtype();
    size_t boxes_num = parser_->getInputDataCount(1);
    float *boxes_ptr =
        (float *)cpu_runtime_.allocate(boxes_num * sizeof(float));
    castDataOut(data_vector_[1].host_ptr, boxes_dtype, (float *)boxes_ptr,
                MLUOP_DTYPE_FLOAT, boxes_num, NO_QUANT, 0, 1, 0);
    host_boxes = boxes_ptr;
  } else {
    host_boxes = cpu_fp32_input_[1];
  }
  auto boxes = host_boxes;
  auto input_desc = parser_->getMetaTensor(0).tensor;
  float spatial_scale =
      parser_->getProtoNode()->roi_align_backward_param().spatial_scale();
  int sampling_ratio =
      parser_->getProtoNode()->roi_align_backward_param().sampling_ratio();
  bool aligned = parser_->getProtoNode()->roi_align_backward_param().aligned();
  int pool_mode =
      parser_->getProtoNode()->roi_align_backward_param().pool_mode();
  int version = parser_->getProtoNode()->roi_align_backward_param().version();

  auto output_desc = parser_->getMetaTensor(2).tensor;
  if (pool_mode == 0) {
    output_desc = parser_->getMetaTensor(4).tensor;
  }

  size_t input_n = input_desc->getDimIndex(0);
  size_t input_h = input_desc->getDimIndex(1);
  size_t input_w = input_desc->getDimIndex(2);
  size_t input_c = input_desc->getDimIndex(3);
  size_t output_n = output_desc->getDimIndex(0);

  for (int idx_n = 0; idx_n < input_n; idx_n++) {
    // check whether box_idx is valid
    int32_t curr_idx = (int32_t)boxes[idx_n * 5];
    if (curr_idx < 0 || curr_idx >= output_n) {
      continue;
    }

    float offset = aligned ? 0.5 : 0;
    float x1 = boxes[idx_n * 5 + 1] * spatial_scale - offset;
    float y1 = boxes[idx_n * 5 + 2] * spatial_scale - offset;
    float x2 = boxes[idx_n * 5 + 3] * spatial_scale - offset;
    float y2 = boxes[idx_n * 5 + 4] * spatial_scale - offset;
    float roi_width = x2 - x1;
    float roi_height = y2 - y1;
    if (!aligned) {
      roi_width = std::max(roi_width, (float)1.0);
      roi_height = std::max(roi_height, (float)1.0);
    }

    int roi_bin_grid_h =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / input_h);
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / input_w);
    if (pool_mode == 1) {
      theory_ops += 14 + 14 * input_h * input_w * input_c * roi_bin_grid_h *
                             roi_bin_grid_w;
    } else if (pool_mode == 0) {
      theory_ops += 11 * input_h * input_w * input_c;
    }
  }

  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
