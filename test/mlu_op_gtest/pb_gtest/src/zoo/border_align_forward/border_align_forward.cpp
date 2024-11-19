/*******************************************************************************
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
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *******************************************************************************/
#include "border_align_forward.h"

#include <string>

namespace mluoptest {

void BorderAlignForwardExecutor::paramCheck() {
  if (parser_->getInputNum() != 2) {
    LOG(ERROR) << "mluOpBorderAlignForward: tensor input number is wrong.";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
  if (parser_->getOutputNum() != 2) {
    LOG(ERROR) << "output number is" << parser_->getOutputNum();
    LOG(ERROR) << "mluOpBorderAlignForward: tensor output number is wrong.";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
}

void BorderAlignForwardExecutor::compute() {
  VLOG(4) << "Border Align Forward Executor compute";
  auto input_desc = tensor_desc_[0].tensor;
  auto input_dev = data_vector_[0].device_ptr;
  auto boxes_desc = tensor_desc_[1].tensor;
  auto boxes_dev = data_vector_[1].device_ptr;
  const int32_t pool_size =
      parser_->getProtoNode()->border_align_param().pool_size();
  auto output_desc = tensor_desc_[2].tensor;
  auto output_dev = data_vector_[2].device_ptr;
  auto argmax_idx_desc = tensor_desc_[3].tensor;
  auto argmax_idx_dev = data_vector_[3].device_ptr;
  VLOG(4) << "call mluOpBorderAlignForward()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpBorderAlignForward(
      handle_, input_desc, input_dev, boxes_desc, boxes_dev, pool_size,
      output_desc, output_dev, argmax_idx_desc, argmax_idx_dev));
  interface_timer_.stop();
}

void BorderAlignForwardExecutor::setMiscellaneousParam() {
  data_vector_[2].alsoServeAsOutput();
  data_vector_[3].alsoServeAsOutput();
}

float bilinear_interpolate(const float *input, const int32_t H, const int32_t W,
                           const int32_t C, float y, float x) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > H || x < -1.0 || x > W) return 0;

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int32_t y_low = (int32_t)y;
  int32_t x_low = (int32_t)x;
  int32_t y_high;
  int32_t x_high;

  if (y_low >= H - 1) {
    y_high = y_low = H - 1;
    y = (float)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= W - 1) {
    x_high = x_low = W - 1;
    x = (float)x_low;
  } else {
    x_high = x_low + 1;
  }

  float ly = y - y_low;
  float lx = x - x_low;
  float hy = 1. - ly, hx = 1. - lx;
  int32_t v1_pos = y_low * W + x_low;
  int32_t v2_pos = y_low * W + x_high;
  int32_t v3_pos = y_high * W + x_low;
  int32_t v4_pos = y_high * W + x_high;

  float v1 = input[v1_pos / W * W * C * 4 + v1_pos % W * C * 4];
  float v2 = input[v2_pos / W * W * C * 4 + v2_pos % W * C * 4];
  float v3 = input[v3_pos / W * W * C * 4 + v3_pos % W * C * 4];
  float v4 = input[v4_pos / W * W * C * 4 + v4_pos % W * C * 4];

  float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

void BorderAlignForwardExecutor::cpuCompute() {
  auto input_desc = parser_->getMetaTensor(0).tensor;
  auto boxes_desc = parser_->getMetaTensor(1).tensor;
  const int32_t N = input_desc->getDimIndex(0);
  const int32_t H = input_desc->getDimIndex(1);
  const int32_t W = input_desc->getDimIndex(2);
  const int32_t C = input_desc->getDimIndex(3) / 4;
  const int32_t K = boxes_desc->getDimIndex(1);
  float x1, x2, y1, y2;
  float x_stride = 0;
  float y_stride = 0;
  int32_t i = 0;
  float x = 0;
  float y = 0;
  const int32_t pool_size =
      parser_->getProtoNode()->border_align_param().pool_size();
  float max_pool_result_temp = 0.0;
  float max_pool_result = 0.0;
  const float *input = cpu_fp32_input_[0];
  const float *boxes = cpu_fp32_input_[1];
  int32_t argmax_idx = 0;
  for (int32_t n = 0; n < N; ++n) {
    for (int32_t k = 0; k < K; ++k) {
      int32_t bbox_offset = n * K * 4 + k * 4;
      x1 = boxes[bbox_offset];
      y1 = boxes[bbox_offset + 1];
      x2 = boxes[bbox_offset + 2];
      y2 = boxes[bbox_offset + 3];
      float bbox_width = x2 - x1;
      float bbox_height = y2 - y1;
      for (int32_t border_loop = 0; border_loop < 4; ++border_loop) {
        for (int32_t c = 0; c < C; ++c) {
          if (pool_size != 0) {
            switch (border_loop) {
              case 0: {
                x_stride = bbox_width / pool_size;
                y_stride = 0;
              } break;
              case 1: {
                x_stride = 0;
                y_stride = bbox_height / pool_size;
              } break;
              case 2: {
                x_stride = -bbox_width / pool_size;
                y_stride = 0;
              } break;
              case 3: {
                x_stride = 0;
                y_stride = -bbox_height / pool_size;
              } break;
              default: {
                VLOG(4) << "Invalid Border Type.";
              } break;
            }
          }
          x = boxes[bbox_offset + border_loop / 2 * 2];
          y = boxes[bbox_offset + border_loop / 2 * 2 + 1];
          int32_t input_offset = n * H * W * C * 4 + border_loop * C + c;
          max_pool_result =
              bilinear_interpolate(input + input_offset, H, W, C, y, x);
          int32_t argmax_idx = 0;
          for (int32_t pool_size_idx = 1; pool_size_idx <= pool_size;
               ++pool_size_idx) {
            x += x_stride;
            y += y_stride;
            max_pool_result_temp =
                bilinear_interpolate(input + input_offset, H, W, C, y, x);
            if (max_pool_result_temp - max_pool_result > 0) {
              max_pool_result = max_pool_result_temp;
              argmax_idx = pool_size_idx;
            }
          }

          cpu_fp32_output_[0][i] = max_pool_result;
          cpu_fp32_output_[1][i] = argmax_idx;
          i = i + 1;
        }
      }
    }
  }
}

int64_t BorderAlignForwardExecutor::getTheoryOps() {
  auto input_desc = parser_->getMetaTensor(0).tensor;
  auto boxes_desc = parser_->getMetaTensor(1).tensor;
  const int32_t N = input_desc->getDimIndex(0);
  const int32_t C = input_desc->getDimIndex(3) / 4;
  const int32_t K = boxes_desc->getDimIndex(1);

  const int64_t theory_ops = N * K * 4 * C * 14;
  return theory_ops;
}

}  // namespace mluoptest
