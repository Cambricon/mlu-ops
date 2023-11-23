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
#include "border_align_backward.h"

#include <string>

namespace mluoptest {

void BorderAlignBackwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->getInputNum() == 3,
              "mluOpBorderAlignBackward: tensor input number is wrong.");
  GTEST_CHECK(parser_->getOutputNum() == 1,
              "mluOpBorderAlignBackward: tensor output number is wrong.");
}

void BorderAlignBackwardExecutor::compute() {
  VLOG(4) << "Border Align Backward Executor compute.";
  auto grad_output_desc = tensor_desc_[0].tensor;
  auto grad_output_dev = data_vector_[0].device_ptr;
  auto boxes_desc = tensor_desc_[1].tensor;
  auto boxes_dev = data_vector_[1].device_ptr;
  auto argmax_idx_desc = tensor_desc_[2].tensor;
  auto argmax_idx_dev = data_vector_[2].device_ptr;

  const int32_t pool_size =
      parser_->getProtoNode()->border_align_param().pool_size();
  auto grad_input_desc = tensor_desc_[3].tensor;
  auto grad_input_dev = data_vector_[3].device_ptr;

  VLOG(4) << "call mluOpBorderAlignBackward().";
  interface_timer_.start();
  MLUOP_CHECK(mluOpBorderAlignBackward(
      handle_, grad_output_desc, grad_output_dev, boxes_desc, boxes_dev,
      argmax_idx_desc, argmax_idx_dev, pool_size, grad_input_desc,
      grad_input_dev));
  interface_timer_.stop();
}

void BorderAlignBackwardExecutor::setMiscellaneousParam() {
  data_vector_[3].alsoServeAsOutput();
}

void bilinear_interpolate_gradient(
    const int32_t height, const int32_t width, float y, float x, float &w1,
    float &w2, float &w3, float &w4, int32_t &x_low, int32_t &x_high,
    int32_t &y_low, int32_t &y_high,
    const int32_t index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;

    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int32_t)y;
  x_low = (int32_t)x;

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

  return;
}

void BorderAlignBackwardExecutor::cpuCompute() {
  auto grad_output_desc = parser_->getMetaTensor(0).tensor;
  auto boxes_desc = parser_->getMetaTensor(1).tensor;
  auto argmax_idx_desc = parser_->getMetaTensor(2).tensor;
  auto grad_input_desc = parser_->getMetaTensor(3).tensor;
  const int32_t grad_output_size = parser_->getInputDataCount(0);
  float *grad_output = cpu_fp32_input_[0];
  float *boxes = cpu_fp32_input_[1];
  float *argmax_idx = cpu_fp32_input_[2];
  float *grad_input = cpu_fp32_output_[0];
  const int32_t box_size = boxes_desc->dims[1];
  const int32_t channels = grad_output_desc->dims[3];
  const int32_t height = grad_input_desc->dims[1];
  const int32_t width = grad_input_desc->dims[2];
  const int32_t N = grad_output_desc->dims[0];
  const int32_t H = grad_output_desc->dims[1];
  const int32_t W = grad_output_desc->dims[2];
  const int32_t C = grad_output_desc->dims[3];

  const int32_t N1 = grad_input_desc->dims[0];
  const int32_t H1 = grad_input_desc->dims[1];
  const int32_t W1 = grad_input_desc->dims[2];
  const int32_t C1 = grad_input_desc->dims[3];
  float x_stride = 0;
  float y_stride = 0;
  float stride = 0;
  auto transData = [&](float *old_data, float *new_data, TensorLayout old_order,
                       TensorLayout new_order, int32_t n, int32_t c, int32_t h,
                       int32_t w) {
    if (old_data == nullptr || new_data == nullptr) {
      LOG(ERROR)
          << "Data address malloc error in border_align_backward cpu compute.";
      return;
    }

    if (LAYOUT_NHWC == old_order && LAYOUT_NCHW == new_order) {
      for (int32_t nn = 0; nn < n; nn++) {
        for (int32_t cc = 0; cc < c; cc++) {
          for (int32_t hh = 0; hh < h; hh++) {
            for (int32_t ww = 0; ww < w; ww++) {
              new_data[nn * c * h * w + cc * h * w + hh * w + ww] =
                  old_data[nn * h * w * c + hh * w * c + ww * c + cc];
            }
          }
        }
      }
    } else if (LAYOUT_NCHW == old_order && LAYOUT_NHWC == new_order) {
      for (int32_t nn = 0; nn < n; nn++) {
        for (int32_t hh = 0; hh < h; hh++) {
          for (int32_t ww = 0; ww < w; ww++) {
            for (int32_t cc = 0; cc < c; cc++) {
              new_data[nn * c * h * w + hh * w * c + ww * c + cc] =
                  old_data[nn * c * h * w + cc * h * w + hh * w + ww];
            }
          }
        }
      }
    }
  };
  float *grad_output_nchw =
      (float *)cpu_runtime_.allocate(N * H * W * C * sizeof(float));
  transData(grad_output, grad_output_nchw, LAYOUT_NHWC, LAYOUT_NCHW, N, C, H,
            W);

  float *argmax_idx_nchw =
      (float *)cpu_runtime_.allocate(N * C * H * W * sizeof(float));
  transData(argmax_idx, argmax_idx_nchw, LAYOUT_NHWC, LAYOUT_NCHW, N, C, H, W);
  float *offset_grad_input_nhwc =
      (float *)cpu_runtime_.allocate(N1 * C1 * H1 * W1 * sizeof(float));
  float *offset_grad_input_nchw =
      (float *)cpu_runtime_.allocate(N1 * C1 * H1 * W1 * sizeof(float));
  const int32_t grad_input_size = parser_->getOutputDataCount(0);
  memset(offset_grad_input_nhwc, 0, grad_input_size * sizeof(float));
  memset(offset_grad_input_nchw, 0, grad_input_size * sizeof(float));
  const int32_t pool_size =
      parser_->getProtoNode()->border_align_param().pool_size();
  for (int32_t index = 0; index < grad_output_size / 4; ++index) {
    int32_t batch_idx = index / channels / box_size;
    int32_t box_idx = index % box_size + batch_idx * box_size;
    int32_t c_idx = (index / box_size) % channels;

    float *offset_box = boxes + box_idx * 4;
    float box_width = *(offset_box + 2) - *offset_box;
    float box_height = *(offset_box + 3) - *(offset_box + 1);
    for (int32_t border_loop = 0; border_loop < 4; ++border_loop) {
      float *offset_grad_output = grad_output_nchw + index * 4 + border_loop;
      float *offset_argmax_idx = argmax_idx_nchw + index * 4 + border_loop;
      int32_t offset_grad_input =
          (batch_idx * channels * 4 + border_loop * channels + c_idx) * height *
          width;
      float *offset_box_x = offset_box + border_loop / 2 * 2;
      switch (border_loop % 4) {
        // top
        case 0:
          stride = box_width / pool_size;
          x_stride = stride;
          y_stride = 0;
          break;
        // left
        case 1:
          stride = box_height / pool_size;
          x_stride = 0;
          y_stride = stride;
          break;
        // bottom
        case 2:
          stride = box_width / pool_size;
          x_stride = -stride;
          y_stride = 0;
          break;
        // right
        case 3:
          stride = box_height / pool_size;
          x_stride = 0;
          y_stride = -stride;
          break;
      }

      // get position (x,y) which has maximum value during forward
      float x = *offset_box_x;
      float y = *(offset_box_x + 1);
      x += x_stride * (float)(*offset_argmax_idx);
      y += y_stride * (float)(*offset_argmax_idx);
      float w1 = 0.0, w2 = 0.0, w3 = 0.0, w4 = 0.0;
      int32_t x_low, x_high, y_low, y_high;
      bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4, x_low,
                                    x_high, y_low, y_high, index);

      int32_t v1_pos = y_low * width + x_low;
      int32_t v2_pos = y_low * width + x_high;
      int32_t v3_pos = y_high * width + x_low;
      int32_t v4_pos = y_high * width + x_high;

      offset_grad_input_nchw[offset_grad_input + v1_pos] +=
          *offset_grad_output * w1;
      offset_grad_input_nchw[offset_grad_input + v2_pos] +=
          *offset_grad_output * w2;
      offset_grad_input_nchw[offset_grad_input + v3_pos] +=
          *offset_grad_output * w3;
      offset_grad_input_nchw[offset_grad_input + v4_pos] +=
          *offset_grad_output * w4;
    }
  }
  transData(offset_grad_input_nchw, offset_grad_input_nhwc, LAYOUT_NCHW,
            LAYOUT_NHWC, N1, C1, H1, W1);
  for (int32_t i = 0; i < N1 * H1 * C1 * W1; ++i) {
    cpu_fp32_output_[0][i] = offset_grad_input_nhwc[i];
  }
  cpu_runtime_.deallocate(grad_output_nchw);
  cpu_runtime_.deallocate(argmax_idx_nchw);
  cpu_runtime_.deallocate(offset_grad_input_nhwc);
  cpu_runtime_.deallocate(offset_grad_input_nchw);
}

int64_t BorderAlignBackwardExecutor::getTheoryOps() {
  auto input_desc = parser_->getMetaTensor(0).tensor;
  auto boxes_desc = parser_->getMetaTensor(1).tensor;
  const int32_t N = input_desc->dims[0];
  const int32_t C = input_desc->dims[3] / 4;
  const int32_t K = boxes_desc->dims[1];

  const int64_t theory_ops = N * K * 4 * C * 3;
  return theory_ops;
}

}  // namespace mluoptest
