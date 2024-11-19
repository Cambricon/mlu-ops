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
#include <string>
#include <algorithm>
#include "roialign_forward.h"
#include "mlu_op.h"

namespace mluoptest {

void RoialignForwardExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_roialign_param()) {
    LOG(ERROR) << "mluOpRoiAlignForward_v2: lose roialign_param. ";
  }
  if (parser_->getInputNum() != 2) {
    LOG(ERROR) << "mluOpRoiAlignForward_v2: tensor input number is wrong.";
  }
  if (parser_->getOutputNum() != 1 && parser_->getOutputNum() != 3) {
    LOG(ERROR) << "output number is" << parser_->getOutputNum();
    LOG(ERROR) << "mluOpRoiAlignForward_v2: tensor output number is wrong.";
  }
}

void RoialignForwardExecutor::compute() {
  float spatial_scale =
      parser_->getProtoNode()->roialign_param().spatial_scale();
  int sampling_ratio =
      parser_->getProtoNode()->roialign_param().sampling_ratio();
  int pool_mode = parser_->getProtoNode()->roialign_param().pool_mode();
  int verison = parser_->getProtoNode()->roialign_param().version();
  bool aligned = parser_->getProtoNode()->roialign_param().aligned();
  auto input_desc = parser_->getMetaTensor(0).tensor;
  auto input_rois_desc = parser_->getMetaTensor(1).tensor;
  auto output_desc = parser_->getMetaTensor(2).tensor;
  int pooled_height = output_desc->getDimIndex(1);
  int pooled_width = output_desc->getDimIndex(2);

  mluOpRoiAlignForwardDescriptor_t roialign_desc;
  mluOpCreateRoiAlignForwardDescriptor(&roialign_desc);

  void *input_dev = data_vector_[0].device_ptr;
  void *input_rois_dev = data_vector_[1].device_ptr;
  void *output_dev = data_vector_[2].device_ptr;

  VLOG(4) << "call mluOpRoiAlignForward";
  interface_timer_.start();

  {
    VLOG(4) << "verison = " << verison;
    mluOpSetRoiAlignForwardDescriptor_v2(roialign_desc, pooled_height,
                                         pooled_width, sampling_ratio,
                                         spatial_scale, pool_mode, aligned);
    if (pool_mode == 0) {
      auto output_argmax_x_desc = parser_->getMetaTensor(3).tensor;
      auto output_argmax_y_desc = parser_->getMetaTensor(4).tensor;
      void *output_argmax_x_dev = data_vector_[3].device_ptr;
      void *output_argmax_y_dev = data_vector_[4].device_ptr;
      MLUOP_CHECK(mluOpRoiAlignForward_v2(
          handle_, roialign_desc, input_desc, input_dev, input_rois_desc,
          input_rois_dev, output_desc, output_dev, output_argmax_x_desc,
          output_argmax_x_dev, output_argmax_y_desc, output_argmax_y_dev));

    } else if (pool_mode == 1) {
      mluOpTensorDescriptor_t output_argmax_x_desc = nullptr;
      mluOpTensorDescriptor_t output_argmax_y_desc = nullptr;
      void *output_argmax_x_dev = NULL;
      void *output_argmax_y_dev = NULL;
      MLUOP_CHECK(mluOpRoiAlignForward_v2(
          handle_, roialign_desc, input_desc, input_dev, input_rois_desc,
          input_rois_dev, output_desc, output_dev, output_argmax_x_desc,
          output_argmax_x_dev, output_argmax_y_desc, output_argmax_y_dev));
    }
  }
  interface_timer_.stop();
  mluOpDestroyRoiAlignForwardDescriptor(roialign_desc);
}

void RoialignForwardExecutor::bilinear_interpolate(
    int height, int width, float y, float x, float &w1, float &w2, float &w3,
    float &w4, int &x_low, int &x_high, int &y_low, int &y_high, int &empty) {
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    empty = 1;
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

void RoialignForwardExecutor::cpuCompute() {
  float spatial_scale =
      parser_->getProtoNode()->roialign_param().spatial_scale();
  int sampling_ratio =
      parser_->getProtoNode()->roialign_param().sampling_ratio();
  bool aligned = parser_->getProtoNode()->roialign_param().aligned();
  auto input_desc = parser_->getMetaTensor(0).tensor;
  auto input_rois_desc = parser_->getMetaTensor(1).tensor;
  auto output_desc = parser_->getMetaTensor(2).tensor;
  int verison = parser_->getProtoNode()->roialign_param().version();
  int pool_mode = parser_->getProtoNode()->roialign_param().pool_mode();

  int input_height = input_desc->getDimIndex(1);
  int input_width = input_desc->getDimIndex(2);
  int pooled_height = output_desc->getDimIndex(1);
  int pooled_width = output_desc->getDimIndex(2);
  int channels = input_desc->getDimIndex(3);
  int num_rois = input_rois_desc->getDimIndex(0);
  int roi_offset = input_rois_desc->getDimIndex(1);
  int input_n = input_desc->getDimIndex(0);

  float *input = cpu_fp32_input_[0];
  float *input_rois = cpu_fp32_input_[1];  // (n, 5) { n, x0, y0, x1, y1}
  float *output = cpu_fp32_output_[0];
  if (pool_mode == 1) {
    // roialign cpu
    VLOG(4) << "BEGIN CPU pool_mode avg";

    float *pooled_value =
        (float *)cpu_runtime_.allocate(channels * sizeof(float));
    int channel_idx = 0;

    for (int roi_idx = 0; roi_idx < num_rois; roi_idx++) {
      int batch_idx = int(input_rois[roi_idx * roi_offset]);
      if (batch_idx < 0 || batch_idx >= input_n) {
        LOG(ERROR) << "RoiAlign cpu : batch_id should be in [0," << input_n - 1
                   << "]. But batch_id is " << batch_idx;
      }
      float roi_x1 = input_rois[roi_idx * roi_offset + 1];
      float roi_y1 = input_rois[roi_idx * roi_offset + 2];
      float roi_x2 = input_rois[roi_idx * roi_offset + 3];
      float roi_y2 = input_rois[roi_idx * roi_offset + 4];

      int height = input_height;
      int width = input_width;
      float offset = aligned ? 0.5 : 0.0;

      float roi_start_w = roi_x1 * spatial_scale - offset;
      float roi_start_h = roi_y1 * spatial_scale - offset;
      float roi_end_w = roi_x2 * spatial_scale - offset;
      float roi_end_h = roi_y2 * spatial_scale - offset;

      float roi_width = roi_end_w - roi_start_w;
      float roi_height = roi_end_h - roi_start_h;

      if (!aligned) {
        roi_width = roi_width > 1 ? roi_width : 1;
        roi_height = roi_height > 1 ? roi_height : 1;
      }

      float bin_size_h = (float)roi_height / pooled_height;
      float bin_size_w = (float)roi_width / pooled_width;

      int roi_bin_grid_h =
          (sampling_ratio > 0) ? sampling_ratio : ceil(bin_size_h);
      int roi_bin_grid_w =
          (sampling_ratio > 0) ? sampling_ratio : ceil(bin_size_w);
      float count = (roi_bin_grid_h * roi_bin_grid_w) > 1
                        ? roi_bin_grid_h * roi_bin_grid_w
                        : 1;
      float count_value = 1.0f / count;

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          float *output_channel_ptr =
              output + roi_idx * pooled_height * pooled_width * channels +
              ph * pooled_width * channels + pw * channels;

          memset(pooled_value, 0, channels * sizeof(float));

          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            float y =
                roi_start_h + ph * bin_size_h +
                (iy + 0.5) * bin_size_h / (roi_bin_grid_h);  // center_point y
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              float x =
                  roi_start_w + pw * bin_size_w +
                  (ix + 0.5) * bin_size_w / (roi_bin_grid_w);  // center_point x
              int empty = 0;
              float w1, w2, w3, w4;
              int x_low, x_high, y_low, y_high;
              bilinear_interpolate(height, width, y, x, w1, w2, w3, w4, x_low,
                                   x_high, y_low, y_high, empty);

              float *input_temp = input + batch_idx * width * height * channels;
              float *input_channel_ptr_1 =
                  input_temp + (y_low * width + x_low) * channels;
              float *input_channel_ptr_2 =
                  input_temp + (y_low * width + x_high) * channels;
              float *input_channel_ptr_3 =
                  input_temp + (y_high * width + x_low) * channels;
              float *input_channel_ptr_4 =
                  input_temp + (y_high * width + x_high) * channels;

              for (channel_idx = 0; channel_idx < channels; channel_idx++) {
                float value;
                if (empty == 1) {
                  value = 0;
                } else {
                  float v1 = input_channel_ptr_1[channel_idx];
                  float v2 = input_channel_ptr_2[channel_idx];
                  float v3 = input_channel_ptr_3[channel_idx];
                  float v4 = input_channel_ptr_4[channel_idx];
                  // bilinear interpolatation
                  value = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                }
                pooled_value[channel_idx] += value;  // sum
              }                                      // channels
            }                                        // roi_bin_grid_w
          }                                          // roi_bin_grid_h
          for (channel_idx = 0; channel_idx < channels; channel_idx++) {
            pooled_value[channel_idx] = pooled_value[channel_idx] * count_value;
          }
          memcpy(output_channel_ptr, pooled_value, channels * sizeof(float));
        }  // pw
      }    // ph
    }      // roi
    cpu_runtime_.deallocate(pooled_value);
  } else if (pool_mode == 0) {
    // roialign cpu

    VLOG(4) << "BEGIN CPU API version 1 and pool_mode max";
    auto output_argmax_x_desc = parser_->getMetaTensor(3).tensor;
    auto output_argmax_y_desc = parser_->getMetaTensor(4).tensor;

    float *output_argmax_x = cpu_fp32_output_[1];
    float *output_argmax_y = cpu_fp32_output_[2];
    float *pooled_value =
        (float *)cpu_runtime_.allocate(channels * sizeof(float));
    float *argmax_x_value =
        (float *)cpu_runtime_.allocate(channels * sizeof(float));
    float *argmax_y_value =
        (float *)cpu_runtime_.allocate(channels * sizeof(float));

    for (int roi_idx = 0; roi_idx < num_rois; roi_idx++) {
      int batch_idx = int(input_rois[roi_idx * roi_offset + 0]);
      if (batch_idx < 0 || batch_idx >= input_n) {
        LOG(ERROR) << "RoiAlign cpu : batch_id should be in [0," << input_n - 1
                   << "]. But batch_id is " << batch_idx;
      }
      float roi_x1 = input_rois[roi_idx * roi_offset + 1];
      float roi_y1 = input_rois[roi_idx * roi_offset + 2];
      float roi_x2 = input_rois[roi_idx * roi_offset + 3];
      float roi_y2 = input_rois[roi_idx * roi_offset + 4];
      int height = input_height;
      int width = input_width;
      float offset = aligned ? 0.5 : 0.0;
      float roi_start_w = roi_x1 * spatial_scale - offset;
      float roi_start_h = roi_y1 * spatial_scale - offset;
      float roi_end_w = roi_x2 * spatial_scale - offset;
      float roi_end_h = roi_y2 * spatial_scale - offset;

      float roi_width = roi_end_w - roi_start_w;
      float roi_height = roi_end_h - roi_start_h;

      if (!aligned) {
        roi_width = roi_width > 1 ? roi_width : 1;
        roi_height = roi_height > 1 ? roi_height : 1;
      }

      float bin_size_h = (float)roi_height / pooled_height;
      float bin_size_w = (float)roi_width / pooled_width;

      int roi_bin_grid_h =
          (sampling_ratio > 0) ? sampling_ratio : ceil(bin_size_h);
      int roi_bin_grid_w =
          (sampling_ratio > 0) ? sampling_ratio : ceil(bin_size_w);

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          float *output_channel_ptr =
              output + roi_idx * pooled_height * pooled_width * channels +
              ph * pooled_width * channels + pw * channels;
          float *output_argmax_x_channel_ptr =
              output_argmax_x +
              roi_idx * pooled_height * pooled_width * channels +
              ph * pooled_width * channels + pw * channels;
          float *output_argmax_y_channel_ptr =
              output_argmax_y +
              roi_idx * pooled_height * pooled_width * channels +
              ph * pooled_width * channels + pw * channels;

          std::fill(pooled_value, pooled_value + channels, -FLT_MAX);
          std::fill(argmax_x_value, argmax_x_value + channels, -1);
          std::fill(argmax_y_value, argmax_y_value + channels, -1);
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            float y = roi_start_h + ph * bin_size_h +
                      (iy + 0.5) * bin_size_h / (roi_bin_grid_h);
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              float x = roi_start_w + pw * bin_size_w +
                        (ix + 0.5) * bin_size_w / (roi_bin_grid_w);
              float w1, w2, w3, w4;
              int x_low, x_high, y_low, y_high;
              int empty = 0;
              bilinear_interpolate(input_height, input_width, y, x, w1, w2, w3,
                                   w4, x_low, x_high, y_low, y_high, empty);
              float *input_temp = input + batch_idx * width * height * channels;
              float *input_channel_ptr_1 =
                  input_temp + (y_low * width + x_low) * channels;
              float *input_channel_ptr_2 =
                  input_temp + (y_low * width + x_high) * channels;
              float *input_channel_ptr_3 =
                  input_temp + (y_high * width + x_low) * channels;
              float *input_channel_ptr_4 =
                  input_temp + (y_high * width + x_high) * channels;
              for (int channel_idx = 0; channel_idx < channels; channel_idx++) {
                float value;
                if (empty == 1) {
                  value = 0;
                } else {
                  float v1 = input_channel_ptr_1[channel_idx];
                  float v2 = input_channel_ptr_2[channel_idx];
                  float v3 = input_channel_ptr_3[channel_idx];
                  float v4 = input_channel_ptr_4[channel_idx];
                  // bilinear interpolatation
                  value = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                }
                if (value > pooled_value[channel_idx]) {
                  pooled_value[channel_idx] = value;
                  argmax_x_value[channel_idx] = x;
                  argmax_y_value[channel_idx] = y;
                }
              }  // channels
            }    // sample w
          }      // sample h
          memcpy(output_channel_ptr, pooled_value, channels * sizeof(float));
          memcpy(output_argmax_x_channel_ptr, argmax_x_value,
                 channels * sizeof(float));
          memcpy(output_argmax_y_channel_ptr, argmax_y_value,
                 channels * sizeof(float));
        }  // pw
      }    // ph
    }      // roi
    cpu_runtime_.deallocate(pooled_value);
    cpu_runtime_.deallocate(argmax_x_value);
    cpu_runtime_.deallocate(argmax_y_value);
  }
}

int64_t RoialignForwardExecutor::getTheoryOps() {
  float spatial_scale =
      parser_->getProtoNode()->roialign_param().spatial_scale();
  int sampling_ratio =
      parser_->getProtoNode()->roialign_param().sampling_ratio();
  bool aligned = parser_->getProtoNode()->roialign_param().aligned();
  auto input_desc = parser_->getMetaTensor(0).tensor;
  auto input_rois_desc = parser_->getMetaTensor(1).tensor;
  auto output_desc = parser_->getMetaTensor(2).tensor;

  int input_height = input_desc->getDimIndex(1);
  int input_width = input_desc->getDimIndex(2);
  int pooled_height = output_desc->getDimIndex(1);
  int pooled_width = output_desc->getDimIndex(2);
  int channels = input_desc->getDimIndex(3);
  int num_rois = input_rois_desc->getDimIndex(0);
  int roi_offset = input_rois_desc->getDimIndex(1);
  int64_t theory_ops = 0;

  Device device = parser_->device();
  float *input_rois = NULL;

  auto rois_dtype = input_rois_desc->getDtype();
  int rois_count_num = num_rois * input_rois_desc->getDimIndex(1);
  float *rois_host =
      (float *)cpu_runtime_.allocate(rois_count_num * sizeof(float));
  castDataOut(data_vector_[1].host_ptr, rois_dtype, (float *)rois_host,
              MLUOP_DTYPE_FLOAT, rois_count_num, NO_QUANT, 0, 1, 0);
  input_rois = rois_host;

  // roialign theory
  for (int roi_idx = 0; roi_idx < num_rois; roi_idx++) {
    int batch_idx = int(input_rois[roi_idx * roi_offset]);
    float roi_x1 = input_rois[roi_idx * roi_offset + 1];
    float roi_y1 = input_rois[roi_idx * roi_offset + 2];
    float roi_x2 = input_rois[roi_idx * roi_offset + 3];
    float roi_y2 = input_rois[roi_idx * roi_offset + 4];

    int height = input_height;
    int width = input_width;
    float ss = spatial_scale;
    float offset = aligned ? 0.5 : 0.0;

    float roi_start_w = roi_x1 * ss - offset;
    float roi_start_h = roi_y1 * ss - offset;
    float roi_end_w = roi_x2 * ss - offset;
    float roi_end_h = roi_y2 * ss - offset;

    float roi_width = roi_end_w - roi_start_w;
    float roi_height = roi_end_h - roi_start_h;

    if (!aligned) {
      roi_width = roi_width > 1 ? roi_width : 1;
      roi_height = roi_height > 1 ? roi_height : 1;
    }

    float bin_size_h = (float)roi_height / pooled_height;
    float bin_size_w = (float)roi_width / pooled_width;
    int roi_bin_grid_h =
        (sampling_ratio > 0) ? sampling_ratio : ceil(bin_size_h);
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(bin_size_w);

    for (int ph = 0; ph < pooled_height; ph++) {
      for (int pw = 0; pw < pooled_width; pw++) {
        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
          for (int ix = 0; ix < roi_bin_grid_w; ix++) {
            for (int channel_idx = 0; channel_idx < channels; channel_idx++) {
              theory_ops += 8;
            }  // channels
          }    // input w
        }      // input h
      }        // pw
    }          // ph
  }            // roi
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  cpu_runtime_.deallocate(input_rois);

  return theory_ops;
}

int64_t RoialignForwardExecutor::getTheoryIoSize() {
  float spatial_scale =
      parser_->getProtoNode()->roialign_param().spatial_scale();
  int sampling_ratio =
      parser_->getProtoNode()->roialign_param().sampling_ratio();
  bool aligned = parser_->getProtoNode()->roialign_param().aligned();
  auto input_desc = parser_->getMetaTensor(0).tensor;
  auto input_rois_desc = parser_->getMetaTensor(1).tensor;
  auto output_desc = parser_->getMetaTensor(2).tensor;
  int pool_mode = parser_->getProtoNode()->roialign_param().pool_mode();

  int input_height = input_desc->getDimIndex(1);
  int input_width = input_desc->getDimIndex(2);
  int pooled_height = output_desc->getDimIndex(1);
  int pooled_width = output_desc->getDimIndex(2);
  int channels = input_desc->getDimIndex(3);
  int num_rois = input_rois_desc->getDimIndex(0);
  int roi_offset = input_rois_desc->getDimIndex(1);
  int64_t theory_io_size = 0;

  Device device = parser_->device();
  float *input_rois = NULL;

  auto rois_dtype = input_rois_desc->getDtype();
  int rois_count_num = num_rois * input_rois_desc->getDimIndex(1);
  float *rois_host =
      (float *)cpu_runtime_.allocate(rois_count_num * sizeof(float));
  castDataOut(data_vector_[1].host_ptr, rois_dtype, (float *)rois_host,
              MLUOP_DTYPE_FLOAT, rois_count_num, NO_QUANT, 0, 1, 0);
  input_rois = rois_host;

  // roialign theory_io
  for (int roi_idx = 0; roi_idx < num_rois; roi_idx++) {
    int batch_idx = input_rois[roi_idx * roi_offset + 0];
    float roi_x1 = input_rois[roi_idx * roi_offset + 1];
    float roi_y1 = input_rois[roi_idx * roi_offset + 2];
    float roi_x2 = input_rois[roi_idx * roi_offset + 3];
    float roi_y2 = input_rois[roi_idx * roi_offset + 4];

    int height = input_height;
    int width = input_width;
    float ss = spatial_scale;
    float offset = aligned ? 0.5 : 0.0;

    float roi_start_w = roi_x1 * ss - offset;
    float roi_start_h = roi_y1 * ss - offset;
    float roi_end_w = roi_x2 * ss - offset;
    float roi_end_h = roi_y2 * ss - offset;

    float roi_width = roi_end_w - roi_start_w;
    float roi_height = roi_end_h - roi_start_h;

    if (!aligned) {
      roi_width = roi_width > 1 ? roi_width : 1;
      roi_height = roi_height > 1 ? roi_height : 1;
    }

    float bin_size_h = (float)roi_height / pooled_height;
    float bin_size_w = (float)roi_width / pooled_width;
    int roi_bin_grid_h =
        (sampling_ratio > 0) ? sampling_ratio : ceil(bin_size_h);
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(bin_size_w);

    theory_io_size += 20;
    for (int ph = 0; ph < pooled_height; ph++) {
      for (int pw = 0; pw < pooled_width; pw++) {
        for (int iy = 0; iy < roi_bin_grid_h; iy++) {
          for (int ix = 0; ix < roi_bin_grid_w; ix++) {
            for (int channel_idx = 0; channel_idx < channels; channel_idx++) {
              theory_io_size += 16;
            }  // channels
          }    // input w
        }      // input h
        theory_io_size = pool_mode == 1
                             ? theory_io_size + channels * sizeof(float)
                             : theory_io_size + channels * sizeof(float) * 3;
      }  // pw
    }    // ph
  }      // roi
  VLOG(4) << "theory_io_size: " << theory_io_size << "ops";
  cpu_runtime_.deallocate(input_rois);

  return theory_io_size;
}

}  // namespace mluoptest
