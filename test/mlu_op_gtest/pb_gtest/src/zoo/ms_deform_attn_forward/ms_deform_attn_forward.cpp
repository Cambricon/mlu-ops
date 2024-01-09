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
#include "ms_deform_attn_forward.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include "math.h"

namespace mluoptest {

float MsDeformAttnForwardExecutor::ms_deform_attn_im2col_bilinear(
    const float *&bottom_data,
    const int &height,
    const int &width,
    const int &nheads,
    const int &channels,
    const float &h,
    const float &w,
    const int &m,
    const int &c) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;  const float lh = h - h_low;
  const float lw = w - w_low;
  const float hh = 1 - lh, hw = 1 - lw;
  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;
  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }
  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

void MsDeformAttnForwardExecutor::cpuMsDeformAttnForward(
    const float *data_value,
    const float *data_spatial_shapes,
    const float *data_level_start_index,
    const float *data_sampling_loc,
    const float *data_attn_weight,
    const int batch_size,
    const int num_keys,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    float *data_col) {
  const int n = batch_size * num_query * num_heads * channels;
  for (int index = 0; index < n; ++index) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;
    float *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * num_keys * qid_stride;
    float col = 0;
    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const float *data_value_ptr =
          data_value +
          (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_point; ++p_col) {
        const float loc_w = data_sampling_loc[data_loc_w_ptr];
        const float loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const float weight = data_attn_weight[data_weight_ptr];
        const float h_im = loc_h * spatial_h - 0.5;
        const float w_im = loc_w * spatial_w - 0.5;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          col += ms_deform_attn_im2col_bilinear(
                     data_value_ptr, spatial_h, spatial_w, num_heads,
                     channels, h_im, w_im, m_col, c_col) *
                 weight;
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
  return;
}

void MsDeformAttnForwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->getInputNum() == 5,
              "[GTEST_MSDEFORMATTN_FORWARD] Input num must be 5.");
  GTEST_CHECK(parser_->getOutputNum() == 1,
              "[GTEST_MSDEFORMATTN_FORWARD] Output num must be 1.");
  if (!parser_->getProtoNode()->has_ms_deform_attn_forward_param()) {
    LOG(ERROR) << "[GTEST_MSDEFORMATTN_FORWARD] Missing ms_deform_attn param.";
    throw std::invalid_argument(std::string(__FILE__) + " +" + std::to_string(__LINE__));  // NOLINT
  }
}

void MsDeformAttnForwardExecutor::compute() {
  VLOG(4) << "MsDeformAttnForwardExecutor::compute() Begin.";
  auto tensor_data_value = tensor_desc_[0].tensor;
  auto tensor_data_spatial_shapes = tensor_desc_[1].tensor;
  auto tensor_data_level_start_index = tensor_desc_[2].tensor;
  auto tensor_data_sampling_loc = tensor_desc_[3].tensor;
  auto tensor_data_attn_weight = tensor_desc_[4].tensor;
  auto tensor_data_col = tensor_desc_[5].tensor;
  auto dev_data_value = data_vector_[0].device_ptr;
  auto dev_data_spatial_shapes = data_vector_[1].device_ptr;
  auto dev_data_level_start_index = data_vector_[2].device_ptr;
  auto dev_data_sampling_loc = data_vector_[3].device_ptr;
  auto dev_data_attn_weight = data_vector_[4].device_ptr;
  auto dev_data_col = data_vector_[5].device_ptr;
  // get params
  auto param_proto_desc =
      parser_->getProtoNode()->ms_deform_attn_forward_param();
  int im2col_step_ = param_proto_desc.im2col_step();  interface_timer_.start();
  MLUOP_CHECK(mluOpMsDeformAttnForward(
      handle_, tensor_data_value, dev_data_value, tensor_data_spatial_shapes,
      dev_data_spatial_shapes, tensor_data_level_start_index,
      dev_data_level_start_index, tensor_data_sampling_loc,
      dev_data_sampling_loc, tensor_data_attn_weight,
      dev_data_attn_weight, im2col_step_, tensor_data_col, dev_data_col));
  interface_timer_.stop();
  VLOG(4) << "MsDeformAttnForwardExecutor::compute() End.";
}

void MsDeformAttnForwardExecutor::cpuCompute() {
  VLOG(4) << "MsDeformAttnForwardExecutor::cpuCompute() Begin.";
  auto tensor_data_value = tensor_desc_[0].tensor;
  auto tensor_data_spatial_shapes = tensor_desc_[1].tensor;
  auto tensor_data_level_start_index = tensor_desc_[2].tensor;
  auto tensor_data_sampling_loc = tensor_desc_[3].tensor;
  auto tensor_data_attn_weight = tensor_desc_[4].tensor;
  auto tensor_data_col = tensor_desc_[5].tensor;
  int batch_size = tensor_data_value->dims[0];
  int num_keys = tensor_data_value->dims[1];
  int num_heads = tensor_data_value->dims[2];
  int channels = tensor_data_value->dims[3];
  int num_levels = tensor_data_spatial_shapes->dims[0];
  int num_query = tensor_data_sampling_loc->dims[1];
  int num_point = tensor_data_sampling_loc->dims[4];
  auto data_value = cpu_fp32_input_[0];
  auto data_spatial_shapes = cpu_fp32_input_[1];
  auto data_level_start_index = cpu_fp32_input_[2];
  auto data_sampling_loc = cpu_fp32_input_[3];
  auto data_attn_weight = cpu_fp32_input_[4];
  auto data_col = cpu_fp32_output_[0];
  cpuMsDeformAttnForward(
      data_value, data_spatial_shapes, data_level_start_index,
      data_sampling_loc, data_attn_weight, batch_size, num_keys, num_heads,
      channels, num_levels, num_query, num_point, data_col);
  VLOG(4) << "MsDeformAttnForwardExecutor::cpuCompute() End.";
}

// The calculated theory IO size here may not equal to real IO size,
// because it's up to sample points location to decide whether the
// neighbor points need memcpy. We assume all the neighbor points are
// within the spatial feature map, which means theory IO size here are
// max possible IO size. So IO efficiency may be over 1
// since it's not accurate.
int64_t MsDeformAttnForwardExecutor::getTheoryIoSize() {
  auto tensor_data_value = tensor_desc_[0].tensor;
  auto tensor_data_spatial_shapes = tensor_desc_[1].tensor;
  auto tensor_data_sampling_loc = tensor_desc_[3].tensor;
  size_t batch_size = tensor_data_value->dims[0];
  size_t num_heads = tensor_data_value->dims[2];
  size_t channels = tensor_data_value->dims[3];
  size_t num_levels = tensor_data_spatial_shapes->dims[0];
  size_t num_query = tensor_data_sampling_loc->dims[1];
  size_t num_point = tensor_data_sampling_loc->dims[4];
  size_t total_size = 0;
  total_size += 4 * batch_size * num_query * num_heads * num_levels *
                num_point * channels *
                parser_->input(0)->sizeof_dtype;
  for (size_t i = 1; i < parser_->inputs().size(); ++i) {
    MetaTensor *ts = parser_->input(i);
    total_size += ts->shape_count * ts->sizeof_dtype;
  }
  for (size_t i = 0; i < parser_->outputs().size(); ++i) {
    MetaTensor *ts = parser_->output(i);
    total_size += ts->shape_count * ts->sizeof_dtype;
  }
  return total_size;
}

// Theory ops may also be inaccurate.
int64_t MsDeformAttnForwardExecutor::getTheoryOps() {
  auto tensor_data_value = tensor_desc_[0].tensor;
  auto tensor_data_spatial_shapes = tensor_desc_[1].tensor;
  auto tensor_data_sampling_loc = tensor_desc_[3].tensor;
  size_t batch_size = tensor_data_value->dims[0];
  size_t num_heads = tensor_data_value->dims[2];
  size_t num_levels = tensor_data_spatial_shapes->dims[0];
  size_t num_query = tensor_data_sampling_loc->dims[1];
  size_t num_point = tensor_data_sampling_loc->dims[4];
  int64_t count = 11;
  int64_t theory_ops = batch_size * num_query * num_heads * num_levels *
                       num_point * count;
  return theory_ops;
}
}  // namespace mluoptest

