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
#include "ms_deform_attn_backward.h"

#include <memory>
#include <string>

namespace mluoptest {

void msDeformAttnCol2imBilinear(
    const float *bottom_data, const int32_t &height, const int32_t &width,
    const int32_t &nheads, const int32_t &channels, const float &h,
    const float &w, const int32_t &m, const int32_t &c, const float &top_grad,
    const float &attn_weight, float *grad_data_value, float *grad_sampling_loc,
    float *grad_attn_weight) {
  const int32_t h_low = floorf(h);
  const int32_t w_low = floorf(w);
  const int32_t h_high = h_low + 1;
  const int32_t w_high = w_low + 1;

  const float lh = h - h_low;
  const float lw = w - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int32_t w_stride = nheads * channels;
  const int32_t h_stride = width * w_stride;
  const int32_t h_low_ptr_offset = h_low * h_stride;
  const int32_t h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int32_t w_low_ptr_offset = w_low * w_stride;
  const int32_t w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int32_t base_ptr = m * channels + c;

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const float top_grad_value = top_grad * attn_weight;
  float grad_h_weight = 0, grad_w_weight = 0;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    int32_t ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;

    *(grad_data_value + ptr1) = *(grad_data_value + ptr1) + w1 * top_grad_value;
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    int32_t ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    *(grad_data_value + ptr2) = *(grad_data_value + ptr2) + w2 * top_grad_value;
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    int32_t ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    *(grad_data_value + ptr3) = *(grad_data_value + ptr3) + w3 * top_grad_value;
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    int32_t ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    *(grad_data_value + ptr4) = *(grad_data_value + ptr4) + w4 * top_grad_value;
  }

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  *grad_attn_weight += top_grad * val;
  *grad_sampling_loc += width * grad_w_weight * top_grad_value;
  *(grad_sampling_loc + 1) += height * grad_h_weight * top_grad_value;
}

void MsDeformAttnBackwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->getInputNum() == 6);
  GTEST_CHECK(parser_->getOutputNum() == 3);

  if (!parser_->getProtoNode()->has_ms_deform_attn_backward_param()) {
    LOG(ERROR) << "[GTEST_MSDEFORMATTN] Missing ms_deform_attn param.";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
}

void MsDeformAttnBackwardExecutor::compute() {
  auto value_desc = tensor_desc_[0].tensor;
  auto spatial_shapes_desc = tensor_desc_[1].tensor;
  auto level_start_index_desc = tensor_desc_[2].tensor;
  auto sampling_loc_desc = tensor_desc_[3].tensor;
  auto attn_weight_desc = tensor_desc_[4].tensor;
  auto grad_output_desc = tensor_desc_[5].tensor;
  auto grad_value_desc = tensor_desc_[6].tensor;
  auto grad_sampling_loc_desc = tensor_desc_[7].tensor;
  auto grad_attn_weight_desc = tensor_desc_[8].tensor;

  const int32_t im2col_step =
      parser_->getProtoNode()->ms_deform_attn_backward_param().im2col_step();
  void *value = data_vector_[0].device_ptr;
  void *spatial_shapes = data_vector_[1].device_ptr;
  void *level_start_index = data_vector_[2].device_ptr;
  void *sampling_loc = data_vector_[3].device_ptr;
  void *attn_weight = data_vector_[4].device_ptr;
  void *grad_output = data_vector_[5].device_ptr;
  void *grad_value = data_vector_[6].device_ptr;
  void *grad_sampling_loc = data_vector_[7].device_ptr;
  void *grad_attn_weight = data_vector_[8].device_ptr;

  interface_timer_.start();
  MLUOP_CHECK(mluOpMsDeformAttnBackward(
      handle_, value_desc, value, spatial_shapes_desc, spatial_shapes,
      level_start_index_desc, level_start_index, sampling_loc_desc,
      sampling_loc, attn_weight_desc, attn_weight, grad_output_desc,
      grad_output, im2col_step, grad_value_desc, grad_value,
      grad_sampling_loc_desc, grad_sampling_loc, grad_attn_weight_desc,
      grad_attn_weight));
  interface_timer_.stop();
}

void MsDeformAttnBackwardExecutor::cpuCompute() {
  float *cpu_value = cpu_fp32_input_[0];
  float *cpu_spatial_shapes = cpu_fp32_input_[1];
  float *cpu_level_start_index = cpu_fp32_input_[2];
  float *cpu_sampling_loc = cpu_fp32_input_[3];
  float *cpu_attn_weight = cpu_fp32_input_[4];
  float *cpu_grad_output = cpu_fp32_input_[5];
  float *cpu_grad_value = cpu_fp32_output_[0];
  float *cpu_grad_sampling_loc = cpu_fp32_output_[1];
  float *cpu_grad_attn_weight = cpu_fp32_output_[2];

  mluOpTensorDescriptor_t value_desc = tensor_desc_[0].tensor;
  mluOpTensorDescriptor_t sampling_loc_desc = tensor_desc_[3].tensor;
  const int32_t batch = value_desc->dims[0];
  const int32_t channels = value_desc->dims[3];

  const int32_t num_query = sampling_loc_desc->dims[1];
  const int32_t num_heads = sampling_loc_desc->dims[2];
  const int32_t num_levels = sampling_loc_desc->dims[3];
  const int32_t num_point = sampling_loc_desc->dims[4];
  const int32_t qid_stride = num_heads * channels;
  const int32_t spatial_size = value_desc->dims[1];

  const int32_t grad_weight_stride = 1;
  const int32_t grad_loc_stride = 2;
  for (int32_t i = 0; i < batch * num_query * num_heads * channels; ++i) {
    int32_t temp = i;
    int32_t c_col = i % channels;
    temp /= channels;
    int32_t sampling_index = temp;
    int32_t m_col = temp % num_heads;
    temp /= num_heads;
    temp /= num_query;
    const int32_t b_col = temp;
    const int32_t per_sample_loc_size =
        b_col * num_query * num_heads * num_levels * num_point * 2;
    const int32_t per_attn_weight_size =
        b_col * num_query * num_heads * num_levels * num_point;

    float *data_value = cpu_value + 0;
    float *grad_data_value = cpu_grad_value + 0;
    float *data_sampling_loc = cpu_sampling_loc + 0;
    float *data_attn_weight = cpu_attn_weight + 0;
    float *grad_sampling_loc = cpu_grad_sampling_loc + 0;
    float *grad_attn_weight = cpu_grad_attn_weight + 0;
    int32_t grad_output_offset = b_col * num_query * num_heads * channels +
                                 i % (num_query * num_heads * channels);
    float top_grad = cpu_grad_output[grad_output_offset];
    int32_t data_weight_ptr = sampling_index * num_levels * num_point;
    int32_t data_loc_w_ptr = data_weight_ptr << 1;
    int32_t grad_sampling_ptr = data_weight_ptr;
    float *grad_sampling_loc_out = grad_sampling_loc + (grad_sampling_ptr << 1);
    float *grad_attn_weight_out = grad_attn_weight + grad_sampling_ptr;
    const int32_t data_value_ptr_init_offset =
        b_col * spatial_size * qid_stride;
    for (int32_t l_col = 0; l_col < num_levels; ++l_col) {
      int32_t level_start_id = cpu_level_start_index[l_col];
      int32_t spatial_h_ptr = l_col << 1;
      int32_t spatial_h = cpu_spatial_shapes[spatial_h_ptr];
      int32_t spatial_w = cpu_spatial_shapes[spatial_h_ptr + 1];
      int32_t value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      float *data_value_ptr = data_value + value_ptr_offset;

      float *grad_value_ptr = grad_data_value + value_ptr_offset;

      for (int32_t p_col = 0; p_col < num_point; ++p_col) {
        float loc_w = data_sampling_loc[data_loc_w_ptr];
        float loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        float weight = data_attn_weight[data_weight_ptr];

        float h_im = loc_h * spatial_h - 0.5;
        float w_im = loc_w * spatial_w - 0.5;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          msDeformAttnCol2imBilinear(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              grad_sampling_loc_out, grad_attn_weight_out);
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight_out += grad_weight_stride;
        grad_sampling_loc_out += grad_loc_stride;
      }
    }
  }
}

int64_t MsDeformAttnBackwardExecutor::getTheoryOps() {
  auto grad_value_desc = tensor_desc_[6].tensor;
  auto grad_sampling_loc_desc = tensor_desc_[7].tensor;

  const int32_t batch = grad_value_desc->dims[0];
  const int32_t channels = grad_value_desc->dims[3];
  const int32_t num_query = grad_sampling_loc_desc->dims[1];
  const int32_t num_heads = grad_sampling_loc_desc->dims[2];
  const int32_t num_levels = grad_sampling_loc_desc->dims[3];
  const int32_t num_point = grad_sampling_loc_desc->dims[4];

  const int64_t count = 48;
  const int64_t theory_ops =
      batch * channels * num_query * num_heads * num_levels * num_point * count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
