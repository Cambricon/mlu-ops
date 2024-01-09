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
#include "psamask_forward.h"

#include <algorithm>
#include <string>

namespace mluoptest {

typedef enum {
  COLLECT = 0,
  DISTRIBUTE = 1,
} psamaskType_t;

template <typename T>
void PsamaskForwardExecutor::psamaskCollectForwardCPU(
    const T *const input, T *output, const int num_, const int h_feature,
    const int w_feature, const int h_mask, const int w_mask,
    const int half_h_mask, const int half_w_mask) {
  int c_o = h_feature * w_feature;
  int h_o_offset = w_feature * c_o;
  int w_o_offset = c_o;
  int c_i = h_mask * w_mask;
  int h_i_offset = w_feature * c_i;
  int w_i_offset = c_i;
  int n_o_offset = h_feature * w_feature * c_o;
  int n_i_offset = h_feature * w_feature * c_i;
  int output_dim = num_ * n_o_offset;

  for (int i = 0; i < output_dim; ++i) {
    output[i] = 0;
  }
  for (int n = 0; n < num_; ++n) {
    for (int h = 0; h < h_feature; ++h) {
      for (int w = 0; w < w_feature; ++w) {
        const int hstart = std::max(0, half_h_mask - h);
        const int hend = std::min(h_mask, h_feature + half_h_mask - h);
        const int wstart = std::max(0, half_w_mask - w);
        const int wend = std::min(w_mask, w_feature + half_w_mask - w);
        // (hidx,                   widx                   ) with mask-indexed
        // (hidx + h - half_h_mask, widx + w - half_w_mask) with feature-indexed
        for (int hidx = hstart; hidx < hend; ++hidx) {
          for (int widx = wstart; widx < wend; ++widx) {
            output[n * n_o_offset + h * h_o_offset + w * w_o_offset +
                   ((hidx + h - half_h_mask) * w_feature +
                    (widx + w - half_w_mask))] =
                input[n * n_i_offset + h * h_i_offset + w * w_i_offset +
                      (hidx * w_mask + widx)];
          }
        }
      }
    }
  }
}

template <typename T>
void PsamaskForwardExecutor::psamaskDistributeForwardCPU(
    const T *const input, T *output, const int num_, const int h_feature,
    const int w_feature, const int h_mask, const int w_mask,
    const int half_h_mask, const int half_w_mask) {
  int c_o = h_feature * w_feature;
  int n_o_offset = h_feature * w_feature * c_o;
  int h_o_offset = w_feature * c_o;
  int w_o_offset = c_o;
  int c_o_offset = 1;

  int c_i = h_mask * w_mask;
  int n_i_offset = h_feature * w_feature * c_i;
  int h_i_offset = w_feature * c_i;
  int w_i_offset = c_i;
  int c_i_offset = 1;

  int output_dim = num_ * n_o_offset;
  for (int i = 0; i < output_dim; ++i) {
    output[i] = 0;
  }
  for (int n = 0; n < num_; n++) {
    for (int h = 0; h < h_feature; h++) {
      for (int w = 0; w < w_feature; w++) {
        // effective mask region : [hstart, hend) x [wstart, wend) with
        // mask-indexed
        const int hstart = std::max(0, half_h_mask - h);
        const int hend = std::min(h_mask, h_feature + half_h_mask - h);
        const int wstart = std::max(0, half_w_mask - w);
        const int wend = std::min(w_mask, w_feature + half_w_mask - w);
        // (hidx,                    widx                   ) with mask-indexed
        // (hidx + h - half_h_mask, widx + w - half_w_mask) with
        // feature-indexed
        for (int hidx = hstart; hidx < hend; hidx++) {
          for (int widx = wstart; widx < wend; widx++) {
            output[n * n_o_offset + (hidx + h - half_h_mask) * h_o_offset +
                   (widx + w - half_w_mask) * w_o_offset + h * w_feature + w] =
                input[n * n_i_offset + h * h_i_offset + w * w_i_offset +
                      (hidx * w_mask + widx)];
          }
        }
      }
    }
  }
}

void PsamaskForwardExecutor::paramCheck() {
  if (parser_->getInputNum() != 1) {
    LOG(ERROR) << "psamask_forward input number is wrong. ";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }

  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "psamask_forward output number is wrong. ";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
}

void PsamaskForwardExecutor::compute() {
  VLOG(4) << "PsamaskForwardExecutor compute ";
  mluOpTensorDescriptor_t input_desc, output_desc;
  void *input = data_vector_[0].device_ptr;
  input_desc = tensor_desc_[0].tensor;
  void *output = data_vector_[1].device_ptr;
  output_desc = tensor_desc_[1].tensor;
  int h_mask = parser_->getProtoNode()->psamask_forward_param().h_mask();
  int w_mask = parser_->getProtoNode()->psamask_forward_param().w_mask();
  int psa_type = parser_->getProtoNode()->psamask_forward_param().psa_type();
  VLOG(4) << "h_mask:" << h_mask << ", w_mask:" << w_mask
          << ", psa_type:" << psa_type;
  VLOG(4) << "call mluOpPsamaskForward()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpPsamaskForward(handle_, psa_type, input_desc, input, h_mask,
                                  w_mask, output_desc, output));
  interface_timer_.stop();
}

void PsamaskForwardExecutor::cpuCompute() {
  VLOG(4) << "PsamaskForwardExecutor cpuCompute ";
  mluOpTensorDescriptor_t input_desc, output_desc;
  input_desc = tensor_desc_[0].tensor;
  output_desc = tensor_desc_[1].tensor;
  int h_mask = parser_->getProtoNode()->psamask_forward_param().h_mask();
  int w_mask = parser_->getProtoNode()->psamask_forward_param().w_mask();
  psamaskType_t psa_type = (psamaskType_t)parser_->getProtoNode()
                               ->psamask_forward_param()
                               .psa_type();
  auto batch = input_desc->dims[0];
  auto input_c = input_desc->dims[3];
  auto h_feature = input_desc->dims[1];
  auto w_feature = input_desc->dims[2];
  auto output_c = output_desc->dims[3];

  int half_h_mask = (h_mask - 1) / 2;
  int half_w_mask = (w_mask - 1) / 2;
  auto input_data_type = input_desc->dtype;
  psamaskType_t psamask_type = (psamaskType_t)psa_type;

  void *input = (void *)cpu_fp32_input_[0];
  void *output = (void *)cpu_fp32_output_[0];
  switch (psa_type) {
    default: {
      VLOG(4) << "Wrong type of psamask.";
    }; break;
    case COLLECT: {
      VLOG(4) << "COLLECT";
      psamaskCollectForwardCPU((float *)input, (float *)output, batch,
                               h_feature, w_feature, h_mask, w_mask,
                               half_h_mask, half_w_mask);
    }; break;
    case DISTRIBUTE: {
      VLOG(4) << "DISTRIBUTE";
      psamaskDistributeForwardCPU((float *)input, (float *)output, batch,
                                  h_feature, w_feature, h_mask, w_mask,
                                  half_h_mask, half_w_mask);
    }; break;
  }
}

int64_t PsamaskForwardExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->getOutputDataCount(0);
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
