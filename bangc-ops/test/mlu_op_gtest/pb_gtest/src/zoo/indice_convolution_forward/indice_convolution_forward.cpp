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
#include "indice_convolution_forward.h"
#include "mlu_op.h"

namespace mluoptest {

void IndiceConvolutionForwardExecutor::paramInit() {
  features_desc_ = tensor_desc_[0].tensor;
  filters_desc_ = tensor_desc_[1].tensor;
  indice_pairs_desc_ = tensor_desc_[2].tensor;
  features_out_desc_ = tensor_desc_[3].tensor;
  auto op_param = parser_->getProtoNode()->indice_convolution_forward_param();
  for (int i = 0; i < op_param.indice_num_size(); i++) {
    indice_num_.push_back(op_param.indice_num(i));
  }
  num_active_out_ = (int64_t)op_param.num_active_out();
  sub_m_ = (int64_t)op_param.sub_m();
  inverse_ = (int64_t)op_param.inverse();
}

void IndiceConvolutionForwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->getInputNum() == 3 && parser_->getOutputNum() == 1,
              op_name + ": wrong input or output number.")
}

void IndiceConvolutionForwardExecutor::workspaceMalloc() {
  paramInit();
  MLUOP_CHECK(mluOpGetIndiceConvolutionForwardWorkspaceSize(
      handle_, features_desc_, filters_desc_, indice_pairs_desc_,
      features_out_desc_, indice_num_.data(), num_active_out_, inverse_, sub_m_,
      &workspace_size_));
  void *dev_workspace = nullptr;
  if (workspace_size_ != 0) {
    VLOG(4) << "Malloc workspace for indice convolution forward.";
    dev_workspace = mlu_runtime_.allocate(workspace_size_);
  }
  workspace_.push_back(dev_workspace);
  eva_->setMluWorkspaceSize(workspace_size_);
  VLOG(4) << "Malloc workspace addr:" << dev_workspace
          << ", size:" << workspace_size_;
}

void IndiceConvolutionForwardExecutor::workspaceFree() {
  for (auto ptr : workspace_) {
    if (ptr) {
      mlu_runtime_.deallocate(ptr);
    }
  }
}

void IndiceConvolutionForwardExecutor::compute() {
  auto features_dev = data_vector_[0].device_ptr;
  auto filters_dev = data_vector_[1].device_ptr;
  auto indice_pairs_dev = data_vector_[2].device_ptr;
  auto features_out_dev = data_vector_[3].device_ptr;
  interface_timer_.start();
  MLUOP_CHECK(mluOpIndiceConvolutionForward(
      handle_, features_desc_, features_dev, filters_desc_, filters_dev,
      indice_pairs_desc_, indice_pairs_dev, indice_num_.data(), num_active_out_,
      inverse_, sub_m_, workspace_[0], workspace_size_, features_out_desc_,
      features_out_dev));
  interface_timer_.stop();
}

void IndiceConvolutionForwardExecutor::cpuCompute() {
  float *features = cpu_fp32_input_[0];
  float *filters = cpu_fp32_input_[1];
  int32_t *indice_pairs = (int32_t *)(data_vector_[2].host_ptr);
  float *features_out = cpu_fp32_output_[0];
  float *temp_features_put = features_out;

  if (features == nullptr || filters == nullptr || indice_pairs == nullptr ||
      features_out == nullptr) {
    return;  // skip zero element
  }

  int64_t num_active_in = features_desc_->dims[0];
  int64_t ci = features_desc_->dims[1];
  int64_t co = features_out_desc_->dims[1];
  int64_t kd = mluOpGetTensordimD(filters_desc_);
  int64_t kh = mluOpGetTensordimH(filters_desc_);
  int64_t kw = mluOpGetTensordimW(filters_desc_);
  int64_t num_filters = kd * kh * kw;

  bool filters_need_transpose = true;
  int32_t stride[3];
  float *filters_transed = filters;
  if (filters_need_transpose) {
    filters_transed = (float *)cpu_runtime_.allocate(
        filters_desc_->total_element_num * sizeof(float));
    if (filters_desc_->layout == MLUOP_LAYOUT_NCDHW) {
      stride[0] = 1;
      stride[1] = num_filters;
      stride[2] = num_filters * ci;
    } else if (filters_desc_->layout == MLUOP_LAYOUT_NDHWC) {
      stride[0] = ci;
      stride[1] = 1;
      stride[2] = ci * num_filters;
    } else {
      LOG(ERROR) << "Unsupported filters layout.";
    }
    for (int32_t DHWi = 0; DHWi < num_filters; ++DHWi) {
      for (int32_t cii = 0; cii < ci; ++cii) {
        for (int32_t coi = 0; coi < co; ++coi) {
          filters_transed[DHWi * co * ci + cii * co + coi] =
              filters[DHWi * stride[0] + cii * stride[1] + coi * stride[2]];
        }
      }
    }
  }

  int32_t features_out_data_count = parser_->getOutputDataCount(0);
  memset(
      cpu_fp32_output_[0], 0x00,
      mluOpDataTypeBytes(features_out_desc_->dtype) * features_out_data_count);

  for (int64_t kdi = 0; kdi < kd; ++kdi) {
    for (int64_t khi = 0; khi < kh; ++khi) {
      for (int64_t kwi = 0; kwi < kw; ++kwi) {
        int64_t filters_index = kdi * kh * kw + khi * kw + kwi;
        for (int64_t ipi = 0; ipi < indice_num_[filters_index]; ++ipi) {
          int64_t input_offset =
              indice_pairs[filters_index * 2 * num_active_in + ipi];
          int64_t output_offset =
              indice_pairs[filters_index * 2 * num_active_in + num_active_in +
                           ipi];
          if (output_offset < 0 || input_offset < 0) continue;
          for (int64_t cii = 0; cii < ci; ++cii) {
            for (int64_t coi = 0; coi < co; ++coi) {
              float temp_res = 0.0;
              temp_res +=
                  features[input_offset * ci + cii] *
                  filters_transed[filters_index * ci * co + cii * co + coi];
              features_out[output_offset * co + coi] += temp_res;
            }
          }
        }
      }
    }
  }
  if (filters_need_transpose) {
    cpu_runtime_.deallocate(filters_transed);
  }
  return;
}

int64_t IndiceConvolutionForwardExecutor::getTheoryOps() {
  int64_t ci = features_desc_->dims[1];
  int64_t co = features_out_desc_->dims[1];
  int64_t num_filters = indice_pairs_desc_->dims[0];
  int64_t total_ops = 0;

  // initialize output to 0
  total_ops += features_out_desc_->total_element_num;
  for (int64_t i = 0; i < num_filters; ++i) {
    if (indice_num_[i] < 0) {
      continue;
    }
    // gather input ops
    total_ops += indice_num_[i] * ci;
    // matmul output ops
    total_ops += ci * co * indice_num_[i];
    // scatter_update and add output ops
    total_ops += indice_num_[i] * co * 2;
  }
  // transpose filters ops
  bool filters_need_transpose = true;
  if (filters_need_transpose) {
    total_ops += filters_desc_->total_element_num;
  }
  return total_ops;
}

int64_t IndiceConvolutionForwardExecutor::getTheoryIoSize() {
  int32_t *indice_pair = (int32_t *)(data_vector_[2].host_ptr);
  int64_t ci = features_desc_->dims[1];
  int64_t co = features_out_desc_->dims[2];
  int64_t num_active_in = features_desc_->dims[0];
  int64_t num_active_out = features_out_desc_->dims[0];
  int64_t num_filters = indice_pairs_desc_->dims[0];
  int64_t theory_ios = 0;
  auto features_dwidth = mluop::getSizeOfDataType(features_desc_->dtype);
  auto filters_dwidth = mluop::getSizeOfDataType(filters_desc_->dtype);
  auto indice_pairs_dwith = mluop::getSizeOfDataType(indice_pairs_desc_->dtype);
  auto features_out_dwith = mluop::getSizeOfDataType(features_out_desc_->dtype);

  auto gather_scatter_ios = [&](const int64_t index, const int64_t num,
                                const int64_t channel,
                                const int32_t is_gather) -> int64_t {
    int64_t gs_theory_ios = 0;
    auto data_dwidth = is_gather ? features_dwidth : features_out_dwith;
    // indice_pairs size
    gs_theory_ios += num * indice_pairs_dwith;
    // gather or scatter output size
    gs_theory_ios += num * channel * data_dwidth;
    // gather or scatter input size
    gs_theory_ios +=
        data_dwidth * channel * (is_gather ? num_active_in : num_active_out);
    return gs_theory_ios;
  };

  // fill ios
  theory_ios += filters_desc_->total_tensor_size;

  // transpose ios
  theory_ios += filters_desc_->total_element_num * 2;

  for (int64_t i = 0; i < num_filters; ++i) {
    if (indice_num_[i] <= 0) {
      continue;
    }

    // gather ios
    theory_ios += gather_scatter_ios(i, indice_num_[i], ci, 0);

    // matmul ios
    theory_ios +=
        indice_num_[i] * (ci * features_dwidth + co * filters_dwidth) +
        ci * co * features_out_dwith;

    // scatter ios
    theory_ios += gather_scatter_ios(i, indice_num_[i], co, 1);
  }

  return theory_ios;
}

}  // namespace mluoptest
