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
#include "indice_convolution_backward_data.h"

#include <vector>

#include "test/mlu_op_gtest/pb_gtest/include/pb_test_tools.h"

namespace mluoptest {

void IndiceConvolutionBackwardDataExecutor::getFilterDims() {
  const mluOpTensorDescriptor_t filters_desc = tensor_desc_[1].tensor;
  const mluOpTensorLayout_t layout = filters_desc->layout;
  kd = 1;
  filter_4d = true;
  if (layout == MLUOP_LAYOUT_NCHW) {
    dyc = filters_desc->dims[0];
    dxc = filters_desc->dims[1];
    kh = filters_desc->dims[2];
    kw = filters_desc->dims[3];
  } else if (layout == MLUOP_LAYOUT_NHWC) {
    dyc = filters_desc->dims[0];
    dxc = filters_desc->dims[3];
    kh = filters_desc->dims[1];
    kw = filters_desc->dims[2];
  } else if (layout == MLUOP_LAYOUT_HWCN) {
    dyc = filters_desc->dims[3];
    dxc = filters_desc->dims[2];
    kh = filters_desc->dims[0];
    kw = filters_desc->dims[1];
  } else if (layout == MLUOP_LAYOUT_NDHWC) {
    dyc = filters_desc->dims[0];
    dxc = filters_desc->dims[4];
    kd = filters_desc->dims[1];
    kh = filters_desc->dims[2];
    kw = filters_desc->dims[3];
    filter_4d = false;
  } else if (layout == MLUOP_LAYOUT_NCDHW) {
    dyc = filters_desc->dims[0];
    dxc = filters_desc->dims[1];
    kd = filters_desc->dims[2];
    kh = filters_desc->dims[3];
    kw = filters_desc->dims[4];
    filter_4d = false;
  } else if (layout == MLUOP_LAYOUT_ARRAY) {
    dyc = filters_desc->dims[4];
    dxc = filters_desc->dims[3];
    kd = filters_desc->dims[0];
    kh = filters_desc->dims[1];
    kw = filters_desc->dims[2];
    filter_4d = false;
  }
}

void IndiceConvolutionBackwardDataExecutor::setSpconvdataParams() {
  inverse_ = parser_->getProtoNode()
                 ->indice_convolution_backward_data_param()
                 .inverse();
  sub_m_ =
      parser_->getProtoNode()->indice_convolution_backward_data_param().sub_m();
}

void IndiceConvolutionBackwardDataExecutor::paramCheck() {
  if (parser_->getInputNum() != 3) {
    LOG(ERROR) << "indice_convolution_backward_data input number is wrong. ";
  }
  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "indice_convolution_backward_data output number is wrong. ";
  }
  if (parser_->getProtoNode()
              ->indice_convolution_backward_data_param()
              .inverse() != 0 &&
      parser_->getProtoNode()
              ->indice_convolution_backward_data_param()
              .inverse() != 1) {
    LOG(ERROR) << "indice_convolution_backward_data inverse param is wrong. ";
  }
  if (parser_->getProtoNode()
              ->indice_convolution_backward_data_param()
              .sub_m() != 0 &&
      parser_->getProtoNode()
              ->indice_convolution_backward_data_param()
              .sub_m() != 1) {
    LOG(ERROR) << "indice_convolution_backward_data sub_m param is wrong. ";
  }
}

void IndiceConvolutionBackwardDataExecutor::workspaceMalloc() {
  getFilterDims();
  setSpconvdataParams();
  mluOpTensorDescriptor_t output_grad_desc = tensor_desc_[0].tensor;
  mluOpTensorDescriptor_t filters_desc = tensor_desc_[1].tensor;
  mluOpTensorDescriptor_t indice_pairs_desc = tensor_desc_[2].tensor;
  mluOpTensorDescriptor_t input_grad_desc = tensor_desc_[3].tensor;
  int K = filter_4d ? kh * kw : kd * kh * kw;
  std::vector<int64_t> indice_num_;

  // get indice_num info
  for (int kk = 0; kk < K; ++kk) {
    indice_num_.push_back(parser_->getProtoNode()
                              ->indice_convolution_backward_data_param()
                              .indice_num(kk));
  }
  MLUOP_CHECK(mluOpGetIndiceConvolutionBackwardDataWorkspaceSize(
      handle_, output_grad_desc, filters_desc, indice_pairs_desc,
      input_grad_desc, indice_num_.data(), inverse_, &workspace_size));
  char *workspace_ptr;
  workspace_.push_back(workspace_ptr);
  workspace_[0] = mlu_runtime_.allocate(workspace_size);
  VLOG(4) << "workspace_[0] = " << workspace_[0]
          << "; workspace size: " << workspace_size;
  eva_->setMluWorkspaceSize(workspace_size);
}

void IndiceConvolutionBackwardDataExecutor::workspaceFree() {
  mlu_runtime_.deallocate(workspace_[0]);
}

void IndiceConvolutionBackwardDataExecutor::compute() {
  VLOG(4) << "IndiceConvolutionBackwardDataExecutor compute ";
  getFilterDims();
  setSpconvdataParams();
  mluOpTensorDescriptor_t output_grad_desc = tensor_desc_[0].tensor;
  mluOpTensorDescriptor_t filters_desc = tensor_desc_[1].tensor;
  mluOpTensorDescriptor_t indice_pairs_desc = tensor_desc_[2].tensor;
  mluOpTensorDescriptor_t input_grad_desc = tensor_desc_[3].tensor;
  auto output_grad = data_vector_[0].device_ptr;
  auto filter = data_vector_[1].device_ptr;
  auto indice_pairs = data_vector_[2].device_ptr;
  auto input_grad = data_vector_[3].device_ptr;
  data_vector_[3].is_output = true;
  int K = filter_4d ? kh * kw : kd * kh * kw;
  std::vector<int64_t> indice_num_;

  // get indice_num info
  for (int kk = 0; kk < K; ++kk) {
    indice_num_.push_back(parser_->getProtoNode()
                              ->indice_convolution_backward_data_param()
                              .indice_num(kk));
  }
  VLOG(4) << "call mluOpIndiceConvolutionBackwardData()";
  VLOG(4) << "<gtest> output_grad " << output_grad << ", filter " << filter
          << ", indice_pairs " << indice_pairs << ", input_grad " << input_grad;
  VLOG(4) << "<gtest> workspace " << workspace_[0];
  interface_timer_.start();
  MLUOP_CHECK(mluOpIndiceConvolutionBackwardData(
      handle_, output_grad_desc, output_grad, filters_desc, filter,
      indice_pairs_desc, indice_pairs, indice_num_.data(), inverse_, sub_m_,
      workspace_[0], workspace_size, input_grad_desc, input_grad));
  interface_timer_.stop();
  VLOG(4) << "finish calling mluOpIndiceConvolutionBackwardData()";
}

/* transpose filter to HWCN / DHWCN */
void IndiceConvolutionBackwardDataExecutor::cpuTransposeFilter(
    float *filter_transpose_cpu, mluOpTensorLayout_t origin_layout) {
  int origin_strides[5];  // origin strides
  int dims[5];            // target dims
  int slice[5];           // target slice
  if (filter_4d) {
    dims[0] = kh;
    dims[1] = kw;
    dims[2] = dxc;
    dims[3] = dyc;
    slice[0] = kw * dxc * dyc;
    slice[1] = dxc * dyc;
    slice[2] = dyc;
    slice[3] = 1;
    if (origin_layout == MLUOP_LAYOUT_NCHW) {
      origin_strides[0] = kw;  // HWCNX
      origin_strides[1] = 1;
      origin_strides[2] = kh * kw;
      origin_strides[3] = dxc * kh * kw;
    } else if (origin_layout == MLUOP_LAYOUT_NHWC) {
      origin_strides[0] = kw * dxc;
      origin_strides[1] = dxc;
      origin_strides[2] = 1;
      origin_strides[3] = kh * kw * dxc;
    }
  } else {
    dims[0] = kd;
    dims[1] = kh;
    dims[2] = kw;
    dims[3] = dxc;
    dims[4] = dyc;
    slice[0] = kh * kw * dxc * dyc;
    slice[1] = kw * dxc * dyc;
    slice[2] = dxc * dyc;
    slice[3] = dyc;
    slice[4] = 1;
    if (origin_layout == MLUOP_LAYOUT_NCDHW) {
      origin_strides[0] = kh * kw;  // DHWCN
      origin_strides[1] = kw;
      origin_strides[2] = 1;
      origin_strides[3] = kd * kh * kw;
      origin_strides[4] = dxc * kd * kh * kw;
    } else if (origin_layout == MLUOP_LAYOUT_NDHWC) {
      origin_strides[0] = kh * kw * dxc;
      origin_strides[1] = kw * dxc;
      origin_strides[2] = dxc;
      origin_strides[3] = 1;
      origin_strides[4] = kd * kh * kw * dxc;
    }
  }
  for (int i = 0; i < dims[0]; ++i) {
    for (int j = 0; j < dims[1]; ++j) {
      for (int p = 0; p < dims[2]; ++p) {
        for (int q = 0; q < dims[3]; ++q) {
          if (!filter_4d) {
            for (int r = 0; r < dims[4]; ++r) {
              int index_origin = i * origin_strides[0] + j * origin_strides[1] +
                                 p * origin_strides[2] + q * origin_strides[3] +
                                 r * origin_strides[4];
              int index = i * slice[0] + j * slice[1] + p * slice[2] +
                          q * slice[3] + r * slice[4];
              filter_transpose_cpu[index] = cpu_fp32_input_[1][index_origin];
            }
          } else {
            int index_origin = i * origin_strides[0] + j * origin_strides[1] +
                               p * origin_strides[2] + q * origin_strides[3];
            int index =
                i * slice[0] + j * slice[1] + p * slice[2] + q * slice[3];
            filter_transpose_cpu[index] = cpu_fp32_input_[1][index_origin];
          }
        }
      }
    }
  }
}

void IndiceConvolutionBackwardDataExecutor::cpuCompute() {
  assert(parser_->getInputNum() == 3);
  assert(parser_->getOutputNum() == 1);
  VLOG(4) << "compute cpu IndiceConvolutionBackwardData";
  auto count = parser_->getOutputDataCount(0);
  assert(count != 0);
  getFilterDims();
  setSpconvdataParams();
  int K = kd * kh * kw;
  int filter_num = K * dyc * dxc;
  const mluOpTensorDescriptor_t filters_desc = tensor_desc_[1].tensor;
  const mluOpTensorLayout_t layout = filters_desc->layout;
  float *filter_transpose_cpu;
  if (!(layout == MLUOP_LAYOUT_HWCN)) {
    filter_transpose_cpu =
        (float *)cpu_runtime_.allocate(filter_num * sizeof(float));
    cpuTransposeFilter(filter_transpose_cpu, layout);
  } else {
    filter_transpose_cpu = cpu_fp32_input_[1];
  }

  // get indice_num info
  std::vector<int64_t> indice_num_;
  for (int kk = 0; kk < K; ++kk) {
    indice_num_.push_back(parser_->getProtoNode()
                              ->indice_convolution_backward_data_param()
                              .indice_num(kk));
  }

  // get index pair param
  const mluOpTensorDescriptor_t indice_pairs_desc = tensor_desc_[2].tensor;
  int L = indice_pairs_desc->dims[2];

  // main calculation
  // set input data to 0
  int input_grad_data_count = parser_->getOutputDataCount(0);
  memset(cpu_fp32_output_[0], 0x00,
         mluOpDataTypeBytes(indice_pairs_desc->dtype) * input_grad_data_count);
  float *output_grad = cpu_fp32_input_[0];
  float *indice_pairs = cpu_fp32_input_[2];
  float *input_grad = cpu_fp32_output_[0];
  bool is_float = (filters_desc->dtype == MLUOP_DTYPE_FLOAT);
  for (int i = 0; i < input_grad_data_count; ++i) {
    input_grad[i] = 0;
  }
  // main loop: filter_transpose K in [K, dxc, dyc]
  for (int kk = 0; kk < K; ++kk) {
    int filter_offset = kk * dxc * dyc;
    int index_num = (int)(indice_num_[kk]);
    assert(L >= index_num);
    for (int l = 0; l < index_num; ++l) {  // index_pair data loop
      int input_idx = indice_pairs[kk * 2 * L + l];
      int output_idx = indice_pairs[kk * 2 * L + L + l];
      float *sub_filter = filter_transpose_cpu + filter_offset;
      float *input_slice = input_grad + input_idx * dxc;
      float *output_slice = output_grad + output_idx * dyc;
      for (int dxc_i = 0; dxc_i < dxc; ++dxc_i) {
        float *input_grad_result = input_slice + dxc_i;
        float input_grad_accumulate = 0;
        for (int dyc_i = 0; dyc_i < dyc; ++dyc_i) {
          float input_grad_tmp = 0;
          float output_grad_tmp = output_slice[dyc_i];
          float filter_tmp = sub_filter[dxc_i * dyc + dyc_i];
          if (is_float) {
            input_grad_tmp = output_grad_tmp * filter_tmp;
            input_grad_accumulate += input_grad_tmp;
          } else {
            // half
            uint16_t temp;
            wrapRtConvertFloatToHalf(&temp, output_grad_tmp);
            wrapRtConvertHalfToFloat(&output_grad_tmp, temp);
            wrapRtConvertFloatToHalf(&temp, filter_tmp);
            wrapRtConvertHalfToFloat(&filter_tmp, temp);
            input_grad_tmp = output_grad_tmp * filter_tmp;
            wrapRtConvertFloatToHalf(&temp, input_grad_tmp);
            wrapRtConvertHalfToFloat(&input_grad_tmp, temp);
            input_grad_accumulate += input_grad_tmp;
            wrapRtConvertFloatToHalf(&temp, input_grad_accumulate);
            wrapRtConvertHalfToFloat(&input_grad_accumulate, temp);
          }
        }
        *input_grad_result += input_grad_accumulate;
      }
    }
  }
  if (!(layout == MLUOP_LAYOUT_HWCN)) {
    cpu_runtime_.deallocate(filter_transpose_cpu);
  }
}

int64_t IndiceConvolutionBackwardDataExecutor::getTheoryOps() {
  getFilterDims();
  setSpconvdataParams();
  int K = kd * kh * kw;
  // get indice_num info
  std::vector<int64_t> indice_num_;
  for (int kk = 0; kk < K; ++kk) {
    indice_num_.push_back(parser_->getProtoNode()
                              ->indice_convolution_backward_data_param()
                              .indice_num(kk));
  }
  int64_t theory_ops = 0;
  for (int kk = 0; kk < K; ++kk) {
    theory_ops += indice_num_[kk] * dxc * dyc;
  }
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}
}  // namespace mluoptest
