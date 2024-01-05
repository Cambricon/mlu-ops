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
#include "indice_convolution_backward_filter.h"
#include <vector>
#include <string>
#include <set>
#include "mlu_op.h"

namespace mluoptest {

void IndiceConvolutionBackwardFilterExecutor::initParam() {
  // init necessary param
  input_indice_desc_ = tensor_desc_[0].tensor;
  diffy_indice_desc_ = tensor_desc_[1].tensor;
  indice_pair_desc_ = tensor_desc_[2].tensor;
  diffw_desc_ = tensor_desc_[3].tensor;
  auto op_param = parser_->getProtoNode()->indice_convolution_backward_param();
  for (int i = 0; i < op_param.indice_num_size(); i++) {
    indice_num_.push_back(op_param.indice_num(i));
  }
  inverse_ = op_param.inverse();
  subm_ = op_param.sub_m();

  diffw_trans_ = false;
  // if (MLUOP_LAYOUT_HWCN != diffw_desc_->layout) {
  //   diffw_trans_ = true;
  // }
}

void IndiceConvolutionBackwardFilterExecutor::paramCheck() {
  GTEST_CHECK(parser_->getProtoNode()->has_indice_convolution_backward_param(),
              op_name_ + ": sparse conv back param missing.");
  GTEST_CHECK(parser_->getInputNum() == 3 && parser_->getOutputNum() == 1,
              op_name_ + ": input or output number error.");
}

void IndiceConvolutionBackwardFilterExecutor::workspaceMalloc() {
  initParam();
  MLUOP_CHECK(mluOpGetIndiceConvolutionBackwardFilterWorkspaceSize(
      handle_, input_indice_desc_, diffy_indice_desc_, indice_pair_desc_,
      diffw_desc_, indice_num_.data(), inverse_, subm_, &workspace_size_));
  void *dev_workspace = nullptr;
  if (workspace_size_ != 0) {
    dev_workspace = mlu_runtime_.allocate(workspace_size_);
    eva_->setMluWorkspaceSize(workspace_size_);
  }

  workspace_.push_back(dev_workspace);
}

void IndiceConvolutionBackwardFilterExecutor::workspaceFree() {
  for (auto ptr : workspace_) {
    if (ptr) {
      mlu_runtime_.deallocate(ptr);
    }
  }
}

void IndiceConvolutionBackwardFilterExecutor::compute() {
  auto input_indice_dev = data_vector_[0].device_ptr;
  auto diffy_indice_dev = data_vector_[1].device_ptr;
  auto indice_pair_dev = data_vector_[2].device_ptr;
  auto diffw_dev = data_vector_[3].device_ptr;
  interface_timer_.start();
  MLUOP_CHECK(mluOpIndiceConvolutionBackwardFilter(
      handle_, input_indice_desc_, input_indice_dev, diffy_indice_desc_,
      diffy_indice_dev, indice_pair_desc_, indice_pair_dev, indice_num_.data(),
      inverse_, subm_, workspace_[0], workspace_size_, diffw_desc_, diffw_dev));
  interface_timer_.stop();
}

void IndiceConvolutionBackwardFilterExecutor::cpuTranspose(
    float *output, const float *input, const int64_t kernel_volume,
    const int64_t ci, const int64_t co,
    const mluOpTensorLayout_t diffw_layout) {
  int64_t in_shape[3] = {kernel_volume, ci, co};
  int64_t dim[3] = {0};
  int64_t permute[3] = {2, 0, 1};
  if (MLUOP_LAYOUT_NCHW == diffw_layout || MLUOP_LAYOUT_NCDHW == diffw_layout) {
    permute[0] = 2;
    permute[1] = 1;
    permute[2] = 0;
  }

  int64_t in_index = 0, out_index = 0;
  for (dim[0] = 0; dim[0] < in_shape[0]; dim[0]++) {
    for (dim[1] = 0; dim[1] < in_shape[1]; dim[1]++) {
      for (dim[2] = 0; dim[2] < in_shape[2]; dim[2]++) {
        in_index =
            dim[0] * in_shape[1] * in_shape[2] + dim[1] * in_shape[2] + dim[2];
        out_index =
            dim[permute[0]] * in_shape[permute[1]] * in_shape[permute[2]] +
            dim[permute[1]] * in_shape[permute[2]] + dim[permute[2]];
        output[out_index] = input[in_index];
      }
    }
  }
  return;
}

void IndiceConvolutionBackwardFilterExecutor::cpuCompute() {
  float *input_indices = cpu_fp32_input_[0];
  float *diffy_indices = cpu_fp32_input_[1];
  int32_t *indice_pair = (int32_t *)(data_vector_[2].host_ptr);
  float *diffw = cpu_fp32_output_[0];
  float *temp_diffw = diffw;

  if (input_indices == nullptr || diffy_indices == nullptr ||
      indice_pair == nullptr || diffw == nullptr) {
    return;  // skip zero element num
  }

  if (diffw_trans_) {
    temp_diffw = (float *)cpu_runtime_.allocate(diffw_desc_->total_element_num *
                                                sizeof(float));
  }

  int64_t in_active_num = input_indice_desc_->dims[0];
  int64_t ci = input_indice_desc_->dims[1];
  int64_t co = diffy_indice_desc_->dims[1];
  int64_t kd = diffw_desc_->dim == 4 ? 1 : mluOpGetTensordimD(diffw_desc_);
  int64_t kh = mluOpGetTensordimH(diffw_desc_);
  int64_t kw = mluOpGetTensordimH(diffw_desc_);
  int64_t kernel_volume = kd * kh * kw;

  for (int64_t kd_index = 0; kd_index < kd; ++kd_index) {
    for (int64_t kh_index = 0; kh_index < kh; ++kh_index) {
      for (int64_t kw_index = 0; kw_index < kw; ++kw_index) {
        int64_t kernel_index = kd_index * kh * kw + kh_index * kw + kw_index;
        for (int64_t ci_index = 0; ci_index < ci; ++ci_index) {
          for (int64_t co_index = 0; co_index < co; ++co_index) {
            float temp_res = 0.0;
            for (int64_t indice_i = 0; indice_i < indice_num_[kernel_index];
                 ++indice_i) {
              int64_t input_pos =
                  indice_pair[kernel_index * 2 * in_active_num + indice_i];
              int64_t diffy_pos = indice_pair[kernel_index * 2 * in_active_num +
                                              1 * in_active_num + indice_i];
              temp_res += input_indices[input_pos * ci + ci_index] *
                          diffy_indices[diffy_pos * co + co_index];
            }

            temp_diffw[kernel_index * ci * co + ci_index * co + co_index] =
                temp_res;
          }
        }
      }
    }
  }
  // trans
  if (diffw_trans_) {
    cpuTranspose(diffw, temp_diffw, kernel_volume, ci, co, diffw_desc_->layout);
    cpu_runtime_.deallocate(temp_diffw);
  }

  return;
}

int64_t IndiceConvolutionBackwardFilterExecutor::getTheoryOps() {
  int64_t ci = input_indice_desc_->dims[1];
  int64_t co = diffy_indice_desc_->dims[1];
  int64_t kernel_volume = indice_pair_desc_->dims[0];
  int64_t total_ops = 0;

  // fill theory ops
  total_ops += diffw_desc_->total_tensor_size;
  for (int64_t i = 0; i < kernel_volume; ++i) {
    if (indice_num_[0] <= 0) {
      continue;
    }
    total_ops += indice_num_[i] * ci;  // gather input_indice theory ops
    total_ops += indice_num_[i] * co;  // gather diffy_inddice theory ops
    total_ops += ci * co * indice_num_[i] * 2;  // matmul theory ops
  }
  // transpose theory ops
  if (diffw_trans_) {
    total_ops += diffw_desc_->total_element_num;
  }
  return total_ops;
}

int64_t IndiceConvolutionBackwardFilterExecutor::getTheoryIoSize() {
  int32_t *indice_pair = (int32_t *)(data_vector_[2].host_ptr);
  int64_t ci = input_indice_desc_->dims[1];
  int64_t co = diffy_indice_desc_->dims[1];
  int64_t in_active_num = input_indice_desc_->dims[0];
  int64_t kernel_volume = indice_pair_desc_->dims[0];
  int64_t theory_ios = 0;
  auto input_indice_dwidth =
      mluop::getSizeOfDataType(input_indice_desc_->dtype);
  auto diffy_indice_dwidth =
      mluop::getSizeOfDataType(diffy_indice_desc_->dtype);
  auto indice_pair_dwidth = mluop::getSizeOfDataType(indice_pair_desc_->dtype);
  auto diffw_dwidth = mluop::getSizeOfDataType(diffw_desc_->dtype);

  auto gather_nd_ios = [&](const int64_t kernel_index, const int64_t gather_num,
                           const int64_t channel,
                           const int is_output) -> int64_t {
    int64_t gather_theory_ios = 0;
    auto data_dwidth = is_output ? diffy_indice_dwidth : input_indice_dwidth;
    gather_theory_ios += gather_num * indice_pair_dwidth;  // indice_size
    gather_theory_ios +=
        gather_num * channel * data_dwidth;  // gather output size
                                             // get unique indice
    int32_t *data_begin = indice_pair + kernel_index * 2 * in_active_num +
                          is_output * in_active_num;
    std::vector<int32_t> indice_data(data_begin,
                                     data_begin + indice_num_[kernel_index]);
    std::set<int32_t> indice_set(indice_data.begin(), indice_data.end());
    auto unique_index_count = indice_set.size();
    gather_theory_ios +=
        unique_index_count * channel * data_dwidth;  // gather input size
    return gather_theory_ios;
  };

  // fill theory ios
  theory_ios += diffw_desc_->total_tensor_size;

  for (int64_t i = 0; i < kernel_volume; ++i) {
    if (indice_num_[i] <= 0) {
      continue;
    }

    // gatherNd theory ios
    theory_ios += gather_nd_ios(i, indice_num_[i], ci, 0);
    theory_ios += gather_nd_ios(i, indice_num_[i], co, 1);
    // matmul theory ios
    theory_ios +=
        indice_num_[i] * (ci * input_indice_dwidth + co * diffy_indice_dwidth) +
        ci * co * diffw_dwidth;
  }
  // transpose theory ios
  if (diffw_trans_) {
    theory_ios += diffw_desc_->total_tensor_size * 2;
  }

  return theory_ios;
}

}  // namespace mluoptest
