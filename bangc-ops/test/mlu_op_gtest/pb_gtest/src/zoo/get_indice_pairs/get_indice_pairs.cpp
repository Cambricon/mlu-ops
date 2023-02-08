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

#include <stdio.h>
#include <vector>
#include <string>
#include <algorithm>
#include "get_indice_pairs.h"
#include "mlu_op.h"

namespace mluoptest {
GetIndicePairsExecutor::GetIndicePairsExecutor() {
  sparse_conv_desc_ =
      cpu_runtime_.allocate(mluOpCreateSparseConvolutionDescriptor,
                            mluOpDestroySparseConvolutionDescriptor);
}

GetIndicePairsExecutor::~GetIndicePairsExecutor() {
  if (sparse_conv_desc_) {
    cpu_runtime_.deallocate(sparse_conv_desc_);
    sparse_conv_desc_ = nullptr;
  }
}

void GetIndicePairsExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_get_indice_pairs_param()) {
    LOG(ERROR) << op_name_ << "::get indice pairs missing.";
  }
  if (parser_->getInputNum() != 1 || parser_->getOutputNum() != 3) {
    LOG(ERROR) << op_name_ << ":: input or output number is wrong. ";
  }
}

void GetIndicePairsExecutor::initParam() {
  indice_in_desc_ = tensor_desc_[0].tensor;
  indice_out_desc_ = tensor_desc_[1].tensor;
  indice_pairs_desc_ = tensor_desc_[2].tensor;
  indice_num_desc_ = tensor_desc_[3].tensor;
  auto op_param = parser_->getProtoNode()->get_indice_pairs_param();
  for (int i = 0; i < op_param.pad_size(); ++i) {
    pad_.push_back(op_param.pad(i));
  }
  for (int i = 0; i < op_param.stride_size(); ++i) {
    stride_.push_back(op_param.stride(i));
  }
  for (int i = 0; i < op_param.dilation_size(); ++i) {
    dilation_.push_back(op_param.dilation(i));
  }
  for (int i = 0; i < op_param.input_space_size(); ++i) {
    input_space_.push_back(op_param.input_space(i));
  }
  for (int i = 0; i < op_param.filter_space_size(); ++i) {
    filter_space_.push_back(op_param.filter_space(i));
  }
  for (int i = 0; i < op_param.output_space_size(); ++i) {
    output_space_.push_back(op_param.output_space(i));
  }
  dimNb_ = op_param.dimnb() == 5 ? 5 : 4;
  batch_ = op_param.batch();
  sub_m_ = op_param.sub_m();

  MLUOP_CHECK(mluOpSetSparseConvolutionDescriptor(
      sparse_conv_desc_, dimNb_, batch_, pad_.data(), stride_.data(),
      dilation_.data(), input_space_.data(), filter_space_.data(),
      output_space_.data(), sub_m_, transpose_, inverse_));
}

void GetIndicePairsExecutor::workspaceMalloc() {
  initParam();
  MLUOP_CHECK(mluOpGetIndicePairsWorkspaceSize(
      handle_, sparse_conv_desc_, indice_in_desc_, indice_pairs_desc_,
      indice_out_desc_, indice_num_desc_, &workspace_size_));
  void *dev_workspace = nullptr;
  if (workspace_size_ != 0) {
    dev_workspace = mlu_runtime_.allocate(workspace_size_);
    eva_->setMluWorkspaceSize(workspace_size_);
  }
  workspace_.push_back(dev_workspace);
}

void GetIndicePairsExecutor::workspaceFree() {
  for (auto ptr : workspace_) {
    if (ptr) {
      mlu_runtime_.deallocate(ptr);
    }
  }
}

void GetIndicePairsExecutor::compute() {
  VLOG(4) << "GetIndicePairsExecutor compute";
  auto dev_indice_in = data_vector_[0].device_ptr;
  auto dev_indice_out = data_vector_[1].device_ptr;
  auto dev_indice_pairs = data_vector_[2].device_ptr;
  auto dev_indice_num = data_vector_[3].device_ptr;

  interface_timer_.start();
  MLUOP_CHECK(mluOpGetIndicePairs(
      handle_, sparse_conv_desc_, indice_in_desc_, dev_indice_in, workspace_[0],
      workspace_size_, indice_pairs_desc_, dev_indice_pairs, indice_out_desc_,
      dev_indice_out, indice_num_desc_, dev_indice_num));
  interface_timer_.stop();
}

void GetIndicePairsExecutor::castIn() {
  int64_t elements = parser_->input(0)->total_count;
  auto dst_data = (int32_t *)(data_vector_[0].host_ptr);
  auto src_data = (float *)(cpu_fp32_input_[0]);
  int64_t idx = 0;
  for (idx = 0; idx < elements; idx++) {
    dst_data[idx] = (int32_t)(src_data[idx]);
  }
  MetaTensor *ts = parser_->input(0);
  input_host_ = cpu_runtime_.allocate(ts->shape_count * ts->sizeof_dtype);
  memcpy(input_host_, data_vector_[0].host_ptr,
         ts->total_count * ts->sizeof_dtype);
}

// void GetIndicePairsExecutor::castOut() {}

void GetIndicePairsExecutor::diffPreprocess() {
  float *cpu_input = (float *)cpu_fp32_output_[1];
  int32_t input_active_in = indice_pairs_desc_->dims[2];
  int32_t kernel_volume = 1;
  for (int i = 0; i < filter_space_.size(); i++) {
    kernel_volume *= filter_space_[i];
  }
  std::vector<float> input;
  std::vector<float> input_copy;
  std::vector<float> output;
  for (int i = 0; i < kernel_volume; i++) {
    for (int j = 0; j < input_active_in; j++) {
      if (cpu_input[i * input_active_in * 2 + j] != -1.0) {
        input.push_back(cpu_input[i * input_active_in * 2 + j]);
        input_copy.push_back(cpu_input[i * input_active_in * 2 + j]);
        output.push_back(
            cpu_input[i * input_active_in * 2 + input_active_in + j]);
      }
    }
    sort(input.begin(), input.end());
    for (int k = 0; k < input.size(); k++) {
      cpu_input[i * input_active_in * 2 + k] = input[k];
      int32_t index = -1;
      for (int j = 0; j < input_copy.size(); j++) {
        if (input_copy[j] == input[k]) {
          index = j;
          break;
        }
      }
      cpu_input[i * input_active_in * 2 + input_active_in + k] = output[index];
    }
    input.clear();
    output.clear();
    input_copy.clear();
  }
}

void GetIndicePairsExecutor::cpuCompute() {
  int *cpu_indice_in = (int *)input_host_;
  int *cpu_indice_out = (int *)cpu_fp32_output_[0];
  int *cpu_indice_pairs = (int *)cpu_fp32_output_[1];
  int *cpu_indice_num = (int *)cpu_fp32_output_[2];
  int32_t grid_out_size = batch_;
  for (int i = 0; i < (dimNb_ - 2); i++) {
    grid_out_size = grid_out_size * output_space_[i];
  }
  int *cpu_grid_out =
      (int *)cpu_runtime_.allocate((uint64_t)sizeof(int) * grid_out_size);
  for (int32_t i = 0; i < grid_out_size; i++) {
    cpu_grid_out[i] = -1;
  }
  int32_t indice_pairs_size = mluOpGetTensorElementNum(indice_pairs_desc_);
  for (int i = 0; i < indice_pairs_size; i++) {
    cpu_indice_pairs[i] = -1;
  }
  int32_t indice_num_size = mluOpGetTensorElementNum(indice_num_desc_);
  for (int i = 0; i < indice_num_size; i++) {
    cpu_indice_num[i] = 0;
  }
  int32_t indice_out_size = mluOpGetTensorElementNum(indice_out_desc_);
  for (int i = 0; i < indice_out_size; i++) {
    cpu_indice_out[i] = -1;
  }

  VLOG(4) << "call cpuGetIndicePairs()";
  cpuGetIndicePairs(cpu_indice_in, cpu_indice_pairs, cpu_indice_out,
                    cpu_indice_num, cpu_grid_out, indice_in_desc_,
                    filter_space_, pad_, stride_, dilation_, output_space_,
                    dimNb_, sub_m_, batch_);

  int32_t elements =
      std::max(indice_pairs_size, std::max(indice_num_size, indice_out_size));
  float *cpu_result32 =
      (float *)cpu_runtime_.allocate(elements * sizeof(float));
  for (int i = 0; i < indice_pairs_size; i++) {
    cpu_result32[i] = (float)cpu_indice_pairs[i];
  }
  memcpy(cpu_indice_pairs, cpu_result32, indice_pairs_size * sizeof(float));

  for (int i = 0; i < indice_num_size; i++) {
    cpu_result32[i] = (float)cpu_indice_num[i];
  }
  memcpy(cpu_indice_num, cpu_result32, indice_num_size * sizeof(float));
  for (int i = 0; i < indice_out_size; i++) {
    cpu_result32[i] = (float)cpu_indice_out[i];
  }
  memcpy(cpu_indice_out, cpu_result32, indice_out_size * sizeof(float));

  cpu_runtime_.deallocate(cpu_result32);
  cpu_runtime_.deallocate(cpu_grid_out);
  cpu_runtime_.deallocate(input_host_);
  return;
}

int32_t GetIndicePairsExecutor::getValidOutPos(
    int32_t *input_pos, std::vector<int32_t> kernel_size,
    std::vector<int32_t> pad, std::vector<int32_t> stride,
    std::vector<int32_t> dilation, std::vector<int32_t> out_spatail_shape,
    int32_t *out, int32_t NDim) {
  int32_t lowers[NDim];
  int32_t uppers[NDim];
  int32_t counter[NDim];
  int32_t counter_size[NDim];
  int32_t point_counter = 0;
  int32_t val;
  int32_t num_points = 1;
  int32_t m, offset;
  bool valid = false;
  for (int i = 0; i < NDim; ++i) {
    lowers[i] = (input_pos[i] - (kernel_size[i] - 1) * dilation[i] - 1 +
                 stride[i] + pad[i]) /
                stride[i];
    uppers[i] = (input_pos[i] + pad[i]) / stride[i];
  }
  for (int i = 0; i < NDim; ++i) {
    counter_size[i] = ((uppers[i] - lowers[i]) / dilation[i] + 1);
    num_points *= counter_size[i];
  }
  for (int i = 0; i < NDim; ++i) {
    counter[i] = 0;
  }

  for (int i = 0; i < num_points; ++i) {
    valid = true;
    m = 1;
    offset = 0;
    for (int j = NDim - 1; j >= 0; --j) {
      val = uppers[j] - counter[j] * dilation[j];
      out[point_counter * (NDim + 1) + j] = val;
      if (val < 0 || val > (out_spatail_shape[j] - 1)) {
        valid = false;
      }
      offset += m * (input_pos[j] - val * stride[j] + pad[j]) / dilation[j];
      m *= kernel_size[j];
    }  // NDim
    out[point_counter * (NDim + 1) + NDim] = offset;
    if (valid) point_counter++;
    counter[NDim - 1] += 1;
    for (int c = NDim - 1; c >= 0; --c) {
      if (counter[c] == counter_size[c] && c > 0) {
        counter[c - 1] += 1;
        counter[c] = 0;
      }
    }
  }  // num_points
  return point_counter;
}

void GetIndicePairsExecutor::cpuGetIndicePairs(
    int32_t *indice_in, int32_t *indice_pairs, int32_t *indice_out,
    int32_t *indice_num, int32_t *grid_out,
    const mluOpTensorDescriptor_t indice_in_desc,
    std::vector<int32_t> kernel_size, std::vector<int32_t> pad,
    std::vector<int32_t> stride, std::vector<int32_t> dilation,
    std::vector<int32_t> out_spatail_shape, const int32_t dimNb,
    const int32_t sub_m, const int32_t batch_size) {
  int32_t num_act = 0;
  int32_t num_act_in = indice_in_desc->dims[0];
  int32_t batch_idx = 0;
  int32_t spatail_volume = 1;
  int32_t NDim = dimNb - 2;
  for (int i = 0; i < NDim; ++i) {
    spatail_volume *= out_spatail_shape[i];
  }
  int32_t kernel_volume = 1;
  for (int i = 0; i < NDim; ++i) {
    kernel_volume *= kernel_size[i];
  }
  int32_t num_valid_points = 0;
  int *valid_points_ =
      (int *)cpu_runtime_.allocate(kernel_volume * (NDim + 1) * sizeof(int));
  int *indicePairUnqie =
      (int *)cpu_runtime_.allocate(kernel_volume * num_act_in * sizeof(int));
  int32_t *point_ptr = nullptr;
  int32_t index = 0;
  int32_t output_space_size = batch_size * spatail_volume + 1;
  for (int i = 0; i < kernel_volume * num_act_in; i++) {
    indicePairUnqie[i] = output_space_size;
  }

  auto getIndex = [](int32_t *indice_in_temp, std::vector<int32_t> kernel_size,
                     int32_t NDim) -> int32_t {
    int32_t index_return = 0;
    int32_t size_temp = 1;
    for (int k = NDim - 1; k >= 0; --k) {
      index_return += indice_in_temp[k] * size_temp;
      size_temp *= kernel_size[k];
    }
    return index_return;
  };

  if (sub_m) {
    for (int j = 0; j < num_act_in; ++j) {
      index =
          getIndex(indice_in + j * (NDim + 1) + 1, out_spatail_shape, NDim) +
          spatail_volume * (indice_in + j * (NDim + 1))[0];
      grid_out[index] = j;
    }
    indice_out = indice_in;
  }

  for (int j = 0; j < num_act_in; ++j) {
    int32_t batchIdx = (indice_in + j * (NDim + 1))[0];
    num_valid_points =
        getValidOutPos(indice_in + j * (NDim + 1) + 1, kernel_size, pad, stride,
                       dilation, out_spatail_shape, valid_points_, NDim);
    for (int i = 0; i < num_valid_points; ++i) {
      point_ptr = valid_points_ + i * (NDim + 1);
      int32_t offset = point_ptr[NDim];  // filter_index
      index = getIndex(point_ptr, out_spatail_shape, NDim) +
              spatail_volume * batchIdx;
      if (sub_m) {
        if (grid_out[index] > 1) {
          indice_pairs[offset * 2 * num_act_in + indice_num[offset]] = j;
          indice_pairs[offset * 2 * num_act_in + num_act_in +
                       indice_num[offset]] = grid_out[index];
          indice_num[offset]++;
        }
      } else {
        int32_t oldNum = indice_num[offset];
        indice_num[offset]++;
        indice_pairs[offset * 2 * num_act_in + oldNum] = j;
        indice_pairs[offset * 2 * num_act_in + num_act_in + oldNum] = index;
        indicePairUnqie[offset * num_act_in + oldNum] = index;
      }  //  sub_m
    }    // num_valid_points
  }      //  num_act_in(L)

  if (!sub_m) {
    std::vector<int> indice_unique;
    for (int i = 0; i < kernel_volume * num_act_in; i++) {
      indice_unique.push_back(indicePairUnqie[i]);
    }
    sort(indice_unique.begin(), indice_unique.end());
    auto num_act = unique(indice_unique.begin(), indice_unique.end());
    indice_unique.erase(num_act, indice_unique.end());
    auto num_act_out = indice_unique.size() - 1;
    for (int j = 0; j < num_act_out; ++j) {
      index = indice_unique[j];
      grid_out[index] = j;
      indice_out[j * (NDim + 1)] = index / spatail_volume;  //  n
      index -= indice_out[j * (NDim + 1)] * spatail_volume;
      indice_out[j * (NDim + 1) + 1] =
          index / (out_spatail_shape[2] * out_spatail_shape[1]);  //  d
      index -= indice_out[j * (NDim + 1) + 1] *
               (out_spatail_shape[2] * out_spatail_shape[1]);
      indice_out[j * (NDim + 1) + 2] = index / out_spatail_shape[2];  //  h
      index -= indice_out[j * (NDim + 1) + 2] * out_spatail_shape[2];
      indice_out[j * (NDim + 1) + 3] = index;  //  w
    }

    for (int j = 0; j < num_act_in; ++j) {
      for (int k = 0; k < kernel_volume; k++) {
        index = indice_pairs[k * 2 * num_act_in + num_act_in + j];
        if (index > -1) {
          indice_pairs[k * 2 * num_act_in + num_act_in + j] = grid_out[index];
        }
      }
    }
  }  //  !sub_m
  cpu_runtime_.deallocate(valid_points_);
  cpu_runtime_.deallocate(indicePairUnqie);
}

int64_t GetIndicePairsExecutor::getTheoryOps() {
  int64_t kernel_volume = indice_pairs_desc_->dims[0];
  int64_t active_input_in = indice_pairs_desc_->dims[2];
  int64_t dims = dimNb_ - 2 + 1;
  int64_t total_op_size = 0;
  int64_t kernel1_op_size = 0, kernel2_op_size = 0, kernel3_op_size = 0,
          kernel4_op_size = 0;
  int64_t scatter_op_size = 0, gather_op_size = 0, reduce_op_size = 0,
          unique_op_size = 0;
  int64_t fill_op_size = 0;
  int64_t num_act_out = active_input_in * kernel_volume;
  int64_t output_size = batch_;
  for (int i = 0; i < output_space_.size(); i++) {
    output_size *= output_space_[i];
  }
  output_size++;

  if (sub_m_) {
    kernel1_op_size = active_input_in * kernel_volume * 43 +
                      active_input_in * 4;                  // default 1
    kernel2_op_size = active_input_in;                      // default 2
    kernel3_op_size = active_input_in * kernel_volume;      // default 3
    kernel4_op_size = active_input_in * kernel_volume * 2;  // subm2
    scatter_op_size = active_input_in;
    gather_op_size = active_input_in * kernel_volume;
    reduce_op_size = active_input_in * kernel_volume;
  } else {
    kernel1_op_size = active_input_in * kernel_volume * 43;  // default 1
    kernel2_op_size = num_act_out;                           // default 2
    kernel3_op_size = active_input_in * kernel_volume * 2;   // default 3
    kernel4_op_size = num_act_out * 6;                       // default 4
    scatter_op_size = num_act_out;
    gather_op_size = active_input_in * kernel_volume;
    reduce_op_size = active_input_in * kernel_volume;
    unique_op_size = active_input_in * kernel_volume;
  }
  fill_op_size =
      batch_ * filter_space_[0] * filter_space_[1] * filter_space_[2] + 1;
  total_op_size = kernel1_op_size + kernel2_op_size + kernel3_op_size +
                  kernel4_op_size + scatter_op_size + gather_op_size +
                  reduce_op_size + unique_op_size + fill_op_size;
  return total_op_size;
}

int64_t GetIndicePairsExecutor::getTheoryIoSize() {
  int64_t kernel_volume = indice_pairs_desc_->dims[0];
  int64_t active_input_in = indice_pairs_desc_->dims[2];
  int64_t dims = dimNb_ - 2 + 1;
  int64_t total_io_size = 0;
  int64_t kernel1_io_size = 0, kernel2_io_size = 0, kernel3_io_size = 0,
          kernel4_io_size = 0;
  int64_t scatter_io_size = 0, gather_io_size = 0, reduce_io_size = 0,
          unique_io_size = 0;
  int64_t fill_io_size = 0;
  int64_t num_act_out = active_input_in * kernel_volume;
  int64_t output_size = batch_;
  for (int i = 0; i < output_space_.size(); i++) {
    output_size *= output_space_[i];
  }
  output_size++;

  if (sub_m_) {
    kernel1_io_size =
        dims * active_input_in + 4 * kernel_volume * active_input_in;
    kernel2_io_size = active_input_in;                      //   default 2
    kernel3_io_size = active_input_in * kernel_volume * 3;  //  default 3
    kernel4_io_size = active_input_in * 3;                  // subm2
    scatter_io_size = active_input_in + active_input_in + output_size;
    gather_io_size = active_input_in * kernel_volume * 2 + output_size;
    reduce_io_size = active_input_in * kernel_volume + active_input_in;
  } else {
    //  kernel 1 2 3 4
    kernel1_io_size =
        dims * active_input_in + 3 * kernel_volume * active_input_in;
    kernel2_io_size = num_act_out;
    kernel3_io_size = active_input_in * kernel_volume * 3;  // in 1  out  2
    kernel4_io_size = num_act_out * 5;                      // in 1 out 4
    scatter_io_size = num_act_out + num_act_out + output_size;
    gather_io_size = active_input_in * kernel_volume * 2 + output_size;
    reduce_io_size = active_input_in * kernel_volume + active_input_in;
    unique_io_size = active_input_in * kernel_volume * 2;
  }
  fill_io_size =
      batch_ * filter_space_[0] * filter_space_[1] * filter_space_[2] + 1;
  total_io_size = kernel1_io_size + kernel2_io_size + kernel3_io_size +
                  kernel4_io_size + scatter_io_size + gather_io_size +
                  reduce_io_size + unique_io_size + fill_io_size;
  return total_io_size;
}

}  // namespace mluoptest
