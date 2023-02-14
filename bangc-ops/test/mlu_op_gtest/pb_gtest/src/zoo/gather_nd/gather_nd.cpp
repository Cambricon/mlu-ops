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
#include <set>
#include <string>
#include "gather_nd.h"
namespace mluoptest {

void GatherNdExecutor::paramCheck() {
  if (parser_->getInputNum() != 2) {
    LOG(ERROR) << "gather_nd input number is wrong. ";
  }
}

void GatherNdExecutor::compute() {
  VLOG(4) << "GatherNdExecutor compute ";
  assert(parser_->getInputNum() == 2);
  assert(parser_->getOutputNum() == 1);
  auto desc_params = tensor_desc_[0].tensor;
  auto desc_indices = tensor_desc_[1].tensor;
  auto desc_output = tensor_desc_[2].tensor;
  auto params = data_vector_[0].device_ptr;
  auto indices = data_vector_[1].device_ptr;
  auto output = data_vector_[2].device_ptr;
  VLOG(4) << "call mluOp GatherNd()";
  interface_timer_.start();
  MLUOP_CHECK(
      mluOpGatherNd(handle_, desc_params, params, desc_indices, indices,
                    desc_output, output));
  interface_timer_.stop();
  data_vector_[2].is_output = true;
}

void GatherNdExecutor::cpuCompute() {
  assert(parser_->getInputNum() == 2);
  assert(parser_->getOutputNum() == 1);
  auto output_count = parser_->getOutputDataCount(0);
  auto desc_params = tensor_desc_[0].tensor;
  auto desc_indices = tensor_desc_[1].tensor;  // indices_size
  int indices_size;
  if (desc_indices->dim == 1) {
    indices_size = 1;
  } else {
    indices_size = desc_indices->dims[desc_indices->dim - 1];
  }  // s_size
  bool single_digital =
      desc_params->dim == 1 || desc_params->dim == indices_size;
  int s_size = 1;
  if (!single_digital) {
    for (int i = indices_size; i < desc_params->dim; i++) {
      s_size *= desc_params->dims[i];
    }
  }  // batch_strides
  int64_t batch_strides[indices_size + 1];  // NOLINT (runtime/arrays)
  batch_strides[indices_size] = s_size;
  for (int i = indices_size; i > 0; i--) {
    batch_strides[i - 1] = batch_strides[i] * desc_params->dims[i - 1];
  }
  int total_iterations = output_count / s_size;
  size_t SizeSlice = s_size * sizeof(float);
  int64_t offset = 0;
  int64_t indice = 0;
  for (int i = 0; i < total_iterations; i++) {
    for (int j = 0; j < indices_size; j++) {
      indice = (int32_t)(cpu_fp32_input_[1][i * indices_size + j]) * batch_strides[j + 1];  // NOLINT
      indice = indice < 0? indice +  batch_strides[j] : indice;
      offset += indice;
    }
    memcpy(cpu_fp32_output_[0] + i * s_size,
           cpu_fp32_input_[0] + offset, SizeSlice);
    offset = 0;
  }
}

int64_t GatherNdExecutor::getTheoryOps() {
  int64_t theory_ops = 0;
  assert(parser_->getInputNum() == 2);
  assert(parser_->getOutputNum() == 1);
  auto output_count = parser_->getOutputDataCount(0);
  auto desc_params = tensor_desc_[0].tensor;
  auto desc_indices = tensor_desc_[1].tensor;  // indices_size
  int indices_size;
  if (desc_indices->dim == 1) {
    indices_size = 1;
  } else {
    indices_size = desc_indices->dims[desc_indices->dim - 1];
  }  // s_size
  bool single_digital =
      desc_params->dim == 1 || desc_params->dim == indices_size;
  int s_size = 1;
  if (!single_digital) {
    for (int i = indices_size; i < desc_params->dim; i++) {
      s_size *= desc_params->dims[i];
    }
  }  int total_iterations = output_count / s_size;
  for (int i = 0; i < total_iterations; i++) {
    theory_ops += s_size;
  }
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

int64_t GatherNdExecutor::getTheoryIoSize() {
  // theory_ios = index + output + unique_index_count * tranfor_num
  auto output_count = parser_->getOutputDataCount(0);
  auto desc_params = tensor_desc_[0].tensor;
  auto desc_indices = tensor_desc_[1].tensor;  // index_dim
  int index_dim;
  if (desc_indices->dim == 1) {
    index_dim = 1;
  } else {
    index_dim = desc_indices->dims[desc_indices->dim - 1];
  }  // s_size
  bool single_digital = desc_params->dim == 1 || desc_params->dim == index_dim;
  int s_size = 1;
  if (!single_digital) {
    for (int i = index_dim; i < desc_params->dim; i++) {
      s_size *= desc_params->dims[i];
    }
  }
  // for gpu pb test
  Device device = parser_-> device();
  float *indices_ptr = NULL;
  if (device == Device::GPU) {
    int64_t index_count = desc_indices->total_element_num;
    float *indices_host =
        (float *)cpu_runtime_.allocate(index_count * sizeof(float));
    castDataOut(data_vector_[1].host_ptr, desc_indices->dtype,
                (float *)indices_host, MLUOP_DTYPE_FLOAT, index_count,
                NO_QUANT, 0, 1, 0);
    indices_ptr = indices_host;
  } else {
    indices_ptr = cpu_fp32_input_[1];
  }
/*
 * remove duplicate indexs for multidimensional coordinates, eg:
 * coordinates array = (1, 10, 3)  change to  string key = 1_10_3_
 *                     (11, 0, 3)                          11_0_3_
 *                     (110, 0, 0)                         110_0_0_
 * use std::set compute unique index count
 */
  auto index_count = desc_indices->total_element_num / index_dim;
  std::set<std::string> unique_index;
  for (int i = 0; i < index_count; i++) {
    std::string index_str = "";
    for (int j = 0; j < index_dim; j++) {
      float indice = indices_ptr[i * index_dim + j];
      index_str = index_str + std::to_string(int(indice)) + "_";
    }
    unique_index.insert(index_str);
  }
  auto unique_index_count = unique_index.size();
  auto desc_output = tensor_desc_[2].tensor;
  auto params_dtype_size = mluop::getSizeOfDataType(desc_params->dtype);
  int64_t theory_ios = desc_indices->total_tensor_size +
                       desc_output->total_tensor_size +
                       unique_index_count * s_size * params_dtype_size;
  VLOG(4) << "getTheoryIOs: " << theory_ios << " bytes";
  if (device == Device::GPU) {
    cpu_runtime_.deallocate(indices_ptr);
  }
  return theory_ios;
}
}  // namespace mluoptest
