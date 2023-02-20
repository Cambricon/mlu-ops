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
#include <map>
#include <set>
#include <utility>
#include "scatter_nd.h"
namespace mluoptest {

void ScatterNdExecutor::paramCheck() {
  m_version = parser_->getProtoNode()->scatter_nd_param().version();
  if (m_version == 1) {
    if (parser_->getInputNum() != 2) {
      LOG(ERROR) << "scatter_nd input number is wrong. ";
    }
  } else if (m_version == 2) {
    if (parser_->getInputNum() != 3) {
      LOG(ERROR) << "scatter_nd input number is wrong. ";
    }
  } else {
      LOG(ERROR) << "scatter_nd input parameter is wrong. version = "
                 << m_version;
  }
  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "scatter_nd output number is wrong. ";
  }
}

std::set<Evaluator::Formula> ScatterNdExecutor::getCriterionsUse() const {
  ScatterNdMode mode = parser_->getProtoNode()->scatter_nd_param().mode();
  auto input_dtype = parser_->getInputDataType(1);
  bool is_diff3 =
      ((mode == SCATTERND_UPDATE) || (input_dtype == MLUOP_DTYPE_INT32)) &&
      (m_version == 2);
  if (is_diff3) {
    return { Evaluator::DIFF3 };
  } else if (exe_context_->handle->arch < 300) {
    return { Evaluator::DIFF1, Evaluator::DIFF2 };
  } else {
    return { Evaluator::DIFF1, Evaluator::DIFF2, Evaluator::DIFF4 };
  }
}

void ScatterNdExecutor::compute() {
  VLOG(4) << "ScatterNdExecutor compute ";
  auto desc_indices = tensor_desc_[0].tensor;
  auto desc_updates = tensor_desc_[1].tensor;
  auto indices = data_vector_[0].device_ptr;
  auto updates = data_vector_[1].device_ptr;
  mluOpScatterNdMode_t mode =
    (mluOpScatterNdMode_t)parser_-> getProtoNode()-> scatter_nd_param().mode();
  if (m_version == 1) {
    auto desc_output = tensor_desc_[2].tensor;
    auto output = data_vector_[2].device_ptr;
    VLOG(4) << "call mluOp ScatterNd()";
    interface_timer_.start();
    MLUOP_CHECK(mluOpScatterNd(handle_, desc_indices, indices, desc_updates,
                               updates, desc_output, output));
    interface_timer_.stop();
    data_vector_[2].is_output = true;
  } else {
    auto desc_input = tensor_desc_[2].tensor;
    auto desc_output = tensor_desc_[3].tensor;
    auto input = data_vector_[2].device_ptr;
    auto output = data_vector_[3].device_ptr;
    VLOG(4) << "call mluOp ScatterNd_v2()";
    interface_timer_.start();
    MLUOP_CHECK(mluOpScatterNd_v2(handle_, mode, desc_indices, indices,
                                  desc_updates, updates, desc_input, input,
                                  desc_output, output));
    interface_timer_.stop();
    data_vector_[3].is_output = true;
  }
}

void ScatterNdExecutor::diffPreprocess() {
  int32_t version = m_version;
  ScatterNdMode mode = parser_->getProtoNode()->scatter_nd_param().mode();
  if (version == 1 || mode != SCATTERND_UPDATE) {
    return;
  }
  auto output_count = parser_->getOutputDataCount(0);
  auto input_count = parser_->getInputDataCount(1);
  auto desc_indices = tensor_desc_[0].tensor;
  auto desc_updates = tensor_desc_[1].tensor;
  auto desc_output = tensor_desc_[2].tensor;
  if (m_version == 2) {
    desc_output = tensor_desc_[3].tensor;
  }
  // indices_size
  int indices_size = desc_indices->dims[desc_indices->dim - 1];
  // inner_dim_size
  int inner_dim_size = 1;
  for (int i = desc_indices->dim - 1; i < desc_updates->dim; ++i) {
    inner_dim_size *= desc_updates->dims[i];
  }
  //  // batch_strides
  int batch_strides[8];
  // max dims of a tensor
  for (int i = 0; i < 8; i++) {
    batch_strides[i] = 1;
  }
  batch_strides[indices_size - 1] = inner_dim_size;
  for (int i = indices_size - 1; i > 0; i--) {
    batch_strides[i - 1] = batch_strides[i] * desc_output->dims[i];
  }
  auto indices_dtype = desc_indices->dtype;
  int indices_num = parser_-> getInputDataCount(0);
  float *indices_host =
      (float *)cpu_runtime_.allocate(indices_num * sizeof(float));
  castDataOut(data_vector_[0].host_ptr, indices_dtype, (float *)indices_host,
              MLUOP_DTYPE_FLOAT, indices_num, NO_QUANT, 0, 1, 0);
  auto update_dtype = desc_updates->dtype;
  int update_num = input_count;
  float *update_host =
      (float *)cpu_runtime_.allocate(update_num * sizeof(float));
  castDataOut(data_vector_[1].host_ptr, update_dtype, (float *)update_host,
              MLUOP_DTYPE_FLOAT, update_num, NO_QUANT, 0, 1, 0);
  std::multimap<int, int> indices_map;
  std::set<int> indices_set;
  int total_iterations = input_count / inner_dim_size;
  for (int i = 0; i < total_iterations; i++) {
    int offset = 0;
    bool invalid_index = false;
    for (int j = 0; j < indices_size; j++) {
      if (indices_host[i * indices_size + j] < 0 ||
          indices_host[i * indices_size + j] >= desc_output->dims[j]) {
        invalid_index = true;
        break;
      }
      offset += (int32_t)(indices_host[i * indices_size + j]) *
                batch_strides[j];
    }
    if (invalid_index) {
      continue;
    }
    indices_map.insert(std::make_pair(offset, i));
    indices_set.insert(offset);
  }
  std::set<int>::iterator it_set = indices_set.begin();
  while (it_set != indices_set.end()) {
    int offset = *it_set;
    int count = indices_map.count(offset);
    if (count < 2) {
      it_set++;
      continue;
    }
    for (int k = 0; k < inner_dim_size; k++) {
      if (*(int *)&cpu_fp32_output_[0][offset + k] != *(int *)&mlu_fp32_output_[0][offset + k]) {  // NOLINT
        auto it_map = indices_map.find(offset);
        while (count) {
          int i = it_map -> second;
          if (*(int *)&mlu_fp32_output_[0][offset + k] ==
              *(int *)&update_host[i * inner_dim_size + k]) {
            cpu_fp32_output_[0][offset + k] = mlu_fp32_output_[0][offset + k];
            break;
          }
          count--;
          it_map++;
        }
        count = indices_map.count(offset);
      }
    }
    it_set++;
  }
  cpu_runtime_.deallocate(indices_host);
  cpu_runtime_.deallocate(update_host);
}

void ScatterNdExecutor::cpuCompute() {
  ScatterNdMode mode = parser_->getProtoNode()->scatter_nd_param().mode();
  auto output_count = parser_->getOutputDataCount(0);
  auto input_count = parser_->getInputDataCount(1);
  auto desc_indices = tensor_desc_[0].tensor;
  auto desc_updates = tensor_desc_[1].tensor;
  auto desc_output = tensor_desc_[2].tensor;
  if (m_version == 2) {
    desc_output = tensor_desc_[3].tensor;
  }
  if (m_version == 1 || parser_->getInputDataCount(2) == 0) {
    for (int i = 0; i < output_count; i++) {
      cpu_fp32_output_[0][i] = 0;
    }
  } else {
    for (int i = 0; i< output_count; i++) {
      cpu_fp32_output_[0][i] = cpu_fp32_input_[2][i];
    }
  }
  // indices_size
  int indices_size = desc_indices->dims[desc_indices->dim - 1];
  // inner_dim_size
  int inner_dim_size = 1;
  for (int i = desc_indices->dim - 1; i < desc_updates->dim; ++i) {
    inner_dim_size *= desc_updates->dims[i];
  }
  //  // batch_strides
  int batch_strides[8];
  // max dims of a tensor
  for (int i = 0; i < 8; i++) {
    batch_strides[i] = 1;
  }
  batch_strides[indices_size - 1] = inner_dim_size;
  for (int i = indices_size - 1; i > 0; i--) {
    batch_strides[i - 1] = batch_strides[i] * desc_output->dims[i];
  }
  int total_iterations = input_count / inner_dim_size;
  for (int i = 0; i < total_iterations; i++) {
    int offset = 0;
    bool invalid_index = false;
    for (int j = 0; j < indices_size; j++) {
      if (cpu_fp32_input_[0][i * indices_size + j] < 0 ||
          cpu_fp32_input_[0][i * indices_size + j] >= desc_output->dims[j]) {
        invalid_index = true;
        break;
      }
      offset += (int32_t)(cpu_fp32_input_[0][i * indices_size + j]) * batch_strides[j];  // NOLINT
    }
    if (invalid_index) {
      continue;
    }
    for (int k = 0; k < inner_dim_size; k++) {
      switch (mode) {
        case SCATTERND_ADD:
          cpu_fp32_output_[0][offset + k] +=
              cpu_fp32_input_[1][i * inner_dim_size + k];
          break;
        case SCATTERND_SUB:
          break;
        case SCATTERND_MUL:
          break;
        case SCATTERND_UPDATE:
          cpu_fp32_output_[0][offset + k] =
              cpu_fp32_input_[1][i * inner_dim_size + k];
          break;
        default:
          LOG(ERROR) << "scatter_nd not support mode. ";
          break;
      }
    }
  }
}

int64_t ScatterNdExecutor::getTheoryOps() {
  int64_t theory_ops = 0;
  baselineOutputMalloc();
  auto output_count = parser_->getOutputDataCount(0);
  auto input_count = parser_->getInputDataCount(1);
  auto desc_indices = tensor_desc_[0].tensor;
  auto desc_updates = tensor_desc_[1].tensor;
  auto desc_output = tensor_desc_[2].tensor;
  if (m_version == 2) {
    desc_output = tensor_desc_[3].tensor;
  }
  Device device = parser_-> device();
  float *fp32_input = NULL;
  if (device == Device::GPU) {
    auto indices_dtype = desc_indices->dtype;
    int indices_num = parser_-> getInputDataCount(0);
    float *indices_host =
        (float *)cpu_runtime_.allocate(indices_num * sizeof(float));
    castDataOut(data_vector_[0].host_ptr, indices_dtype,
                (float *)indices_host, MLUOP_DTYPE_FLOAT,
                indices_num, NO_QUANT, 0, 1, 0);
    fp32_input = indices_host;
  } else {
    fp32_input = cpu_fp32_input_[0];
  }  for (int i = 0; i < output_count; i++) {
    cpu_fp32_output_[0][i] = 0;
  }
  // indices_size
  int indices_size = desc_indices->dims[desc_indices->dim - 1];
  // inner_dim_size
  int inner_dim_size = 1;
  for (int i = desc_indices->dim - 1; i < desc_updates->dim; ++i) {
    inner_dim_size *= desc_updates->dims[i];
  }
  //  // batch_strides
  int batch_strides[8];
  // max dims of a tensor
  for (int i = 0; i < 8; i++) {
    batch_strides[i] = 1;
  }
  batch_strides[indices_size - 1] = inner_dim_size;
  for (int i = indices_size - 1; i > 0; i--) {
    batch_strides[i - 1] = batch_strides[i] * desc_output->dims[i];
  }
  int total_iterations = input_count / inner_dim_size;
  for (int i = 0; i < total_iterations; i++) {
    int offset = 0;
    bool invalid_index = false;
    for (int j = 0; j < indices_size; j++) {
      if (fp32_input[i * indices_size + j] < 0 ||
          fp32_input[i * indices_size + j] >= desc_output->dims[j]) {
        invalid_index = true;
        theory_ops++;
        break;
      }
      theory_ops += 2;
    }
    if (invalid_index) {
      continue;
    }
    for (int k = 0; k < inner_dim_size; k++) {
      theory_ops++;
    }
  }
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  if (device == Device::GPU) {
    cpu_runtime_.deallocate(fp32_input);
  }
  return theory_ops;
}

int64_t ScatterNdExecutor::getTheoryIoSize() {
  int64_t theory_io_size = 0;
  int32_t input_count_ = 0;
  void* input_ptr = NULL;
  auto index_count = parser_->getInputDataCount(0);
  auto input_count = parser_->getInputDataCount(1);
  auto output_count = parser_->getOutputDataCount(0);
  auto desc_indices = tensor_desc_[0].tensor;
  auto desc_updates = tensor_desc_[1].tensor;
  auto desc_output = tensor_desc_[2].tensor;
  if (m_version == 2) {
    desc_output = tensor_desc_[3].tensor;
    input_count_ = parser_->getInputDataCount(2);
    input_ptr = data_vector_[2].device_ptr;
    void* output_ptr = data_vector_[3].device_ptr;
    if (input_ptr == output_ptr) {
     input_count_ = output_count = 0;
    }
  }
  Device device = parser_-> device();
  float *fp32_input = NULL;
  if (device == Device::GPU) {
    auto indices_dtype = desc_indices->dtype;
    int indices_num = parser_-> getInputDataCount(0);
    float *indices_host =
        (float *)cpu_runtime_.allocate(indices_num * sizeof(float));
    castDataOut(data_vector_[0].host_ptr, indices_dtype,
                (float *)indices_host, MLUOP_DTYPE_FLOAT,
                indices_num, NO_QUANT, 0, 1, 0);
    fp32_input = indices_host;
  } else {
    fp32_input = cpu_fp32_input_[0];
  }
  // indices_size
  int indices_size = desc_indices->dims[desc_indices->dim - 1];
  // inner_dim_size
  int inner_dim_size = 1;
  for (int i = desc_indices->dim - 1; i < desc_updates->dim; ++i) {
    inner_dim_size *= desc_updates->dims[i];
  }
  int total_iterations = input_count / inner_dim_size;
  for (int i = 0; i < total_iterations; i++) {
    int offset = 0;
    bool invalid_index = false;
    for (int j = 0; j < indices_size; j++) {
      if (fp32_input[i * indices_size + j] < 0 ||
          fp32_input[i * indices_size + j] >= desc_output->dims[j]) {
        invalid_index = true;
        break;
      }
    }
    if (invalid_index) {
      continue;
    }
    theory_io_size += inner_dim_size;
  }
  theory_io_size = index_count + theory_io_size + input_count_ + output_count;
  theory_io_size =
      theory_io_size * mluop::getSizeOfDataType(desc_updates -> dtype);
  VLOG(4) << "scatter_nd Executor: getTheoryIOs: "
          << theory_io_size  << " bytes";
  if (device == Device::GPU) {
    cpu_runtime_.deallocate(fp32_input);
  }
  return theory_io_size;
}
}  // namespace mluoptest
