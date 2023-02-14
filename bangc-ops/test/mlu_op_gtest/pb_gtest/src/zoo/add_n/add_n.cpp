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
#include <vector>
#include "add_n.h"
namespace mluoptest {
bool AddNExecutor::canBroadCast(std::vector<int> shape0,
                                std::vector<int> shape1) {
  int ndim = shape1.size();
  int tensor_dim = shape0.size();
  if (tensor_dim == 0)
    return false;
  for (int i = ndim - 1; i >= 0; i--) {
    int offset = ndim - 1 - i;
    int dim = tensor_dim - 1 - offset;
    int size = (dim >= 0) ? shape0[dim] : 1;
    if (shape1[i] == -1)
      shape1[i] = size;
    if (shape1[i] != size)
      if (size != 1)
        return false;
  }
  return true;
}

void AddNExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_addn_param()) {
    LOG(ERROR) << "Missing add_n param. ";
  }  uint32_t num = parser_->getProtoNode()->addn_param().num();
  if (parser_->getInputNum() != num) {
    LOG(ERROR) << "add_n input number is wrong. ";
  }
}

void AddNExecutor::compute() {
  VLOG(4) << "AddNExecutor compute ";
  if (!parser_->getProtoNode()->has_addn_param()) {
    LOG(ERROR) << "Missing add_n param. ";
  }
  uint32_t num = parser_->getProtoNode()->addn_param().num();
  mluOpTensorDescriptor_t inputs_desc[num];  // NOLINT
  void *inputs[num];
  for (int i = 0; i < num; ++i) {
    inputs_desc[i] = tensor_desc_[i].tensor;
    inputs[i] = data_vector_[i].device_ptr;
  }
  auto c_desc = tensor_desc_[num].tensor;
  auto c = data_vector_[num].device_ptr;
  // call
  VLOG(4) << "call mluOp add_n";
  auto workspace = workspace_.at(0);
  interface_timer_.start();
  MLUOP_CHECK(mluOpAddN_v2(handle_, inputs_desc, inputs, num, c_desc,
                           c, workspace, workspace_size));
  interface_timer_.stop();
  data_vector_[num].is_output = true;
}

void AddNExecutor::cpuCompute() {
  uint32_t num = parser_->getProtoNode()->addn_param().num();
  mluOpTensorDescriptor_t inputs_desc[num];  // NOLINT
  const void *inputs[num];
  for (int i = 0; i < num; ++i) {
    inputs_desc[i] = tensor_desc_[i].tensor;
    inputs[i] = data_vector_[i].device_ptr;
    if (parser_->getInputDataCount(i) == 0) {
      return;
    }
  }
  auto c_desc = tensor_desc_[num].tensor;
  auto c = data_vector_[num].device_ptr;
  auto count_output = parser_->getOutputDataCount(0);
  if (count_output == 0) {
    return;
  }  std::vector<double> result(count_output, 0.0);
  std::vector<double> y(count_output, 0.0);
  std::vector<double> temp(count_output, 0.0);
  std::vector<double> comp(count_output, 0.0);
  for (int i = 0; i < count_output; ++i) {
    result[i] = 0.0;
    comp[i] = 0.0;
  }  for (int i = 0; i < num; ++i) {
    float *a_broadcast =
        (float *)cpu_runtime_.allocate(count_output * sizeof(float));
    expand_compute_cpu(
        std::vector<int>(inputs_desc[i]->dims,
                         inputs_desc[i]->dims + inputs_desc[i]->dim),
        std::vector<int>(c_desc->dims, c_desc->dims + c_desc->dim),
        cpu_fp32_input_[i], a_broadcast);
    for (int j = 0; j < count_output; ++j) {
      y[j] = (double)a_broadcast[j] - comp[j];
      temp[j] = result[j] + y[j];
      comp[j] = (temp[j] - result[j]) - y[j];
      result[j] = temp[j];
    }
  }
  switch (c_desc->dtype) {
    default: break;
    case MLUOP_DTYPE_FLOAT:
    case MLUOP_DTYPE_HALF: {
      for (int i = 0; i < count_output; ++i) {
        cpu_fp32_output_[0][i] = (float)result[i];
      }
    }; break;
    case MLUOP_DTYPE_INT32: {
      for (int i = 0; i < count_output; ++i) {
        cpu_fp32_output_[0][i] = (int32_t)result[i];
      }
    }; break;
    case MLUOP_DTYPE_INT16: {
        for (int i = 0; i < count_output; ++i) {
          cpu_fp32_output_[0][i] = (int16_t)result[i];
        }
    }; break;
    case MLUOP_DTYPE_INT8: {
        for (int i = 0; i < count_output; ++i) {
          cpu_fp32_output_[0][i] = (int8_t)result[i];
        }
    }; break;
    case MLUOP_DTYPE_UINT8: {
        for (int i = 0; i < count_output; ++i) {
          cpu_fp32_output_[0][i] = (uint8_t)result[i];
        }
    }; break;
  }
}

void AddNExecutor::workspaceMalloc() {
  uint32_t num = parser_->getProtoNode()->addn_param().num();
  mluOpTensorDescriptor_t inputs_desc[num];  // NOLINT
  void *inputs[num];
  for (int i = 0; i < num; ++i) {
    inputs_desc[i] = tensor_desc_[i].tensor;
    inputs[i] = data_vector_[i].device_ptr;
  }
  auto c_desc = tensor_desc_[num].tensor;
  auto c = data_vector_[num].device_ptr;
  void *workspace = NULL;
  MLUOP_CHECK(mluOpGetAddNWorkspaceSize(handle_, inputs_desc, num,
                                        c_desc, &workspace_size));
  if (workspace_size != 0) {
    workspace = mlu_runtime_.allocate(workspace_size);
  }
  workspace_.push_back(workspace);
  eva_->setMluWorkspaceSize(workspace_size);
}

void AddNExecutor::workspaceFree() {
  if (workspace_[0]) {
    mlu_runtime_.deallocate(workspace_[0]);
  }
}

size_t AddNExecutor::get_size_of_data_type(mluOpDataType_t dtype) {
  GTEST_CHECK(dtype >= 0);  switch (dtype) {
    default: { return 0; }
    case MLUOP_DTYPE_UINT8:
    case MLUOP_DTYPE_INT8: {
      return 1;
    }
    case MLUOP_DTYPE_INT16:
    case MLUOP_DTYPE_HALF: {
      return 2;
    }
    case MLUOP_DTYPE_INT32:
    case MLUOP_DTYPE_FLOAT: {
      return 4;
    }
  }
}

int AddNExecutor::expand_num_after_first(int num) {
  int tmp = 0;
  while (num) {
    num = num >> 1;
    tmp++;
  }
  return tmp - 1;
}

void AddNExecutor::expand_compute_cpu(std::vector<int> shape_a,
                                      std::vector<int> shape_b,
                                      float *input,
                                      float *output) {
  if (shape_a.size() < MLUOP_DIM_MAX) {
    shape_a.insert(shape_a.begin(), MLUOP_DIM_MAX - shape_a.size(), 1);
  }
  if (shape_b.size() < MLUOP_DIM_MAX) {
    shape_b.insert(shape_b.begin(), MLUOP_DIM_MAX - shape_b.size(), 1);
  }  bool can_broadcast = canBroadCast(shape_a, shape_b);
  assert(can_broadcast == 1);  uint64_t sizeA = 1;
  uint64_t sizeB = 1;  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    sizeA = sizeA * shape_a[i];
    sizeB = sizeB * shape_b[i];
  }
  float * tmp = cpu_runtime_.allocate(new float[sizeB]);
  memcpy(tmp, input, sizeA * sizeof(float));  int is_first = true;
  int leftSizeA = 1;
  int rightSizeA = 1;
  int leftSizeB = 1;
  int rightSizeB = 1;
  int E = 1;
  int ExpandA = 1;  int size = MLUOP_DIM_MAX;
  for (int i = size - 1; i >= 0; i--) {
    rightSizeA = rightSizeA * shape_a[i];
    rightSizeB = rightSizeB * shape_b[i];
    leftSizeA = sizeA / rightSizeA;
    leftSizeB = sizeB / rightSizeB;
    if (shape_a[i] != shape_b[i]) {
      E = shape_b[i];
      ExpandA = ExpandA * shape_a[i];
      shape_a[i] = shape_b[i];
      for (int j = 0; j < leftSizeA; j++) {
        int numAfter = expand_num_after_first(E);
        memcpy(output + j * rightSizeB, tmp + j * (rightSizeB / E),
               rightSizeB / E * sizeof(float));
        for (int k = 1; k <= numAfter; k++) {
          memcpy(output + j * rightSizeB + (1 << (k - 1)) * (rightSizeB / E),
                 output + j * rightSizeB,
                 (1 << (k - 1)) * (rightSizeB / E) * sizeof(float));
        }
        int done = 1 << numAfter;
        int rem = E - (1 << numAfter);
        memcpy(output + j * rightSizeB + done * (rightSizeB / E),
               output + j * rightSizeB,
               rem * (rightSizeB / E) * sizeof(float));
      }
      memcpy(tmp, output, sizeB * sizeof(float));
    }
  }
  memcpy(output, tmp, sizeB * sizeof(float));
  cpu_runtime_.deallocate(tmp);
}

int64_t AddNExecutor::getTheoryOps() {
  int input_num = parser_->getInputNum();
  int64_t theory_ops = 0;  // sum of input
  for (int i = 0; i < input_num; i++) {
    theory_ops += parser_->getInputDataCount(0);
  }
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}
}  // namespace mluoptest
