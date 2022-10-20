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

#include "expand.h"

namespace mluoptest {

// judge if shape0 can broadcast to shape1
static bool canBroadCast(std::vector<int> shape0, std::vector<int> shape1) {
  int ndim = shape1.size();
  int tensor_dim = shape0.size();
  if (tensor_dim == 0) return false;
  for (int i = ndim - 1; i >= 0; i--) {
    int offset = ndim - 1 - i;
    int dim = tensor_dim - 1 - offset;
    int size = (dim >= 0) ? shape0[dim] : 1;
    if (shape1[i] == -1) shape1[i] = size;
    if (shape1[i] != size)
      if (size != 1) return false;
  }
  return true;
}

// set dims to 8, set (1,3,224,1) -> (1,3,224,1,1,1,1,1)
void expandToMaxdim(std::vector<std::vector<int>> &expand_dims,
                    std::vector<int> input_dims, std::vector<int> output_dims) {
  int ndim = output_dims.size();
  int tensor_dim = input_dims.size();
  for (int i = ndim - 1; i >= 0; i--) {
    int offset = ndim - 1 - i;
    int dims_in = tensor_dim - 1 - offset;
    int dims_out = ndim - 1 - offset;
    int size_in = dims_in >= 0 ? input_dims[dims_in] : 1;
    int size_out = output_dims[dims_out];
    expand_dims[1][i] = size_out;
    expand_dims[0][i] = size_in;
  }
}

void ExpandExecutor::paramCheck() {
  if (parser_->getInputNum() != 1) {
    LOG(ERROR) << "expand tensor input number is wrong. ";
  }
}

void ExpandExecutor::compute() {
  VLOG(4) << "ExpandExecutor compute ";
  if (parser_->getInputNum() != 1) {
    LOG(ERROR) << "expand tensor input number is wrong. ";
  }
  auto tensor_a = tensor_desc_[0].tensor;
  auto tensor_c = tensor_desc_[1].tensor;
  auto dev_a = data_vector_[0].device_ptr;
  auto dev_c = data_vector_[1].device_ptr;
  VLOG(4) << "call mluOp expandTensor()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpExpand(handle_, tensor_a, dev_a, tensor_c, dev_c));
  interface_timer_.stop();
}

void ExpandExecutor::cpuCompute() {
  assert(parser_->getInputNum() == 1);
  assert(parser_->getOutputNum() == 1);

  auto input_dim = tensor_desc_[0].tensor->dim;
  auto output_dim = tensor_desc_[1].tensor->dim;
  int shape_a[MLUOP_DIM_MAX];
  int shape_b[MLUOP_DIM_MAX];
  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    shape_a[i] = 1;
    shape_b[i] = 1;
  }

  for (int i = 0; i < input_dim; i++) {
    shape_a[i + MLUOP_DIM_MAX - input_dim] = tensor_desc_[0].tensor->dims[i];
  }

  for (int i = 0; i < output_dim; i++) {
    shape_b[i + MLUOP_DIM_MAX - output_dim] = tensor_desc_[1].tensor->dims[i];
  }

  uint64_t size_a = 1;
  uint64_t size_b = 1;

  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    size_a = size_a * shape_a[i];
    size_b = size_b * shape_b[i];
  }

  float *tmp = (float *)cpu_runtime_.allocate(size_b * sizeof(float));
  memcpy(tmp, cpu_fp32_input_[0], size_a * sizeof(float));

  int leftsize_a = 1;
  int rightsize_a = 1;
  int rightsize_b = 1;
  int expand_times = 1;

  int size = MLUOP_DIM_MAX;

  for (int i = size - 1; i >= 0; i--) {
    rightsize_a = rightsize_a * shape_a[i];
    rightsize_b = rightsize_b * shape_b[i];
    leftsize_a = size_a / rightsize_a;
    if (shape_a[i] != shape_b[i]) {
      expand_times = shape_b[i] / shape_a[i];
      shape_a[i] = shape_b[i];
      for (int j = 0; j < leftsize_a; j++) {
        for (int k = 0; k < expand_times; k++) {
          memcpy(cpu_fp32_output_[0] + j * rightsize_b +
                     k * (rightsize_b / expand_times),
                 tmp + j * (rightsize_b / expand_times),
                 rightsize_b / expand_times * sizeof(float));
        }
      }
      memcpy(tmp, cpu_fp32_output_[0], size_b * sizeof(float));
    }
  }
  memcpy(cpu_fp32_output_[0], tmp, size_b * sizeof(float));
  cpu_runtime_.deallocate(tmp);
}

int64_t ExpandExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->getOutputDataCount(0);
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
