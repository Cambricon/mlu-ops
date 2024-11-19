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
#include "div.h"

namespace mluoptest {

void DivExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 2, "div input number is wrong. ");
  GTEST_CHECK(parser_->outputs().size() == 1, "div output number is wrong. ");
}

void DivExecutor::compute() {
  VLOG(4) << "DivExecutor compute ";
  auto tensor_x = tensor_desc_[0].tensor;
  auto tensor_y = tensor_desc_[1].tensor;
  auto tensor_z = tensor_desc_[2].tensor;
  auto dev_x = data_vector_[0].device_ptr;
  auto dev_y = data_vector_[1].device_ptr;
  auto dev_z = data_vector_[2].device_ptr;

  mluOpComputationPreference_t prefer =
      (mluOpComputationPreference_t)parser_->getProtoNode()
          ->div_param()
          .prefer();
  VLOG(4) << "call mluOpDiv";
  interface_timer_.start();
  MLUOP_CHECK(mluOpDiv(handle_, prefer, tensor_x, dev_x, tensor_y, dev_y,
                       tensor_z, dev_z));
  interface_timer_.stop();
  VLOG(4) << "DivExecutor done";
}

void DivExecutor::cpuCompute() {
  auto count1 = parser_->input(0)->shape_count;
  auto count2 = parser_->input(1)->shape_count;
  auto count3 = parser_->output(0)->shape_count;
  if (count1 == 0 || count2 == 0) {
    return;
  }

  auto a_desc = tensor_desc_[0].tensor;
  auto b_desc = tensor_desc_[1].tensor;
  auto c_desc = tensor_desc_[2].tensor;
  float *a_broadcast = (float *)cpu_runtime_.allocate(count3 * sizeof(float));
  float *b_broadcast = (float *)cpu_runtime_.allocate(count3 * sizeof(float));
  expand_compute_cpu(std::vector<int>(a_desc->getDims(), a_desc->getDims() + a_desc->getDim()),
                     std::vector<int>(c_desc->getDims(), c_desc->getDims() + c_desc->getDim()),
                     cpu_fp32_input_[0], a_broadcast);
  expand_compute_cpu(std::vector<int>(b_desc->getDims(), b_desc->getDims() + b_desc->getDim()),
                     std::vector<int>(c_desc->getDims(), c_desc->getDims() + c_desc->getDim()),
                     cpu_fp32_input_[1], b_broadcast);

  for (size_t i = 0; i < count3; ++i) {
    cpu_fp32_output_[0][i] = a_broadcast[i] / b_broadcast[i];
  }
  VLOG(4) << "Div cpu compute done";
  cpu_runtime_.deallocate(a_broadcast);
  cpu_runtime_.deallocate(b_broadcast);
  a_broadcast = NULL;
  b_broadcast = NULL;
}

bool DivExecutor::canBroadCast(std::vector<int> shape0,
                               std::vector<int> shape1) {
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

int DivExecutor::expand_num_after_first(int num) {
  int tmp = 0;
  while (num) {
    num = num >> 1;
    tmp++;
  }
  return tmp - 1;
}

void DivExecutor::expand_compute_cpu(std::vector<int> shape_a,
                                     std::vector<int> shape_b, float *input,
                                     float *output) {
  if (shape_a.size() < MLUOP_DIM_MAX) {
    shape_a.insert(shape_a.begin(), MLUOP_DIM_MAX - shape_a.size(), 1);
  }
  if (shape_b.size() < MLUOP_DIM_MAX) {
    shape_b.insert(shape_b.begin(), MLUOP_DIM_MAX - shape_b.size(), 1);
  }

  bool can_broadcast = canBroadCast(shape_a, shape_b);
  GTEST_CHECK(can_broadcast);

  uint64_t sizeA = 1;
  uint64_t sizeB = 1;

  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    sizeA = sizeA * shape_a[i];
    sizeB = sizeB * shape_b[i];
  }

  float *tmp = cpu_runtime_.allocate(new float[sizeB]);
  memcpy(tmp, input, sizeA * sizeof(float));

  int is_first = true;
  int leftSizeA = 1;
  int rightSizeA = 1;
  int leftSizeB = 1;
  int rightSizeB = 1;
  int E = 1;
  int ExpandA = 1;

  int size = MLUOP_DIM_MAX;

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
               output + j * rightSizeB, rem * (rightSizeB / E) * sizeof(float));
      }
      memcpy(tmp, output, sizeB * sizeof(float));
    }
  }
  memcpy(output, tmp, sizeB * sizeof(float));
  cpu_runtime_.deallocate(tmp);
}

int64_t DivExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->output(0)->total_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
