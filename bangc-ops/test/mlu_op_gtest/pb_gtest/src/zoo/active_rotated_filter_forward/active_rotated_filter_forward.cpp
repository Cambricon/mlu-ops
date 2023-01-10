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

#include "active_rotated_filter_forward.h"

namespace mluoptest {

void ActiveRotatedFilterForwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 2,
              "active_rotated_filter_forward input number is wrong.");
  GTEST_CHECK(parser_->outputs().size() == 1,
              "active_rotated_filter_forward output number is wrong.");
}

void ActiveRotatedFilterForwardExecutor::workspaceMalloc() {
  paramCheck();
  auto input_desc = tensor_desc_[0].tensor;
  MLUOP_CHECK(mluOpGetActiveRotatedFilterForwardWorkspaceSize(
      handle_, input_desc, &workspace_size_));
  VLOG(4) << "workspace_size:" << workspace_size_ << ".";
  eva_->setMluWorkspaceSize(workspace_size_);
  if (workspace_size_ > 0) {
    workspace_ = mlu_runtime_.allocate(workspace_size_);
  }
}

void ActiveRotatedFilterForwardExecutor::workspaceFree() {
  if (workspace_ != nullptr) {
    VLOG(4) << "ActiveRotatedFilterForwardExecutor free workspace memory.";
    mlu_runtime_.deallocate(workspace_);
    workspace_ = nullptr;
  }
}

void ActiveRotatedFilterForwardExecutor::compute() {
  paramCheck();
  VLOG(4) << "ActiveRotatedFilterForwardExecutor compute.";
  // get params
  auto input_desc = tensor_desc_[0].tensor;
  auto indices_desc = tensor_desc_[1].tensor;
  auto output_desc = tensor_desc_[2].tensor;

  auto input = data_vector_[0].device_ptr;
  auto indices = data_vector_[1].device_ptr;
  auto output = data_vector_[2].device_ptr;

  VLOG(4) << "call mluOpActiveRotatedFilterForward().";
  interface_timer_.start();

  MLUOP_CHECK(mluOpActiveRotatedFilterForward(
      handle_, input_desc, input, indices_desc, indices, workspace_,
      workspace_size_, output_desc, output));

  interface_timer_.stop();
}

void ActiveRotatedFilterForwardExecutor::cpuCompute() {
  std::vector<int> input_shape = parser_->input(0)->shape;
  std::vector<int> indices_shape = parser_->input(1)->shape;

  const int output_planes = input_shape[0];
  const int input_planes = input_shape[1];
  const int orientations = input_shape[2];
  const int h = input_shape[3];
  const int w = input_shape[4];
  const int rotations = indices_shape[3];
  auto input = cpu_fp32_input_[0];
  auto indices = cpu_fp32_input_[1];
  auto cpu_output = cpu_fp32_output_[0];

  float *cpu_output_float = (float *)cpu_output;

  const int nEntry = orientations * h * w;
  int i, j, l;
  int k;
  for (i = 0; i < output_planes; i++) {
    for (j = 0; j < input_planes; j++) {
      for (l = 0; l < nEntry; l++) {
        const int weightIndex = i * input_planes * nEntry + j * nEntry + l;
        float val = *(input + weightIndex);
        for (k = 0; k < rotations; k++) {
          const int index = *(indices + l * rotations + k) - 1;
          float *target = cpu_output_float +
                          i * (rotations * input_planes * nEntry) +
                          k * (input_planes * nEntry) + j * nEntry + index;
          *target = val;
        }
      }
    }
  }
}

int64_t ActiveRotatedFilterForwardExecutor::getTheoryIoSize() {
  auto dtype = tensor_desc_[0].tensor->dtype;
  std::vector<int> indices_shape = parser_->input(1)->shape;
  const int rotations = indices_shape[3];

  int dsize = 0;
  if (dtype == MLUOP_DTYPE_FLOAT) {
    dsize = 4;
  } else if (dtype == MLUOP_DTYPE_HALF) {
    dsize = 2;
  } else {
    GTEST_CHECK(false, "ActiveRotatedFilterForward don't support this dtype.");
  }
  int64_t theory_io_size =
      parser_->getInputDataCount(0) * dsize * (rotations + 1);
  VLOG(4) << "getTheoryIoSize: " << theory_io_size << " Bytes.";
  return theory_io_size;
}
}  // namespace mluoptest
