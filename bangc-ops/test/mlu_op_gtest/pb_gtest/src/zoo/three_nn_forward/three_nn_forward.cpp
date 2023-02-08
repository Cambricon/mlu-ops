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

#include "three_nn_forward.h"

#include "mlu_op.h"

namespace mluoptest {

void ThreeNnForwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 2,
              "three_nn_forward input number is wrong.");
  GTEST_CHECK(parser_->outputs().size() == 2,
              "three_nn_forward output number is wrong.");
}

void ThreeNnForwardExecutor::workspaceMalloc() {
  paramCheck();
  auto known_desc = tensor_desc_[1].tensor;
  MLUOP_CHECK(mluOpGetThreeNNForwardWorkspaceSize(handle_, known_desc,
                                                  &workspace_size_));
  VLOG(4) << "workspace_size:" << workspace_size_ << ".";
  eva_->setMluWorkspaceSize(workspace_size_);
  if (workspace_size_ > 0) {
    workspace_ = mlu_runtime_.allocate(workspace_size_);
  }
}

void ThreeNnForwardExecutor::workspaceFree() {
  if (workspace_ != nullptr) {
    VLOG(4) << "ThreeNNForwardExecutor free workspace memory.";
    mlu_runtime_.deallocate(workspace_);
    workspace_ = nullptr;
  }
}

void ThreeNnForwardExecutor::compute() {
  paramCheck();
  VLOG(4) << "ThreeNNForwardExecutor compute.";
  // get params
  auto unknown_desc = tensor_desc_[0].tensor;
  auto known_desc = tensor_desc_[1].tensor;
  auto dist2_desc = tensor_desc_[2].tensor;
  auto idx_desc = tensor_desc_[3].tensor;

  auto unknown = data_vector_[0].device_ptr;
  auto known = data_vector_[1].device_ptr;
  auto dist2 = data_vector_[2].device_ptr;
  auto idx = data_vector_[3].device_ptr;

  VLOG(4) << "call mluOpThreeNNForward().";
  interface_timer_.start();

  MLUOP_CHECK(mluOpThreeNNForward(handle_, unknown_desc, unknown, known_desc,
                                  known, workspace_, workspace_size_,
                                  dist2_desc, dist2, idx_desc, idx));

  interface_timer_.stop();
}

void ThreeNnForwardExecutor::cpuCompute() {
  std::vector<int> unknown_shape = parser_->input(0)->shape;
  std::vector<int> known_shape = parser_->input(1)->shape;

  const int b = unknown_shape[0];
  const int n = unknown_shape[1];
  const int m = known_shape[1];
  auto unknown = cpu_fp32_input_[0];
  auto known = cpu_fp32_input_[1];
  auto dist2 = cpu_fp32_output_[0];
  float *dist2_start = (float *)cpu_fp32_output_[0];
  float *idx = (float *)cpu_fp32_output_[1];
  float *idx_start = (float *)cpu_fp32_output_[1];

  for (int i = 0; i < b; ++i) {
    for (int j = 0; j < n; ++j) {
      float ux = unknown[j * 3 + 0];
      float uy = unknown[j * 3 + 1];
      float uz = unknown[j * 3 + 2];
      double best1 = 1e40;
      double best2 = 1e40;
      double best3 = 1e40;
      int besti1 = 0;
      int besti2 = 0;
      int besti3 = 0;

      for (int k = 0; k < m; ++k) {
        float x = known[k * 3 + 0];
        float y = known[k * 3 + 1];
        float z = known[k * 3 + 2];
        double d =
            (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
        if (d < best1) {
          best3 = best2;
          besti3 = besti2;
          best2 = best1;
          besti2 = besti1;
          best1 = d;
          besti1 = k;
        } else if (d < best2) {
          best3 = best2;
          besti3 = besti2;
          best2 = d;
          besti2 = k;
        } else if (d < best3) {
          best3 = d;
          besti3 = k;
        }
      }
      dist2[j * 3 + 0] = float(best1);
      dist2[j * 3 + 1] = float(best2);
      dist2[j * 3 + 2] = float(best3);
      idx[j * 3 + 0] = besti1;
      idx[j * 3 + 1] = besti2;
      idx[j * 3 + 2] = besti3;
    }
    unknown += n * 3;
    known += m * 3;
    dist2 += n * 3;
    idx += n * 3;
  }
}

int64_t ThreeNnForwardExecutor::getTheoryOps() {
  std::vector<int> unknown_shape = parser_->input(0)->shape;
  std::vector<int> known_shape = parser_->input(1)->shape;
  int64_t theory_ops = 0;
  const int b = unknown_shape[0];
  const int n = unknown_shape[1];
  const int m = known_shape[1];
  theory_ops += b * n * m * 8;
  VLOG(4) << "getTheoryOps: " << theory_ops << "ops.";
  return theory_ops;
}
}  // namespace mluoptest
