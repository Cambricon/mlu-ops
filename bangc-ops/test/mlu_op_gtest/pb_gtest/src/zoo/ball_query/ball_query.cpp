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

#include "ball_query.h"

#include <iostream>
#include <vector>

#include "mlu_op.h"
#include "core/type.h"

namespace mluoptest {
void BallQueryExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 2,
              "[BallQueryExecutor] input number is wrong. ");
  GTEST_CHECK(parser_->outputs().size() == 1,
              "[BallQueryExecutor] output number is wrong. ");
}

void BallQueryExecutor::compute() {
  auto new_xyz_desc = tensor_desc_[0].tensor;
  auto new_xyz_ptr = data_vector_[0].device_ptr;
  auto xyz_desc = tensor_desc_[1].tensor;
  auto xyz_ptr = data_vector_[1].device_ptr;

  min_radius_ = parser_->getProtoNode()->ball_query_param().min_radius();
  max_radius_ = parser_->getProtoNode()->ball_query_param().max_radius();
  nsample_ = parser_->getProtoNode()->ball_query_param().nsample();

  auto idx_desc = tensor_desc_[2].tensor;
  auto idx_ptr = data_vector_[2].device_ptr;
  auto data_vector = (int *)data_vector_[2].host_ptr;
  // set idx to 0
  size_t output_total_bytes = data_vector_[2].count * sizeof(int32_t);
  GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMemset(idx_ptr, 0, output_total_bytes));
  interface_timer_.start();
  MLUOP_CHECK(mluOpBallQuery(handle_, new_xyz_desc, new_xyz_ptr, xyz_desc,
                             xyz_ptr, min_radius_, max_radius_, nsample_,
                             idx_desc, idx_ptr));
  interface_timer_.stop();
  data_vector_[2].is_output = true;
}

void BallQueryExecutor::cpuCompute() {
  VLOG(4) << "[BallQueryExecutor] call cpuCompute";
  auto new_xyz_host = cpu_fp32_input_[0];
  auto xyz_host = cpu_fp32_input_[1];
  auto idx_host = cpu_fp32_output_[0];

  // get b, m, n
  auto new_xyz_shape = data_vector_[0].shape;
  auto xyz_shape = data_vector_[1].shape;
  int b = new_xyz_shape[0];
  int m = new_xyz_shape[1];
  int n = xyz_shape[1];

  auto idx_shape = data_vector_[2].shape;

  VLOG(4) << "new_xyz shape:[" << new_xyz_shape[0] << ", " << new_xyz_shape[1]
          << ", " << new_xyz_shape[2] << "]";
  VLOG(4) << "xyz shape:[" << xyz_shape[0] << ", " << xyz_shape[1] << ", "
          << xyz_shape[2] << "]";
  VLOG(4) << "idx shape:[" << idx_shape[0] << ", " << idx_shape[1] << ", "
          << idx_shape[2] << "]";
  // get min_radius and max_radius
  min_radius_ = parser_->getProtoNode()->ball_query_param().min_radius();
  max_radius_ = parser_->getProtoNode()->ball_query_param().max_radius();
  nsample_ = parser_->getProtoNode()->ball_query_param().nsample();

  float min_radius2 = min_radius_ * min_radius_;
  float max_radius2 = max_radius_ * max_radius_;

  for (int b_idx = 0; b_idx < b; ++b_idx) {
    for (int row = 0; row < m; ++row) {
      int record_idx = 0;
      bool in_ball = false;
      for (int col = 0; col < n; ++col) {
        float sub_x1 = new_xyz_host[b_idx * m * 3 + row * 3 + 0] -
                       xyz_host[b_idx * n * 3 + col * 3 + 0];
        float sub_y1 = new_xyz_host[b_idx * m * 3 + row * 3 + 1] -
                       xyz_host[b_idx * n * 3 + col * 3 + 1];
        float sub_z1 = new_xyz_host[b_idx * m * 3 + row * 3 + 2] -
                       xyz_host[b_idx * n * 3 + col * 3 + 2];
        float distance2 = sub_x1 * sub_x1 + sub_y1 * sub_y1 + sub_z1 * sub_z1;
        if (distance2 == 0 ||
            (distance2 >= min_radius2 && distance2 < max_radius2)) {
          in_ball = true;
          if (record_idx == 0) {
            for (int i = 0; i < nsample_; ++i) {
              idx_host[b_idx * m * nsample_ + row * nsample_ + i] = col;
            }
          }
          idx_host[b_idx * m * nsample_ + row * nsample_ + record_idx] = col;
          ++record_idx;
          if (record_idx >= nsample_) break;
        }
      }
      if (!in_ball) {  // for one nex_xyz point, if xyz points are out of
                       // ball,then set this idx_host to 0
        for (int i = 0; i < nsample_; ++i) {
          idx_host[b_idx * m * nsample_ + row * nsample_ + i] = 0;
        }
      }
    }
  }
  VLOG(4) << "BallQuery cpu compute done";
}

int64_t BallQueryExecutor::getTheoryOps() {
  std::vector<int> new_xyz_shape = parser_->input(0)->shape;
  std::vector<int> xyz_shape = parser_->input(1)->shape;
  const int b = new_xyz_shape[0];
  const int m = new_xyz_shape[1];
  const int n = xyz_shape[1];
  int64_t theory_ops = b * n * m * 10;
  VLOG(4) << "getTheoryops: " << theory_ops << " ops.";
  return theory_ops;
}
}  // namespace mluoptest
