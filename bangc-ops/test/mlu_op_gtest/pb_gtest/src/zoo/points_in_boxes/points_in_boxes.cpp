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
#include "points_in_boxes.h"
#include "mlu_op.h"

namespace mluoptest {

static void lindar_to_local_coords_cpu(float shift_x, float shift_y,
                                       float rot_angle, float &local_x,
                                       float &local_y) {
  float cosa = cos(-rot_angle), sina = sin(-rot_angle);
  local_x = shift_x * cosa + shift_y * (-sina);
  local_y = shift_x * sina + shift_y * cosa;
}

static int check_pt_in_box3d_cpu(const float *pt, const float *box3d,
                                 float &local_x, float &local_y) {
  const float MARGIN = 1e-5;
  float x = pt[0], y = pt[1], z = pt[2];
  float cx = box3d[0], cy = box3d[1], cz = box3d[2];
  float dx = box3d[3], dy = box3d[4], dz = box3d[5], rz = box3d[6];

  if (fabsf(z - cz) > dz / 2.0) return 0;
  lindar_to_local_coords_cpu(x - cx, y - cy, rz, local_x, local_y);
  float in_flag =
      (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
  return in_flag;
}

static void points_in_boxes_cpu(
    const mluOpTensorDescriptor_t points_desc, const void *points,
    const mluOpTensorDescriptor_t boxes_desc, const void *boxes,
    const mluOpTensorDescriptor_t points_indices_desc, void *points_indices) {
  for (int64_t i = 0;
       i < points_indices_desc->dims[0] * points_indices_desc->dims[1]; i++) {
    *((float *)points_indices + i) = -1.0;
  }
  for (int64_t i = 0; i < points_desc->dims[0]; i++) {
    for (int64_t j = 0; j < points_desc->dims[1]; j++) {
      for (int64_t m = 0; m < boxes_desc->dims[1]; m++) {
        float local_x, local_y;
        int cur_in_flag = check_pt_in_box3d_cpu(
            (float *)points + (i * points_desc->dims[1] + j) * 3,
            (float *)boxes + (i * boxes_desc->dims[1] + m) * 7, local_x,
            local_y);
        if (cur_in_flag) {
          *((float *)points_indices + i * points_desc->dims[1] + j) = (float)m;
          break;
        }
      }
    }
  }
}

void PointsInBoxesExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 2,
              "points_in_boxes tensor input number is wrong.");
  GTEST_CHECK(parser_->outputs().size() == 1,
              "points_in_boxes tensor output number is wrong.");
}

void PointsInBoxesExecutor::compute() {
  auto points_desc = tensor_desc_[0].tensor;
  auto boxes_desc = tensor_desc_[1].tensor;
  auto points_indices_desc = tensor_desc_[2].tensor;

  auto pDev_points = data_vector_[0].device_ptr;
  auto pDev_boxes = data_vector_[1].device_ptr;
  auto pDev_points_indices = data_vector_[2].device_ptr;

  VLOG(4) << "PointsInBoxesExecutor::compute() Begin.";
  interface_timer_.start();
  MLUOP_CHECK(mluOpPointsInBoxes(handle_, points_desc, pDev_points, boxes_desc,
                                 pDev_boxes, points_indices_desc,
                                 pDev_points_indices));
  interface_timer_.stop();
  VLOG(4) << "PointsInBoxesExecutor::compute() End.";
}

void PointsInBoxesExecutor::cpuCompute() {
  VLOG(4) << "PointsInBoxesExecutor::cpuCompute() Begin.";
  auto points_desc = tensor_desc_[0].tensor;
  auto boxes_desc = tensor_desc_[1].tensor;
  auto points_indices_desc = tensor_desc_[2].tensor;
  points_in_boxes_cpu(points_desc, cpu_fp32_input_[0], boxes_desc,
                      cpu_fp32_input_[1], points_indices_desc,
                      cpu_fp32_output_[0]);
  VLOG(4) << "PointsInBoxesExecutor::cpuCompute() End.";
}

int64_t PointsInBoxesExecutor::getTheoryOps() {
  int64_t pts = parser_->input(0)->total_count / 3;
  int64_t bs = parser_->input(1)->total_count / 7;
  int64_t theory_ops = 2 * pts * 3 + 21 * pts * bs + 2 * pts;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
