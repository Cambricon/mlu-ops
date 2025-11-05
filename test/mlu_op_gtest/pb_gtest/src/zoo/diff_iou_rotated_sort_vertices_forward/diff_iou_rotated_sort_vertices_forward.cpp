/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
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
#include "diff_iou_rotated_sort_vertices_forward.h"

namespace mluoptest {

void DiffIouRotatedSortVerticesForwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 3,
              "[DiffIouRotatedSortVerticesForwardExecutor] tensor input number "
              "is wrong.");
  GTEST_CHECK(parser_->outputs().size() == 1,
              "[DiffIouRotatedSortVerticesForwardExecutor] tensor output "
              "number is wrong.");
}

void DiffIouRotatedSortVerticesForwardExecutor::compute() {
  VLOG(4)
      << "[DiffIouRotatedSortVerticesForwardExecutor] call compute() begin.";
  // input tensor
  auto vertices_desc = tensor_desc_[0].tensor;
  auto mask_desc = tensor_desc_[1].tensor;
  auto num_valid_desc = tensor_desc_[2].tensor;
  auto dev_vertices = data_vector_[0].device_ptr;
  auto dev_mask = data_vector_[1].device_ptr;
  auto dev_num_valid = data_vector_[2].device_ptr;

  // output tensor
  auto idx_desc = tensor_desc_[3].tensor;
  auto dev_idx = data_vector_[3].device_ptr;

  interface_timer_.start();
  MLUOP_CHECK(mluOpDiffIouRotatedSortVerticesForward(
      handle_, vertices_desc, dev_vertices, mask_desc, dev_mask, num_valid_desc,
      dev_num_valid, idx_desc, dev_idx));
  interface_timer_.stop();
  VLOG(4) << "[DiffIouRotatedSortVerticesForwardExecutor] call compute() end.";
}

/*
compare normalized vertices (vertices around (0,0))
if vertex1 < vertex2 return true.
order: minimum at x-aixs, become larger in anti-clockwise direction
*/
bool DiffIouRotatedSortVerticesForwardExecutor ::compare_vertices(
    const float x1, const float y1, const float x2, const float y2) {
  if (fabs(x1 - x2) < EPSILON && fabs(y2 - y1) < EPSILON)
    return false;  // if equal, return false

  if (y1 > 0 && y2 < 0) return true;
  if (y1 < 0 && y2 > 0) return false;

  float n1 = x1 * x1 + y1 * y1 + EPSILON;
  float n2 = x2 * x2 + y2 * y2 + EPSILON;
  float diff = fabs(x1) * x1 / n1 - fabs(x2) * x2 / n2;

  if (y1 > 0 && y2 > 0) {
    if (diff > EPSILON)
      return true;
    else
      return false;
  }
  if (y1 < 0 && y2 < 0) {
    if (diff < EPSILON)
      return true;
    else
      return false;
  }
  return false;
}

void DiffIouRotatedSortVerticesForwardExecutor::cpuCompute() {
  VLOG(4)
      << "[DiffIouRotatedSortVerticesForwardExecutor] call cpuCompute() begin.";
  float *data_vertices = cpu_fp32_input_[0];
  float *data_mask = cpu_fp32_input_[1];
  float *data_num_valid = (float *)cpu_fp32_input_[2];
  float *data_idx = (float *)cpu_fp32_output_[0];
  auto vertices_desc = tensor_desc_[0].tensor;

  int dim_b = vertices_desc->getDimIndex(0);
  int dim_n = vertices_desc->getDimIndex(1);
  int dim_m = vertices_desc->getDimIndex(2);

  memset(data_idx, 0, dim_b * dim_n * 9 * sizeof(int));
  for (int bi = 0; bi < dim_b; ++bi) {
    float *vertices = data_vertices + bi * dim_n * dim_m * 2;
    float *mask = data_mask + bi * dim_n * dim_m;
    float *num_valid = data_num_valid + bi * dim_n;
    float *idx = data_idx + bi * dim_n * MAX_NUM_VERT_IDX;

    for (int i = 0; i < dim_n; ++i) {
      int pad = 0;  // index of arbitrary invalid intersection point (not box
                    // corner!)
      for (int j = INTERSECTION_OFFSET; j < dim_m; ++j) {
        if (!mask[i * dim_m + j]) {
          pad = j;
          break;
        }
      }
      if ((int)num_valid[i] < 3) {
        // not enough vertices, take an invalid intersection point
        // (zero padding)
        for (int j = 0; j < MAX_NUM_VERT_IDX; ++j) {
          idx[i * MAX_NUM_VERT_IDX + j] = (float)pad;
        }
      } else {
        // sort the valid vertices
        // note the number of valid vertices is known
        // note: check that num_valid[i] < MAX_NUM_VERT_IDX
        for (int j = 0; j < (int)num_valid[i]; ++j) {
          // initialize with a "big" value
          float x_min = 1;
          float y_min = -EPSILON;
          int i_take = 0;
          int i2;
          float x2, y2;
          if (j != 0) {
            i2 = idx[i * MAX_NUM_VERT_IDX + j - 1];
            x2 = vertices[i * dim_m * 2 + i2 * 2 + 0];
            y2 = vertices[i * dim_m * 2 + i2 * 2 + 1];
          }
          for (int k = 0; k < dim_m; ++k) {
            float x = vertices[i * dim_m * 2 + k * 2 + 0];
            float y = vertices[i * dim_m * 2 + k * 2 + 1];
            if (mask[i * dim_m + k] && compare_vertices(x, y, x_min, y_min)) {
              if ((j == 0) || (j != 0 && compare_vertices(x2, y2, x, y))) {
                x_min = x;
                y_min = y;
                i_take = k;
              }
            }
          }
          idx[i * MAX_NUM_VERT_IDX + j] = (float)i_take;
        }
        // duplicate the first idx
        idx[i * MAX_NUM_VERT_IDX + (int)num_valid[i]] =
            idx[i * MAX_NUM_VERT_IDX + 0];

        // pad zeros
        for (int j = (int)num_valid[i] + 1; j < MAX_NUM_VERT_IDX; ++j) {
          idx[i * MAX_NUM_VERT_IDX + j] = (float)pad;
        }

        // for corner case: the two boxes are exactly the same.
        // in this case, idx would have duplicate elements, which makes the
        // shoelace formula broken because of the definition, the duplicate
        // elements only appear in the first 8 positions (they are "corners in
        // box", not "intersection of edges")
        if ((int)num_valid[i] == 8) {
          int counter = 0;
          for (int j = 0; j < 4; ++j) {
            float check = idx[i * MAX_NUM_VERT_IDX + j];
            for (int k = 4; k < INTERSECTION_OFFSET; ++k) {
              if (idx[i * MAX_NUM_VERT_IDX + k] == check) counter++;
            }
          }
          if (counter == 4) {
            idx[i * MAX_NUM_VERT_IDX + 4] = idx[i * MAX_NUM_VERT_IDX + 0];
            for (int j = 5; j < MAX_NUM_VERT_IDX; ++j) {
              idx[i * MAX_NUM_VERT_IDX + j] = (float)pad;
            }
          }
        }
      }
    }
  }

  VLOG(4)
      << "[DiffIouRotatedSortVerticesForwardExecutor] call cpuCompute() end.";
}

int64_t DiffIouRotatedSortVerticesForwardExecutor::getTheoryOps() {
  int64_t theory_ops = 0;
  theory_ops += parser_->getInputDataCount(0) * 3;
  theory_ops += parser_->getInputDataCount(2) * 2;
  VLOG(4) << "[DiffIouRotatedSortVerticesForwardExecutor] getTheoryOps: "
          << theory_ops << " ops.";
  return theory_ops;
}

}  // namespace mluoptest
