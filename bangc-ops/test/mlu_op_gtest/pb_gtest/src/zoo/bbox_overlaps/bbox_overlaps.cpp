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
#include "bbox_overlaps.h"
#include <stdlib.h>

namespace mluoptest {

void BboxOverlapsExecutor::paramCheck() {
  if (parser_->getInputNum() != 2) {
    LOG(ERROR) << "bbox overlaps input number is wrong. ";
  }

  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "bbox overlaps output number is wrong. ";
  }
}

void BboxOverlapsExecutor::compute() {
  VLOG(4) << "bboxOverlapse executor compute ";

  auto bbox1_desc = tensor_desc_[0].tensor;
  auto bbox2_desc = tensor_desc_[1].tensor;
  auto ious_desc = tensor_desc_[2].tensor;

  auto dev_bbox1 = data_vector_[0].device_ptr;
  auto dev_bbox2 = data_vector_[1].device_ptr;
  auto dev_ious = data_vector_[2].device_ptr;

  // get struct param
  int32_t mode = parser_->getProtoNode()->bbox_overlaps_param().mode();
  bool aligned = parser_->getProtoNode()->bbox_overlaps_param().aligned();
  int32_t offset = parser_->getProtoNode()->bbox_overlaps_param().offset();

  // print the input tensor
  int count1 = (int)parser_->getInputDataCount(0);
  int count2 = (int)parser_->getInputDataCount(1);
  int count3 = (int)parser_->getOutputDataCount(0);

  VLOG(4) << "call mluop BboxOverlaps()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpBboxOverlaps(handle_, mode, aligned, offset, bbox1_desc,
                                dev_bbox1, bbox2_desc, dev_bbox2, ious_desc,
                                dev_ious));
  interface_timer_.stop();
}

template <typename T>
void BboxOverlapsExecutor::cpuBboxOverlaps(const T *bbox1, const T *bbox2,
                                           T *ious, const int num_bbox1,
                                           const int num_bbox2, const int mode,
                                           const bool aligned,
                                           const int offset) {
  if (aligned) {
    for (int index = 0; index < num_bbox1; index++) {
      int b1 = index;
      int b2 = index;

      int base1 = b1 * 4;
      T b1_x1 = bbox1[base1];
      T b1_y1 = bbox1[base1 + 1];
      T b1_x2 = bbox1[base1 + 2];
      T b1_y2 = bbox1[base1 + 3];
      T b1_area = (b1_x2 - b1_x1 + offset) * (b1_y2 - b1_y1 + offset);

      int base2 = b2 * 4;
      T b2_x1 = bbox2[base2];
      T b2_y1 = bbox2[base2 + 1];
      T b2_x2 = bbox2[base2 + 2];
      T b2_y2 = bbox2[base2 + 3];
      T b2_area = (b2_x2 - b2_x1 + offset) * (b2_y2 - b2_y1 + offset);

      T left = fmaxf(b1_x1, b2_x1), right = fminf(b1_x2, b2_x2);
      T top = fmaxf(b1_y1, b2_y1), bottom = fminf(b1_y2, b2_y2);
      T width = fmaxf(right - left + offset, 0.f);
      T height = fmaxf(bottom - top + offset, 0.f);
      T interS = width * height;
      T baseS = 1.0;
      if (mode == 0) {
        baseS = fmaxf(b1_area + b2_area - interS, T(offset));
      } else if (mode == 1) {
        baseS = fmaxf(b1_area, T(offset));
      }
      ious[index] = interS / baseS;
    }
  } else {
    for (int index = 0; index < num_bbox1 * num_bbox2; index++) {
      int b1 = index / num_bbox2;
      int b2 = index % num_bbox2;

      int base1 = b1 * 4;
      T b1_x1 = bbox1[base1];
      T b1_y1 = bbox1[base1 + 1];
      T b1_x2 = bbox1[base1 + 2];
      T b1_y2 = bbox1[base1 + 3];
      T b1_area = (b1_x2 - b1_x1 + offset) * (b1_y2 - b1_y1 + offset);

      int base2 = b2 * 4;
      T b2_x1 = bbox2[base2];
      T b2_y1 = bbox2[base2 + 1];
      T b2_x2 = bbox2[base2 + 2];
      T b2_y2 = bbox2[base2 + 3];
      T b2_area = (b2_x2 - b2_x1 + offset) * (b2_y2 - b2_y1 + offset);

      T left = fmaxf(b1_x1, b2_x1), right = fminf(b1_x2, b2_x2);
      T top = fmaxf(b1_y1, b2_y1), bottom = fminf(b1_y2, b2_y2);
      T width = fmaxf(right - left + offset, 0.f);
      T height = fmaxf(bottom - top + offset, 0.f);

      T interS = width * height;
      T baseS = 1.0;
      if (mode == 0) {
        baseS = fmaxf(b1_area + b2_area - interS, T(offset));
      } else if (mode == 1) {
        baseS = fmaxf(b1_area, T(offset));
      }
      ious[index] = interS / baseS;
    }
  }
}

void BboxOverlapsExecutor::cpuCompute() {
  // zero element
  if ((int)parser_->getInputDataCount(0) == 0 ||
      (int)parser_->getInputDataCount(1) == 0) {
    return;
  }
  if ((int)parser_->getInputDataCount(0) == 0 ||
      (int)parser_->getOutputDataCount(0) == 0) {
    return;
  }

  auto bbox1_desc = tensor_desc_[0].tensor;
  auto bbox2_desc = tensor_desc_[1].tensor;

  int rows = bbox1_desc->dims[0];
  int cols = bbox2_desc->dims[0];

  // get struct param
  int mode = parser_->getProtoNode()->bbox_overlaps_param().mode();
  bool aligned = parser_->getProtoNode()->bbox_overlaps_param().aligned();
  int offset = parser_->getProtoNode()->bbox_overlaps_param().offset();

  auto count1 = parser_->getInputDataCount(0);
  auto count2 = parser_->getInputDataCount(1);
  auto count3 = parser_->getOutputDataCount(0);

  cpuBboxOverlaps(cpu_fp32_input_[0], cpu_fp32_input_[1], cpu_fp32_output_[0],
                  rows, cols, mode, aligned, offset);
}

int64_t BboxOverlapsExecutor::getTheoryOps() {
  int64_t theory_ops = 0;
  int32_t mode = parser_->getProtoNode()->bbox_overlaps_param().mode();
  int num_element = (int)parser_->getOutputDataCount(0);

  /*
  2 * 5 for compute b1_area and b2_area:
  b1_area = (b1_x2 - b1_x1 + offset) * (b1_y2 - b1_y1 + offset)
  ...

  4 for compare left, right, top, bottom:
  left = fmaxf(b1_x1, b2_x1), right = fminf(b1_x2, b2_x2)
  ....

  2 *3 for compute width and height:
  width = fmaxf(right - left + offset, 0.f)
  ...

  1 for compute interS:
  interS = width * height
  */

  theory_ops = 2 * 5 + 4 + 2 * 3 + 1;

  if (mode) {
    // 1 for compute baseS
    // baseS = fmaxf(b1_area, T(offset))
    theory_ops = theory_ops + 1;
  } else {
    // 3 for compute baseS
    // baseS = fmaxf(b1_area + b2_area - interS, T(offset))
    theory_ops = theory_ops + 3;
  }

  // 1 for compute ious[i][j]
  // ious[i][j] = interS / baseS
  theory_ops += 1;

  // the number of ops is i * j * theory_ops
  theory_ops = num_element * theory_ops;

  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
