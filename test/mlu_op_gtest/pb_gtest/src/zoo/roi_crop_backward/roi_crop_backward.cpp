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
#include "roi_crop_backward.h"

#include "mlu_op.h"

namespace mluoptest {
void RoiCropBackwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 2,
              "[RoiCropBackwardExecutor] input number is wrong. ");
  GTEST_CHECK(parser_->outputs().size() == 1,
              "[RoiCropBackwardExecutor] output number is wrong. ");
}

void RoiCropBackwardExecutor::initData() {
  VLOG(4) << "[RoiCropBackwardExecutor] call initData() begin.";
  grad_output_data_ptr_ = data_vector_[0].device_ptr;
  grid_data_ptr_ = data_vector_[1].device_ptr;
  grad_input_data_ptr_ = data_vector_[2].device_ptr;
  grad_output_desc_ = tensor_desc_[0].tensor;
  grid_desc_ = tensor_desc_[1].tensor;
  grad_input_desc_ = tensor_desc_[2].tensor;
  grad_output_h_ = grad_output_desc_->getDimIndex(1);
  grad_output_w_ = grad_output_desc_->getDimIndex(2);
  grid_batch_roi_ = grid_desc_->getDimIndex(0);
  grad_input_batch_ = grad_input_desc_->getDimIndex(0);
  grad_input_h_ = grad_input_desc_->getDimIndex(1);
  grad_input_w_ = grad_input_desc_->getDimIndex(2);
  grad_input_c_ = grad_input_desc_->getDimIndex(3);
  VLOG(4) << "[RoiCropBackwardExecutor] call initData() end.";
}

void RoiCropBackwardExecutor::printDataInfo() {
  VLOG(4) << "[RoiCropBackwardExecutor] call printDataInfo() begin.";
  VLOG(4) << "grid_batch_roi_  " << grid_batch_roi_;
  VLOG(4) << "grad_output_h_        " << grad_output_h_;
  VLOG(4) << "grad_output_w_        " << grad_output_w_;
  VLOG(4) << "grad_input_batch_     " << grad_input_batch_;
  VLOG(4) << "grad_input_h_         " << grad_input_h_;
  VLOG(4) << "grad_input_w_         " << grad_input_w_;
  VLOG(4) << "grad_input_c_         " << grad_input_c_;
  VLOG(4) << "[RoiCropBackwardExecutor] call printDataInfo() end.";
}

int RoiCropBackwardExecutor::getTopLeft(const float grid_yx_value,
                                        const int input_h_w, float* weight) {
  float xcoord = (grid_yx_value + 1) * (input_h_w - 1) / 2;
  int point = floor(xcoord);
  *weight = 1 - (xcoord - point);
  return point;
}

void RoiCropBackwardExecutor::compute() {
  VLOG(4) << "[RoiCropBackwardExecutor] call compute() begin.";
  initData();
  printDataInfo();
  interface_timer_.start();
  MLUOP_CHECK(mluOpRoiCropBackward(
      handle_, grad_output_desc_, grad_output_data_ptr_, grid_desc_,
      grid_data_ptr_, grad_input_desc_, grad_input_data_ptr_));
  interface_timer_.stop();
  VLOG(4) << "[RoiCropBackwardExecutor] call compute() end.";
}

void RoiCropBackwardExecutor::cpuCompute() {
  VLOG(4) << "[RoiCropBackwardExecutor] call cpuCompute() begin.";
  float* grad_output_cpu_ptr = cpu_fp32_input_[0];
  float* grid_cpu_ptr = cpu_fp32_input_[1];
  float* grad_input_cpu_ptr = cpu_fp32_output_[0];
  int grad_output_nums =
      grid_batch_roi_ * grad_output_h_ * grad_output_w_ * grad_input_c_;
  int roi_per_img = grid_batch_roi_ / grad_input_batch_;
  int grad_output_stride_batch =
      grad_output_h_ * grad_output_w_ * grad_input_c_;
  int grad_output_stride_h = grad_output_w_ * grad_input_c_;
  int grad_output_stride_w = grad_input_c_;
  int grid_stride_batch = grad_output_h_ * grad_output_w_ * 2;
  int grid_stride_h = grad_output_w_ * 2;
  int gride_stride_w = 2;
  int grad_input_stride_batch = grad_input_h_ * grad_input_w_ * grad_input_c_;
  int grad_input_stride_h = grad_input_w_ * grad_input_c_;
  int grad_input_stride_w = grad_input_c_;
  int i_tl_x = 0;
  int i_tl_y = 0;
  float i_tl_x_weight = 0.0;
  float i_tl_y_weight = 0.0;
  float i_tl = 0;
  float i_tr = 0;
  float i_bl = 0;
  float i_br = 0;
  for (int index = 0; index < grad_output_nums; ++index) {
    // coordinates of each position in grad_output data
    int goc = index % grad_input_c_;
    int gow = (index / grad_output_stride_w) % grad_output_w_;
    int goh = (index / grad_output_stride_h) % grad_output_h_;
    int gon = index / grad_output_stride_batch;
    // data offset in grad_output
    const int output_offset = gon * grad_output_stride_batch +
                              goh * grad_output_stride_h +
                              gow * grad_output_stride_w + goc;
    float grad_output_value = grad_output_cpu_ptr[output_offset];

    // batch dimension index in grad_output
    int grad_input_n = gon / roi_per_img;
    // data value in grid
    float yf = grid_cpu_ptr[gon * grid_stride_batch + goh * grid_stride_h +
                            gow * gride_stride_w];
    float xf = grid_cpu_ptr[gon * grid_stride_batch + goh * grid_stride_h +
                            gow * gride_stride_w + 1];
    // grad_input data information
    i_tl_x = getTopLeft(xf, grad_input_w_, &i_tl_x_weight);
    i_tl_y = getTopLeft(yf, grad_input_h_, &i_tl_y_weight);

    const int i_tl_offset = grad_input_n * grad_input_stride_batch +
                            i_tl_y * grad_input_stride_h +
                            i_tl_x * grad_input_stride_w + goc;
    float i_tl_xy_weight = i_tl_x_weight * i_tl_y_weight;
    bool topLeftIsIn = i_tl_x >= 0 && i_tl_x <= (grad_input_w_ - 1) &&
                       i_tl_y >= 0 && i_tl_y <= (grad_input_h_ - 1);
    if (topLeftIsIn) {
      grad_input_cpu_ptr[i_tl_offset] += i_tl_xy_weight * grad_output_value;
    }

    const int i_tr_offset = i_tl_offset + grad_input_stride_w;
    float i_tr_xy_weight = (1 - i_tl_x_weight) * i_tl_y_weight;
    bool topRightIsIn = i_tl_x >= 0 && i_tl_x <= (grad_input_w_ - 1) &&
                        (i_tl_y + 1) >= 0 &&
                        (i_tl_y + 1) <= (grad_input_h_ - 1);
    if (topRightIsIn) {
      grad_input_cpu_ptr[i_tr_offset] += i_tr_xy_weight * grad_output_value;
    }

    const int i_bl_offset = i_tl_offset + grad_input_stride_h;
    float i_bl_xy_weight = i_tl_x_weight * (1 - i_tl_y_weight);
    bool bottomLeftIsIn = (i_tl_x + 1) >= 0 &&
                          (i_tl_x + 1) <= (grad_input_w_ - 1) && i_tl_y >= 0 &&
                          i_tl_y <= (grad_input_h_ - 1);
    if (bottomLeftIsIn) {
      grad_input_cpu_ptr[i_bl_offset] += i_bl_xy_weight * grad_output_value;
    }

    const int i_br_offset =
        i_tl_offset + grad_input_stride_h + grad_input_stride_w;
    float i_br_xy_weight = (1 - i_tl_x_weight) * (1 - i_tl_y_weight);
    bool bottomRightIsIn =
        (i_tl_x + 1) >= 0 && (i_tl_x + 1) <= (grad_input_w_ - 1) &&
        (i_tl_y + 1) >= 0 && (i_tl_y + 1) <= (grad_input_h_ - 1);
    if (bottomRightIsIn) {
      grad_input_cpu_ptr[i_br_offset] += i_br_xy_weight * grad_output_value;
    }
  }
  VLOG(4) << "[RoiCropBackwardExecutor] call cpuCompute() end.";
}

int64_t RoiCropBackwardExecutor::getTheoryOps() {
  const int cp_count = 8;
  theory_ops_ = parser_->getInputDataCount(0) * cp_count;
  VLOG(4) << "[RoiCropBackwardExecutor] getTheoryOps: " << theory_ops_
          << " ops.";
  return theory_ops_;
}

}  // namespace mluoptest
