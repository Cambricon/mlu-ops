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
#include "roi_crop_forward.h"

#include "mlu_op.h"

namespace mluoptest {
void RoiCropForwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 2,
              "[RoiCropForwardExecutor] input number is wrong. ");
  GTEST_CHECK(parser_->outputs().size() == 1,
              "[RoiCropForwardExecutor] output number is wrong. ");
}

void RoiCropForwardExecutor::initData() {
  VLOG(4) << "[RoiCropForwardExecutor] call initData() Begin.";
  input_data_ptr_ = data_vector_[0].device_ptr;
  grid_data_ptr_ = data_vector_[1].device_ptr;
  output_data_ptr_ = data_vector_[2].device_ptr;
  input_desc_ = tensor_desc_[0].tensor;
  grid_desc_ = tensor_desc_[1].tensor;
  output_desc_ = tensor_desc_[2].tensor;
  input_batch_ = input_desc_->dims[0];
  input_h_ = input_desc_->dims[1];
  input_w_ = input_desc_->dims[2];
  input_c_ = input_desc_->dims[3];
  grid_batch_roi_ = grid_desc_->dims[0];
  output_h_ = output_desc_->dims[1];
  output_w_ = output_desc_->dims[2];
  VLOG(4) << "[RoiCropForwardExecutor] call initData() End.";
}

void RoiCropForwardExecutor::printDataInfo() {
  VLOG(4) << "[RoiCropForwardExecutor] call printDataInfo() Begin.";
  VLOG(4) << "input_batch_     " << input_batch_;
  VLOG(4) << "input_h_         " << input_h_;
  VLOG(4) << "input_w_         " << input_w_;
  VLOG(4) << "input_c_         " << input_c_;
  VLOG(4) << "grid_batch_roi_  " << grid_batch_roi_;
  VLOG(4) << "output_h_        " << output_h_;
  VLOG(4) << "output_w_        " << output_w_;
  VLOG(4) << "[RoiCropForwardExecutor] call printDataInfo() End.";
}

int RoiCropForwardExecutor::getTopLeft(const float grid_yx_value,
                                       const int input_hw, float* weight) {
  float xcoord = (grid_yx_value + 1) * (input_hw - 1) / 2;
  int point = floor(xcoord);
  *weight = 1 - (xcoord - point);
  return point;
}

void RoiCropForwardExecutor::compute() {
  VLOG(4) << "[RoiCropForwardExecutor] call compute() Begin.";
  initData();
  printDataInfo();
  interface_timer_.start();
  MLUOP_CHECK(mluOpRoiCropForward(handle_, input_desc_, input_data_ptr_,
                                  grid_desc_, grid_data_ptr_, output_desc_,
                                  output_data_ptr_));
  interface_timer_.stop();
  VLOG(4) << "[RoiCropForwardExecutor] call compute() End.";
}

void RoiCropForwardExecutor::cpuCompute() {
  VLOG(4) << "[RoiCropForwardExecutor] call cpuCompute() Begin.";
  float* input_c_pu_ptr = cpu_fp32_input_[0];
  float* grid_cpu_ptr = cpu_fp32_input_[1];
  float* output_cpu_ptr = cpu_fp32_output_[0];
  int output_nums = grid_batch_roi_ * output_h_ * output_w_ * input_c_;
  int roi_per_img = grid_batch_roi_ / input_batch_;
  int output_stride_batch = output_h_ * output_w_ * input_c_;
  int output_stride_h = output_w_ * input_c_;
  int output_stride_w = input_c_;
  int grid_stride_batch = output_h_ * output_w_ * 2;
  int grid_stride_h = output_w_ * 2;
  int gride_stride_w = 2;
  int input_stride_batch = input_h_ * input_w_ * input_c_;
  int input_stride_h = input_w_ * input_c_;
  int input_stride_w = input_c_;
  int i_tl_x = 0;
  int i_tl_y = 0;
  float i_tl_x_weight = 0.0;
  float i_tl_y_weight = 0.0;
  float i_tl = 0;
  float i_tr = 0;
  float i_bl = 0;
  float i_br = 0;

  for (int index = 0; index < output_nums; ++index) {
    // coordinates of each position in output data
    int oc = index % input_c_;
    int ow = (index / output_stride_w) % output_w_;
    int oh = (index / output_stride_h) % output_h_;
    int on = index / output_stride_batch;
    // data oddset in output
    const int output_offset = on * output_stride_batch + oh * output_stride_h +
                              ow * output_stride_w + oc;
    // batch dimension index in output
    int input_n = on / roi_per_img;
    // data value in grid
    float yf = grid_cpu_ptr[on * grid_stride_batch + oh * grid_stride_h +
                            ow * gride_stride_w];
    float xf = grid_cpu_ptr[on * grid_stride_batch + oh * grid_stride_h +
                            ow * gride_stride_w + 1];
    // input data information
    i_tl_x = getTopLeft(xf, input_w_, &i_tl_x_weight);
    i_tl_y = getTopLeft(yf, input_h_, &i_tl_y_weight);

    // field information
    const int i_tl_offset = input_n * input_stride_batch +
                            i_tl_y * input_stride_h + i_tl_x * input_stride_w +
                            oc;
    float i_tl_xy_weight = i_tl_x_weight * i_tl_y_weight;
    bool topLeftIsIn = i_tl_x >= 0 && i_tl_x <= (input_w_ - 1) && i_tl_y >= 0 &&
                       i_tl_y <= (input_h_ - 1);
    if (topLeftIsIn) {
      i_tl = input_c_pu_ptr[i_tl_offset];
    }
    const int i_tr_offset = i_tl_offset + input_stride_w;
    float i_tr_xy_weight = (1 - i_tl_x_weight) * i_tl_y_weight;
    bool topRightIsIn = i_tl_x >= 0 && i_tl_x <= (input_w_ - 1) &&
                        (i_tl_y + 1) >= 0 && (i_tl_y + 1) <= (input_h_ - 1);
    if (topRightIsIn) {
      i_tr = input_c_pu_ptr[i_tr_offset];
    }
    const int i_bl_offset = i_tl_offset + input_stride_h;
    float i_bl_xy_weight = i_tl_x_weight * (1 - i_tl_y_weight);
    bool bottomLeftIsIn = (i_tl_x + 1) >= 0 && (i_tl_x + 1) <= (input_w_ - 1) &&
                          i_tl_y >= 0 && i_tl_y <= (input_h_ - 1);
    if (bottomLeftIsIn) {
      i_bl = input_c_pu_ptr[i_bl_offset];
    }
    const int i_br_offset = i_tl_offset + input_stride_h + input_stride_w;
    float i_br_xy_weight = (1 - i_tl_x_weight) * (1 - i_tl_y_weight);
    bool bottomRightIsIn = (i_tl_x + 1) >= 0 &&
                           (i_tl_x + 1) <= (input_w_ - 1) &&
                           (i_tl_y + 1) >= 0 && (i_tl_y + 1) <= (input_h_ - 1);
    if (bottomRightIsIn) {
      i_br = input_c_pu_ptr[i_br_offset];
    }

    output_cpu_ptr[output_offset] =
        i_tl_xy_weight * i_tl + i_tr_xy_weight * i_tr + i_bl_xy_weight * i_bl +
        i_br_xy_weight * i_br;
  }
  VLOG(4) << "[RoiCropForwardExecutor] call cpuCompute() End.";
}

int64_t RoiCropForwardExecutor::getTheoryOps() {
  const int cp_count = 7;
  theory_ops_ = parser_->getInputDataCount(0) * cp_count;
  VLOG(4) << "[RoiCropForwardExecutor] getTheoryOps: " << theory_ops_
          << " ops.";
  return theory_ops_;
}

}  // namespace mluoptest
