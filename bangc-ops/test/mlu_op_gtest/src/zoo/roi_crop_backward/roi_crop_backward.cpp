/*************************************************************************
 * Copyright (C) 2022 by Cambricon, Inc. All rights reserved.
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
  VLOG(4) << "[RoiCropBackwardExecutor] call initData() Begin.";
  gradOutput_data_ptr_ = data_vector_[0].device_ptr;
  grid_data_ptr_ = data_vector_[1].device_ptr;
  gradInput_data_ptr_ = data_vector_[2].device_ptr;
  gradOutput_desc_ = tensor_desc_[0].tensor;
  grid_desc_ = tensor_desc_[1].tensor;
  gradInput_desc_ = tensor_desc_[2].tensor;
  gradOutput_h_ = gradOutput_desc_->dims[1];
  gradOutput_w_ = gradOutput_desc_->dims[2];
  grid_batch_roi_ = grid_desc_->dims[0];
  gradInput_batch_ = gradInput_desc_->dims[0];
  gradInput_h_ = gradInput_desc_->dims[1];
  gradInput_w_ = gradInput_desc_->dims[2];
  gradInput_c_ = gradInput_desc_->dims[3];
  VLOG(4) << "[RoiCropBackwardExecutor] call initData() End.";
}

void RoiCropBackwardExecutor::printDataInfo() {
  VLOG(4) << "[RoiCropBackwardExecutor] call printDataInfo() Begin.";
  VLOG(4) << "grid_batch_roi_  " << grid_batch_roi_;
  VLOG(4) << "gradOutput_h_        " << gradOutput_h_;
  VLOG(4) << "gradOutput_w_        " << gradOutput_w_;
  VLOG(4) << "gradInput_batch_     " << gradInput_batch_;
  VLOG(4) << "gradInput_h_         " << gradInput_h_;
  VLOG(4) << "gradInput_w_         " << gradInput_w_;
  VLOG(4) << "gradInput_c_         " << gradInput_c_;
  VLOG(4) << "[RoiCropBackwardExecutor] call printDataInfo() End.";
}

int RoiCropBackwardExecutor::getTopLeft(const float grid_yx_value,
                                        const int input_h_w, float* weight) {
  float xcoord = (grid_yx_value + 1) * (input_h_w - 1) / 2;
  int point = floor(xcoord);
  *weight = 1 - (xcoord - point);
  return point;
}

void RoiCropBackwardExecutor::compute() {
  VLOG(4) << "[RoiCropBackwardExecutor] call compute() Begin.";
  initData();
  printDataInfo();
  interface_timer_.start();
  MLUOP_CHECK(mluOpRoiCropBackward(
      handle_, gradOutput_desc_, gradOutput_data_ptr_, grid_desc_,
      grid_data_ptr_, gradInput_desc_, gradInput_data_ptr_));
  interface_timer_.stop();
  VLOG(4) << "[RoiCropBackwardExecutor] call compute() End.";
}

void RoiCropBackwardExecutor::cpuCompute() {
  VLOG(4) << "[RoiCropBackwardExecutor] call cpuCompute() Begin.";
  float* gradOutput_cpu_ptr = cpu_fp32_input_[0];
  float* grid_cpu_ptr = cpu_fp32_input_[1];
  float* gradInput_c_pu_ptr = cpu_fp32_output_[0];
  int gradOutput_nums =
      grid_batch_roi_ * gradOutput_h_ * gradOutput_w_ * gradInput_c_;
  int roi_per_img = grid_batch_roi_ / gradInput_batch_;
  int gradOutput_stride_batch = gradOutput_h_ * gradOutput_w_ * gradInput_c_;
  int gradOutput_stride_h = gradOutput_w_ * gradInput_c_;
  int gradOutput_stride_w = gradInput_c_;
  int grid_stride_batch = gradOutput_h_ * gradOutput_w_ * 2;
  int grid_stride_h = gradOutput_w_ * 2;
  int gride_stride_w = 2;
  int gradInput_stride_batch = gradInput_h_ * gradInput_w_ * gradInput_c_;
  int gradInput_stride_h = gradInput_w_ * gradInput_c_;
  int gradInput_stride_w = gradInput_c_;
  int i_tl_x = 0;
  int i_tl_y = 0;
  float i_tl_x_weight = 0.0;
  float i_tl_y_weight = 0.0;
  float i_tl = 0;
  float i_tr = 0;
  float i_bl = 0;
  float i_br = 0;
  for (int index = 0; index < gradOutput_nums; ++index) {
    // coordinates of each position in gradOutput data
    int goc = index % gradInput_c_;
    int gow = (index / gradOutput_stride_w) % gradOutput_w_;
    int goh = (index / gradOutput_stride_h) % gradOutput_h_;
    int gon = index / gradOutput_stride_batch;
    // data offset in gradoutput
    const int output_offset = gon * gradOutput_stride_batch +
                              goh * gradOutput_stride_h +
                              gow * gradOutput_stride_w + goc;
    float gradOutput_value = gradOutput_cpu_ptr[output_offset];

    // batch dimension index in gradOutput
    int gradInput_n = gon / roi_per_img;
    // data value in grid
    float yf = grid_cpu_ptr[gon * grid_stride_batch + goh * grid_stride_h +
                            gow * gride_stride_w];
    float xf = grid_cpu_ptr[gon * grid_stride_batch + goh * grid_stride_h +
                            gow * gride_stride_w + 1];
    // gradInput data information
    i_tl_x = getTopLeft(xf, gradInput_w_, &i_tl_x_weight);
    i_tl_y = getTopLeft(yf, gradInput_h_, &i_tl_y_weight);

    // field information
    const int i_tl_offset = gradInput_n * gradInput_stride_batch +
                            i_tl_y * gradInput_stride_h +
                            i_tl_x * gradInput_stride_w + goc;
    float i_tl_xy_weight = i_tl_x_weight * i_tl_y_weight;
    bool topLeftIsIn = i_tl_x >= 0 && i_tl_x <= (gradInput_w_ - 1) &&
                       i_tl_y >= 0 && i_tl_y <= (gradInput_h_ - 1);
    if (topLeftIsIn) {
      gradInput_c_pu_ptr[i_tl_offset] += i_tl_xy_weight * gradOutput_value;
    }

    const int i_tr_offset = i_tl_offset + gradInput_stride_w;
    float i_tr_xy_weight = (1 - i_tl_x_weight) * i_tl_y_weight;
    bool topRightIsIn = i_tl_x >= 0 && i_tl_x <= (gradInput_w_ - 1) &&
                        (i_tl_y + 1) >= 0 && (i_tl_y + 1) <= (gradInput_h_ - 1);
    if (topRightIsIn) {
      gradInput_c_pu_ptr[i_tr_offset] += i_tr_xy_weight * gradOutput_value;
    }

    const int i_bl_offset = i_tl_offset + gradInput_stride_h;
    float i_bl_xy_weight = i_tl_x_weight * (1 - i_tl_y_weight);
    bool bottomLeftIsIn = (i_tl_x + 1) >= 0 &&
                          (i_tl_x + 1) <= (gradInput_w_ - 1) && i_tl_y >= 0 &&
                          i_tl_y <= (gradInput_h_ - 1);
    if (bottomLeftIsIn) {
      gradInput_c_pu_ptr[i_bl_offset] += i_bl_xy_weight * gradOutput_value;
    }

    const int i_br_offset =
        i_tl_offset + gradInput_stride_h + gradInput_stride_w;
    float i_br_xy_weight = (1 - i_tl_x_weight) * (1 - i_tl_y_weight);
    bool bottomRightIsIn =
        (i_tl_x + 1) >= 0 && (i_tl_x + 1) <= (gradInput_w_ - 1) &&
        (i_tl_y + 1) >= 0 && (i_tl_y + 1) <= (gradInput_h_ - 1);
    if (bottomRightIsIn) {
      gradInput_c_pu_ptr[i_br_offset] += i_br_xy_weight * gradOutput_value;
    }
  }
  VLOG(4) << "[RoiCropBackwardExecutor] call cpuCompute() End.";
}

int64_t RoiCropBackwardExecutor::getTheoryOps() {
  const int cp_count = 8;
  theory_ops_ = parser_->getInputDataCount(0) * cp_count;
  VLOG(4) << "[RoiCropBackwardExecutor] getTheoryOps: " << theory_ops_
          << " ops.";
  return theory_ops_;
}

}  // namespace mluoptest
