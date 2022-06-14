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

#include <iostream>
#include "mlu_op.h"

#define getTensorDesc(data_index) tensor_desc_[data_index].tensor
#define getTensorDim(data_index, dim_index) \
  tensor_desc_[data_index].tensor->dims[dim_index]
#define getTensorData(data_index) data_vector_[data_index].device_ptr

namespace mluoptest {
void RoiCropBackwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 2,
              "[RoiCropBackwardExecutor] input number is wrong. ");
  GTEST_CHECK(parser_->outputs().size() == 1,
              "[RoiCropBackwardExecutor] output number is wrong. ");
}

void RoiCropBackwardExecutor::initData() {
  VLOG(4) << "[RoiCropBackwardExecutor] call initData() Begin.";
  gradOutput_data_ptr = getTensorData(0);
  grid_data_ptr = getTensorData(1);
  gradInput_data_ptr = getTensorData(2);
  gradOutput_desc = getTensorDesc(0);
  grid_desc = getTensorDesc(1);
  gradInput_desc = getTensorDesc(2);
  grid_batch_roi = getTensorDim(1, 0);
  gradOutput_h = getTensorDim(0, 1);
  gradOutput_w = getTensorDim(0, 2);
  gradInput_batch = getTensorDim(2, 0);
  gradInput_h = getTensorDim(2, 1);
  gradInput_w = getTensorDim(2, 2);
  gradInput_c = getTensorDim(2, 3);
  VLOG(4) << "[RoiCropBackwardExecutor] call initData() End.";
}

void RoiCropBackwardExecutor::printDataInfo() {
  VLOG(4) << "[RoiCropBackwardExecutor] call printDataInfo() Begin.";
  VLOG(4) << "grid_batch_roi  " << grid_batch_roi;
  VLOG(4) << "gradOutput_h        " << gradOutput_h;
  VLOG(4) << "gradOutput_w        " << gradOutput_w;
  VLOG(4) << "gradInput_batch     " << gradInput_batch;
  VLOG(4) << "gradInput_h         " << gradInput_h;
  VLOG(4) << "gradInput_w         " << gradInput_w;
  VLOG(4) << "gradInput_c         " << gradInput_c;
  VLOG(4) << "[RoiCropBackwardExecutor] call printDataInfo() End.";
}

int RoiCropBackwardExecutor::getInputTopLeft(float grid_yx_value,
                                             int gradInput_hw, float& weight) {
  float xcoord = (grid_yx_value + 1) * (gradInput_hw - 1) / 2;
  int point = floor(xcoord);
  weight = 1 - (xcoord - point);
  return point;
}

void RoiCropBackwardExecutor::compute() {
  VLOG(4) << "[RoiCropBackwardExecutor] call compute() Begin.";
  paramCheck();
  initData();
  printDataInfo();
  interface_timer_.start();
  MLUOP_CHECK(mluOpRoiCropBackward(
      handle_, gradOutput_desc, gradOutput_data_ptr, grid_desc, grid_data_ptr,
      gradInput_desc, gradInput_data_ptr));
  interface_timer_.stop();
  VLOG(4) << "[RoiCropBackwardExecutor] call compute() End.";
}

void RoiCropBackwardExecutor::cpuCompute() {
  VLOG(4) << "[RoiCropBackwardExecutor] call cpuCompute() Begin.";

  float* gradOutput_cpu_ptr = cpu_fp32_input_[0];
  float* grid_cpu_ptr = cpu_fp32_input_[1];
  float* gradInput_cpu_ptr = cpu_fp32_output_[0];
  int gradOutput_nums =
      grid_batch_roi * gradOutput_h * gradOutput_w * gradInput_c;
  int roi_per_img = grid_batch_roi / gradInput_batch;
  int gradOutput_stride_batch = gradOutput_h * gradOutput_w * gradInput_c;
  int gradOutput_stride_h = gradOutput_w * gradInput_c;
  int gradOutput_stride_w = gradInput_c;
  int grid_stride_batch = gradOutput_h * gradOutput_w * 2;
  int grid_stride_h = gradOutput_w * 2;
  int gride_stride_w = 2;
  int gradInput_stride_batch = gradInput_h * gradInput_w * gradInput_c;
  int gradInput_stride_h = gradInput_w * gradInput_c;
  int gradInput_stride_w = gradInput_c;
  int i_top_left_x = 0;
  int i_top_left_y = 0;
  float i_top_left_x_weight = 0.0;
  float i_top_left_y_weight = 0.0;
  float i_top_left = 0;
  float i_top_right = 0;
  float i_bottom_left = 0;
  float i_bottom_right = 0;
  for (int index = 0; index < gradOutput_nums; ++index) {
    // coordinates of each position in gradOutput data
    int goc = index % gradInput_c;
    int gow = (index / gradOutput_stride_w) % gradOutput_w;
    int goh = (index / gradOutput_stride_h) % gradOutput_h;
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
    i_top_left_x = getInputTopLeft(xf, gradInput_w, i_top_left_x_weight);
    i_top_left_y = getInputTopLeft(yf, gradInput_h, i_top_left_y_weight);

    // field information
    const int i_top_left_offset = gradInput_n * gradInput_stride_batch +
                                  i_top_left_y * gradInput_stride_h +
                                  i_top_left_x * gradInput_stride_w + goc;
    float i_top_left_xy_weight = i_top_left_x_weight * i_top_left_y_weight;
    bool topLeftIsIn = i_top_left_x >= 0 && i_top_left_x <= (gradInput_w - 1) &&
                       i_top_left_y >= 0 && i_top_left_y <= (gradInput_h - 1);
    if (topLeftIsIn) {
      gradInput_cpu_ptr[i_top_left_offset] +=
          i_top_left_xy_weight * gradOutput_value;
      // std::cout<<"gradInput_cpu_ptr[i_top_left_offset]"<<gradInput_cpu_ptr[i_top_left_offset]<<std::endl;
    }

    const int i_top_right_offset = i_top_left_offset + gradInput_stride_w;
    float i_top_right_xy_weight =
        (1 - i_top_left_x_weight) * i_top_left_y_weight;
    bool topRightIsIn =
        i_top_left_x >= 0 && i_top_left_x <= (gradInput_w - 1) &&
        (i_top_left_y + 1) >= 0 && (i_top_left_y + 1) <= (gradInput_h - 1);
    if (topRightIsIn) {
      gradInput_cpu_ptr[i_top_right_offset] +=
          i_top_right_xy_weight * gradOutput_value;
      // std::cout<<"gradInput_cpu_ptr[i_top_right_offset]"<<gradInput_cpu_ptr[i_top_right_offset]<<std::endl;
    }

    const int i_bottom_left_offset = i_top_left_offset + gradInput_stride_h;
    float i_bottom_left_xy_weight =
        i_top_left_x_weight * (1 - i_top_left_y_weight);
    bool bottomLeftIsIn =
        (i_top_left_x + 1) >= 0 && (i_top_left_x + 1) <= (gradInput_w - 1) &&
        i_top_left_y >= 0 && i_top_left_y <= (gradInput_h - 1);
    if (bottomLeftIsIn) {
      gradInput_cpu_ptr[i_bottom_left_offset] +=
          i_bottom_left_xy_weight * gradOutput_value;
      // std::cout<<"gradInput_cpu_ptr[i_bottom_left_offset]"<<gradInput_cpu_ptr[i_bottom_left_offset]<<std::endl;
    }

    const int i_bottom_right_offset =
        i_top_left_offset + gradInput_stride_h + gradInput_stride_w;
    float i_bottom_right_xy_weight =
        (1 - i_top_left_x_weight) * (1 - i_top_left_y_weight);
    bool bottomRightIsIn =
        (i_top_left_x + 1) >= 0 && (i_top_left_x + 1) <= (gradInput_w - 1) &&
        (i_top_left_y + 1) >= 0 && (i_top_left_y + 1) <= (gradInput_h - 1);
    if (bottomRightIsIn) {
      gradInput_cpu_ptr[i_bottom_right_offset] +=
          i_bottom_right_xy_weight * gradOutput_value;
      // std::cout<<"gradInput_cpu_ptr[i_bottom_right_offset]"<<gradInput_cpu_ptr[i_bottom_right_offset]<<std::endl;
    }
  }
  VLOG(4) << "[RoiCropBackwardExecutor] call cpuCompute() End.";
}

int64_t RoiCropBackwardExecutor::getTheoryOps() {
  int cp_count = 8;
  theory_ops = parser_->getInputDataCount(0) * cp_count;
  VLOG(4) << "[RoiCropBackwardExecutor] getTheoryOps: " << theory_ops
          << " ops.";
  return theory_ops;
}

}  // namespace mluoptest
