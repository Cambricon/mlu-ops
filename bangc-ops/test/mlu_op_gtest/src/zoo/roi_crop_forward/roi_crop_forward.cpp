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
#include "roi_crop_forward.h"

#include <iostream>
#include "mlu_op.h"

#define getTensorDesc(data_index) tensor_desc_[data_index].tensor
#define getTensorDim(data_index, dim_index) \
  tensor_desc_[data_index].tensor->dims[dim_index]
#define getTensorData(data_index) data_vector_[data_index].device_ptr

namespace mluoptest {
void RoiCropForwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->inputs().size() == 2,
              "[RoiCropForwardExecutor] input number is wrong. ");
  GTEST_CHECK(parser_->outputs().size() == 1,
              "[RoiCropForwardExecutor] output number is wrong. ");
}

void RoiCropForwardExecutor::initData() {
  VLOG(4) << "[RoiCropForwardExecutor] call initData() Begin.";
  input_data_ptr = getTensorData(0);
  grid_data_ptr = getTensorData(1);
  output_data_ptr = getTensorData(2);
  input_desc = getTensorDesc(0);
  grid_desc = getTensorDesc(1);
  output_desc = getTensorDesc(2);
  input_batch = getTensorDim(0, 0);
  input_h = getTensorDim(0, 1);
  input_w = getTensorDim(0, 2);
  input_c = getTensorDim(0, 3);
  grid_batch_roi = getTensorDim(1, 0);
  output_h = getTensorDim(2, 1);
  output_w = getTensorDim(2, 2);
  VLOG(4) << "[RoiCropForwardExecutor] call initData() End.";
}

void RoiCropForwardExecutor::printDataInfo() {
  VLOG(4) << "[RoiCropForwardExecutor] call printDataInfo() Begin.";
  VLOG(4) << "input_batch     " << input_batch;
  VLOG(4) << "input_h         " << input_h;
  VLOG(4) << "input_w         " << input_w;
  VLOG(4) << "input_c         " << input_c;
  VLOG(4) << "grid_batch_roi  " << grid_batch_roi;
  VLOG(4) << "output_h        " << output_h;
  VLOG(4) << "output_w        " << output_w;
  VLOG(4) << "[RoiCropForwardExecutor] call printDataInfo() End.";
}

int RoiCropForwardExecutor::getInputTopLeft(float grid_yx_value, int input_hw,
                                            float& weight) {
  VLOG(4) << "[RoiCropForwardExecutor] call getInputLeft() Begin.";
  float i_top_left_coord = (grid_yx_value + 1) * (input_hw - 1) / 2;
  int i_top_left = floor(i_top_left_coord);
  weight = 1 - (i_top_left_coord - i_top_left);
  VLOG(4) << "[RoiCropForwardExecutor] call getInputLeft() End.";
  return i_top_left;
}

void RoiCropForwardExecutor::compute() {
  VLOG(4) << "[RoiCropForwardExecutor] call compute() Begin.";
  paramCheck();
  initData();
  printDataInfo();
  interface_timer_.start();
  MLUOP_CHECK(mluOpRoiCropForward(handle_, input_desc, input_data_ptr,
                                  grid_desc, grid_data_ptr, output_desc,
                                  output_data_ptr));
  interface_timer_.stop();
  VLOG(4) << "[RoiCropForwardExecutor] call compute() End.";
}

void RoiCropForwardExecutor::cpuCompute() {
  VLOG(4) << "[RoiCropForwardExecutor] call cpuCompute() Begin.";
  float* input_cpu_ptr = cpu_fp32_input_[0];
  float* grid_cpu_ptr = cpu_fp32_input_[1];
  float* output_cpu_ptr = cpu_fp32_output_[0];
  int output_nums = grid_batch_roi * output_h * output_w * input_c;
  int roi_per_img = grid_batch_roi / input_batch;
  int output_stride_batch = output_h * output_w * input_c;
  int output_stride_h = output_w * input_c;
  int output_stride_w = input_c;
  int grid_stride_batch = output_h * output_w * 2;
  int grid_stride_h = output_w * 2;
  int gride_stride_w = 2;
  int input_stride_batch = input_h * input_w * input_c;
  int input_stride_h = input_w * input_c;
  int input_stride_w = input_c;
  int i_top_left_x = 0;
  int i_top_left_y = 0;
  float i_top_left_x_weight = 0.0;
  float i_top_left_y_weight = 0.0;
  float i_top_left = 0;
  float i_top_right = 0;
  float i_bottom_left = 0;
  float i_bottom_right = 0;

  for (int index = 0; index < output_nums; ++index) {
    // coordinates of each position in output data
    int oc = index % input_c;
    int ow = (index / output_stride_w) % output_w;
    int oh = (index / output_stride_h) % output_h;
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
    i_top_left_x = getInputTopLeft(xf, input_w, i_top_left_x_weight);
    i_top_left_y = getInputTopLeft(yf, input_h, i_top_left_y_weight);

    // field information
    const int i_top_left_offset = input_n * input_stride_batch +
                                  i_top_left_y * input_stride_h +
                                  i_top_left_x * input_stride_w + oc;
    float i_top_left_xy_weight = i_top_left_x_weight * i_top_left_y_weight;
    bool topLeftIsIn = i_top_left_x >= 0 && i_top_left_x <= (input_w - 1) &&
                       i_top_left_y >= 0 && i_top_left_y <= (input_h - 1);
    if (topLeftIsIn) {
      i_top_left = input_cpu_ptr[i_top_left_offset];
    }
    const int i_top_right_offset = i_top_left_offset + input_stride_w;
    float i_top_right_xy_weight =
        (1 - i_top_left_x_weight) * i_top_left_y_weight;
    bool topRightIsIn = i_top_left_x >= 0 && i_top_left_x <= (input_w - 1) &&
                        (i_top_left_y + 1) >= 0 &&
                        (i_top_left_y + 1) <= (input_h - 1);
    if (topRightIsIn) {
      i_top_right = input_cpu_ptr[i_top_right_offset];
    }
    const int i_bottom_left_offset = i_top_left_offset + input_stride_h;
    float i_bottom_left_xy_weight =
        i_top_left_x_weight * (1 - i_top_left_y_weight);
    bool bottomLeftIsIn = (i_top_left_x + 1) >= 0 &&
                          (i_top_left_x + 1) <= (input_w - 1) &&
                          i_top_left_y >= 0 && i_top_left_y <= (input_h - 1);
    if (bottomLeftIsIn) {
      i_bottom_left = input_cpu_ptr[i_bottom_left_offset];
    }
    const int i_bottom_right_offset =
        i_top_left_offset + input_stride_h + input_stride_w;
    float i_bottom_right_xy_weight =
        (1 - i_top_left_x_weight) * (1 - i_top_left_y_weight);
    bool bottomRightIsIn =
        (i_top_left_x + 1) >= 0 && (i_top_left_x + 1) <= (input_w - 1) &&
        (i_top_left_y + 1) >= 0 && (i_top_left_y + 1) <= (input_h - 1);
    if (bottomRightIsIn) {
      i_bottom_right = input_cpu_ptr[i_bottom_right_offset];
    }

    output_cpu_ptr[output_offset] = i_top_left_xy_weight * i_top_left +
                                    i_top_right_xy_weight * i_top_right +
                                    i_bottom_left_xy_weight * i_bottom_left +
                                    i_bottom_right_xy_weight * i_bottom_right;
  }
  VLOG(4) << "[RoiCropForwardExecutor] call cpuCompute() End.";
}

int64_t RoiCropForwardExecutor::getTheoryOps() {
  int cp_count = 58;
  theory_ops = parser_->getInputDataCount(0) * cp_count;
  VLOG(4) << "[RoiCropForwardExecutor] getTheoryOps: " << theory_ops << " ops.";
  return theory_ops;
}

}  // namespace mluoptest
