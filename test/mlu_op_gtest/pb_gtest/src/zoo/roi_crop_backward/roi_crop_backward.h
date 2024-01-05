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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_ROI_CROP_BACKWARD_ROI_CROP_BACKEARD_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_ROI_CROP_BACKWARD_ROI_CROP_BACKEARD_H_

#include "executor.h"

namespace mluoptest {
class RoiCropBackwardExecutor : public Executor {
 public:
  RoiCropBackwardExecutor() {}
  ~RoiCropBackwardExecutor() {}
  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;

 private:
  void initData();
  void printDataInfo();
  int getTopLeft(const float grid_yx_value, const int input_hw, float* weight);
  void* grad_output_data_ptr_;
  void* grid_data_ptr_;
  void* grad_input_data_ptr_;
  mluOpTensorDescriptor_t grad_output_desc_;
  mluOpTensorDescriptor_t grid_desc_;
  mluOpTensorDescriptor_t grad_input_desc_;
  int grad_input_batch_;
  int grad_input_h_;
  int grad_input_w_;
  int grad_input_c_;
  int grid_batch_roi_;
  int grad_output_h_;
  int grad_output_w_;
  int64_t theory_ops_;
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_ROI_CROP_BACKWARD_ROI_CROP_BACKEARD_H_
