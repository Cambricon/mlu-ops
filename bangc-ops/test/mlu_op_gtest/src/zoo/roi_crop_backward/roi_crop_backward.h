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
  void* gradOutput_data_ptr_;
  void* grid_data_ptr_;
  void* gradInput_data_ptr_;
  mluOpTensorDescriptor_t gradOutput_desc_;
  mluOpTensorDescriptor_t grid_desc_;
  mluOpTensorDescriptor_t gradInput_desc_;
  int gradInput_batch_;
  int gradInput_h_;
  int gradInput_w_;
  int gradInput_c_;
  int grid_batch_roi_;
  int gradOutput_h_;
  int gradOutput_w_;
  int64_t theory_ops_;
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_ROI_CROP_BACKWARD_ROI_CROP_BACKEARD_H_
