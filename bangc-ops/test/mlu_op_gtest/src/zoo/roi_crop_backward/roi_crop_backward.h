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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_ROI_CROP_BACKWARD_ROI_CROP_FOREARD_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_ROI_CROP_BACKWARD_ROI_CROP_FOREARD_H_
#include <vector>

#include "core/type.h"
#include "executor.h"

namespace mluoptest {
class RoiCropBackwardExecutor : public Executor {
 public:
  RoiCropBackwardExecutor() {}
  ~RoiCropBackwardExecutor() {}
  void paramCheck();
  void initData();
  void printDataInfo();
  int getInputTopLeft(float grid_yx_value, int gradInput_hw, float& weight);
  void compute();
  void cpuCompute();
  int64_t getTheoryOps() override;

 private:
  void* gradOutput_data_ptr               = NULL;
  void* grid_data_ptr                     = NULL;
  void* gradInput_data_ptr                = NULL;
  mluOpTensorDescriptor_t gradOutput_desc = NULL;
  mluOpTensorDescriptor_t grid_desc       = NULL;
  mluOpTensorDescriptor_t gradInput_desc  = NULL;
  int gradInput_batch;
  int gradInput_h;
  int gradInput_w;
  int gradInput_c;
  int grid_batch_roi;
  int gradOutput_h;
  int gradOutput_w;
  int64_t theory_ops = 0;
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_ROI_CROP_BACKWARD_ROI_CROP_FOREARD_H_
