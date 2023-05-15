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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_NMS_NMS_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_NMS_NMS_H_
#include "executor.h"
#include "nms3D_utils.h"
#include "mlu_op.h"
namespace mluoptest {
class NmsExecutor : public Executor {
 public:
  NmsExecutor() {}
  ~NmsExecutor() {}
  void paramCheck();
  void workspaceMalloc();
  void workspaceFree();
  void compute();
  void cpuCompute();
  void nms3D_detection_cpu(float *output_data, int &output_box_num,
                           float *input_data, int input_box_num,
                           float thresh_iou, int input_layout);

  void nms_detection_cpu(float *output_data, int &output_box_num,
                         float *input_data, float *input_score,
                         int input_box_num, int keepNum, float thresh_iou,
                         float thresh_score, mluOpNmsOutputMode_t output_mode,
                         int input_layout, mluOpNmsAlgo_t algo, float offset,
                         mluOpNmsBoxPointMode_t box_mode,
                         mluOpNmsMethodMode_t method_mode, float soft_nms_sigma,
                         int batch_idx, int class_idx);
  int64_t getTheoryOps() override;

 private:
  void diffPreprocess();
  int output_boxes_;
  int theory_ops = 0;
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_NMS_NMS_H_
