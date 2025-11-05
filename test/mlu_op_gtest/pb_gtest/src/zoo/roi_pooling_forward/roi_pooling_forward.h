/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
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
 * OR IMPLIED, INCLUDING BUT NOKType LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHKType HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef TEST_MLUOP_GTEST_SRC_ZOO_ROI_POOLING_FORWARD_ROI_POOLING_FORWARD_H_
#define TEST_MLUOP_GTEST_SRC_ZOO_ROI_POOLING_FORWARD_ROI_POOLING_FORWARD_H_
#include "executor.h"

namespace mluoptest {

class RoiPoolingForwardExecutor : public Executor {
 public:
  RoiPoolingForwardExecutor() {}
  ~RoiPoolingForwardExecutor() {}

  void compute() override;
  void cpuCompute() override;
  void initData();
  int64_t getTheoryOps() override;

  void cpuRoiPoolingForward(float *input_v,
                            float *rois,
                            int batch_v,
                            int height,
                            int width,
                            int channels,
                            int pool_height,
                            int pool_width,
                            int rois_num,
                            float spatial_scale,
                            float *output,
                            float *argmax);

 private:
  int batch_;
  int height_;
  int width_;
  int channels_;
  int rois_num_;
  int pool_height_;
  int pool_width_;
  float spatial_scale_;
  mluOpPoolingMode_t pooling_mode_     = MLUOP_POOLING_MAX;
  void *input_mlu_                     = NULL;
  void *rois_mlu_                      = NULL;
  void *output_mlu_                    = NULL;
  int *argmax_mlu_                     = NULL;
  mluOpTensorDescriptor_t input_desc_  = NULL;
  mluOpTensorDescriptor_t rois_desc_   = NULL;
  mluOpTensorDescriptor_t output_desc_ = NULL;
  int64_t theory_ops = 0;
};

}  // namespace mluoptest
#endif  // TEST_MLUOP_GTEST_SRC_ZOO_ROI_POOLING_FORWARD_ROI_POOLING_FORWARD_H_

