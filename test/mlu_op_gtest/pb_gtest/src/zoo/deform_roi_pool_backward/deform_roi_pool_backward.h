/*******************************************************************************
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
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *******************************************************************************/
#ifndef TEST_MLUOP_GTEST_SRC_ZOO_DEFORMROIPOOLBACKWARD_DEFORMROIPOOLBACKWARD_H_
#define TEST_MLUOP_GTEST_SRC_ZOO_DEFORMROIPOOLBACKWARD_DEFORMROIPOOLBACKWARD_H_
#include "executor.h"

namespace mluoptest {
class DeformRoiPoolBackwardExecutor : public Executor {
 public:
  DeformRoiPoolBackwardExecutor() {}
  ~DeformRoiPoolBackwardExecutor() {}
  void initData();
  void printDataInfo();
  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;

 private:
  int batchs;
  int height;
  int width;
  int channels;
  int rois_num;
  int pooled_height;
  int pooled_width;
  int sampling_ratio;
  float spatial_scale;
  float gamma;
  mluOpTensorDescriptor_t grad_output_desc = NULL;
  mluOpTensorDescriptor_t input_desc = NULL;
  mluOpTensorDescriptor_t rois_desc = NULL;
  mluOpTensorDescriptor_t offset_desc = NULL;
  mluOpTensorDescriptor_t grad_input_desc = NULL;
  mluOpTensorDescriptor_t grad_offset_desc = NULL;
};
}  // namespace mluoptest
#endif
