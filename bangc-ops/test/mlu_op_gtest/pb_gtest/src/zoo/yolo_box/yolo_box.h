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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_YOLO_BOX_YOLO_BOX_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_YOLO_BOX_YOLO_BOX_H_

#include "executor.h"

namespace mluoptest {
class YoloBoxExecutor : public Executor {
 public:
  YoloBoxExecutor() {}
  ~YoloBoxExecutor() {}
  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;

 private:
  void initData();
  float sigmoid(const float x);
  void getYoloBox(float *box, const float *x, const float *anchors, const int i,
                  const int j, const int an_idx, const int grid_size_h,
                  const int grid_size_w, const int input_size_h,
                  const int input_size_w, const int index, const int stride,
                  const float img_height, const float img_width,
                  const float scale, const float bias);
  int getEntryIndex(const int batch, const int an_idx, const int hw_idx,
                    const int an_num, const int an_stride, const int stride,
                    const int entry, const bool iou_aware);
  int getIoUIndex(const int batch, const int an_idx, const int hw_idx,
                  const int an_num, const int an_stride, const int stride);
  void calcDetectionBox(float *boxes, float *box, const int box_idx,
                        const float img_height, const float img_width,
                        const int stride, const bool clip_bbox);
  void calcLabelScore(float *scores, const float *input, const int label_idx,
                      const int score_idx, const int class_num,
                      const float conf, const int stride);
  int class_num_;
  float conf_thresh_;
  int downsample_ratio_;
  bool clip_bbox_;
  float scale_x_y_;
  bool iou_aware_;
  float iou_aware_factor_;
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_YOLO_BOX_YOLO_BOX_H_
