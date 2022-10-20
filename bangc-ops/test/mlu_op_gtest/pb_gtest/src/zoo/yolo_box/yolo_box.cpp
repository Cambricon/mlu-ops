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
#include "yolo_box.h"

#include "mlu_op.h"

namespace mluoptest {
void YoloBoxExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_yolo_box_param()) {
    LOG(ERROR) << "Lose yolo_box_param. ";
  }
  GTEST_CHECK(parser_->inputs().size() == 3,
              "[YoloBoxExecutor] input number is wrong. ");
  GTEST_CHECK(parser_->outputs().size() == 2,
              "[YoloBoxExecutor] output number is wrong. ");
}

void YoloBoxExecutor::initData() {
  class_num_ = parser_->getProtoNode()->yolo_box_param().class_num();
  conf_thresh_ = parser_->getProtoNode()->yolo_box_param().conf_thresh();
  downsample_ratio_ =
      parser_->getProtoNode()->yolo_box_param().downsample_ratio();
  clip_bbox_ = parser_->getProtoNode()->yolo_box_param().clip_bbox();
  scale_x_y_ = parser_->getProtoNode()->yolo_box_param().scale_x_y();
  iou_aware_ = parser_->getProtoNode()->yolo_box_param().iou_aware();
  iou_aware_factor_ =
      parser_->getProtoNode()->yolo_box_param().iou_aware_factor();
}

void YoloBoxExecutor::compute() {
  VLOG(4) << "[YoloBoxExecutor] call compute() begin.";
  initData();
  // input tensor
  auto x_desc = tensor_desc_[0].tensor;
  auto img_size_desc = tensor_desc_[1].tensor;
  auto anchors_desc = tensor_desc_[2].tensor;
  auto dev_x = data_vector_[0].device_ptr;
  auto dev_img_size = data_vector_[1].device_ptr;
  auto dev_anchors = data_vector_[2].device_ptr;

  // output tensor
  auto boxes_desc = tensor_desc_[3].tensor;
  auto scores_desc = tensor_desc_[4].tensor;
  auto dev_boxes = data_vector_[3].device_ptr;
  auto dev_scores = data_vector_[4].device_ptr;

  interface_timer_.start();
  MLUOP_CHECK(mluOpYoloBox(handle_, x_desc, dev_x, img_size_desc, dev_img_size,
                           anchors_desc, dev_anchors, class_num_, conf_thresh_,
                           downsample_ratio_, clip_bbox_, scale_x_y_,
                           iou_aware_, iou_aware_factor_, boxes_desc, dev_boxes,
                           scores_desc, dev_scores));
  interface_timer_.stop();
  VLOG(4) << "[YoloBoxExecutor] call compute() end.";
}

float YoloBoxExecutor::sigmoid(const float x) {
  return 1.0 / (1.0 + std::exp(-x));
}

void YoloBoxExecutor::getYoloBox(float *box, const float *x,
                                 const float *anchors, const int i, const int j,
                                 const int an_idx, const int grid_size_h,
                                 const int grid_size_w, const int input_size_h,
                                 const int input_size_w, const int index,
                                 const int stride, const float img_height,
                                 const float img_width, const float scale,
                                 const float bias) {
  box[0] = (i + sigmoid(x[index]) * scale + bias) * img_width / grid_size_w;
  box[1] = (j + sigmoid(x[index + stride]) * scale + bias) * img_height /
           grid_size_h;
  box[2] = std::exp(x[index + 2 * stride]) * anchors[2 * an_idx] * img_width /
           input_size_w;
  box[3] = std::exp(x[index + 3 * stride]) * anchors[2 * an_idx + 1] *
           img_height / input_size_h;
}

int YoloBoxExecutor::getEntryIndex(const int batch, const int an_idx,
                                   const int hw_idx, const int an_num,
                                   const int an_stride, const int stride,
                                   const int entry, const bool iou_aware) {
  if (iou_aware) {
    return (batch * an_num + an_idx) * an_stride +
           (batch * an_num + an_num + entry) * stride + hw_idx;
  } else {
    return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
  }
}

int YoloBoxExecutor::getIoUIndex(const int batch, const int an_idx,
                                 const int hw_idx, const int an_num,
                                 const int an_stride, const int stride) {
  return batch * an_num * an_stride + (batch * an_num + an_idx) * stride +
         hw_idx;
}

void YoloBoxExecutor::calcDetectionBox(float *boxes, float *box,
                                       const int box_idx,
                                       const float img_height,
                                       const float img_width, const int stride,
                                       const bool clip_bbox) {
  boxes[box_idx] = box[0] - box[2] / 2;
  boxes[box_idx + stride] = box[1] - box[3] / 2;
  boxes[box_idx + 2 * stride] = box[0] + box[2] / 2;
  boxes[box_idx + 3 * stride] = box[1] + box[3] / 2;

  if (clip_bbox) {
    boxes[box_idx] =
        boxes[box_idx] > 0 ? boxes[box_idx] : static_cast<float>(0);
    boxes[box_idx + stride] = boxes[box_idx + stride] > 0
                                  ? boxes[box_idx + stride]
                                  : static_cast<float>(0);
    boxes[box_idx + 2 * stride] = boxes[box_idx + 2 * stride] < img_width - 1
                                      ? boxes[box_idx + 2 * stride]
                                      : static_cast<float>(img_width - 1);
    boxes[box_idx + 3 * stride] = boxes[box_idx + 3 * stride] < img_height - 1
                                      ? boxes[box_idx + 3 * stride]
                                      : static_cast<float>(img_height - 1);
  }
}

void YoloBoxExecutor::calcLabelScore(float *scores, const float *input,
                                     const int label_idx, const int score_idx,
                                     const int class_num, const float conf,
                                     const int stride) {
  for (int i = 0; i < class_num; i++) {
    scores[score_idx + i * stride] =
        conf * sigmoid(input[label_idx + i * stride]);
  }
}

void YoloBoxExecutor::cpuCompute() {
  VLOG(4) << "[YoloBoxExecutor] call cpuCompute() begin.";
  float bias = -0.5 * (scale_x_y_ - 1);

  auto x_desc = tensor_desc_[0].tensor;
  float *input_data = cpu_fp32_input_[0];
  float *imgsize_data = cpu_fp32_input_[1];
  float *anchors_data = cpu_fp32_input_[2];
  float *boxes_data = cpu_fp32_output_[0];
  float *scores_data = cpu_fp32_output_[1];
  int boxes_size = parser_->getOutputDataCount(0);
  int scores_size = parser_->getOutputDataCount(1);
  memset(boxes_data, 0, boxes_size);
  memset(scores_data, 0, scores_size);

  const int n = x_desc->dims[0];
  const int h = x_desc->dims[2];
  const int w = x_desc->dims[3];
  auto anchors_desc = tensor_desc_[2].tensor;
  uint64_t anchors_tensor_num = mluOpGetTensorElementNum(anchors_desc);
  const int an_num = anchors_tensor_num / 2;
  const int input_size_h = downsample_ratio_ * h;
  const int input_size_w = downsample_ratio_ * w;
  const int stride = h * w;
  const int an_stride = (class_num_ + 5) * stride;

  float box[4] = {0};
  for (int i = 0; i < n; i++) {
    float img_height = imgsize_data[2 * i];
    float img_width = imgsize_data[2 * i + 1];

    for (int j = 0; j < an_num; j++) {
      for (int k = 0; k < h; k++) {
        for (int l = 0; l < w; l++) {
          int obj_idx = getEntryIndex(i, j, k * w + l, an_num, an_stride,
                                      stride, 4, iou_aware_);
          float conf = sigmoid(input_data[obj_idx]);
          if (iou_aware_) {
            int iou_idx =
                getIoUIndex(i, j, k * w + l, an_num, an_stride, stride);
            float iou = sigmoid(input_data[iou_idx]);
            conf = pow(conf, static_cast<float>(1. - iou_aware_factor_)) *
                   pow(iou, static_cast<float>(iou_aware_factor_));
          }
          if (conf < conf_thresh_) {
            continue;
          }

          int box_idx = getEntryIndex(i, j, k * w + l, an_num, an_stride,
                                      stride, 0, iou_aware_);

          getYoloBox(box, input_data, anchors_data, l, k, j, h, w, input_size_h,
                     input_size_w, box_idx, stride, img_height, img_width,
                     scale_x_y_, bias);
          box_idx = (i * an_num + j) * 4 * stride + k * w + l;

          calcDetectionBox(boxes_data, box, box_idx, img_height, img_width,
                           stride, clip_bbox_);

          int label_idx = getEntryIndex(i, j, k * w + l, an_num, an_stride,
                                        stride, 5, iou_aware_);
          int score_idx = (i * an_num + j) * class_num_ * stride + k * w + l;

          calcLabelScore(scores_data, input_data, label_idx, score_idx,
                         class_num_, conf, stride);
        }
      }
    }
    VLOG(4) << "[YoloBoxExecutor] call cpuCompute() end.";
  }
}

int64_t YoloBoxExecutor::getTheoryOps() {
  const int cp_count = 30;
  int64_t theory_ops0_ = parser_->getInputDataCount(0) * cp_count;
  VLOG(4) << "[YoloBoxExecutor] getTheoryOps: " << theory_ops0_ << " ops.";
  return theory_ops0_;
}

}  // namespace mluoptest
