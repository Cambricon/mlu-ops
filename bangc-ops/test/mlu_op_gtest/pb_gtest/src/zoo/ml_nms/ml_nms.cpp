/*******************************************************************************
* Copyright (C) [2022] by Cambricon, Inc.
*
*******************************************************************************/
#include "ml_nms.h"
#include <vector>
#include <algorithm>

namespace mluoptest {

void MlNmsExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_ml_nms_param()) {
    LOG(ERROR) << "Lose ml_nms_param. ";
  }
  GTEST_CHECK(parser_->inputs().size() == 1,
              "[MlNmsExecutor] input number is wrong. ");
  GTEST_CHECK(parser_->outputs().size() == 1,
              "[MlNmsExecutor] output number is wrong. ");
}

void MlNmsExecutor::compute() {
  float iou_threshold =
      parser_->getProtoNode()->ml_nms_param().iou_threshold();
  VLOG(4) << "[mluMlNms] iou_threshold: " << iou_threshold;
  // get tensor by name (in prototxt)
  auto boxes_desc = parser_->getMetaTensor("input").tensor;
  auto output_desc = parser_->getMetaTensor("output").tensor;
  auto boxes_ptr = parser_->getMetaTensor("input").dev_ptr;
  auto output_ptr = parser_->getMetaTensor("output").dev_ptr;
  interface_timer_.start();

  VLOG(4) << "[mluOpMlNms] call mluOpMlNms()";
  MLUOP_CHECK(mluOpMlNms(handle_, boxes_desc, boxes_ptr,
    iou_threshold, (uint8_t*)output_ptr));
  interface_timer_.stop();
  VLOG(4) << "[mluOpMlNms] mluOpMlNms end.";
}

static float iouCompute(std::vector<float> box1, std::vector<float> box2) {
  float x1 = std::max(box1[0], box2[0]);
  float y1 = std::min(box1[1], box2[1]);
  float x2 = std::min(box1[2], box2[2]);
  float y2 = std::max(box1[3], box2[3]);

  float area1 = abs(box1[0] - box1[2]) * abs(box1[1] - box1[3]);
  float area2 = abs(box2[0] - box2[2]) * abs(box2[1] - box2[3]);
  float inter = abs(x1 - x2) * abs(y1 - y2);

  float iou = inter / (area1 + area2 - inter);

  return iou;
}

void MlNmsExecutor::cpuCompute() {
  float iou_threshold =
    parser_->getProtoNode()->ml_nms_param().iou_threshold();
  VLOG(4) << "mluMlNms iou_threshold:" << iou_threshold;
  auto input_desc = tensor_desc_[0].tensor;
  auto boxes_ptr = parser_->getMetaTensor(0).cpu_ptr;
  auto output_ptr = parser_->getMetaTensor(1).cpu_ptr;
  int input_boxes_num = input_desc->dims[0];
  std::vector<std::vector<float>> boxes_data_ptr;
  for (int i = 0; i < input_boxes_num * 4; i+=4) {
    std::vector<float> data_ptr;
    for (int j = 0; j < 4; j++) {
      data_ptr.push_back(boxes_ptr[j + i]);
    }
    boxes_data_ptr.push_back(data_ptr);
  }
  for (int i = 0; i < input_boxes_num ; i++) {
    float iou = iouCompute(boxes_data_ptr[0], boxes_data_ptr[i]);
    if (iou <= iou_threshold) {
      output_ptr[i] = 1;
    } else {
      output_ptr[i] = 0;
    }
  }
}

int64_t MlNmsExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->input(0)->total_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}
}  // namespace mluoptest
