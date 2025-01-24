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
#include <sys/time.h>
#include "nms.h"
#include "mlu_op.h"

namespace mluoptest {
void NmsExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_nms_param()) {
    LOG(ERROR) << "mluOpNms: Lose nms_param. ";
  }
  if (parser_->getInputNum() != 2) {
    LOG(ERROR) << "mluOpNms: tensor input number is wrong. ";
  }
  if (parser_->getOutputNum() != 1 && parser_->getOutputNum() != 2) {
    LOG(ERROR) << "mluOpNms: tensor output number is wrong. ";
  }
}

void NmsExecutor::workspaceMalloc() {
  auto tensor_boxes = parser_->getMetaTensor("input1").tensor;
  auto tensor_confi = parser_->getMetaTensor("input2").tensor;
  auto input_layout = parser_->getProtoNode()->nms_param().input_layout();
  if (tensor_boxes->getDim() == 2) {
    if (input_layout == 0) {
      box_dim_ = tensor_boxes->getDimIndex(1);
    } else {
      box_dim_ = tensor_boxes->getDimIndex(0);
    }
  }
  float iou_threshold = parser_->getProtoNode()->nms_param().iou_threshold();
  mluOpNmsOutputMode_t mode =
      (mluOpNmsOutputMode_t)parser_->getProtoNode()->nms_param().mode();
  mluOpNmsAlgo_t algo =
      (mluOpNmsAlgo_t)parser_->getProtoNode()->nms_param().algo();
  float offset = parser_->getProtoNode()->nms_param().offset();
  int max_output_size = parser_->getProtoNode()->nms_param().max_output_boxes();
  float confidence_threshold =
      parser_->getProtoNode()->nms_param().confidence_threshold();
  bool pad_to_max_output_size =
      parser_->getProtoNode()->nms_param().pad_to_max_output_size();

  mluOpNmsBoxPointMode_t box_mode =
      (mluOpNmsBoxPointMode_t)parser_->getProtoNode()->nms_param().box_mode();
  float soft_nms_sigma = parser_->getProtoNode()->nms_param().soft_nms_sigma();
  mluOpNmsMethodMode_t method_mode =
      (mluOpNmsMethodMode_t)parser_->getProtoNode()->nms_param().method_mode();
  mluOpNmsDescriptor_t nms_desc;
  nms_desc = cpu_runtime_.allocate(mluOpCreateNmsDescriptor,
                                   mluOpDestroyNmsDescriptor);
  MLUOP_CHECK(mluOpSetNmsDescriptor(
      nms_desc, box_mode, mode, algo, method_mode, iou_threshold,
      soft_nms_sigma, max_output_size, confidence_threshold, offset,
      input_layout, pad_to_max_output_size));
  VLOG(4) << "box_dim_: " << box_dim_;
  if (box_dim_ == 4) {
    MLUOP_CHECK(mluOpGetNmsWorkspaceSize(handle_, nms_desc, tensor_boxes, tensor_confi,
                                         &workspace_size_));
  } else {
    // box_dim_ = 7
    MLUOP_CHECK(mluOpGetNmsWorkspaceSize(handle_, nms_desc, tensor_boxes, nullptr,
                                         &workspace_size_));
  }
  VLOG(4) << "Malloc workspace space.";
  if (workspace_size_ > 0) {
    workspace_ = mlu_runtime_.allocate(workspace_size_);
  }
  VLOG(4) << "Malloc addr=" << workspace_ << ", size=" << workspace_size_;

  // this op will modify input data.
  // when repeat != 1, after second compute(), input data has been modified.
  // input data changed, the result of this op ("result_num",
  // aka output->getDimIndex(0)) is changed.
  // so we don't know result_num when repeat finished.
  // so ignore what result_num(valid data number in output0) is,
  // set all data in output0 as 0.
  // actually, if we know what result_num is, just set other data as 0.
  size_t output_size = parser_->getMetaTensor("output1").size_in_bytes;
  if ((!pad_to_max_output_size ||
       parser_->getMetaTensor("input1").size_in_bytes == 0) &&
      output_size > 0) {
    void *output_ptr = parser_->getMetaTensor("output1").dev_origin_ptr;
    GTEST_CHECK(cnrtSuccess == cnrtMemset(output_ptr, 0, output_size));
  }

  void *output2_ptr = parser_->getMetaTensor("output2").dev_origin_ptr;
  size_t output2_size = parser_->getMetaTensor("output2").size_in_bytes;
  GTEST_CHECK(cnrtSuccess == cnrtMemset(output2_ptr, 0, output2_size));

  eva_->setMluWorkspaceSize(workspace_size_);
}

void NmsExecutor::workspaceFree() {
  if (workspace_) {
    VLOG(4) << "Free device workspace space.";
    mlu_runtime_.deallocate(workspace_);
  }
}

void NmsExecutor::compute() {
  float iou_threshold = parser_->getProtoNode()->nms_param().iou_threshold();
  mluOpNmsOutputMode_t mode =
      (mluOpNmsOutputMode_t)parser_->getProtoNode()->nms_param().mode();
  mluOpNmsAlgo_t algo =
      (mluOpNmsAlgo_t)parser_->getProtoNode()->nms_param().algo();
  float offset = parser_->getProtoNode()->nms_param().offset();
  auto input_layout = parser_->getProtoNode()->nms_param().input_layout();
  int max_output_size = parser_->getProtoNode()->nms_param().max_output_boxes();
  float confidence_threshold =
      parser_->getProtoNode()->nms_param().confidence_threshold();
  bool pad_to_max_output_size =
      parser_->getProtoNode()->nms_param().pad_to_max_output_size();

  mluOpNmsBoxPointMode_t box_mode =
      (mluOpNmsBoxPointMode_t)parser_->getProtoNode()->nms_param().box_mode();
  float soft_nms_sigma = parser_->getProtoNode()->nms_param().soft_nms_sigma();
  mluOpNmsMethodMode_t method_mode =
      (mluOpNmsMethodMode_t)parser_->getProtoNode()->nms_param().method_mode();

  // get tensor by name (in prototxt)
  auto tensor_boxes = parser_->getMetaTensor("input1").tensor;

  auto tensor_confi =
      box_dim_ == 7 ? nullptr : parser_->getMetaTensor("input2").tensor;
  auto tensor_output = parser_->getMetaTensor("output1").tensor;
  auto dev_boxes = parser_->getMetaTensor("input1").dev_ptr;
  auto dev_confi =
      box_dim_ == 7 ? nullptr : parser_->getMetaTensor("input2").dev_ptr;
  auto dev_output = parser_->getMetaTensor("output1").dev_ptr;
  VLOG(4) << "call mluop NmsTensor()";

  mluOpNmsDescriptor_t nms_desc;
  nms_desc = cpu_runtime_.allocate(mluOpCreateNmsDescriptor,
                                   mluOpDestroyNmsDescriptor);
  VLOG(5) << "tensor_boxes->getDim(): " << tensor_boxes->getDim();
  MLUOP_CHECK(mluOpSetNmsDescriptor(
      nms_desc, box_mode, mode, algo, method_mode, iou_threshold,
      soft_nms_sigma, max_output_size, confidence_threshold, offset,
      input_layout, pad_to_max_output_size));
  VLOG(4) << "algo: " << algo;
  VLOG(4) << "offset: " << offset;
  VLOG(4) << "sigma: " << soft_nms_sigma;
  VLOG(4) << "box_mode: " << box_mode;
  VLOG(4) << "method_mode: " << method_mode;
  auto dev_output_size = parser_->getMetaTensor("output2").dev_ptr;
  interface_timer_.start();
  VLOG(4) << "tensor_confi=" << tensor_confi << ", dev_confi=" << dev_confi
          << ", workspace_size_=" << workspace_size_;
  MLUOP_CHECK(mluOpNms(handle_, nms_desc, tensor_boxes, dev_boxes, tensor_confi,
                       dev_confi, workspace_, workspace_size_, tensor_output,
                       dev_output, dev_output_size));
  interface_timer_.stop();
  VLOG(4) << "mluOpNms-end";

  cpu_runtime_.deallocate(nms_desc);
}

void NmsExecutor::nms3D_detection_cpu(float *output_data, int &output_box_num,
                                      float *input_data, int input_box_num,
                                      float thresh_iou, int input_layout) {
  float *x1 = cpu_runtime_.allocate(new float[input_box_num]);
  float *y1 = cpu_runtime_.allocate(new float[input_box_num]);
  float *dx = cpu_runtime_.allocate(new float[input_box_num]);
  float *dy = cpu_runtime_.allocate(new float[input_box_num]);
  float *angle = cpu_runtime_.allocate(new float[input_box_num]);
  float *score = cpu_runtime_.allocate(new float[input_box_num]);
  float *box_a = cpu_runtime_.allocate(new float[7]);
  float *box_b = cpu_runtime_.allocate(new float[7]);
  memset(score, 1, input_box_num * sizeof(int));
  if (input_layout == 0) {
    // x,y,z,dx,dy,dz,angle
    for (int i = 0; i < input_box_num; i++) {
      x1[i] = input_data[0 + i * 7];
      y1[i] = input_data[1 + i * 7];
      dx[i] = input_data[3 + i * 7];
      dy[i] = input_data[4 + i * 7];
      angle[i] = input_data[6 + i * 7];
    }
  } else if (input_layout == 1) {
    memcpy(x1, input_data, input_box_num * sizeof(float));
    memcpy(y1, input_data + 1 * input_box_num, input_box_num * sizeof(float));
    memcpy(dx, input_data + 3 * input_box_num, input_box_num * sizeof(float));
    memcpy(dy, input_data + 4 * input_box_num, input_box_num * sizeof(float));
    memcpy(angle, input_data + 6 * input_box_num,
           input_box_num * sizeof(float));
  } else {
    VLOG(4) << "unsupport data layout now.";
  }

  for (int cur_box = 0; cur_box < input_box_num; cur_box++) {
    if (score[cur_box] == 0) {
      continue;
    }
    output_data[output_box_num] = cur_box;
    output_box_num++;
    // params box_a: [x, y, z, dx, dy, dz, heading]
    box_a[0] = x1[cur_box], box_a[1] = y1[cur_box];
    box_a[3] = dx[cur_box], box_a[4] = dy[cur_box];
    box_a[6] = angle[cur_box];

    for (int i = 0; i < input_box_num; i++) {
      box_b[0] = x1[i], box_b[1] = y1[i];
      box_b[3] = dx[i], box_b[4] = dy[i];
      box_b[6] = angle[i];
      // get IOU
      float iou = Nms3DUtils::UtilsFunctions::iou_bev(box_a, box_b);
      if (iou > thresh_iou) {
        score[i] = 0;
      }
    }
  }
  VLOG(4) << "ouput_boxes_num:" << output_box_num;

  cpu_runtime_.deallocate(score);
  cpu_runtime_.deallocate(x1);
  cpu_runtime_.deallocate(y1);
  cpu_runtime_.deallocate(dx);
  cpu_runtime_.deallocate(dy);
  cpu_runtime_.deallocate(angle);
  cpu_runtime_.deallocate(box_a);
  cpu_runtime_.deallocate(box_b);
}

void NmsExecutor::nms_detection_cpu(
    float *output_data, int &output_box_num, float *input_data,
    float *input_score, int input_box_num, int keepNum, float thresh_iou,
    float thresh_score, mluOpNmsOutputMode_t output_mode, int input_layout,
    mluOpNmsAlgo_t algo, float offset, mluOpNmsBoxPointMode_t box_mode,
    mluOpNmsMethodMode_t method_mode, float soft_nms_sigma, int batch_idx,
    int class_idx) {
  float *score = cpu_runtime_.allocate(new float[input_box_num]);
  float *x1 = cpu_runtime_.allocate(new float[input_box_num]);
  float *y1 = cpu_runtime_.allocate(new float[input_box_num]);
  float *x2 = cpu_runtime_.allocate(new float[input_box_num]);
  float *y2 = cpu_runtime_.allocate(new float[input_box_num]);
  if (input_layout == 0) {
    // input layout is [boxes_num, 4]
    for (int i = 0; i < input_box_num; i++) {
      x1[i] = input_data[0 + i * 4];
      y1[i] = input_data[1 + i * 4];
      x2[i] = input_data[2 + i * 4];
      y2[i] = input_data[3 + i * 4];
      // VLOG(4) << i << " x1:" << x1[i] << "y1:" << y1[i] << "x2:" << x2[i] <<
      // "y2:" << y2[i];
    }
  } else if (input_layout == 1) {
    // input layout is [4, boxes_num]
    memcpy(x1, input_data, input_box_num * sizeof(float));
    memcpy(y1, input_data + 1 * input_box_num, input_box_num * sizeof(float));
    memcpy(x2, input_data + 2 * input_box_num, input_box_num * sizeof(float));
    memcpy(y2, input_data + 3 * input_box_num, input_box_num * sizeof(float));
  } else {
    VLOG(4) << "unsupport data layout now.";
  }
  memcpy(score, input_score, input_box_num * sizeof(float));
  for (int keep = 0; keep < keepNum; keep++) {
    // find the max score
    float max_score = score[0];
    int max_index = 0;
    float max_x1, max_y1, max_x2, max_y2;
    float output_x1, output_y1, output_x2, output_y2;
    float max_area = 0;

    for (int i = 1; i < input_box_num; i++) {
      if (score[i] > max_score) {
        max_score = score[i];
        max_index = i;
      }
    }
    max_x1 = x1[max_index];
    max_y1 = y1[max_index];
    max_x2 = x2[max_index];
    max_y2 = y2[max_index];

    if (max_score <= thresh_score) {
      break;
    }

    // VLOG(4) << "max_index:" << max_index;
    // VLOG(4) << "max_score: " << max_score << "x1: " << max_x1 << "y1: " <<
    // max_y1
    //        << "x2: " << max_x2 << "y2: " << max_y2;
    if (output_mode == 0) {
      // save index of max score
      output_data[output_box_num] = max_index;
    } else if (output_mode == 1) {
      output_data[output_box_num * 5 + 0] = max_score;
      output_data[output_box_num * 5 + 1] = max_x1;
      output_data[output_box_num * 5 + 2] = max_y1;
      output_data[output_box_num * 5 + 3] = max_x2;
      output_data[output_box_num * 5 + 4] = max_y2;
    } else if (output_mode == 2) {
      output_data[0 * keepNum + output_box_num] = max_score;
      output_data[1 * keepNum + output_box_num] = max_x1;
      output_data[2 * keepNum + output_box_num] = max_y1;
      output_data[3 * keepNum + output_box_num] = max_x2;
      output_data[4 * keepNum + output_box_num] = max_y2;
    } else if (output_mode == 3) {
      output_data[output_box_num * 3 + 0] = batch_idx;
      output_data[output_box_num * 3 + 1] = class_idx;
      output_data[output_box_num * 3 + 2] = max_index;
    } else {
      VLOG(4) << "unsupport output mode now.";
    }
    output_box_num++;
    score[max_index] = 0;

    if (box_mode == 0) {
      if (max_x1 > max_x2) {
        float tmp = max_x1;
        max_x1 = max_x2;
        max_x2 = tmp;
      }
      if (max_y1 > max_y2) {
        float tmp = max_y1;
        max_y1 = max_y2;
        max_y2 = tmp;
      }
    } else if (box_mode == 1) {
      max_x1 = max_x1 - max_x2 * 0.5;
      max_x2 = max_x1 + max_x2;
      max_y1 = max_y1 - max_y2 * 0.5;
      max_y2 = max_y1 + max_y2;
    }
    if (algo == 0 || offset == 0.0) {
      max_area = (max_x2 - max_x1) * (max_y2 - max_y1);
    } else {
      max_area = (max_x2 - max_x1 + offset) * (max_y2 - max_y1 + offset);
    }

    for (int i = 0; i < input_box_num; i++) {
      // compute the IOU
      float x1_cur = x1[i];
      float y1_cur = y1[i];
      float x2_cur = x2[i];
      float y2_cur = y2[i];
      float area_cur = 0.0;
      if (box_mode == 0) {
        if (x1_cur > x2_cur) {
          float tmp = x1_cur;
          x1_cur = x2_cur;
          x2_cur = tmp;
        }
        if (y1_cur > y2_cur) {
          float tmp = y1_cur;
          y1_cur = y2_cur;
          y2_cur = tmp;
        }
      } else if (box_mode == 1) {
        x1_cur = x1_cur - x2_cur * 0.5;
        x2_cur = x1_cur + x2_cur;
        y1_cur = y1_cur - y2_cur * 0.5;
        y2_cur = y1_cur + y2_cur;
      }
      if (algo == 1) {
        area_cur = (x2_cur - x1_cur + offset) * (y2_cur - y1_cur + offset);
      } else {
        area_cur = (x2_cur - x1_cur) * (y2_cur - y1_cur);
      }

      // get the area_I
      float inter_x1 = (max_x1 > x1_cur ? max_x1 : x1_cur);
      float inter_y1 = (max_y1 > y1_cur ? max_y1 : y1_cur);
      float inter_x2 = (max_x2 > x2_cur ? x2_cur : max_x2);
      float inter_y2 = (max_y2 > y2_cur ? y2_cur : max_y2);
      float inter_w = 0.0, inter_h = 0.0;
      if (algo == 1) {
        inter_w = inter_x2 - inter_x1 + offset;
        inter_h = inter_y2 - inter_y1 + offset;
      } else {
        inter_w = inter_x2 - inter_x1;
        inter_h = inter_y2 - inter_y1;
      }
      if (inter_w < 0) {
        inter_w = 0;
      }
      if (inter_h < 0) {
        inter_h = 0;
      }
      float area_I = inter_w * inter_h;
      // get the area_U
      float area_U = max_area + area_cur - area_I;
      float iou = area_I / area_U;
      // update the score
      if (method_mode == 0) {
        if (iou > thresh_iou) {
          score[i] = 0;
        }
      } else if (method_mode == 1) {
        score[i] = iou > thresh_iou ? score[i] * (1 - iou) : score[i];
      } else {
        // assert method_mode == 2
        // TODO(wch): make sure the formula
        if (soft_nms_sigma > 0.0) {
          score[i] *= exp(-iou * iou / (2 * soft_nms_sigma));
        } else {
          score[i] = (iou > thresh_iou) ? 0.0 : score[i];
        }
      }
    }
  }
  // VLOG(4) << "ouput_boxes_num:" << output_box_num;
  cpu_runtime_.deallocate(score);
  cpu_runtime_.deallocate(x1);
  cpu_runtime_.deallocate(y1);
  cpu_runtime_.deallocate(x2);
  cpu_runtime_.deallocate(y2);
}

void NmsExecutor::cpuCompute() {
  GTEST_CHECK(parser_->getInputNum() == 2);
  // assert(parser_->getOutputNum() == 1);

  int max_output_boxes =
      parser_->getProtoNode()->nms_param().max_output_boxes();
  float iou_thresh = parser_->getProtoNode()->nms_param().iou_threshold();
  mluOpNmsOutputMode_t mode =
      (mluOpNmsOutputMode_t)parser_->getProtoNode()->nms_param().mode();
  float confidence_threshold =
      parser_->getProtoNode()->nms_param().confidence_threshold();
  auto input_layout = parser_->getProtoNode()->nms_param().input_layout();
  mluOpNmsAlgo_t algo =
      (mluOpNmsAlgo_t)parser_->getProtoNode()->nms_param().algo();
  float offset = parser_->getProtoNode()->nms_param().offset();
  mluOpNmsBoxPointMode_t box_mode =
      (mluOpNmsBoxPointMode_t)parser_->getProtoNode()->nms_param().box_mode();
  float soft_nms_sigma = parser_->getProtoNode()->nms_param().soft_nms_sigma();
  mluOpNmsMethodMode_t method_mode =
      (mluOpNmsMethodMode_t)parser_->getProtoNode()->nms_param().method_mode();
  bool pad_to_max_output_size =
      parser_->getProtoNode()->nms_param().pad_to_max_output_size();

  auto input_box_desc = tensor_desc_[0].tensor;
  auto input_conf_desc = tensor_desc_[1].tensor;
  int input_boxes_num = 0;
  int input_batches_num = 1;
  int input_classes_num = 1;
  int box_dim = 4;
  if (input_box_desc->getDim() == 2) {
    if (input_layout == 0) {
      // when layout is [boxes_num, 4], dims[0] represets the input number.
      input_boxes_num = input_box_desc->getDimIndex(0);
      box_dim = input_box_desc->getDimIndex(1);
    } else if (input_layout == 1) {
      input_boxes_num = input_box_desc->getDimIndex(1);
      box_dim = input_box_desc->getDimIndex(0);
    } else {
      VLOG(4) << "unsupport input layout now.";
    }
  } else {
    // assert input_box_desc->getDim() == 3
    input_batches_num = input_box_desc->getDimIndex(0);
    input_classes_num = input_conf_desc->getDimIndex(1);
    // keep content of algo and offset, algo is deprecated at
    // setNmsDescriptor_v4
    algo = mluOpNmsAlgo_t::MLUOP_NMS_ALGO_INCLUDE_BOUNDARY;
    if (input_layout == 0) {
      // when layout is [boxes_num, 4], dims[0] represets the input number.
      input_boxes_num = input_box_desc->getDimIndex(1);
    } else if (input_layout == 1) {
      input_boxes_num = input_box_desc->getDimIndex(2);
    } else {
      VLOG(4) << "unsupport input layout now.";
    }
  }

  VLOG(4) << "input_boxes_num:" << input_boxes_num;

  auto input_boxes = parser_->getMetaTensor(0).cpu_ptr;
  auto input_conf = parser_->getMetaTensor(1).cpu_ptr;
  auto output_info = parser_->getMetaTensor(2).cpu_ptr;

  int total_output_boxes_num = 0;
  if (box_dim == 7) {
    // NMS3D
    nms3D_detection_cpu(output_info, total_output_boxes_num, input_boxes,
                        input_boxes_num, iou_thresh, input_layout);
  } else {
    for (int batch_idx = 0; batch_idx < input_batches_num; ++batch_idx) {
      for (int class_idx = 0; class_idx < input_classes_num; ++class_idx) {
        int boxes_offset = input_boxes_num * 4 * batch_idx;
        int conf_offset = input_classes_num * input_boxes_num * batch_idx +
                          input_boxes_num * class_idx;
        int output_offset = mode == 3 ? 3 * total_output_boxes_num : 0;
        int output_boxes_num = 0;
        nms_detection_cpu((float *)output_info + output_offset,
                          output_boxes_num, (float *)input_boxes + boxes_offset,
                          (float *)input_conf + conf_offset, input_boxes_num,
                          max_output_boxes, iou_thresh, confidence_threshold,
                          mode, input_layout, algo, offset, box_mode,
                          method_mode, soft_nms_sigma, batch_idx, class_idx);
        total_output_boxes_num += output_boxes_num;
        // VLOG(4) << "total_output_boxes:" << total_output_boxes_num;
      }
    }
  }
  // save the output boxes num, computed by CPU
  VLOG(4) << "total_output_boxes_num:" << total_output_boxes_num;
  auto output_size = parser_->getMetaTensor(3).cpu_ptr;
  output_size[0] = total_output_boxes_num;
}

void NmsExecutor::diffPreprocess() {
  // The valid output data will be no more than the max_output_size and output
  // shape size. So before the diff computation, we should set the invalid data
  // to zeros.
  int max_output_boxes =
      parser_->getProtoNode()->nms_param().max_output_boxes();
  mluOpNmsOutputMode_t mode =
      (mluOpNmsOutputMode_t)parser_->getProtoNode()->nms_param().mode();
  auto output_size = parser_->getMetaTensor(3).cpu_ptr;
  output_boxes_ = output_size[0];
  VLOG(4) << "max_output_size:" << max_output_boxes;
  VLOG(4) << "output_size:" << output_boxes_;
  auto output1_desc = parser_->getMetaTensor("output1").tensor;
  // output ram size, which could hold max_output_size number of boxes.
  uint64_t output_shape = mluOpGetTensorElementNum(output1_desc);
  VLOG(4) << "output_shape:" << output_shape;
  auto tensor_boxes = parser_->getMetaTensor("input1").tensor;
  auto input_layout = parser_->getProtoNode()->nms_param().input_layout();
  bool pad_to_max_output_size =
      parser_->getProtoNode()->nms_param().pad_to_max_output_size();
  int output_mode_num = 1;
  int box_dim = 4;
  if (tensor_boxes->getDim() == 2) {
    box_dim = input_layout == 0 ? tensor_boxes->getDimIndex(1)
                                : tensor_boxes->getDimIndex(0);
  }
  if (box_dim == 7) {
    mode = static_cast<mluOpNmsOutputMode_t>(0);
  }
  // if mode = 1, output_mode_num = 1
  if (mode == 1 || mode == 2) {
    output_mode_num = 5;
  } else if (mode == 3) {
    // mode = 3
    output_mode_num = 3;
  }

  if (output_boxes_ * output_mode_num != output_shape) {
    if (mode == 0) {
      // The valid output number is output_boxes_, the whole ram contains
      // output_shape number.
      /* __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
       *|i1|i2|i3|..|..|..|in|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
       *|__|__ __|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
       *|-----valid data-----|
       *|---------------output shape size------------------|
       */
      int rem_size = output_shape - output_boxes_;
      for (int i = 0; i < rem_size; i++) {
        cpu_fp32_output_[0][i + output_boxes_] = 0.0f;
        // mlu_fp32_output_[0][i + output_boxes_] = 0.0f;
      }
      return;
    } else if (mode == 1) {
      /* __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
       *|s1|x1|y1|x2|y2|..|sn|x1|y1|x2|y2|0 |0 |0 |0 |0 |0 |
       *|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
       *|----------valid data------------|
       *|---------------output shape size------------------|
       */
      int rem_size = output_shape - 5 * output_boxes_;
      for (int i = 0; i < rem_size; i++) {
        cpu_fp32_output_[0][i + output_boxes_ * 5] = 0.0f;
        // mlu_fp32_output_[0][i + output_boxes_ * 5] = 0.0f;
      }
      return;
    } else if (mode == 2) {
      /* __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
       *|s1|s2|s3|..|..|..|sn|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
       *|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
       * __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
       *|x1|x1|x1|..|..|..|x1|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
       *|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
       * __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
       *|y1|y1|y1|..|..|..|y1|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
       *|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
       * __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
       *|x2|x2|x2|..|..|..|x2|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |
       *|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
       * __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
       *|y2|y2|y2|..|..|..|y2|0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |0 |..|..|0 |0 |
       *|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
       *|----valid data------|
       *|------------------max_output_size-----------------|
       *                                                   |-----tail-----|
       *  the tail equals to output_shape minus 4 * max_output_size
       */
      VLOG(4) << "cpu_fp32";
      for (int i = 0; i < 5; i++) {
        int rem_size = max_output_boxes - output_boxes_;
        int per_rem_size =
            (i < 4 ? rem_size
                   : (output_shape - 4 * max_output_boxes - output_boxes_));
        for (int j = 0; j < per_rem_size; j++) {
          cpu_fp32_output_[0][i * max_output_boxes + output_boxes_ + j] = 0.0f;
          // mlu_fp32_output_[0][i * max_output_boxes + output_boxes_ + j] =
          // 0.0f;
        }
      }
      return;
    } else if (mode == 3) {
      /* __________ __________ ________ ___ __________ __________ ________ __ __
       *__ __ __ __
       *|batch_idx1|class_idx1|box_idx1|...|batch_idxn|class_idxn|box_idxn|0 |0
       *|0 |0 |0 |0 |
       *|__________|__________|________|___|__________|__________|________|__|__|__|__|__|__|
       *|-----------------------------valid data--------------------------|
       *|----------------------------------output shape
       *size--------------------------------|
       */
      int rem_size = output_shape - 3 * output_boxes_;
      for (int i = 0; i < rem_size; i++) {
        cpu_fp32_output_[0][i + output_boxes_ * 3] = 0.0f;
        if (!pad_to_max_output_size) {
          mlu_fp32_output_[0][i + output_boxes_ * 3] = 0.0f;
        }
      }
    } else {
      VLOG(4) << "Unsupport output mode!";
      return;
    }
  } else {
    VLOG(4) << "Don`t need to preprocess!";
    return;
  }
}

int64_t NmsExecutor::getTheoryOps() {
  VLOG(4) << "getTheoryOps";
  int cp_count = 24;
  int64_t theory_ops = parser_->getInputDataCount(1);
  auto input_layout = parser_->getProtoNode()->nms_param().input_layout();
  mluOpNmsAlgo_t algo =
      (mluOpNmsAlgo_t)parser_->getProtoNode()->nms_param().algo();
  float offset = parser_->getProtoNode()->nms_param().offset();
  if (offset != 0.0) {
    cp_count += 4;
  }

  VLOG(4) << "get the output boxes num:";
  int max_output_boxes =
      parser_->getProtoNode()->nms_param().max_output_boxes();
  float iou_thresh = parser_->getProtoNode()->nms_param().iou_threshold();
  mluOpNmsOutputMode_t mode =
      (mluOpNmsOutputMode_t)parser_->getProtoNode()->nms_param().mode();
  float confidence_threshold =
      parser_->getProtoNode()->nms_param().confidence_threshold();
  mluOpNmsBoxPointMode_t box_mode =
      (mluOpNmsBoxPointMode_t)parser_->getProtoNode()->nms_param().box_mode();
  float soft_nms_sigma = parser_->getProtoNode()->nms_param().soft_nms_sigma();
  mluOpNmsMethodMode_t method_mode =
      (mluOpNmsMethodMode_t)parser_->getProtoNode()->nms_param().method_mode();

  auto input_desc = tensor_desc_[0].tensor;
  auto input_conf_desc = tensor_desc_[1].tensor;
  int input_boxes_num = 0;
  int input_batches_num = 0;
  int input_classes_num = 0;
  int box_dim = 4;
  if (input_desc->getDim() == 2) {
    if (input_layout == 0) {
      input_boxes_num = input_desc->getDimIndex(0);
      box_dim = input_desc->getDimIndex(1);
    } else if (input_layout == 1) {
      input_boxes_num = input_desc->getDimIndex(1);
      box_dim = input_desc->getDimIndex(0);
    } else {
      VLOG(4) << "unsupport input layout now.";
    }
  } else {
    // assert input_desc->getDim() == 3
    input_batches_num = input_desc->getDimIndex(0);
    input_classes_num = input_conf_desc->getDimIndex(1);
    if (input_layout == 0) {
      // when layout is [boxes_num, 4], dims[0] represets the input number.
      input_boxes_num = input_desc->getDimIndex(1);
    } else if (input_layout == 1) {
      input_boxes_num = input_desc->getDimIndex(2);
    } else {
      VLOG(4) << "unsupport input layout now.";
    }
  }
  Device device = parser_->device();
  float *input_boxes = NULL;
  float *input_conf = NULL;
  if (device == Device::GPU) {
    auto boxes_dtype = input_desc->getDtype();
    auto conf_dtype = input_conf_desc->getDtype();
    int boxes_count_num = mluOpGetTensorElementNum(input_desc);
    int conf_count_num = mluOpGetTensorElementNum(input_conf_desc);
    float *boxes_host =
        (float *)cpu_runtime_.allocate(boxes_count_num * sizeof(float));
    float *conf_host =
        (float *)cpu_runtime_.allocate(conf_count_num * sizeof(float));
    castDataOut(data_vector_[0].host_ptr, boxes_dtype, (float *)boxes_host,
                MLUOP_DTYPE_FLOAT, boxes_count_num, NO_QUANT, 0, 1, 0);
    castDataOut(data_vector_[1].host_ptr, conf_dtype, (float *)conf_host,
                MLUOP_DTYPE_FLOAT, conf_count_num, NO_QUANT, 0, 1, 0);
    input_boxes = boxes_host;
    input_conf = conf_host;
  } else {
    input_boxes = parser_->getMetaTensor(0).cpu_ptr;
    input_conf = parser_->getMetaTensor(1).cpu_ptr;
  }

  // allocate memory for output, because when perf testing gtest framework won`t
  // allocate memory for output which resulting segmentation fault.
  auto output1_desc = parser_->getMetaTensor("output1").tensor;

  float *output_info =
      cpu_runtime_.allocate(new float[mluOpGetTensorElementNum(output1_desc)]);

  int total_output_boxes_num = 0;
  if (box_dim == 7) {
    // NMS3D
    nms3D_detection_cpu(output_info, total_output_boxes_num, input_boxes,
                        input_boxes_num, iou_thresh, input_layout);
  } else {
    for (int batch_idx = 0; batch_idx < input_batches_num; ++batch_idx) {
      for (int class_idx = 0; class_idx < input_classes_num; ++class_idx) {
        int boxes_offset = input_boxes_num * 4 * batch_idx;
        int conf_offset = input_classes_num * input_boxes_num * batch_idx +
                          input_boxes_num * class_idx;
        int output_offset = mode == 3 ? 3 * total_output_boxes_num : 0;
        int output_boxes_num = 0;
        nms_detection_cpu((float *)output_info + output_offset,
                          output_boxes_num, (float *)input_boxes + boxes_offset,
                          (float *)input_conf + conf_offset, input_boxes_num,
                          max_output_boxes, iou_thresh, confidence_threshold,
                          mode, input_layout, algo, offset, box_mode,
                          method_mode, soft_nms_sigma, batch_idx, class_idx);
        total_output_boxes_num += output_boxes_num;
      }
    }
  }
  cpu_runtime_.deallocate(output_info);
  cp_count *= total_output_boxes_num;
  theory_ops *= cp_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  if (device == Device::GPU) {
    cpu_runtime_.deallocate(input_boxes);
    cpu_runtime_.deallocate(input_conf);
  }
  return theory_ops;
}

}  // namespace mluoptest
