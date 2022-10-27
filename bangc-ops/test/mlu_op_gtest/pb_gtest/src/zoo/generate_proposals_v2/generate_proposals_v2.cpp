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
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *******************************************************************************/

#include "generate_proposals_v2.h"

#include "generate_proposals_v2_impl.h"
#include "mlu_op.h"

namespace mluoptest {

void GenerateProposalsV2Executor::paramCheck() {
  if (!parser_->getProtoNode()->has_generate_proposals_v2_param()) {
    LOG(ERROR) << "Lose poly_nms_param. ";
  }
  GTEST_CHECK(parser_->inputs().size() == 5,
              "[GenerateProposalsV2Executor] input number is wrong. ");
  GTEST_CHECK(parser_->outputs().size() == 4,
              "[GenerateProposalsV2Executor] output number is wrong. ");
}

void GenerateProposalsV2Executor::workspaceMalloc() {
  size_t workspace_size = 0;
  auto tensor_scores = parser_->getMetaTensor("input1").tensor;
  mluOpGetGenerateProposalsV2WorkspaceSize(handle_, tensor_scores,
                                           &workspace_size);

  VLOG(4) << "Malloc workspace space.";
  void *temp = mlu_runtime_.allocate(workspace_size);
  workspace_.push_back(temp);
  VLOG(4) << "Malloc addr: " << temp << " , size: " << workspace_size;
  eva_->setMluWorkspaceSize(workspace_size);

  void *output_ptr = parser_->getMetaTensor("output1").dev_origin_ptr;
  size_t output_size = parser_->getMetaTensor("output1").size_in_bytes;
  GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMemset(output_ptr, 0, output_size));

  void *output_ptr1 = parser_->getMetaTensor("output2").dev_origin_ptr;
  size_t output_size1 = parser_->getMetaTensor("output2").size_in_bytes;
  GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMemset(output_ptr1, 0, output_size1));

  void *output_ptr2 = parser_->getMetaTensor("output3").dev_origin_ptr;
  size_t output_size2 = parser_->getMetaTensor("output3").size_in_bytes;
  GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMemset(output_ptr2, 0, output_size2));

  void *output_ptr3 = parser_->getMetaTensor("output4").dev_origin_ptr;
  size_t output_size3 = parser_->getMetaTensor("output4").size_in_bytes;
  GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMemset(output_ptr3, 0, output_size3));
}

void GenerateProposalsV2Executor::workspaceFree() {
  if (workspace_[0]) {
    VLOG(4) << "Free device workspace space.";
    GTEST_CHECK(CNRT_RET_SUCCESS == mlu_runtime_.deallocate(workspace_[0]));
    workspace_[0] = nullptr;
  }
}

void GenerateProposalsV2Executor::compute() {
  int pre_nms_top_n =
      parser_->getProtoNode()->generate_proposals_v2_param().pre_nms_top_n();
  int post_nms_top_n =
      parser_->getProtoNode()->generate_proposals_v2_param().post_nms_top_n();
  float nms_thresh =
      parser_->getProtoNode()->generate_proposals_v2_param().nms_thresh();
  float min_size =
      parser_->getProtoNode()->generate_proposals_v2_param().min_size();
  float eta = parser_->getProtoNode()->generate_proposals_v2_param().eta();
  bool pixel_offset =
      parser_->getProtoNode()->generate_proposals_v2_param().pixel_offset();

  VLOG(4) << "[mluOpGenerateProposalsV2] pre_nms_top_n: " << pre_nms_top_n;
  VLOG(4) << "[mluOpGenerateProposalsV2] post_nms_top_n: " << post_nms_top_n;
  VLOG(4) << "[mluOpGenerateProposalsV2] nms_thresh: " << nms_thresh;
  VLOG(4) << "[mluOpGenerateProposalsV2] min_size: " << min_size;
  VLOG(4) << "[mluOpGenerateProposalsV2] eta: " << eta;
  VLOG(4) << "[mluOpGenerateProposalsV2] pixel_offset: " << pixel_offset;

  // get tensor by name (in prototxt)
  auto tensor_scores = parser_->getMetaTensor("input1").tensor;
  auto tensor_deltas = parser_->getMetaTensor("input2").tensor;
  auto tensor_anchors = parser_->getMetaTensor("input3").tensor;
  auto tensor_variances = parser_->getMetaTensor("input4").tensor;
  auto tensor_img_shape = parser_->getMetaTensor("input5").tensor;

  auto scores_ptr = parser_->getMetaTensor("input1").dev_ptr;
  auto deltas_ptr = parser_->getMetaTensor("input2").dev_ptr;
  auto anchors_ptr = parser_->getMetaTensor("input3").dev_ptr;
  auto variances_ptr = parser_->getMetaTensor("input4").dev_ptr;
  auto img_shape_ptr = parser_->getMetaTensor("input5").dev_ptr;

  auto tensor_rois = parser_->getMetaTensor("output1").tensor;
  auto tensor_roi_probs = parser_->getMetaTensor("output2").tensor;
  auto tensor_rois_num = parser_->getMetaTensor("output3").tensor;

  auto rois_ptr = parser_->getMetaTensor("output1").dev_ptr;
  auto roi_probs_ptr = parser_->getMetaTensor("output2").dev_ptr;
  auto rois_num_ptr = parser_->getMetaTensor("output3").dev_ptr;
  auto rpn_rois_batch_size_ptr = parser_->getMetaTensor("output4").dev_ptr;

  VLOG(4) << "[mluOpGenerateProposalsV2] call "
             "mluOpGetGenerateProposalsV2WorkspaceSize()";
  size_t workspace_size = 0;
  MLUOP_CHECK(mluOpGetGenerateProposalsV2WorkspaceSize(handle_, tensor_scores,
                                                       &workspace_size));
  interface_timer_.start();

  VLOG(4) << "[mluOpGenerateProposalsV2] call "
             "mluOpGetGenerateProposalsV2WorkspaceSize()";

  MLUOP_CHECK(mluOpGenerateProposalsV2(
      handle_, pre_nms_top_n, post_nms_top_n, nms_thresh, min_size, eta,
      pixel_offset, tensor_scores, scores_ptr, tensor_deltas, deltas_ptr,
      tensor_img_shape, img_shape_ptr, tensor_anchors, anchors_ptr,
      tensor_variances, variances_ptr, workspace_[0], workspace_size,
      tensor_rois, rois_ptr, tensor_roi_probs, roi_probs_ptr, tensor_rois_num,
      rois_num_ptr, rpn_rois_batch_size_ptr));
  interface_timer_.stop();
  VLOG(4) << "[mluOpGenerateProposalsV2] mluOpGenerateProposalsV2 end.";
}

void GenerateProposalsV2Executor::cpuCompute() {
  const int pre_nms_top_n =
      parser_->getProtoNode()->generate_proposals_v2_param().pre_nms_top_n();
  const int post_nms_top_n =
      parser_->getProtoNode()->generate_proposals_v2_param().post_nms_top_n();
  const float nms_thresh =
      parser_->getProtoNode()->generate_proposals_v2_param().nms_thresh();
  const float min_size =
      parser_->getProtoNode()->generate_proposals_v2_param().min_size();
  const float eta =
      parser_->getProtoNode()->generate_proposals_v2_param().eta();
  bool pixel_offset =
      parser_->getProtoNode()->generate_proposals_v2_param().pixel_offset();

  auto tensor_scores = parser_->getMetaTensor("input1").tensor;

  const int N = tensor_scores->dims[0];
  const int H = tensor_scores->dims[1];
  const int W = tensor_scores->dims[2];
  const int A = tensor_scores->dims[3];

  auto scores_ptr = parser_->getMetaTensor("input1").cpu_ptr;
  auto deltas_ptr = parser_->getMetaTensor("input2").cpu_ptr;
  auto anchors_ptr = parser_->getMetaTensor("input3").cpu_ptr;
  auto variances_ptr = parser_->getMetaTensor("input4").cpu_ptr;
  auto img_shape_ptr = parser_->getMetaTensor("input5").cpu_ptr;

  auto rois_ptr = parser_->getMetaTensor("output1").cpu_ptr;
  auto roi_probs_ptr = parser_->getMetaTensor("output2").cpu_ptr;
  auto rois_num_ptr = parser_->getMetaTensor("output3").cpu_ptr;
  auto rois_batch_size_ptr = parser_->getMetaTensor("output4").cpu_ptr;

  VLOG(4) << "[mluOpGenerateProposalsV2] cpu compute start. ";
  GenerateProposalsV2::generateProposalsV2CPUImpl(
      scores_ptr, deltas_ptr, img_shape_ptr, anchors_ptr, variances_ptr,
      pre_nms_top_n, post_nms_top_n, nms_thresh, min_size, eta, pixel_offset, N,
      H, W, A, rois_ptr, roi_probs_ptr, rois_num_ptr, rois_batch_size_ptr);
  VLOG(4) << "[mluOpGenerateProposalsV2] cpu compute end, rois_batch_size: "
          << rois_batch_size_ptr[0];
}

int64_t GenerateProposalsV2Executor::getTheoryOps() {
  VLOG(4) << "getTheoryOps";
  //   int dims = parser_->getMetaTensor("input1").tensor->dims[0];
  auto tensor_scores = parser_->getMetaTensor("input1").tensor;
  const int N = tensor_scores->dims[0];
  const int H = tensor_scores->dims[1];
  const int W = tensor_scores->dims[2];
  const int A = tensor_scores->dims[3];
  int64_t theory_ops = 39 * N * A * H * W;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
