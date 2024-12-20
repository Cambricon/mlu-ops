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
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "masked_im2col_forward.h"

namespace mluoptest {

void MaskedIm2colForwardExecutor::printDataInfo() {
  VLOG(4) << "############################### printfDataInfo() Begin ##";
  VLOG(4) << "# batchs_:         " << batchs_;
  VLOG(4) << "# height_:         " << height_;
  VLOG(4) << "# width_:          " << width_;
  VLOG(4) << "# channels_:       " << channels_;
  VLOG(4) << "# kernel_h:       " << kernel_h;
  VLOG(4) << "# kernel_w:       " << kernel_w;
  VLOG(4) << "# pad_h:          " << pad_h;
  VLOG(4) << "# pad_w:          " << pad_w;
  VLOG(4) << "# mask_cnt_:       " << mask_cnt_;
  VLOG(4) << "############################### printfDataInfo() End ##";
}

void MaskedIm2colForwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->getInputNum() == 3,
              "masked_im2col_forward input number is wrong.");
  GTEST_CHECK(parser_->getOutputNum() == 1,
              "masked_im2col_forward output number is wrong.");
}
void MaskedIm2colForwardExecutor::init() {
  auto input_desc = tensor_desc_[0].tensor;
  auto mask_desc = tensor_desc_[1].tensor;
  batchs_ = input_desc->getDimIndex(0);
  channels_ = input_desc->getDimIndex(1);
  height_ = input_desc->getDimIndex(2);
  width_ = input_desc->getDimIndex(3);
  mask_cnt_ = mask_desc->getDimIndex(0);
  auto masked_im2col_forward_proto_desc =
      parser_->getProtoNode()->masked_im2col_forward_param();
  kernel_h = masked_im2col_forward_proto_desc.kernel_h();
  kernel_w = masked_im2col_forward_proto_desc.kernel_w();
  pad_h = masked_im2col_forward_proto_desc.pad_h();
  pad_w = masked_im2col_forward_proto_desc.pad_w();
}

void MaskedIm2colForwardExecutor::workspaceMalloc() {
  paramCheck();
  init();
  printDataInfo();
  auto tensor_feature = tensor_desc_[0].tensor;
  auto tensor_mask_h_idx = tensor_desc_[1].tensor;
  auto tensor_mask_w_idx = tensor_desc_[2].tensor;
  auto tensor_data_col = tensor_desc_[3].tensor;
  auto dev_feature = data_vector_[0].device_ptr;
  auto dev_mask_h_idx = data_vector_[1].device_ptr;
  auto dev_mask_w_idx = data_vector_[2].device_ptr;
  auto dev_data_col = data_vector_[3].device_ptr;
  MLUOP_CHECK(mluOpGetMaskedIm2colForwardWorkspaceSize(
      handle_, tensor_feature, tensor_mask_h_idx, tensor_mask_w_idx, kernel_h,
      kernel_w, tensor_data_col, &workspace_size_));
  eva_->setMluWorkspaceSize(workspace_size_);
  if (workspace_size_ > 0) {
    workspace_ = mlu_runtime_.allocate(workspace_size_);
  }
}

void MaskedIm2colForwardExecutor::workspaceFree() {
  if (workspace_ != nullptr) {
    VLOG(4) << "MaskedIm2colForwardExecutor free workspace memory.";
    mlu_runtime_.deallocate(workspace_);
  }
}

void MaskedIm2colForwardExecutor::compute() {
  VLOG(4) << "MaskedIm2colForwardExecutor compute ";
  auto tensor_feature = tensor_desc_[0].tensor;
  auto tensor_mask_h_idx = tensor_desc_[1].tensor;
  auto tensor_mask_w_idx = tensor_desc_[2].tensor;
  auto tensor_data_col = tensor_desc_[3].tensor;
  auto dev_feature = data_vector_[0].device_ptr;
  auto dev_mask_h_idx = data_vector_[1].device_ptr;
  auto dev_mask_w_idx = data_vector_[2].device_ptr;
  auto dev_data_col = data_vector_[3].device_ptr;
  VLOG(4) << "call mluOpMaskedIm2colForward().";
  interface_timer_.start();

  MLUOP_CHECK(mluOpMaskedIm2colForward(
      handle_, tensor_feature, dev_feature, tensor_mask_h_idx, dev_mask_h_idx,
      tensor_mask_w_idx, dev_mask_w_idx, kernel_h, kernel_w, pad_h, pad_w,
      workspace_, workspace_size_, tensor_data_col, dev_data_col));

  interface_timer_.stop();
  VLOG(4) << "call mluOpMaskedIm2colForward() finish!";
}

void MaskedIm2colForwardExecutor::cpuCompute() {
  int count = channels_ * mask_cnt_;
  for (int index = 0; index < count; ++index) {
    const int m_index = index % mask_cnt_;
    const int h_col = cpu_fp32_input_[1][m_index];
    const int w_col = cpu_fp32_input_[2][m_index];
    const int c_im = index / mask_cnt_;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col - pad_h;
    const int w_offset = w_col - pad_w;
    float *data_col_ptr = cpu_fp32_output_[0] + c_col * mask_cnt_ + m_index;
    for (int i = 0; i < kernel_h; ++i) {
      int h_im = h_offset + i;
      for (int j = 0; j < kernel_w; ++j) {
        int w_im = w_offset + j;
        if (h_im >= 0 && w_im >= 0 && h_im < height_ && w_im < width_) {
          *data_col_ptr =
              cpu_fp32_input_[0][(c_im * height_ + h_im) * width_ + w_im];
        } else {
          *data_col_ptr = 0.0;
        }
        data_col_ptr += mask_cnt_;
      }
    }
  }
}

}  // namespace mluoptest
