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
#include "masked_col2im_forward.h"

namespace mluoptest {

void MaskedCol2imForwardExecutor::printDataInfo() {
  VLOG(4) << "############################### printfDataInfo() Begin ##";
  VLOG(4) << "# batchs:         " << batchs;
  VLOG(4) << "# height:         " << height;
  VLOG(4) << "# width:          " << width;
  VLOG(4) << "# channels:       " << channels;
  VLOG(4) << "# mask_cnt:       " << mask_cnt;
  VLOG(4) << "############################### printfDataInfo() End ##";
}

void MaskedCol2imForwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->getInputNum() == 3,
              "masked_col2im_forward input number is wrong.");
  GTEST_CHECK(parser_->getOutputNum() == 1,
              "masked_col2im_forward output number is wrong.");
}
void MaskedCol2imForwardExecutor::init() {
  auto col_desc = tensor_desc_[0].tensor;
  auto im_desc = tensor_desc_[3].tensor;
  batchs = im_desc->dims[0];
  channels = im_desc->dims[1];
  height = im_desc->dims[2];
  width = im_desc->dims[3];
  mask_cnt = col_desc->dims[1];
}

void MaskedCol2imForwardExecutor::workspaceMalloc() {
  paramCheck();
  init();
  printDataInfo();
  auto tensor_col = tensor_desc_[0].tensor;
  auto tensor_mask_h_idx = tensor_desc_[1].tensor;
  auto tensor_mask_w_idx = tensor_desc_[2].tensor;
  auto tensor_im = tensor_desc_[3].tensor;
  MLUOP_CHECK(mluOpGetMaskedCol2imForwardWorkspaceSize(
      handle_, tensor_col, tensor_mask_h_idx, tensor_mask_w_idx, tensor_im,
      &workspace_size_));
  eva_->setMluWorkspaceSize(workspace_size_);
  if (workspace_size_ > 0) {
    workspace_ = mlu_runtime_.allocate(workspace_size_);
  }
}

void MaskedCol2imForwardExecutor::workspaceFree() {
  if (workspace_ != nullptr) {
    VLOG(4) << "MaskedCol2imForwardExecutor free workspace memory.";
    mlu_runtime_.deallocate(workspace_);
    workspace_ = nullptr;
  }
}

void MaskedCol2imForwardExecutor::compute() {
  VLOG(4) << "MaskedCol2imForwardExecutor compute.";
  auto tensor_col = tensor_desc_[0].tensor;
  auto tensor_mask_h_idx = tensor_desc_[1].tensor;
  auto tensor_mask_w_idx = tensor_desc_[2].tensor;
  auto tensor_im = tensor_desc_[3].tensor;
  auto dev_col = data_vector_[0].device_ptr;
  auto dev_mask_h_idx = data_vector_[1].device_ptr;
  auto dev_mask_w_idx = data_vector_[2].device_ptr;
  auto dev_im = data_vector_[3].device_ptr;
  VLOG(4) << "Call mluOpMaskedCol2imForward().";
  interface_timer_.start();

  MLUOP_CHECK(mluOpMaskedCol2imForward(
      handle_, tensor_col, dev_col, tensor_mask_h_idx, dev_mask_h_idx,
      tensor_mask_w_idx, dev_mask_w_idx, workspace_size_, workspace_, tensor_im,
      dev_im));
  interface_timer_.stop();
  VLOG(4) << "call mluOpMaskedCol2imForward() finish!";
}

void MaskedCol2imForwardExecutor::cpuCompute() {
  int output_size = parser_->getOutputDataCount(0);
  for (int index = 0; index < output_size; index++) {
    cpu_fp32_output_[0][index] = 0;
  }
  int count = channels * mask_cnt;
  for (int index = 0; index < count; ++index) {
    const int m_index = index % mask_cnt;
    const int h_im = cpu_fp32_input_[1][m_index];
    const int w_im = cpu_fp32_input_[2][m_index];
    const int c_im = index / mask_cnt;
    cpu_fp32_output_[0][(c_im * height + h_im) * width + w_im] =
        cpu_fp32_input_[0][index];
  }
}

int64_t MaskedCol2imForwardExecutor::getTheoryIoSize() {
  int input_size = parser_->getInputDataCount(0);
  auto dtype = tensor_desc_[0].tensor->dtype;
  int dsize = 0;
  if (dtype == MLUOP_DTYPE_FLOAT) {
    dsize = 4;
  } else if (dtype == MLUOP_DTYPE_HALF) {
    dsize = 2;
  } else {
    GTEST_CHECK(false, "MaskedCol2imForward don't support this dtype.");
  }
  int64_t theory_io_size = parser_->getInputDataCount(0) * dsize;
  VLOG(4) << "getTheoryIoSize: " << theory_io_size << " Bytes.";
  return theory_io_size;
}

}  // namespace mluoptest
