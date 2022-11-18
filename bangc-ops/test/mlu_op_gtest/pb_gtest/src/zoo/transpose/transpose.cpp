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
#include "transpose.h"
#include "transpose_cpu.h"

namespace mluoptest {

void TransposeExecutor::paramCheck() {
  if (parser_->getInputNum() != 1) {
    LOG(ERROR) << "transpose input number is wrong. ";
  }

  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "transpose output number is wrong. ";
  }
  flag_quant_mode_ = NO_QUANT;
}

void TransposeExecutor::compute() {
  VLOG(4) << "TransposeExecutor compute ";
  auto d = parser_->getProtoNode()->transpose_param().dim();
  auto x = tensor_desc_[0].tensor;
  auto y = tensor_desc_[1].tensor;
  auto x_ptr = data_vector_[0].device_ptr;
  // data_vector_[1].device_ptr = data_vector_[0].device_ptr;
  auto y_ptr = data_vector_[1].device_ptr;
  VLOG(4) << "call mluOpTranspose()";

  int permute[8];
  for (int i = 0; i < d; i++) {
    permute[i] = parser_->getProtoNode()->transpose_param().permute(i);
  }

  mluOpTransposeDescriptor_t trans_desc = nullptr;
  trans_desc = cpu_runtime_.allocate(mluOpCreateTransposeDescriptor,
                                     mluOpDestroyTransposeDescriptor);
  MLUOP_CHECK(mluOpSetTransposeDescriptor(trans_desc, d, permute));
  auto workspace = workspace_.at(0);
  interface_timer_.start();
  MLUOP_CHECK(mluOpTranspose_v2(handle_, trans_desc, x, x_ptr, y, y_ptr,
                                workspace, size_workspace_));
  interface_timer_.stop();
  cpu_runtime_.deallocate(trans_desc);
}

void TransposeExecutor::cpuCompute() {
  assert(parser_->getInputNum() == 1);
  assert(parser_->getOutputNum() == 1);
  auto d = parser_->getProtoNode()->transpose_param().dim();
  auto x = tensor_desc_[0].tensor;
  auto y = tensor_desc_[1].tensor;

  auto count1 = parser_->getInputDataCount(0);
  auto count2 = parser_->getOutputDataCount(0);
  assert(count1 == count2);

  int permute[8];
  for (int i = 0; i < d; i++) {
    permute[i] = parser_->getProtoNode()->transpose_param().permute(i);
  }
  mluOpTransposeDescriptor_t trans_desc;
  trans_desc = cpu_runtime_.allocate(mluOpCreateTransposeDescriptor,
                                     mluOpDestroyTransposeDescriptor);
  MLUOP_CHECK(mluOpSetTransposeDescriptor(trans_desc, d, permute));
  VLOG(4) << "call mluOpTransposeHost()";
  MLUOP_CHECK(mluOpTransposeCpu(trans_desc, x, cpu_fp32_input_[0], y,
                                cpu_fp32_output_[0]));
  cpu_runtime_.deallocate(trans_desc);
}
void TransposeExecutor::workspaceMalloc() {
  auto x_desc = tensor_desc_[0].tensor;
  auto d = parser_->getProtoNode()->transpose_param().dim();
  int permute[8];
  for (int i = 0; i < d; i++) {
    permute[i] = parser_->getProtoNode()->transpose_param().permute(i);
  }
  mluOpTransposeDescriptor_t trans_desc;
  trans_desc = cpu_runtime_.allocate(mluOpCreateTransposeDescriptor,
                                     mluOpDestroyTransposeDescriptor);
  MLUOP_CHECK(mluOpSetTransposeDescriptor(trans_desc, d, permute));

  MLUOP_CHECK(mluOpGetTransposeWorkspaceSize(handle_, x_desc, trans_desc,
                                             &size_workspace_));
  VLOG(4) << "Malloc workspace space.";
  void* temp = nullptr;
  if (size_workspace_ != 0) {
    temp = mlu_runtime_.allocate(size_workspace_);
  }
  workspace_.push_back(temp);
  VLOG(4) << "Malloc addr: " << temp << " , size: " << size_workspace_;
  eva_->setMluWorkspaceSize(size_workspace_);
}

void TransposeExecutor::workspaceFree() {
  auto temp = workspace_.at(0);
  mlu_runtime_.deallocate(temp);
}

int64_t TransposeExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->getOutputDataCount(0);
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
