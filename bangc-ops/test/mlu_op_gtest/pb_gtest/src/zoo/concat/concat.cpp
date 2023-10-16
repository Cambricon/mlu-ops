/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "concat.h"

#include <memory>

namespace mluoptest {

void ConcatExecutor::paramCheck() {
  assert(parser_->getInputNum() > 0);
  assert(parser_->getOutputNum() == 1);
  if (!parser_->getProtoNode()->has_concat_param()) {
    LOG(ERROR) << "Lose concat param. ";
  }
}

void ConcatExecutor::workspaceMalloc() {
  input_num_ = parser_->getInputNum();
  MLUOP_CHECK(
      mluOpGetConcatWorkspaceSize(handle_, input_num_, &workspace_size_));
  VLOG(4) << "Malloc workspace space.";
  void *temp = mlu_runtime_.allocate(workspace_size_);
  workspace_.push_back(temp);
  VLOG(4) << "Malloc addr: " << temp << " , size: " << workspace_size_;

  eva_->setMluWorkspaceSize(workspace_size_);
}

void ConcatExecutor::compute() {
  VLOG(4) << "ConcatExecutor compute ";
  if (!parser_->getProtoNode()->has_concat_param()) {
    LOG(ERROR) << "Lose concat param. ";
  }
  axis_ = parser_->getProtoNode()->concat_param().axis();

  std::vector<void *> pdev_input_h(input_num_);
  for (int i = 0; i < input_num_; i++) {
    pdev_input_h[i] = data_vector_[i].device_ptr;
  }

  mluOpTensorDescriptor_t *in_desc =
      cpu_runtime_.allocate(new mluOpTensorDescriptor_t[input_num_]);
  for (int i = 0; i < input_num_; i++) {
    in_desc[i] = tensor_desc_[i].tensor;
  }
  auto out_desc = tensor_desc_[input_num_].tensor;

  VLOG(4) << "call mluOpconcatTensor()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpConcat(handle_, input_num_, axis_, in_desc,
                          pdev_input_h.data(), workspace_[0], workspace_size_,
                          out_desc, data_vector_[input_num_].device_ptr));
  interface_timer_.stop();

  if (in_desc) {
    cpu_runtime_.deallocate(in_desc);
    in_desc = nullptr;
  }
}

void ConcatExecutor::workspaceFree() {
  VLOG(4) << "Free device workspace space.";
  if (workspace_[0] != nullptr) {
    mlu_runtime_.deallocate(workspace_[0]);
  }
}

void ConcatExecutor::cpuConcat(std::vector<TensorPair> input_desc,
                               std::vector<float *> input, int input_num,
                               int axis_t, float *output) {
  int dim_num = input_desc[0].tensor->dim;
  size_t axis = axis_t < 0 ? axis_t + dim_num : axis_t;
  size_t high_size = 1;
  for (size_t i = 0; i < axis; i++) {
    high_size *= input_desc[0].tensor->dims[i];
  }
  size_t low_low_size = 1;
  for (size_t i = dim_num - 1; i > axis; i--) {
    low_low_size *= input_desc[0].tensor->dims[i];
  }
  size_t *low_sizes = cpu_runtime_.allocate(new size_t[input_num]);
  for (size_t i = 0; i < input_num; i++) {
    low_sizes[i] = input_desc[i].tensor->dims[axis] * low_low_size;
  }

  size_t offset = 0;
  for (size_t j = 0; j < high_size; j++) {
    for (size_t i = 0; i < input_num; i++) {
      memcpy(output + offset, input[i] + j * low_sizes[i],
             low_sizes[i] * sizeof(float));
      offset += low_sizes[i];
    }
  }
  cpu_runtime_.deallocate(low_sizes);
}

void ConcatExecutor::cpuCompute() {
  assert(parser_->getInputNum() > 0);
  assert(parser_->getOutputNum() == 1);

  cpuConcat(tensor_desc_, cpu_fp32_input_, input_num_, axis_,
            cpu_fp32_output_[0]);
}

int64_t ConcatExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->getOutputDataCount(0);
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
