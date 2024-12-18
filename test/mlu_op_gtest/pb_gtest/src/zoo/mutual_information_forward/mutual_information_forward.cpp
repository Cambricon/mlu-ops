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

#include "mutual_information_forward.h"

namespace mluoptest {

void MutualInformationForwardExecutor::initParam() {
  px_desc_ = tensor_desc_[0].tensor;
  py_desc_ = tensor_desc_[1].tensor;
  float *host_p_in = nullptr;
  if (tensor_desc_.size() == max_tensor_num_) {
    opt_boundary_desc_ = tensor_desc_[2].tensor;
    p_desc_ = tensor_desc_[3].tensor;
    ans_desc_ = tensor_desc_[5].tensor;
    host_p_in = (float *)data_vector_[3].host_ptr;
  } else {
    p_desc_ = tensor_desc_[2].tensor;
    ans_desc_ = tensor_desc_[4].tensor;

    host_p_in = (float *)data_vector_[2].host_ptr;
  }

  B_ = px_desc_->getDimIndex(0);
  S_ = px_desc_->getDimIndex(1);
  T_ = py_desc_->getDimIndex(2);

  px_index_ = MutualInformationForward::Index3D(S_, T_ + 1);
  py_index_ = MutualInformationForward::Index3D(S_ + 1, T_);
  p_index_ = MutualInformationForward::Index3D(S_ + 1, T_ + 1);

  int p_size = B_ * (S_ + 1) * (T_ + 1);
  p_in_ = (float *)cpu_runtime_.allocate(p_size * sizeof(float));
  memcpy(p_in_, host_p_in, p_size  * sizeof(float));
}

void MutualInformationForwardExecutor::workspaceMalloc() {
  initParam();
  MLUOP_CHECK(mluOpGetMutualInformationForwardWorkspaceSize(
      handle_, px_desc_, py_desc_, opt_boundary_desc_, p_desc_, ans_desc_,
      &workspace_size_));
  void *dev_workspace = nullptr;
  if (workspace_size_ != 0) {
    dev_workspace = mlu_runtime_.allocate(workspace_size_);
    eva_->setMluWorkspaceSize(workspace_size_);
  }
  workspace_.push_back(dev_workspace);
}

void MutualInformationForwardExecutor::workspaceFree() {
  for (auto ptr : workspace_) {
    if (ptr) {
      mlu_runtime_.deallocate(ptr);
    }
  }
}

void MutualInformationForwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->getInputNum() == 4 || parser_->getInputNum() == 3,
              "[MutualInformationForwardExecutor] Input number is wrong.");
  GTEST_CHECK(parser_->getOutputNum() == 2,
              "[MutualInformationForwardExecutor] Output number is wrong.");
}

void MutualInformationForwardExecutor::compute() {
  VLOG(4) << "MutualInformationForwardExecutor compute.";
  void *dev_px = data_vector_[0].device_ptr;
  void *dev_py = data_vector_[1].device_ptr;
  void *dev_opt_boundary = nullptr;
  void *dev_p = nullptr;
  void *dev_ans = nullptr;

  if (tensor_desc_.size() == max_tensor_num_) {
    dev_opt_boundary = data_vector_[2].device_ptr;
    dev_p = data_vector_[3].device_ptr;
    dev_ans = data_vector_[5].device_ptr;
  } else {
    dev_p = data_vector_[2].device_ptr;
    dev_ans = data_vector_[4].device_ptr;
  }

  interface_timer_.start();
  MLUOP_CHECK(mluOpMutualInformationForward(
      handle_, px_desc_, dev_px, py_desc_, dev_py, opt_boundary_desc_,
      dev_opt_boundary, p_desc_, dev_p, workspace_[0], workspace_size_,
      ans_desc_, dev_ans));
  interface_timer_.stop();
}

void MutualInformationForwardExecutor::setMiscellaneousParam() {
  if (tensor_desc_.size() == max_tensor_num_) {
    data_vector_[3].alsoServeAsOutput();
    data_vector_[4].onlyServeAsInput();
    data_vector_[5].alsoServeAsOutput();
  } else {
    data_vector_[2].alsoServeAsOutput();
    data_vector_[3].onlyServeAsInput();
    data_vector_[4].alsoServeAsOutput();
  }
}

void MutualInformationForwardExecutor::cpuCompute() {
  float *host_px = (float *)data_vector_[0].host_ptr;
  float *host_py = (float *)data_vector_[1].host_ptr;
  int64_t *host_opt_boundary = nullptr;

  if (tensor_desc_.size() == max_tensor_num_) {
    host_opt_boundary = (int64_t *)data_vector_[2].host_ptr;
  }

  float *host_p_out = cpu_fp32_output_[0];
  memcpy(host_p_out, p_in_, B_ * (S_ + 1) * (T_ + 1) * sizeof(float));
  float *host_ans = cpu_fp32_output_[1];

  int s_begin = 0;
  int t_begin = 0;
  int s_end = S_;
  int t_end = T_;
  for (int b = 0; b < B_; ++b) {
    if (host_opt_boundary != nullptr) {
      s_begin = (int)host_opt_boundary[b * 4];
      t_begin = (int)host_opt_boundary[b * 4 + 1];
      s_end = (int)host_opt_boundary[b * 4 + 2];
      t_end = (int)host_opt_boundary[b * 4 + 3];
    }
    computeMutualInformation(b, s_begin, s_end, t_begin, t_end, host_px,
                             host_py, host_p_out, host_ans);
  }

  if (p_in_) {
    cpu_runtime_.deallocate(p_in_);
  }
}

float MutualInformationForwardExecutor::logAdd(float x, float y) {
  float diff;
  if (x < y) {
    diff = x - y;
    x = y;
    theory_ops_ += 2;
  } else {
    diff = y - x;
    theory_ops_++;
  }

  if (diff >= min_log_diff_float) {
    float res;
    res = x + log1pf(expf(diff));
    theory_ops_ += 4;
    return res;
  }

  return x;
}

void MutualInformationForwardExecutor::computeMutualInformation(
    const int b, const int s_begin, const int s_end, const int t_begin,
    const int t_end, float *px, float *py, float *p, float *ans) {
  p[p_index_(b, s_begin, t_begin)] = 0;
  theory_ops_++;

  for (int s = s_begin + 1; s <= s_end; ++s) {
    p[p_index_(b, s, t_begin)] = logAdd(p[p_index_(b, s - 1, t_begin)] +
                                        px[px_index_(b, s - 1, t_begin)],
                                        -INFINITY);
    theory_ops_++;
  }

  for (int t = t_begin + 1; t <= t_end; ++t) {
    p[p_index_(b, s_begin, t)] = logAdd(-INFINITY,
                                        p[p_index_(b, s_begin, t - 1)] +
                                        py[py_index_(b, s_begin, t - 1)]);
    theory_ops_++;
  }

  for (int s = s_begin + 1; s <= s_end; ++s) {
    for (int t = t_begin + 1; t <= t_end; ++t) {
      p[p_index_(b, s, t)] = logAdd(
          p[p_index_(b, s - 1, t)] + px[px_index_(b, s - 1, t)],
          p[p_index_(b, s, t - 1)] + py[py_index_(b, s, t - 1)]);
      theory_ops_ += 2;
    }
  }

  ans[b] = p[p_index_(b, s_end, t_end)];
  theory_ops_++;
}

int64_t MutualInformationForwardExecutor::getTheoryOps() {
  if (parser_->device() != Device::CPU) {
    theory_ops_ = 0;
    if (exe_config_->mlu_only) {
      baselineOutputMalloc();
    }
    cpuCompute();
  }
  return theory_ops_;
}

}  // namespace mluoptest
