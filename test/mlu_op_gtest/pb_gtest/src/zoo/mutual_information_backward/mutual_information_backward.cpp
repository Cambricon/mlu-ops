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

#include "mutual_information_backward.h"

namespace mluoptest {

void MutualInformationBackwardExecutor::initParam() {
  overwrite_ans_grad_ = parser_->getProtoNode()
                            ->mutual_information_backward_param()
                            .overwrite_ans_grad();
  px_desc_ = tensor_desc_[0].tensor;
  py_desc_ = tensor_desc_[1].tensor;
  float *host_ans_grad_in = nullptr;
  if (tensor_desc_.size() == max_tensor_num_) {
    opt_boundary_desc_ = tensor_desc_[2].tensor;
    p_desc_ = tensor_desc_[3].tensor;
    ans_grad_desc_ = tensor_desc_[4].tensor;
    px_grad_desc_ = tensor_desc_[6].tensor;
    py_grad_desc_ = tensor_desc_[7].tensor;

    host_ans_grad_in = (float *)data_vector_[4].host_ptr;
  } else {
    p_desc_ = tensor_desc_[2].tensor;
    ans_grad_desc_ = tensor_desc_[3].tensor;
    px_grad_desc_ = tensor_desc_[5].tensor;
    py_grad_desc_ = tensor_desc_[6].tensor;

    host_ans_grad_in = (float *)data_vector_[3].host_ptr;
  }

  B_ = px_desc_->getDimIndex(0);
  S_ = px_desc_->getDimIndex(1);
  T_ = py_desc_->getDimIndex(2);

  px_index_ = Index3D(S_, T_ + 1);
  py_index_ = Index3D(S_ + 1, T_);
  p_index_ = Index3D(S_ + 1, T_ + 1);

  ans_grad_in_ = (float *)cpu_runtime_.allocate(B_ * sizeof(float));
  memcpy(ans_grad_in_, host_ans_grad_in, B_ * sizeof(float));
}

void MutualInformationBackwardExecutor::workspaceMalloc() {
  initParam();
  MLUOP_CHECK(mluOpGetMutualInformationBackwardWorkspaceSize(
      handle_, px_desc_, py_desc_, opt_boundary_desc_, p_desc_, ans_grad_desc_,
      overwrite_ans_grad_, &workspace_size_));
  void *dev_workspace = nullptr;
  if (workspace_size_ != 0) {
    dev_workspace = mlu_runtime_.allocate(workspace_size_);
    eva_->setMluWorkspaceSize(workspace_size_);
  }
  workspace_.push_back(dev_workspace);
}

void MutualInformationBackwardExecutor::workspaceFree() {
  for (auto ptr : workspace_) {
    if (ptr) {
      mlu_runtime_.deallocate(ptr);
    }
  }
}

void MutualInformationBackwardExecutor::paramCheck() {
  GTEST_CHECK(parser_->getProtoNode()->has_mutual_information_backward_param(),
              "[MutualInformationBackwardExecutor] Missing param.");
  GTEST_CHECK(parser_->getInputNum() == 4 || parser_->getInputNum() == 5,
              "[MutualInformationBackwardExecutor] Input number is wrong.");
  GTEST_CHECK(parser_->getOutputNum() == 3,
              "[MutualInformationBackwardExecutor] Output number is wrong.");
}

void MutualInformationBackwardExecutor::compute() {
  VLOG(4) << "MutualInformationBackwardExecutor compute.";
  void *dev_px = data_vector_[0].device_ptr;
  void *dev_py = data_vector_[1].device_ptr;
  void *dev_opt_boundary = nullptr;
  void *dev_p = nullptr;
  void *dev_ans_grad = nullptr;
  void *dev_px_grad = nullptr;
  void *dev_py_grad = nullptr;

  if (tensor_desc_.size() == max_tensor_num_) {
    dev_opt_boundary = data_vector_[2].device_ptr;
    dev_p = data_vector_[3].device_ptr;
    dev_ans_grad = data_vector_[4].device_ptr;
    dev_px_grad = data_vector_[6].device_ptr;
    dev_py_grad = data_vector_[7].device_ptr;
  } else {
    dev_p = data_vector_[2].device_ptr;
    dev_ans_grad = data_vector_[3].device_ptr;
    dev_px_grad = data_vector_[5].device_ptr;
    dev_py_grad = data_vector_[6].device_ptr;
  }

  interface_timer_.start();
  MLUOP_CHECK(mluOpMutualInformationBackward(
      handle_, px_desc_, dev_px, py_desc_, dev_py, opt_boundary_desc_,
      dev_opt_boundary, p_desc_, dev_p, ans_grad_desc_, dev_ans_grad,
      overwrite_ans_grad_, workspace_[0], workspace_size_, px_grad_desc_,
      dev_px_grad, py_grad_desc_, dev_py_grad));
  interface_timer_.stop();
}

void MutualInformationBackwardExecutor::setMiscellaneousParam() {
  if (tensor_desc_.size() == max_tensor_num_) {
    data_vector_[4].alsoServeAsOutput();
    data_vector_[5].onlyServeAsInput();
    data_vector_[6].alsoServeAsOutput();
    data_vector_[7].alsoServeAsOutput();
  } else {
    data_vector_[3].alsoServeAsOutput();
    data_vector_[4].onlyServeAsInput();
    data_vector_[5].alsoServeAsOutput();
    data_vector_[6].alsoServeAsOutput();
  }
}

void MutualInformationBackwardExecutor::cpuCompute() {
  float *host_px = (float *)data_vector_[0].host_ptr;
  float *host_py = (float *)data_vector_[1].host_ptr;
  int64_t *host_opt_boundary = nullptr;
  float *host_p = nullptr;

  float *host_ans_grad_in = nullptr;
  if (tensor_desc_.size() == max_tensor_num_) {
    host_opt_boundary = (int64_t *)data_vector_[2].host_ptr;
    host_p = (float *)data_vector_[3].host_ptr;
    host_ans_grad_in = (float *)data_vector_[4].host_ptr;
  } else {
    host_p = (float *)data_vector_[2].host_ptr;
    host_ans_grad_in = (float *)data_vector_[3].host_ptr;
  }

  float *host_ans_grad_out = nullptr;
  float *host_px_grad = nullptr;
  float *host_py_grad = nullptr;
  host_ans_grad_out = cpu_fp32_output_[0];
  memcpy(host_ans_grad_out, ans_grad_in_, B_ * sizeof(float));

  host_px_grad = cpu_fp32_output_[1];
  host_py_grad = cpu_fp32_output_[2];

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

    computeTerm1AndTerm2(b, s_begin, s_end, t_begin, t_end, host_px, host_py,
                         host_p);
    computePGrad(b, s_begin, s_end, t_begin, t_end, host_px, host_py, host_p,
                 ans_grad_in_, host_ans_grad_out);
    computePxGradAndPyGrad(b, s_begin, s_end, t_begin, t_end, host_px, host_py,
                           host_p, host_px_grad, host_py_grad);
  }

  if (ans_grad_in_) {
    cpu_runtime_.deallocate(ans_grad_in_);
  }
}

float MutualInformationBackwardExecutor::safeExp(float x) {
  if (x - x != 0) {
    theory_ops_ += 2;
    return 0;
  } else {
    float ans = std::exp(x);
    theory_ops_ += 5;
    if (ans - ans != 0.0) {
      return 0;
    }
    return ans;
  }
}

void MutualInformationBackwardExecutor::computeTerm1AndTerm2(
    const int b, const int s_begin, const int s_end, const int t_begin,
    const int t_end, float *px, float *py, float *p) {
  for (int s = s_begin; s <= s_end; ++s) {
    for (int t = t_begin; t <= t_end; ++t) {
      theory_ops_++;
      if (p[p_index_(b, s, t)] < large_neg_num_) {
        p[p_index_(b, s, t)] = large_neg_num_;
        theory_ops_++;
      }
    }
  }

  for (int s = s_begin; s <= s_end; ++s) {
    for (int t = t_begin; t <= t_end; ++t) {
      if (s < s_end) {
        // compute term1
        px[px_index_(b, s, t)] =
            safeExp(p[p_index_(b, s, t)] + px[px_index_(b, s, t)] -
                    p[p_index_(b, s + 1, t)]);
        theory_ops_ += 2;
      }

      if (t < t_end) {
        // compute term2
        py[py_index_(b, s, t)] =
            safeExp(p[p_index_(b, s, t)] + py[py_index_(b, s, t)] -
                    p[p_index_(b, s, t + 1)]);
        theory_ops_ += 2;
      }
    }
  }
}

void MutualInformationBackwardExecutor::computePGrad(
    const int b, const int s_begin, const int s_end, const int t_begin,
    const int t_end, float *term1, float *term2, float *p, float *ans_grad_in,
    float *ans_grad_out) {
  // compute p_grad[b][s_end][t_end]
  p[p_index_(b, s_end, t_end)] = ans_grad_in[b];
  theory_ops_++;

  // compute p_grad[b][s_end][0:t_end]
  for (int t = t_end - 1; t >= t_begin; --t) {
    p[p_index_(b, s_end, t)] =
        term2[py_index_(b, s_end, t)] * p[p_index_(b, s_end, t + 1)];
    theory_ops_++;
  }

  // compute p_grad[b][0:s_end][t_end]
  for (int s = s_end - 1; s >= s_begin; --s) {
    p[p_index_(b, s, t_end)] =
        term1[px_index_(b, s, t_end)] * p[p_index_(b, s + 1, t_end)];
    theory_ops_++;
  }

  for (int s = s_end - 1; s >= s_begin; --s) {
    for (int t = t_end - 1; t >= t_begin; --t) {
      p[p_index_(b, s, t)] =
          term1[px_index_(b, s, t)] * p[p_index_(b, s + 1, t)] +
          term2[py_index_(b, s, t)] * p[p_index_(b, s, t + 1)];
      theory_ops_ += 3;
    }
  }

  if (overwrite_ans_grad_ && s_begin <= s_end && t_begin <= t_end) {
    ans_grad_out[b] = p[p_index_(b, s_begin, t_begin)];
    theory_ops_++;
  }
}

void MutualInformationBackwardExecutor::computePxGradAndPyGrad(
    const int b, const int s_begin, const int s_end, const int t_begin,
    const int t_end, float *term1, float *term2, float *p_grad, float *px_grad,
    float *py_grad) {
  for (int s = s_begin; s <= s_end; ++s) {
    for (int t = t_begin; t <= t_end; ++t) {
      if (s < s_end) {
        // compute px_grad
        px_grad[px_index_(b, s, t)] =
            p_grad[p_index_(b, s + 1, t)] * term1[px_index_(b, s, t)];
      }

      if (t < t_end) {
        // compute py_grad
        py_grad[py_index_(b, s, t)] =
            p_grad[p_index_(b, s, t + 1)] * term2[py_index_(b, s, t)];
      }
    }
  }

  for (int s = 0; s <= S_; ++s) {
    for (int t = 0; t <= T_; ++t) {
      if (s < S_ && (s < s_begin || s >= s_end) && (t < t_begin || t > t_end)) {
        px_grad[px_index_(b, s, t)] = 0;
      }

      if (t < T_ && (s < s_begin || s > s_end) && (t < t_begin || t >= t_end)) {
        py_grad[py_index_(b, s, t)] = 0;
      }
    }
  }

  theory_ops_ += S_ * (T_ + 1) + (S_ + 1) * T_;
}

int64_t MutualInformationBackwardExecutor::getTheoryOps() {
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
