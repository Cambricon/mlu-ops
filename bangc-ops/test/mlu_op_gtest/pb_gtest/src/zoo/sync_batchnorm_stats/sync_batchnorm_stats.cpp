/*************************************************************************
 * Copyright (C) [2019-2023] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "sync_batchnorm_stats.h"

namespace mluoptest {

void SyncBatchnormStatsExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_sync_batchnorm_stats_param()) {
    LOG(ERROR) << "Lose sync_batchnorm_stats param.";
  }
}

void SyncBatchnormStatsExecutor::workspaceMalloc() {
  auto tensor_x = tensor_desc_[0].tensor;
  void *tmp = nullptr;
  // allocate extra nram space for deletion of CDMA
  MLUOP_CHECK(mluOpGetSyncBatchNormStatsWorkspaceSize(handle_, tensor_x,
                                                      &workspace_size_));
  if (workspace_size_ > 0) {
    VLOG(4) << "Malloc workspace space for deletion of CDMA.";
    tmp = mlu_runtime_.allocate(workspace_size_);
    VLOG(4) << "Mallocated addr: " << tmp << ", size: " << workspace_size_;
  } else {
    VLOG(4) << "Don't need to Malloc workspace space.";
  }
  workspace_.push_back(tmp);
  eva_->setMluWorkspaceSize(workspace_size_);
}

void SyncBatchnormStatsExecutor::workspaceFree() {
  if (workspace_[0]) {
    VLOG(4) << "Free device workspace space.";
    mlu_runtime_.deallocate(workspace_[0]);
  }
}

void SyncBatchnormStatsExecutor::compute() {
  float eps = parser_->getProtoNode()->sync_batchnorm_stats_param().eps();

  mluOpTensorDescriptor_t x_desc = tensor_desc_[0].tensor;
  mluOpTensorDescriptor_t mean_desc = tensor_desc_[1].tensor;
  mluOpTensorDescriptor_t invstd_desc = tensor_desc_[2].tensor;

  VLOG(4) << "call mluOpSyncBatchNormStats()";
  interface_timer_.start();
#if 1
  VLOG(4) << "launch mluOpSyncBatchNormStats_v2.";
  MLUOP_CHECK(mluOpSyncBatchNormStats_v2(
      handle_, x_desc, data_vector_[0].device_ptr, workspace_[0],
      workspace_size_, eps, mean_desc, data_vector_[1].device_ptr, invstd_desc,
      data_vector_[2].device_ptr));
#else
  VLOG(4) << "launch mluOpSyncBatchNormStats.";
  MLUOP_CHECK(mluOpSyncBatchNormStats(
      handle_, x_desc, data_vector_[0].device_ptr, eps, mean_desc,
      data_vector_[1].device_ptr, invstd_desc, data_vector_[2].device_ptr));
#endif
  interface_timer_.stop();
}

void kahan_stats(float input, float &sum, float &delta) {
  float y = input - delta;
  float t = sum + y;
  delta = t - sum - y;
  sum = t;
}

void cpuSyncBatchNormStats(const float *x, const float eps, float *mean,
                           float *invstd, const int len_x, const int len_c) {
  float len_nhw = len_x / len_c;

  bool flag_free = false;
  if (mean == nullptr && invstd == nullptr) {
    mean = new float[len_c];
    invstd = new float[len_c];
    flag_free = true;
  }

  for (int ci = 0; ci < len_c; ++ci) {
    float sum = 0, ssum = 0;
    float c_sum = 0.0, c_ssum = 0.0;
    const float *xc = x + ci;
    for (int xi = 0; xi < len_nhw; ++xi) {
      kahan_stats(xc[xi * len_c], sum, c_sum);
      kahan_stats(xc[xi * len_c] * xc[xi * len_c], ssum, c_ssum);
    }
    mean[ci] = sum / len_nhw;
    invstd[ci] = 1.0f / sqrt(ssum / len_nhw - (mean[ci] * mean[ci]) + eps);
  }

  if (flag_free == true) {
    delete[] mean;
    delete[] invstd;
  }
}

void SyncBatchnormStatsExecutor::cpuCompute() {
  float eps = parser_->getProtoNode()->sync_batchnorm_stats_param().eps();

  int idx_c = tensor_desc_[0].tensor->dim - 1;
  int len_c = tensor_desc_[0].tensor->dims[idx_c];
  int len_x = 1;
  for (int i = 0; i < tensor_desc_[0].tensor->dim; ++i) {
    len_x *= tensor_desc_[0].tensor->dims[i];
  }
  if (len_x == 0 || len_c == 0) {
    return;
  }
  VLOG(4) << "Start to run cpuSyncBatchNormStats().";
  cpuSyncBatchNormStats(cpu_fp32_input_[0], eps, cpu_fp32_output_[0],
                        cpu_fp32_output_[1], len_x, len_c);
}

int64_t SyncBatchnormStatsExecutor::getTheoryOps() {
  int cp_count = 8;
  int64_t theory_ops = parser_->getOutputDataCount(0) * cp_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
