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
#include "sync_batchnorm_backward_reduce.h"

namespace mluoptest {

void SyncBatchnormBackwardReduceExecutor::paramCheck() {
  GTEST_CHECK(parser_->node()->has_sync_batchnorm_backward_reduce_param(),
              "Lose sync_batchnorm_backward_reduce param.");
}

void SyncBatchnormBackwardReduceExecutor::workspaceMalloc() {
  auto tensor_x = tensor_desc_[1].tensor;
  void *tmp = nullptr;
  // allocate extra nram space for deletion of CDMA
  MLUOP_CHECK(mluOpGetSyncBatchNormBackwardReduceWorkspaceSize(
      handle_, tensor_x, &workspace_size_));
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

void SyncBatchnormBackwardReduceExecutor::workspaceFree() {
  if (workspace_[0]) {
    VLOG(4) << "Free device workspace space.";
    mlu_runtime_.deallocate(workspace_[0]);
  }
}

void SyncBatchnormBackwardReduceExecutor::compute() {
  const bool needs_input_grad0 = parser_->getProtoNode()
                                     ->sync_batchnorm_backward_reduce_param()
                                     .needs_input_grad0();
  const bool needs_input_grad1 = parser_->getProtoNode()
                                     ->sync_batchnorm_backward_reduce_param()
                                     .needs_input_grad1();
  const bool needs_input_grad2 = parser_->getProtoNode()
                                     ->sync_batchnorm_backward_reduce_param()
                                     .needs_input_grad2();
  // input tensor description
  mluOpTensorDescriptor_t desc_dz = tensor_desc_[0].tensor;
  mluOpTensorDescriptor_t desc_x = tensor_desc_[1].tensor;
  mluOpTensorDescriptor_t desc_mean = tensor_desc_[2].tensor;
  mluOpTensorDescriptor_t desc_invstd = tensor_desc_[3].tensor;
  mluOpTensorDescriptor_t desc_sum_dy = NULL;
  mluOpTensorDescriptor_t desc_sum_dy_xmu = NULL;
  mluOpTensorDescriptor_t desc_dweight = NULL;
  mluOpTensorDescriptor_t desc_dbias = NULL;

  if (needs_input_grad0 == 1 && needs_input_grad1 == 0 &&
      needs_input_grad2 == 0) {
    GTEST_CHECK(parser_->outputs().size() == 2,
                "[Output MISMATCHED]: Only sum_dy and sum_dy_xmu will be "
                "compute currently.");
  }
  if (needs_input_grad0 == 0 && needs_input_grad1 == 1 &&
      needs_input_grad2 == 0) {
    GTEST_CHECK(parser_->outputs().size() == 1,
                "[Output MISMATCHED]: Only dweight will be compute currently.");
  }
  if (needs_input_grad0 == 0 && needs_input_grad1 == 0 &&
      needs_input_grad2 == 1) {
    GTEST_CHECK(parser_->outputs().size() == 1,
                "[Output MISMATCHED]: Only dbias will be compute currently.");
  }
  if (needs_input_grad0 == 1 && needs_input_grad1 == 1 &&
      needs_input_grad2 == 0) {
    GTEST_CHECK(parser_->outputs().size() == 3,
                "[Output MISMATCHED]: Only sum_dy, sum_dy_xmu, dweight will be "
                "compute currently.");
  }
  if (needs_input_grad0 == 1 && needs_input_grad1 == 0 &&
      needs_input_grad2 == 1) {
    GTEST_CHECK(parser_->outputs().size() == 3,
                "[Output MISMATCHED]: Only sum_dy, sum_dy_xmu, dbias will be "
                "compute currently.");
  }
  if (needs_input_grad0 == 0 && needs_input_grad1 == 1 &&
      needs_input_grad2 == 1) {
    GTEST_CHECK(parser_->outputs().size() == 2,
                "[Output MISMATCHED]: Only dweight and dbias will be compute "
                "currently.");
  }
  if (needs_input_grad0 == 1 && needs_input_grad1 == 1 &&
      needs_input_grad2 == 1) {
    GTEST_CHECK(parser_->outputs().size() == 4,
                "[Output MISMATCHED]: All of the four outputs will be compute "
                "currently.");
  }
  // input pointer for device
  void *dev_dz = data_vector_[0].device_ptr;
  void *dev_x = data_vector_[1].device_ptr;
  void *dev_mean = data_vector_[2].device_ptr;
  void *dev_invstd = data_vector_[3].device_ptr;
  void *dev_sum_dy = NULL;
  void *dev_sum_dy_xmu = NULL;
  void *dev_dweight = NULL;
  void *dev_dbias = NULL;

  if (needs_input_grad0) {
    desc_sum_dy = tensor_desc_[5].tensor;
    desc_sum_dy_xmu = tensor_desc_[6].tensor;
    dev_sum_dy = data_vector_[5].device_ptr;
    dev_sum_dy_xmu = data_vector_[6].device_ptr;
    if (needs_input_grad1) {
      desc_dweight = tensor_desc_[7].tensor;
      dev_dweight = data_vector_[7].device_ptr;
      if (needs_input_grad2) {
        desc_dbias = tensor_desc_[8].tensor;
        dev_dbias = data_vector_[8].device_ptr;
      }
    } else {
      if (needs_input_grad2) {
        desc_dbias = tensor_desc_[7].tensor;
        dev_dbias = data_vector_[7].device_ptr;
      }
    }
  } else {
    if (needs_input_grad1) {
      desc_dweight = tensor_desc_[5].tensor;
      dev_dweight = data_vector_[5].device_ptr;
      if (needs_input_grad2) {
        desc_dbias = tensor_desc_[6].tensor;
        dev_dbias = data_vector_[6].device_ptr;
      }
    } else {
      if (needs_input_grad2) {
        desc_dbias = tensor_desc_[5].tensor;
        dev_dbias = data_vector_[5].device_ptr;
      }
    }
  }

  VLOG(4) << "Start to run mluOpSyncBatchNormBackwardReduce().";
  interface_timer_.start();
#if 1
  VLOG(4) << "launch mluOpSyncBatchNormBackwardReduce_v2.";
  MLUOP_CHECK(mluOpSyncBatchNormBackwardReduce_v2(
      handle_, desc_dz, dev_dz, desc_x, dev_x, desc_mean, dev_mean, desc_invstd,
      dev_invstd, workspace_[0], workspace_size_, desc_dweight, dev_dweight,
      desc_dbias, dev_dbias, desc_sum_dy, dev_sum_dy, desc_sum_dy_xmu,
      dev_sum_dy_xmu, needs_input_grad0, needs_input_grad1, needs_input_grad2));
#else
  VLOG(4) << "launch mluOpSyncBatchNormBackwardReduce.";
  MLUOP_CHECK(mluOpSyncBatchNormBackwardReduce(
      handle_, desc_dz, dev_dz, desc_x, dev_x, desc_mean, dev_mean, desc_invstd,
      dev_invstd, desc_dweight, dev_dweight, desc_dbias, dev_dbias, desc_sum_dy,
      dev_sum_dy, desc_sum_dy_xmu, dev_sum_dy_xmu, needs_input_grad0,
      needs_input_grad1, needs_input_grad2));
#endif

  interface_timer_.stop();
}

void cpuGetSyncBnBkwReduceOuput(
    const float *x, const float *diff_z, const float *mean, const float *invstd,
    float *diff_weight, float *diff_bias, float *sum_dy, float *sum_dy_xmu,
    const int len_x, const int len_c, const bool needs_input_grad0,
    const bool needs_input_grad1, const bool needs_input_grad2) {
  if (len_x == 0 || len_c == 0) {
    LOG(ERROR) << "SyncBnBackwardReduce: the element number of input tensor "
                  "should not be zero";
    return;
  }
  int len_nhw = len_x / len_c;
  float *x_hat = new float[len_x];
  float *xmu = new float[len_x];

  for (int ci = 0; ci < len_c; ++ci) {
    const float *xc = x + ci;
    float *x_hat_c = x_hat + ci;
    float *xmu_c = xmu + ci;
    for (int xi = 0; xi < len_nhw; ++xi) {
      xmu_c[xi * len_c] = xc[xi * len_c] - mean[ci];
      x_hat_c[xi * len_c] = xmu_c[xi * len_c] * invstd[ci];
    }
  }

  for (int ci = 0; ci < len_c; ++ci) {
    const float *x_hat_c = x_hat + ci;
    const float *xmu_c = xmu + ci;
    const float *dzc = diff_z + ci;
    double dweight = 0, dbias = 0, meandyxmu = 0;
    for (int i = 0; i < len_nhw; i++) {
      dweight = dweight + x_hat_c[i * len_c] * dzc[i * len_c];
      dbias = dbias + dzc[i * len_c];
      meandyxmu = meandyxmu + xmu_c[i * len_c] * dzc[i * len_c];
    }
    if (needs_input_grad0 == true) {
      // diff_weight[ci] = dweight;
      // diff_bias[ci] = dbias;
      sum_dy[ci] = dbias;
      sum_dy_xmu[ci] = meandyxmu;
    }
    if (needs_input_grad1 == true) {
      diff_weight[ci] = dweight;
    }
    if (needs_input_grad2 == true) {
      diff_bias[ci] = dbias;
    }
  }
  delete[] x_hat;
  delete[] xmu;
}

void SyncBatchnormBackwardReduceExecutor::cpuCompute() {
  int len_c = tensor_desc_[0].tensor->getDimIndex(tensor_desc_[0].tensor->getDim() - 1);
  int len_x = parser_->getInputDataCount(0);
  const bool needs_input_grad0 = parser_->getProtoNode()
                                     ->sync_batchnorm_backward_reduce_param()
                                     .needs_input_grad0();
  const bool needs_input_grad1 = parser_->getProtoNode()
                                     ->sync_batchnorm_backward_reduce_param()
                                     .needs_input_grad1();
  const bool needs_input_grad2 = parser_->getProtoNode()
                                     ->sync_batchnorm_backward_reduce_param()
                                     .needs_input_grad2();

  auto tensor_dz = cpu_fp32_input_[0];
  auto tensor_x = cpu_fp32_input_[1];
  auto tensor_mean = cpu_fp32_input_[2];
  auto tensor_invstd = cpu_fp32_input_[3];

  auto tensor_sum_dy = cpu_fp32_output_[0];
  auto tensor_sum_dy_xmu = cpu_fp32_output_[1];
  auto tensor_dweight = cpu_fp32_output_[2];
  auto tensor_dbias = cpu_fp32_output_[3];
  if (needs_input_grad0) {
    tensor_sum_dy = cpu_fp32_output_[0];
    tensor_sum_dy_xmu = cpu_fp32_output_[1];
    if (needs_input_grad1) {
      tensor_dweight = cpu_fp32_output_[2];
      if (needs_input_grad2) {
        tensor_dbias = cpu_fp32_output_[3];
      }
    } else {
      if (needs_input_grad2) {
        tensor_dbias = cpu_fp32_output_[2];
      }
    }
  } else {
    if (needs_input_grad1) {
      tensor_dweight = cpu_fp32_output_[0];
      if (needs_input_grad2) {
        tensor_dbias = cpu_fp32_output_[1];
      }
    } else {
      if (needs_input_grad2) {
        tensor_dbias = cpu_fp32_output_[0];
      }
    }
  }

  // const bool needs_input_grad[3] = {1,1,1};
  // call the cup compute function to get:-> grad weight, grad bias, sum_dy,
  // sum_dy_xmu
  cpuGetSyncBnBkwReduceOuput(tensor_x, tensor_dz, tensor_mean, tensor_invstd,
                             tensor_dweight, tensor_dbias, tensor_sum_dy,
                             tensor_sum_dy_xmu, len_x, len_c, needs_input_grad0,
                             needs_input_grad1, needs_input_grad2);
}

int64_t SyncBatchnormBackwardReduceExecutor::getTheoryOps() {
  int cp_count = 8;
  int64_t theory_ops = parser_->getOutputDataCount(0) * cp_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
