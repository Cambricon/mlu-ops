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
#include "sync_batchnorm_gather_stats_with_counts.h"

namespace mluoptest {

void SyncBatchnormGatherStatsWithCountsExecutor::paramCheck() {
  GTEST_CHECK(parser_->getProtoNode()
              ->has_sync_batchnorm_gather_stats_with_counts_param(),
              "Lose sync_batchnorm_gather_stats_with_counts param.");

  // set flag
  flag_input_reuse_ = false;
}

void SyncBatchnormGatherStatsWithCountsExecutor::compute() {
  float eps = parser_->getProtoNode()
                  ->sync_batchnorm_gather_stats_with_counts_param()
                  .eps();
  float momentum = parser_->getProtoNode()
                       ->sync_batchnorm_gather_stats_with_counts_param()
                       .momentum();

  mluOpTensorDescriptor_t mean_all_desc;
  mluOpTensorDescriptor_t invstd_all_desc;
  mluOpTensorDescriptor_t count_all_desc;
  mean_all_desc = tensor_desc_[1].tensor;
  invstd_all_desc = tensor_desc_[2].tensor;

  // if num_inputs = 3,
  // then [mean_all, invstd_all, count_all] -> [mean, invstd]
  // if num_inputs = 4,
  // then [input, mean_all, invstd_all, count_all] -> [mean,invstd]
  // if num_inputs = 5,
  // then [mean_all, invstd_all, moving_mean, moving_var, count_all]
  //   -> [moving_mean, moving_var, mean, invstd]
  // if num_inputs = 6,
  // then [input, mean_all, invstd_all, moving_mean, moving_var, count_all]
  //   -> [moving_mean, moving_var, mean, invstd]
  VLOG(4) << "Start to run mluOpSyncBatchNormGatherStatsWithCounts().";
  if (parser_->getInputNum() == 3) {
    // for case without "input" param
    mean_all_desc = tensor_desc_[0].tensor;
    invstd_all_desc = tensor_desc_[1].tensor;
    count_all_desc = tensor_desc_[2].tensor;
    mluOpTensorDescriptor_t mean_desc = tensor_desc_[3].tensor;
    mluOpTensorDescriptor_t invstd_desc = tensor_desc_[4].tensor;
    interface_timer_.start();
    MLUOP_CHECK(mluOpSyncBatchNormGatherStatsWithCounts(
        handle_, mean_all_desc, data_vector_[0].device_ptr, invstd_all_desc,
        data_vector_[1].device_ptr, nullptr, nullptr, nullptr, nullptr,
        momentum, eps, count_all_desc, data_vector_[2].device_ptr, mean_desc,
        data_vector_[3].device_ptr, invstd_desc, data_vector_[4].device_ptr));
    interface_timer_.stop();
  } else if (parser_->getInputNum() == 4) {
    count_all_desc = tensor_desc_[3].tensor;
    mluOpTensorDescriptor_t mean_desc = tensor_desc_[4].tensor;
    mluOpTensorDescriptor_t invstd_desc = tensor_desc_[5].tensor;
    interface_timer_.start();
    MLUOP_CHECK(mluOpSyncBatchNormGatherStatsWithCounts(
        handle_, mean_all_desc, data_vector_[1].device_ptr, invstd_all_desc,
        data_vector_[2].device_ptr, nullptr, nullptr, nullptr, nullptr,
        momentum, eps, count_all_desc, data_vector_[3].device_ptr, mean_desc,
        data_vector_[4].device_ptr, invstd_desc, data_vector_[5].device_ptr));
    interface_timer_.stop();
  } else if (parser_->getInputNum() == 5) {
    // for case without "input" param
    mean_all_desc = tensor_desc_[0].tensor;
    invstd_all_desc = tensor_desc_[1].tensor;
    mluOpTensorDescriptor_t moving_mean_desc = tensor_desc_[2].tensor;
    mluOpTensorDescriptor_t moving_var_desc = tensor_desc_[3].tensor;
    count_all_desc = tensor_desc_[4].tensor;
    if (parser_->getOutputNum() == 2) {
      mluOpTensorDescriptor_t mean_desc = tensor_desc_[5].tensor;
      mluOpTensorDescriptor_t invstd_desc = tensor_desc_[6].tensor;
      interface_timer_.start();
      MLUOP_CHECK(mluOpSyncBatchNormGatherStatsWithCounts(
          handle_, mean_all_desc, data_vector_[0].device_ptr, invstd_all_desc,
          data_vector_[1].device_ptr, moving_mean_desc,
          data_vector_[2].device_ptr, moving_var_desc,
          data_vector_[3].device_ptr, momentum, eps, count_all_desc,
          data_vector_[4].device_ptr, mean_desc, data_vector_[5].device_ptr,
          invstd_desc, data_vector_[6].device_ptr));
      interface_timer_.stop();
    } else {
      mluOpTensorDescriptor_t mean_desc = tensor_desc_[7].tensor;
      mluOpTensorDescriptor_t invstd_desc = tensor_desc_[8].tensor;
      interface_timer_.start();
      MLUOP_CHECK(mluOpSyncBatchNormGatherStatsWithCounts(
          handle_, mean_all_desc, data_vector_[0].device_ptr, invstd_all_desc,
          data_vector_[1].device_ptr, moving_mean_desc,
          data_vector_[2].device_ptr, moving_var_desc,
          data_vector_[3].device_ptr, momentum, eps, count_all_desc,
          data_vector_[4].device_ptr, mean_desc, data_vector_[7].device_ptr,
          invstd_desc, data_vector_[8].device_ptr));
      interface_timer_.stop();
    }
  } else if (parser_->getInputNum() == 6) {
    mluOpTensorDescriptor_t moving_mean_desc = tensor_desc_[3].tensor;
    mluOpTensorDescriptor_t moving_var_desc = tensor_desc_[4].tensor;
    count_all_desc = tensor_desc_[5].tensor;
    if (parser_->getOutputNum() == 2) {
      mluOpTensorDescriptor_t mean_desc = tensor_desc_[6].tensor;
      mluOpTensorDescriptor_t invstd_desc = tensor_desc_[7].tensor;
      interface_timer_.start();
      MLUOP_CHECK(mluOpSyncBatchNormGatherStatsWithCounts(
          handle_, mean_all_desc, data_vector_[1].device_ptr, invstd_all_desc,
          data_vector_[2].device_ptr, moving_mean_desc,
          data_vector_[3].device_ptr, moving_var_desc,
          data_vector_[4].device_ptr, momentum, eps, count_all_desc,
          data_vector_[5].device_ptr, mean_desc, data_vector_[6].device_ptr,
          invstd_desc, data_vector_[7].device_ptr));
      interface_timer_.stop();
    } else {
      mluOpTensorDescriptor_t mean_desc = tensor_desc_[8].tensor;
      mluOpTensorDescriptor_t invstd_desc = tensor_desc_[9].tensor;
      interface_timer_.start();
      MLUOP_CHECK(mluOpSyncBatchNormGatherStatsWithCounts(
          handle_, mean_all_desc, data_vector_[1].device_ptr, invstd_all_desc,
          data_vector_[2].device_ptr, moving_mean_desc,
          data_vector_[3].device_ptr, moving_var_desc,
          data_vector_[4].device_ptr, momentum, eps, count_all_desc,
          data_vector_[5].device_ptr, mean_desc, data_vector_[8].device_ptr,
          invstd_desc, data_vector_[9].device_ptr));
      interface_timer_.stop();
    }
  }
}

void SyncBatchnormGatherStatsWithCountsExecutor::setMiscellaneousParam() {
  if (parser_->getInputNum() == 6) {
    if (parser_->getOutputNum() == 2) {
      data_vector_[3].alsoServeAsVolatile();
      data_vector_[4].alsoServeAsVolatile();
    } else {
      data_vector_[3].alsoServeAsOutput();
      data_vector_[4].alsoServeAsOutput();
      data_vector_[6].onlyServeAsInput();
      data_vector_[7].onlyServeAsInput();
    }
  } else if (parser_->getInputNum() == 5) {
    if (parser_->getOutputNum() == 2) {
      data_vector_[2].alsoServeAsVolatile();
      data_vector_[3].alsoServeAsVolatile();
    } else {
      data_vector_[2].alsoServeAsOutput();
      data_vector_[3].alsoServeAsOutput();
      data_vector_[5].onlyServeAsInput();
      data_vector_[6].onlyServeAsInput();
    }
  }
}

void kahan(float input, float &sum, float &delta) {
  float y = input - delta;
  float t = sum + y;
  delta = t - sum - y;
  sum = t;
}

void cpuBatchNormForwardTraining(float *mean_all, float *invstd_all,
                                 float *moving_mean, float *moving_var,
                                 const float momentum, const float eps,
                                 float *count_all, float *m_mean, float *m_var,
                                 float *mean, float *invstd,
                                 const int len_mean_all, const int len_c,
                                 const int output_num) {
  int len_n = len_mean_all / len_c;
  int len_all = 0;
  for (int i = 0; i < len_n; ++i) {
    len_all += count_all[i];
  }

  // B.P.Welford algo
  for (int ci = 0; ci < len_c; ++ci) {
    float c_sum = 0.0, c_ssum = 0.0;
    const float *meanc = mean_all + ci;
    const float *invstdc = invstd_all + ci;
    float sum = 0.0, ssum = 0.0, temp = 0.0;
    for (int xi = 0; xi < len_n; ++xi) {
      kahan(meanc[xi * len_c] * count_all[xi], sum, c_sum);
      temp = 1.0f / (invstdc[xi * len_c] * invstdc[xi * len_c]) +
             meanc[xi * len_c] * meanc[xi * len_c] - eps;
      kahan(temp * count_all[xi], ssum, c_ssum);
    }
    mean[ci] = sum / len_all;
    invstd[ci] = 1.0f / sqrt(ssum / len_all - mean[ci] * mean[ci] + eps);
    float unbiased_var =
        (1.0f / (invstd[ci] * invstd[ci]) - eps) * len_all / (len_all - 1);
    if (moving_mean != nullptr && moving_var != nullptr && output_num == 4) {
      m_mean[ci] = momentum * mean[ci] + (1 - momentum) * moving_mean[ci];
      m_var[ci] = momentum * unbiased_var + (1 - momentum) * moving_var[ci];
    }
  }
}

void SyncBatchnormGatherStatsWithCountsExecutor::cpuCompute() {
  float eps = parser_->getProtoNode()
                  ->sync_batchnorm_gather_stats_with_counts_param()
                  .eps();
  float momentum = parser_->getProtoNode()
                       ->sync_batchnorm_gather_stats_with_counts_param()
                       .momentum();

  int idx_c = tensor_desc_[0].tensor->getDim() - 1;
  int len_c = tensor_desc_[0].tensor->getDimIndex(idx_c);
  int len_count_all = 1;
  int len_mean_all = 1;
  int len_invstd_all = 1;
  if (parser_->getInputNum() == 3) {
    len_count_all = tensor_desc_[2].tensor->getDimIndex(0);
  } else if (parser_->getInputNum() == 4) {
    len_count_all = tensor_desc_[3].tensor->getDimIndex(0);
  } else if (parser_->getInputNum() == 5) {
    len_count_all = tensor_desc_[4].tensor->getDimIndex(0);
  } else if (parser_->getInputNum() == 6) {
    len_count_all = tensor_desc_[5].tensor->getDimIndex(0);
  }
  if (parser_->getInputNum() == 3 || parser_->getInputNum() == 5) {
    for (int i = 0; i < tensor_desc_[0].tensor->getDim(); ++i) {
      len_mean_all *= tensor_desc_[0].tensor->getDimIndex(i);
    }
    for (int i = 0; i < tensor_desc_[1].tensor->getDim(); ++i) {
      len_invstd_all *= tensor_desc_[1].tensor->getDimIndex(i);
    }
  } else {
    for (int i = 0; i < tensor_desc_[1].tensor->getDim(); ++i) {
      len_mean_all *= tensor_desc_[1].tensor->getDimIndex(i);
    }
    for (int i = 0; i < tensor_desc_[2].tensor->getDim(); ++i) {
      len_invstd_all *= tensor_desc_[2].tensor->getDimIndex(i);
    }
  }
  if (len_mean_all == 0 || len_c == 0 || len_count_all == 0 ||
      len_mean_all != len_invstd_all) {
    return;
  }
  int output_num = parser_->getOutputNum();
  VLOG(4) << "Start to run cpuBatchNormForwardTraining().";
  if (parser_->getInputNum() == 3) {
    cpuBatchNormForwardTraining(
        cpu_fp32_input_[0], cpu_fp32_input_[1], nullptr, nullptr, momentum, eps,
        cpu_fp32_input_[2], nullptr, nullptr, cpu_fp32_output_[0],
        cpu_fp32_output_[1], len_mean_all, len_c, output_num);
  } else if (parser_->getInputNum() == 4) {
    cpuBatchNormForwardTraining(
        cpu_fp32_input_[1], cpu_fp32_input_[2], nullptr, nullptr, momentum, eps,
        cpu_fp32_input_[3], nullptr, nullptr, cpu_fp32_output_[0],
        cpu_fp32_output_[1], len_mean_all, len_c, output_num);
  } else if (parser_->getInputNum() == 5) {
    if (parser_->getOutputNum() == 2) {
      cpuBatchNormForwardTraining(
          cpu_fp32_input_[0], cpu_fp32_input_[1], cpu_fp32_input_[2],
          cpu_fp32_input_[3], momentum, eps, cpu_fp32_input_[4], nullptr,
          nullptr, cpu_fp32_output_[0], cpu_fp32_output_[1], len_mean_all,
          len_c, output_num);
    } else {
      cpuBatchNormForwardTraining(
          cpu_fp32_input_[0], cpu_fp32_input_[1], cpu_fp32_input_[2],
          cpu_fp32_input_[3], momentum, eps, cpu_fp32_input_[4],
          cpu_fp32_output_[0], cpu_fp32_output_[1], cpu_fp32_output_[2],
          cpu_fp32_output_[3], len_mean_all, len_c, output_num);
    }
  } else if (parser_->getInputNum() == 6) {
    if (parser_->getOutputNum() == 2) {
      cpuBatchNormForwardTraining(
          cpu_fp32_input_[1], cpu_fp32_input_[2], cpu_fp32_input_[3],
          cpu_fp32_input_[4], momentum, eps, cpu_fp32_input_[5], nullptr,
          nullptr, cpu_fp32_output_[0], cpu_fp32_output_[1], len_mean_all,
          len_c, output_num);
    } else {
      cpuBatchNormForwardTraining(
          cpu_fp32_input_[1], cpu_fp32_input_[2], cpu_fp32_input_[3],
          cpu_fp32_input_[4], momentum, eps, cpu_fp32_input_[5],
          cpu_fp32_output_[0], cpu_fp32_output_[1], cpu_fp32_output_[2],
          cpu_fp32_output_[3], len_mean_all, len_c, output_num);
    }
  }
}

int64_t SyncBatchnormGatherStatsWithCountsExecutor::getTheoryOps() {
  int cp_count = 8;
  int64_t theory_ops = parser_->getOutputDataCount(0) * cp_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

std::set<Evaluator::Formula>
SyncBatchnormGatherStatsWithCountsExecutor::getCriterionsUse() const {
  return {Evaluator::DIFF1, Evaluator::DIFF2, Evaluator::DIFF3};
}

}  // namespace mluoptest
