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
#include "sync_batchnorm_elemt.h"

namespace mluoptest {

void SyncBatchnormElemtExecutor::paramCheck() {
  if (parser_->getInputNum() != 3 && parser_->getInputNum() != 5) {
    LOG(ERROR) << "SyncBatchnormElemtExecutor: input number is wrong. ";
  }
  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "SyncBatchnormElemtExecutor: output number is wrong. ";
  }
}

void SyncBatchnormElemtExecutor::compute() {
  VLOG(4) << "SyncBatchnormElemtExecutor compute begin";
  auto x_desc = tensor_desc_[0].tensor;
  auto dev_x = data_vector_[0].device_ptr;
  auto mean_desc = tensor_desc_[1].tensor;
  auto dev_mean = data_vector_[1].device_ptr;
  auto invstd_desc = tensor_desc_[2].tensor;
  auto dev_invstd = data_vector_[2].device_ptr;

  if (parser_->getInputNum() == 3) {
    auto y_desc = tensor_desc_[3].tensor;
    auto dev_y = data_vector_[3].device_ptr;
    interface_timer_.start();
    MLUOP_CHECK(mluOpSyncBatchNormElemt(
        handle_, x_desc, dev_x, mean_desc, dev_mean, invstd_desc, dev_invstd,
        nullptr, nullptr, nullptr, nullptr, y_desc, dev_y));
    interface_timer_.stop();
  } else if (parser_->getInputNum() == 5) {
    auto weight_desc = tensor_desc_[3].tensor;
    auto dev_weight = data_vector_[3].device_ptr;
    auto bias_desc = tensor_desc_[4].tensor;
    auto dev_bias = data_vector_[4].device_ptr;
    auto y_desc = tensor_desc_[5].tensor;
    auto dev_y = data_vector_[5].device_ptr;
    interface_timer_.start();
    MLUOP_CHECK(mluOpSyncBatchNormElemt(
        handle_, x_desc, dev_x, mean_desc, dev_mean, invstd_desc, dev_invstd,
        weight_desc, dev_weight, bias_desc, dev_bias, y_desc, dev_y));
    interface_timer_.stop();
  }
  VLOG(4) << "SyncBatchnormElemtExecutor compute end";
}

void cpuSyncBNElemt(const float *x, const float *mean, const float *invstd,
                    float *weight, float *bias, float *y, const int len_x,
                    const int len_c) {
  int len_nhw = len_x / len_c;

  for (int h = 0; h < len_nhw; ++h) {
    for (int c = 0; c < len_c; ++c) {
      y[h * len_c + c] = (x[h * len_c + c] - mean[c]) * invstd[c];
      if (weight != nullptr && bias != nullptr) {
        y[h * len_c + c] = y[h * len_c + c] * weight[c] + bias[c];
      }
    }
  }
}

void SyncBatchnormElemtExecutor::cpuCompute() {
  int len_c = tensor_desc_[0].tensor->dims[tensor_desc_[0].tensor->dim - 1];
  int len_x = parser_->getInputDataCount(0);

  VLOG(4) << "SyncBatchnormElemtExecutor: cpu compute begin";
  // actually len_c = 0, then len_x must be 0
  if (len_c == 0 || len_x == 0) {
    VLOG(4) << "SyncBatchnormElemtExecutor: cpu compute zero elemt";
    return;
  }

  if (parser_->getInputNum() == 3) {
    VLOG(4) << "weight and bias is nullptr";
    cpuSyncBNElemt(cpu_fp32_input_[0], cpu_fp32_input_[1], cpu_fp32_input_[2],
                   nullptr, nullptr, cpu_fp32_output_[0], len_x, len_c);
  } else if (parser_->getInputNum() == 5) {
    cpuSyncBNElemt(cpu_fp32_input_[0], cpu_fp32_input_[1], cpu_fp32_input_[2],
                   cpu_fp32_input_[3], cpu_fp32_input_[4], cpu_fp32_output_[0],
                   len_x, len_c);
  }
  VLOG(4) << "SyncBatchnormElemtExecutor: cpu compute end";
}

int64_t SyncBatchnormElemtExecutor::getTheoryOps() {
  int64_t theory_ops = 0;
  int len_x = parser_->getInputDataCount(0);
  if (parser_->getInputNum() == 3) {
    theory_ops = len_x * 2;
  } else if (parser_->getInputNum() == 5) {
    theory_ops = len_x * 4;
  }
  VLOG(4) << "SyncBatchnormElemtExecutor: getTheoryOps: " << theory_ops
          << " ops";
  return theory_ops;
}

}  // namespace mluoptest
