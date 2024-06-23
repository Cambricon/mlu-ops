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
#include "logcumsumexp.h"

#include <algorithm>
#include <memory>

#define CUMSUM_SIZE 1000

namespace mluoptest {

void LogcumsumexpExecutor::compute() {
    VLOG(4) << "LogcumsumexpExecutor compute";
    auto tensor_x = tensor_desc_[0].tensor;
    auto tensor_y = tensor_desc_[1].tensor;
    auto dev_x = data_vector_[0].device_ptr;
    auto dev_y = data_vector_[1].device_ptr;
    int32_t dim = parser_->getProtoNode()->logcumsumexp_param().dim();

    VLOG(4) << "call mluOpLogcumsumexp()";
    interface_timer_.start();
    MLUOP_CHECK(mluOpLogcumsumexp(handle_, tensor_x,
                                  dev_x, tensor_y, dev_y, dim));
    interface_timer_.stop();
}

void LogcumsumexpExecutor::cpuCompute() {
  GTEST_CHECK(parser_->getInputNum() == 1);
  GTEST_CHECK(parser_->getOutputNum() == 1);
  int32_t dim = parser_->getProtoNode()->logcumsumexp_param().dim();
  int32_t data_size = parser_->input(0)->total_count;

  // Pamameter Preparing
  auto self_dim_size = tensor_desc_[0].tensor->dims[dim];
  int32_t self_dim_stride = 1;
  int32_t batches = 1;
  for (int i = dim + 1; i < tensor_desc_[0].tensor->dim; i++) {
    self_dim_stride *= tensor_desc_[0].tensor->dims[i];
  }
  for (int i = dim - 1; i >= 0; i--) {
    batches *= tensor_desc_[0].tensor->dims[i];
  }
  for (int i = 0; i < data_size; i++) {
    cpu_fp32_input_[0][i] = exp(cpu_fp32_input_[0][i]);
  }
  // Cumsum Computing
  for (int b = 0; b < batches; b++) {
    float* x_ptr =  cpu_fp32_input_[0] + self_dim_size * self_dim_stride * b;
    float* y_ptr =  cpu_fp32_output_[0] + self_dim_size * self_dim_stride * b;
    // To minimize distortion in floating-point calculations
    for (int i = 0; i < self_dim_stride; i++) {
      int32_t segNum = (self_dim_size + CUMSUM_SIZE - 1) / CUMSUM_SIZE;
      std::unique_ptr<float[]> cum_number = std::make_unique<float[]>(segNum);
      for (int j = 0; j < segNum; j++) {
        cum_number[j] = 0;
      }
      for (int j = 0; j < segNum; j++) {
        int32_t dealLength;
        if (self_dim_size % CUMSUM_SIZE == 0)
          dealLength = CUMSUM_SIZE;
        else if (j < segNum - 1)
          dealLength = CUMSUM_SIZE;
        else
          dealLength = self_dim_size % CUMSUM_SIZE;
        cum_number[j] = x_ptr[j * CUMSUM_SIZE * self_dim_stride];
        y_ptr[j * CUMSUM_SIZE * self_dim_stride] = cum_number[j];

        // cumsum in every seg
        for (int k = 1; k < dealLength; k++) {
          auto add_exe = [](float x, float y) -> float{
            float min = std::isnan(y) ? y : std::min(x, y);
            float max = std::isnan(y) ? y : std::max(x, y);
            if (min != max || std::isfinite(min)) {
              return min + max;
            } else {
              return x;
            }
          };
        cum_number[j]
          = add_exe(x_ptr[(j * CUMSUM_SIZE + k) * self_dim_stride], cum_number[j]);
        y_ptr[(j * CUMSUM_SIZE + k) * self_dim_stride] = cum_number[j];
        }
      }
      // computing the offsets
      for (int j = 1; j < segNum; j++) {
        cum_number[j] = cum_number[j] + cum_number[j-1];
      }
      // offsets to result
      for (int j = 1; j < segNum; j++) {
        int32_t dealLength;
        if (self_dim_size % CUMSUM_SIZE == 0)
          dealLength = CUMSUM_SIZE;
        else if (j < segNum - 1)
          dealLength = CUMSUM_SIZE;
        else
          dealLength = self_dim_size % CUMSUM_SIZE;
        for (int k = 0; k < dealLength; k++) {
          y_ptr[(j * CUMSUM_SIZE + k) * self_dim_stride] += cum_number[j - 1];
        }
      }
      x_ptr += 1;
      y_ptr += 1;
    }
  }
  for (int i = 0; i < data_size; i++) {
    cpu_fp32_output_[0][i] = log(cpu_fp32_output_[0][i]);
  }
}

int64_t LogcumsumexpExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->input(0)->total_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}
}  // namespace mluoptest
