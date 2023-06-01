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
#include "tin_shift_backward.h"

#include <string>

namespace mluoptest {
void TinShiftBackwardExecutor::paramCheck() {
  if (parser_->getInputNum() != 2) {
    LOG(ERROR) << "TinShiftBackward input number is wrong. ";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "TinShiftBackward output number is wrong. ";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
}

void TinShiftBackwardExecutor::compute() {
  auto in_tensor = tensor_desc_[0].tensor;
  auto input_ptr = data_vector_[0].device_ptr;
  auto shift_tensor = tensor_desc_[1].tensor;
  auto shift_ptr = data_vector_[1].device_ptr;
  auto out_tensor = tensor_desc_[2].tensor;
  auto output_ptr = data_vector_[2].device_ptr;
  interface_timer_.start();
  MLUOP_CHECK(mluOpTinShiftBackward(handle_, in_tensor, input_ptr, shift_tensor,
                                    shift_ptr, out_tensor, output_ptr));
  interface_timer_.stop();
}

void TinShiftBackwardExecutor::cpuCompute() {
  auto x = tensor_desc_[0].tensor;
  auto x1 = tensor_desc_[1].tensor;
  auto count1 = parser_->getInputDataCount(0);
  int batch_size = x->dims[0];
  int t_size = x->dims[1];
  int channels = x->dims[2];
  int hw_size = x->dims[3];
  int group_batch = x1->dims[0];
  int group_size = x1->dims[1];
  int group_channel = channels / group_size;

  for (int index = 0; index < count1; index++) {
    const int hw_index = index % hw_size;
    const int j = index / hw_size % channels;
    const int n_index = index / hw_size / channels % batch_size;
    int group_id = j / group_channel;
    int t_shift = cpu_fp32_input_[1][n_index * group_size + group_id];
    int offset = n_index * t_size * hw_size * channels + hw_size * j + hw_index;
    for (int i = 0; i < t_size; i++) {
      int now_t = i + t_shift;
      int data_id = i * hw_size * channels + offset;
      if (now_t < 0 || now_t >= t_size) {
        continue;
      }
      int out_id = now_t * hw_size * channels + offset;
      cpu_fp32_output_[0][out_id] = cpu_fp32_input_[0][data_id];
    }
  }
}

int64_t TinShiftBackwardExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->getOutputDataCount(0);
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}
}  // namespace mluoptest
