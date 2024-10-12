/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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
#include "logspace.h"

namespace mluoptest {

void LogspaceExecutor::initData() {
  start_num_ = parser_->getProtoNode()->logspace_param().start();
  end_num_ = parser_->getProtoNode()->logspace_param().end();
  steps_num_ = parser_->getProtoNode()->logspace_param().steps();
  base_num_ = parser_->getProtoNode()->logspace_param().base();
}

void LogspaceExecutor::paramCheck() {
  GTEST_CHECK(parser_->outputs().size() == 1,
              "logspace tensor output number is wrong.");
}

void LogspaceExecutor::compute() {
  VLOG(4) << "LogspaceExecutor compute ";
  initData();

  auto tensor_y = tensor_desc_[1].tensor;
  auto dev_y = data_vector_[1].device_ptr;

  VLOG(4) << "call mluOpLogspace()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpLogspace(handle_, start_num_, end_num_, (int64_t)steps_num_,
                            base_num_, tensor_y, dev_y));
  interface_timer_.stop();
}

void LogspaceExecutor::cpuCompute() {
  if (steps_num_ == 1) {
    cpu_fp32_output_[0][0] = (half)::powf(base_num_, start_num_);
  } else {
    auto count = parser_->output(0)->shape_count;
    float step = (end_num_ - start_num_) / (steps_num_ - 1);

    switch (tensor_desc_[1].tensor->dtype) {
      case MLUOP_DTYPE_FLOAT: {
        for (int i = 0; i < count; ++i) {
          cpu_fp32_output_[0][i] = ::powf(base_num_, start_num_ + step * i);
        }
      }; break;
      case MLUOP_DTYPE_HALF: {
        half step =
            ((half)end_num_ - (half)start_num_) / (half)((half)steps_num_ - 1);
        int halfway = steps_num_ / 2;
        for (int i = 0; i < count; ++i) {
          if (i < halfway) {
            cpu_fp32_output_[0][i] =
                (half)::pow((half)base_num_, (half)start_num_ + step * i);
          } else {
            cpu_fp32_output_[0][i] = (half)::pow(
                (half)base_num_, (half)end_num_ - step * (steps_num_ - i - 1));
          }
        }
      }; break;
      case MLUOP_DTYPE_INT32: {
        for (int i = 0; i < count; ++i) {
          cpu_fp32_output_[0][i] =
              (int)::powf(base_num_, start_num_ + step * i);
        }
      }; break;
      default:
        break;
    }
  }
}

int64_t LogspaceExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->output(0)->total_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

}  // namespace mluoptest
