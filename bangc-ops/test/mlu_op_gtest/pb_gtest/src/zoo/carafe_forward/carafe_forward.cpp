/*******************************************************************************
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
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS self.tcp LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *******************************************************************************/
#include <string>
#include <vector>
#include <set>
#include "carafe_forward.h"
#include "mlu_op.h"

namespace mluoptest {
std::set<Evaluator::Formula> CarafeForwardExecutor::getCriterionsUse() const {
  return {Evaluator::DIFF1, Evaluator::DIFF2, Evaluator::DIFF4};
}

void CarafeForwardExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_carafe_forward_param()) {
    LOG(ERROR) << "Missing carafe param. ";
  }
}

void CarafeForwardExecutor::compute() {
  auto carafe_desc_node = parser_->getProtoNode()->carafe_forward_param();

  int dim_num = carafe_desc_node.dimnb();
  int kernel_size = carafe_desc_node.kernel_size();
  int group_size = carafe_desc_node.group_size();
  int scale_factor = carafe_desc_node.scale_factor();
  mluOpCarafeDescriptor_t carafe_desc = cpu_runtime_.allocate(
      mluOpCreateCarafeDescriptor, mluOpDestroyCarafeDescriptor);
  MLUOP_CHECK(mluOpSetCarafeDescriptor(carafe_desc, dim_num, kernel_size,
                                       group_size, scale_factor));

  auto input_desc = tensor_desc_[0].tensor;
  auto mask_desc = tensor_desc_[1].tensor;
  auto output_desc = tensor_desc_[2].tensor;

  void *dev_input = data_vector_[0].device_ptr;
  void *dev_mask = data_vector_[1].device_ptr;
  void *dev_output = data_vector_[2].device_ptr;

  interface_timer_.start();

  MLUOP_CHECK(mluOpCarafeForward(handle_, carafe_desc, input_desc, dev_input,
                                 mask_desc, dev_mask, output_desc, dev_output));
  interface_timer_.stop();
}

void CarafeForwardExecutor::cpuCompute() {
  assert(parser_->getInputNum() == 2);
  assert(parser_->getOutputNum() == 1);

  auto carafe_desc_node = parser_->getProtoNode()->carafe_forward_param();

  int kernel_size = carafe_desc_node.kernel_size();
  int group_size = carafe_desc_node.group_size();
  int scale_factor = carafe_desc_node.scale_factor();

  assert(kernel_size >= 1 && (kernel_size - 1) % 2 == 0);
  assert(scale_factor >= 1);
  assert(group_size >= 1);

  int half_kernel_size = (kernel_size - 1) / 2;

  auto input_desc = tensor_desc_[0].tensor;
  auto mask_desc = tensor_desc_[1].tensor;
  auto output_desc = tensor_desc_[2].tensor;

  int input_dimN = mluOpGetTensordimN(input_desc);
  int input_dimH = mluOpGetTensordimH(input_desc);
  int input_dimW = mluOpGetTensordimW(input_desc);
  int input_dimC = mluOpGetTensordimC(input_desc);

  int mask_dimN = mluOpGetTensordimN(mask_desc);
  int mask_dimH = mluOpGetTensordimH(mask_desc);
  int mask_dimW = mluOpGetTensordimW(mask_desc);
  int mask_dimC = mluOpGetTensordimC(mask_desc);

  int output_dimN = mluOpGetTensordimN(output_desc);
  int output_dimH = mluOpGetTensordimH(output_desc);
  int output_dimW = mluOpGetTensordimW(output_desc);
  int output_dimC = mluOpGetTensordimC(output_desc);

  assert(input_dimN == mask_dimN);
  assert(input_dimN == output_dimN);
  assert(input_dimC == output_dimC);
  assert(mask_dimC == kernel_size * kernel_size * group_size);
  assert(mask_dimH == scale_factor * input_dimH);
  assert(mask_dimW == scale_factor * input_dimW);
  assert(mask_dimH == output_dimH);
  assert(mask_dimW == output_dimW);
  assert(input_dimC % group_size == 0);

  int channels_per_group = input_dimC / group_size;

  float *host_input = cpu_fp32_input_[0];
  float *host_mask = cpu_fp32_input_[1];
  float *host_output = cpu_fp32_output_[0];

  int output_size = output_dimN * output_dimH * output_dimW * output_dimC;
  for (int i = 0; i < output_size; i++) {
    host_output[i] = 0.0;
  }

  // calculate weighted sum on each output location
  for (int index1_output = 0; index1_output < output_size; index1_output++) {
    // output[index1_output] -> output[no,ho,wo,co]
    int co = index1_output % output_dimC;
    int wo = (index1_output / output_dimC) % output_dimW;
    int ho = (index1_output / output_dimC / output_dimW) % output_dimH;
    int no = index1_output / output_dimC / output_dimW / output_dimH;
    // co -> mask_group
    int mask_group = co / channels_per_group;
    // kernel window's bottom-left location on the input feature map
    int min_hi = ho / scale_factor - half_kernel_size;
    int min_wi = wo / scale_factor - half_kernel_size;
    // calculate weighted sum over the kernel window
    for (int kh = 0; kh < kernel_size; kh++) {
      // corresponding input location
      int hi = min_hi + kh;
      // skip elements outside of the input feature map
      if (hi < 0 || hi > input_dimH - 1) {
        continue;
      }
      for (int kw = 0; kw < kernel_size; kw++) {
        // corresponding input location
        int wi = min_wi + kw;
        // skip elements outside of the input feature map
        if (wi < 0 || wi > input_dimW - 1) {
          continue;
        }
        // corresponding mask location: index1(group,kh,kw)
        int mask_c = (mask_group * kernel_size + kh) * kernel_size + kw;
        // calculate the weighted sum
        // output[no,ho,wo,co] = input[no,hi,wi,co] * mask[no,hi,wi,mask_c]
        int index1_input =
            co + input_dimC * (wi + input_dimW * (hi + input_dimH * no));
        int index1_mask =
            mask_c + mask_dimC * (wo + mask_dimW * (ho + mask_dimH * no));
        host_output[index1_output] +=
            host_input[index1_input] * host_mask[index1_mask];
        theory_ops_ += 2;  // one is multiply, the other is addition
      }                    // kernel_width
    }                      // kernel_height
  }                        // each output element
}

int64_t CarafeForwardExecutor::getTheoryOps() {
  VLOG(4) << "getTheoryOps: " << theory_ops_ << " ops";
  return theory_ops_;
}

}  // namespace mluoptest
