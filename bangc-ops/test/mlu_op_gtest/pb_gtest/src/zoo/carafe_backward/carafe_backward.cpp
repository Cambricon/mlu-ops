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
#include "carafe_backward.h"
#include "mlu_op.h"

namespace mluoptest {
void CarafeBackwardExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_carafe_backward_param()) {
    LOG(ERROR) << "Missing carafe param. ";
  }
}

void CarafeBackwardExecutor::compute() {
  auto carafe_desc_node = parser_->getProtoNode()->carafe_backward_param();
  int dimnb = carafe_desc_node.dimnb();
  int kernel_size = carafe_desc_node.kernel_size();
  int group_size = carafe_desc_node.group_size();
  int scale_factor = carafe_desc_node.scale_factor();

  mluOpCarafeDescriptor_t carafe_desc = cpu_runtime_.allocate(
      mluOpCreateCarafeDescriptor, mluOpDestroyCarafeDescriptor);
  MLUOP_CHECK(mluOpSetCarafeDescriptor(carafe_desc, dimnb, kernel_size,
                                       group_size, scale_factor));

  void *dev_input = data_vector_[0].device_ptr;
  void *dev_mask = data_vector_[1].device_ptr;
  void *dev_grad_output = data_vector_[2].device_ptr;
  void *dev_grad_input = data_vector_[3].device_ptr;
  void *dev_grad_mask = data_vector_[4].device_ptr;

  auto input_desc = tensor_desc_[0].tensor;
  auto mask_desc = tensor_desc_[1].tensor;
  auto grad_output_desc = tensor_desc_[2].tensor;
  auto grad_input_desc = tensor_desc_[3].tensor;
  auto grad_mask_desc = tensor_desc_[4].tensor;

  interface_timer_.start();
  MLUOP_CHECK(mluOpCarafeBackward(
      handle_, carafe_desc, input_desc, dev_input, mask_desc, dev_mask,
      grad_output_desc, dev_grad_output, grad_input_desc, dev_grad_input,
      grad_mask_desc, dev_grad_mask));
  interface_timer_.stop();
}

void CarafeBackwardExecutor::cpuCompute() {
  auto carafe_desc_node = parser_->getProtoNode()->carafe_backward_param();

  int kernel_size = carafe_desc_node.kernel_size();
  int group_size = carafe_desc_node.group_size();
  int scale_factor = carafe_desc_node.scale_factor();

  float *host_input = cpu_fp32_input_[0];
  float *host_mask = cpu_fp32_input_[1];
  float *host_grad_output = cpu_fp32_input_[2];
  float *host_grad_input = cpu_fp32_output_[0];
  float *host_grad_mask = cpu_fp32_output_[1];

  auto input_desc = tensor_desc_[0].tensor;
  auto mask_desc = tensor_desc_[1].tensor;
  auto grad_output_desc = tensor_desc_[2].tensor;

  int n = mluOpGetTensordimN(input_desc);
  int hi = mluOpGetTensordimH(input_desc);
  int wi = mluOpGetTensordimW(input_desc);
  int ci = mluOpGetTensordimC(input_desc);
  int c_per_group = ci / group_size;

  int ho = mluOpGetTensordimH(grad_output_desc);
  int wo = mluOpGetTensordimW(grad_output_desc);

  for (int i = 0; i < n * hi * wi * ci; i++) {
    host_grad_input[i] = 0;
  }
  for (int i = 0; i < n * ho * wo * group_size * kernel_size * kernel_size;
       i++) {
    host_grad_mask[i] = 0.0;
  }
  for (int i_iter = 0; i_iter < n * ho * wo * group_size; i_iter++) {
    int group_iter = i_iter % group_size;
    int w_iter = (i_iter / group_size) % wo;
    int h_iter = (i_iter / wo / group_size) % ho;
    int n_iter = (i_iter / ho / wo / group_size) % n;

    int down_h_iter = h_iter / scale_factor;
    int down_w_iter = w_iter / scale_factor;

    int start_h = down_h_iter - (kernel_size - 1) / 2;
    int end_h = down_h_iter + (kernel_size - 1) / 2 + 1;
    int start_w = down_w_iter - (kernel_size - 1) / 2;
    int end_w = down_w_iter + (kernel_size - 1) / 2 + 1;

    for (int ih_iter = start_h; ih_iter < end_h; ih_iter++) {
      for (int iw_iter = start_w; iw_iter < end_w; iw_iter++) {
        if (ih_iter < 0 || ih_iter > hi - 1 || iw_iter < 0 ||
            iw_iter > wi - 1) {
          continue;
        }
        int mask_ih = ih_iter - down_h_iter + (kernel_size - 1) / 2;
        int mask_iw = iw_iter - down_w_iter + (kernel_size - 1) / 2;
        int mask_c = group_iter * kernel_size * kernel_size +
                     mask_ih * kernel_size + mask_iw;
        int mask_offset =
            n_iter * ho * wo * group_size * kernel_size * kernel_size +
            h_iter * wo * group_size * kernel_size * kernel_size +
            w_iter * group_size * kernel_size * kernel_size + mask_c;
        float mask_value = host_mask[mask_offset];
        int input_offset = n_iter * hi * wi * ci + ih_iter * wi * ci +
                           iw_iter * ci + group_iter * c_per_group;
        int grad_output_offset = n_iter * ho * wo * ci + h_iter * wo * ci +
                                 w_iter * ci + group_iter * c_per_group;
        for (int iter = 0; iter < c_per_group; iter++) {
          host_grad_input[input_offset + iter] +=
              mask_value * host_grad_output[grad_output_offset + iter];
          host_grad_mask[mask_offset] +=
              host_input[input_offset + iter] *
              host_grad_output[grad_output_offset + iter];
          theory_ops_ += 2;
        }  // iter
      }    // iw iter
    }      // ih iter
  }        // i iter
}

int64_t CarafeBackwardExecutor::getTheoryOps() {
  VLOG(4) << "getTheoryOps: " << theory_ops_ << " ops";
  return theory_ops_;
}
}  // namespace mluoptest
