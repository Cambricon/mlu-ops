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
#include "roi_pooling_backward.h"

namespace mluoptest {

void RoiPoolingBackwardExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_roi_pooling_backward_param()) {
    LOG(ERROR) << "mluOpRoiPoolingBackward: lose param.";
  }
  if (parser_->getInputNum() != 3) {
    LOG(ERROR) << "mluOpRoiPoolingBackward: tensor grads number is wrong.";
  }
  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "mluOpRoiPoolingBackward: grads_image number is wrong.";
  }
}

void RoiPoolingBackwardExecutor::compute() {
  auto grads = parser_->getMetaTensor(0).dev_ptr;
  auto rois = parser_->getMetaTensor(1).dev_ptr;
  auto argmax = parser_->getMetaTensor(2).dev_ptr;
  auto grads_image = parser_->getMetaTensor(3).dev_ptr;

  auto grads_desc = parser_->getMetaTensor(0).tensor;
  auto rois_desc = parser_->getMetaTensor(1).tensor;
  auto argmax_desc = parser_->getMetaTensor(2).tensor;
  auto grads_image_desc = parser_->getMetaTensor(3).tensor;
  float spatial_scale =
      parser_->getProtoNode()->roi_pooling_backward_param().spatial_scale();
  PoolingForwardMode mode =
      parser_->getProtoNode()->roi_pooling_backward_param().mode();
  mluOpPoolingMode_t cmode;
  if (mode == POOLING_MAX) {
    cmode = MLUOP_POOLING_MAX;
  }

  VLOG(4) << "call mluop mluOpRoiPoolingBackward()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpRoiPoolingBackward(
      handle_, cmode, grads_desc, grads, rois_desc, rois, argmax_desc,
      (int *)argmax, spatial_scale, grads_image_desc, grads_image));
  interface_timer_.stop();
}

void RoiPoolingBackwardExecutor::cpuCompute() {
  VLOG(4) << "call cpuCompute()";
  auto grads = parser_->getMetaTensor(0).cpu_ptr;
  auto rois = parser_->getMetaTensor(1).cpu_ptr;
  auto argmax = parser_->getMetaTensor(2).cpu_ptr;
  auto grads_image = parser_->getMetaTensor(3).cpu_ptr;
  auto grads_desc = parser_->getMetaTensor(0).tensor;
  auto rois_desc = parser_->getMetaTensor(1).tensor;
  auto argmax_desc = parser_->getMetaTensor(2).tensor;
  auto grads_image_desc = parser_->getMetaTensor(3).tensor;
  float spatial_scale =
      parser_->getProtoNode()->roi_pooling_backward_param().spatial_scale();
  PoolingForwardMode mode =
      parser_->getProtoNode()->roi_pooling_backward_param().mode();

  size_t grads_n = grads_desc->getDimIndex(0);
  size_t grads_h = grads_desc->getDimIndex(1);
  size_t grads_w = grads_desc->getDimIndex(2);
  size_t grads_c = grads_desc->getDimIndex(3);
  size_t num1 = rois_desc->getDimIndex(0);
  size_t num2 = rois_desc->getDimIndex(1);
  size_t argmax_n = argmax_desc->getDimIndex(0);
  size_t argmax_h = argmax_desc->getDimIndex(1);
  size_t argmax_w = argmax_desc->getDimIndex(2);
  size_t argmax_c = argmax_desc->getDimIndex(3);
  size_t grads_image_n = grads_image_desc->getDimIndex(0);
  size_t grads_image_h = grads_image_desc->getDimIndex(1);
  size_t grads_image_w = grads_image_desc->getDimIndex(2);
  size_t grads_image_c = grads_image_desc->getDimIndex(3);

  const int batch_size = grads_image_n;
  const int channels = grads_image_c;
  const int height = grads_image_h;
  const int width = grads_image_w;
  const int pooled_height = grads_h;
  const int pooled_width = grads_w;
  const int num_rois = argmax_n;

  auto transData = [&](float *old_data, float *new_data, TensorLayout old_order,
                       TensorLayout new_order, int n, int c, int d, int h,
                       int w) {
    if (old_data == nullptr || new_data == nullptr) {
      LOG(ERROR) << "data address do not malloc in cpu compute.";
      return;
    }

    if (LAYOUT_NDHWC == old_order && LAYOUT_NCDHW == new_order) {
      for (int nn = 0; nn < n; nn++) {
        for (int cc = 0; cc < c; cc++) {
          for (int dd = 0; dd < d; dd++) {
            for (int hh = 0; hh < h; hh++) {
              for (int ww = 0; ww < w; ww++) {
                new_data[nn * c * d * h * w + cc * d * h * w + dd * h * w +
                         hh * w + ww] =
                    old_data[nn * d * h * w * c + dd * h * w * c + hh * w * c +
                             ww * c + cc];
              }
            }
          }
        }
      }
    } else if (LAYOUT_NCDHW == old_order && LAYOUT_NDHWC == new_order) {
      for (int nn = 0; nn < n; nn++) {
        for (int cc = 0; cc < c; cc++) {
          for (int dd = 0; dd < d; dd++) {
            for (int hh = 0; hh < h; hh++) {
              for (int ww = 0; ww < w; ww++) {
                new_data[nn * c * d * h * w + dd * h * w * c + hh * w * c +
                         ww * c + cc] =
                    old_data[nn * d * h * w * c + cc * d * h * w + dd * h * w +
                             hh * w + ww];
              }
            }
          }
        }
      }
    }
  };

  float *top_diff = (float *)cpu_runtime_.allocate(
      grads_n * 1 * grads_c * grads_h * grads_w * sizeof(float));
  transData(grads, top_diff, LAYOUT_NDHWC, LAYOUT_NCDHW, grads_n, grads_c, 1,
            grads_h, grads_w);
  float *argmax_data = (float *)cpu_runtime_.allocate(
      argmax_n * 1 * argmax_c * argmax_h * argmax_w * sizeof(float));
  transData(argmax, argmax_data, LAYOUT_NDHWC, LAYOUT_NCDHW, argmax_n, argmax_c,
            1, argmax_h, argmax_w);
  float *bottom_diff = (float *)cpu_runtime_.allocate(
      grads_image_n * 1 * grads_image_c * grads_image_h * grads_image_w *
      sizeof(float));

  for (int i = 0; i < batch_size * channels * height * width; i++) {
    *((float *)bottom_diff + i) = 0;
  }
  for (int ind = 0; ind < num_rois * channels * pooled_height * pooled_width;
       ind++) {
    int pw = ind % pooled_width;
    int ph = (ind / pooled_width) % pooled_height;
    int c = (ind / pooled_width / pooled_height) % channels;
    int n = ind / pooled_width / pooled_height / channels;
    const float *offset_bottom_rois = rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    int bottom_offset = (roi_batch_ind * channels + c) * height * width;
    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const float *offset_top_diff = top_diff + top_offset;
    float *offset_bottom_diff = bottom_diff + bottom_offset;
    const float *offset_argmax_data = argmax_data + top_offset;
    int argmax = offset_argmax_data[ph * pooled_width + pw];

    offset_bottom_diff += argmax;
    if (argmax != -1) {
      *offset_bottom_diff += offset_top_diff[ph * pooled_width + pw];
    }
  }

  transData(bottom_diff, grads_image, LAYOUT_NCDHW, LAYOUT_NDHWC, grads_image_n,
            grads_image_c, 1, grads_image_h, grads_image_w);
  cpu_runtime_.deallocate(bottom_diff);
  cpu_runtime_.deallocate(top_diff);
  cpu_runtime_.deallocate(argmax_data);
  return;
}

int64_t RoiPoolingBackwardExecutor::getTheoryOps() {
  int64_t theory_ops = 0;
  float *host_argmax = nullptr;
  Device device = parser_->device();
  if (device != Device::CPU) {
    auto argmax_desc = tensor_desc_[2].tensor;
    auto argmax_dtype = argmax_desc->getDtype();
    size_t argmax_num = parser_->getInputDataCount(2);
    float *argmax = (float *)cpu_runtime_.allocate(argmax_num * sizeof(float));
    castDataOut(data_vector_[2].host_ptr, argmax_dtype, (float *)argmax,
                MLUOP_DTYPE_FLOAT, argmax_num, NO_QUANT, 0, 1, 0);
    host_argmax = argmax;
  } else {
    host_argmax = cpu_fp32_input_[2];
  }
  auto argmax = host_argmax;

  auto grads_desc = parser_->getMetaTensor(0).tensor;
  auto grads_image_desc = parser_->getMetaTensor(3).tensor;
  size_t grads_n = grads_desc->getDimIndex(0);
  size_t grads_h = grads_desc->getDimIndex(1);
  size_t grads_w = grads_desc->getDimIndex(2);
  size_t grads_c = grads_desc->getDimIndex(3);
  size_t grads_image_n = grads_image_desc->getDimIndex(0);
  size_t grads_image_h = grads_image_desc->getDimIndex(1);
  size_t grads_image_w = grads_image_desc->getDimIndex(2);
  size_t grads_image_c = grads_image_desc->getDimIndex(3);

  theory_ops += grads_image_n * grads_image_h * grads_image_w * grads_image_c;
  for (size_t i = 0; i < grads_n * grads_h * grads_w * grads_c; i++) {
    if (argmax[i] != -1) {
      theory_ops++;
    }
  }
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

std::set<Evaluator::Formula> RoiPoolingBackwardExecutor::getCriterionsUse()
    const {
  return {Evaluator::DIFF1, Evaluator::DIFF2, Evaluator::DIFF4};
}

}  // namespace mluoptest
