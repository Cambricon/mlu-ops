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
#include "stride.h"

namespace mluoptest {

void stride_map(void *dst,                              // dst ptr
                void *src,                              // src ptr
                const std::vector<size_t> &shape,       // shape
                const std::vector<size_t> &dst_stride,  // stride
                const std::vector<size_t> &src_stride,  // stride
                size_t dst_offset, size_t src_offset, size_t d,
                size_t sizeof_dtype, const size_t dst_max,
                const size_t src_max) {
  if (d == shape.size() - 1) {  // the last dim
    for (size_t i = 0; i < shape[d]; ++i) {
      size_t dst_idx = src_offset + i * src_stride[d];
      size_t src_idx = dst_offset + i * dst_stride[d];
      memcpy((char *)dst + dst_idx * sizeof_dtype,
             (char *)src + src_idx * sizeof_dtype, sizeof_dtype);
    }
  } else {
    for (size_t i = 0; i < shape[d]; ++i) {
      stride_map(dst, src, shape, dst_stride, src_stride,
                 dst_offset + i * dst_stride[d], src_offset + i * src_stride[d],
                 d + 1, sizeof_dtype, dst_max, src_max);
    }
  }
}

// src(strided) -> dst(shape)
// dst should malloc by shape_count
// src should malloc by stride_count
void tensor_stride_in(void *dst, void *src, const std::vector<size_t> &shape,
                      const std::vector<size_t> &dst_stride,  // dst_stride
                      size_t sizeof_dtype) {
  GTEST_CHECK(shape.size() == dst_stride.size(),
              "shape's size is not equal to stride's size.");

  size_t shape_total = std::accumulate(shape.begin(), shape.end(), (size_t)1,
                                       std::multiplies<size_t>());
  size_t stride_total = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    stride_total += (shape[i] - 1) * dst_stride[i];
  }

  std::vector<size_t> src_stride(shape.size());
  size_t stride_base = 1;
  for (ssize_t i = shape.size() - 1; i >= 0; --i) {
    src_stride[i] = stride_base;
    stride_base *= shape[i];
  }
  stride_map(dst, src, shape, dst_stride, src_stride, 0, 0, 0, sizeof_dtype,
             stride_total, shape_total);
}

// src(shape) -> dst(strided)
// dst should malloc by stride_count
// src should malloc by shape_count
void tensor_stride_out(void *dst, void *src, const std::vector<size_t> &shape,
                       const std::vector<size_t> &src_stride,  // src_stride
                       size_t sizeof_dtype) {
  GTEST_CHECK(shape.size() == src_stride.size(),
              "shape's size is not equal to stride's size.");

  size_t shape_total = std::accumulate(shape.begin(), shape.end(), (size_t)1,
                                       std::multiplies<size_t>());
  size_t stride_total = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    stride_total += (shape[i] - 1) * src_stride[i];
  }

  std::vector<size_t> dst_stride(shape.size());
  size_t stride_base = 1;
  for (ssize_t i = shape.size() - 1; i >= 0; --i) {
    dst_stride[i] = stride_base;
    stride_base *= shape[i];
  }
  stride_map(dst, src, shape, dst_stride, src_stride, 0, 0, 0, sizeof_dtype,
             shape_total, stride_total);
}

class Stride::StrideImpl {
 public:
  StrideImpl() = default;
  // output reuse input tensor
  bool init_by_input_ = false;
  bool have_stride_ = false;
  // store input tensor used for init output when reuse
  void *tensor_copy_ = nullptr;
  void *tensor_in_ = nullptr;
  MetaTensor *ts_ = nullptr;
  StrideDirection sd_;
  CPURuntime *cpu_runtime_;

  void *strideOutputByDtype();
  inline bool tensor_have_stride(MetaTensor *ts) { return !ts->stride.empty(); }
};

Stride::Stride(
    CPURuntime *cpu_runtime)  // should use same cpu_runtime with executor
    : pimpl_(std::make_unique<StrideImpl>()) {
  pimpl_->cpu_runtime_ = cpu_runtime;
}
Stride::~Stride() {}

void Stride::setStrideAttr(void *tensor_in, void *tensor_copy, MetaTensor *ts,
                           bool init_by_input) {
  pimpl_->init_by_input_ = init_by_input;
  pimpl_->ts_ = ts;
  pimpl_->have_stride_ = pimpl_->tensor_have_stride(ts);
  pimpl_->tensor_in_ = tensor_in;
  pimpl_->tensor_copy_ = tensor_copy;
}

void *Stride::strideOutputByDtype() { return pimpl_->strideOutputByDtype(); }

void *Stride::StrideImpl::strideOutputByDtype() {
  if (have_stride_) {
    size_t dtype_size;
    MLUOP_CHECK(mluOpGetSizeOfDataType(ts_->dtype, &dtype_size));
    void *tensor_out = cpu_runtime_->allocate(ts_->total_count * dtype_size);
    if (init_by_input_) {
      memcpy(tensor_out, tensor_copy_, ts_->total_count * dtype_size);
    } else {
      memset(tensor_out, 0x0, ts_->total_count * dtype_size);
    }
    tensor_stride_out(tensor_out, tensor_in_, getTensorShapeSizeT(ts_),
                      getTensorStrideSizeT(ts_), dtype_size);
    cpu_runtime_->deallocate(tensor_in_);
    return tensor_out;
  } else {
    return tensor_in_;
  }
}
}  // namespace mluoptest
