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
#ifndef TEST_MLU_OP_GTEST_INCLUDE_STRIDE_H_
#define TEST_MLU_OP_GTEST_INCLUDE_STRIDE_H_

#include <memory>
#include <vector>

#include "runtime.h"
#include "parser.h"

// stride for one tensor

namespace mluoptest {

// TODO(niewenchang): currently useless,
// use it to integrage stride_in and stride_out in the future
enum StrideDirection {
  STRIDE_IN = 1,   // total_count -> shape_count
  STRIDE_OUT = 2,  // shape_count -> total_count
};

void tensor_stride_in(void *dst, void *src, const std::vector<size_t> &shape,
                      const std::vector<size_t> &dst_stride,
                      size_t sizeof_dtype);

void tensor_stride_out(void *dst, void *src, const std::vector<size_t> &shape,
                       const std::vector<size_t> &src_stride,
                       size_t sizeof_dtype);

inline std::vector<size_t> getTensorShapeSizeT(MetaTensor *ts) {
  std::vector<size_t> shape_vec(ts->shape.begin(), ts->shape.end());
  return shape_vec;
}

inline std::vector<size_t> getTensorStrideSizeT(MetaTensor *ts) {
  std::vector<size_t> stride_vec(ts->stride.begin(), ts->stride.end());
  return stride_vec;
}

class Stride {
 public:
  // this class use the same cpu_runtime object with executor
  explicit Stride(CPURuntime *cpu_runtime);
  ~Stride();
  Stride(const Stride &) = delete;
  Stride &operator=(const Stride &) = delete;
  Stride(Stride &&) = default;
  Stride &operator=(Stride &&) = default;
  void setStrideAttr(void *tensor_in, void *tensor_copy, MetaTensor *ts,
                     bool init_by_input);
  void *strideOutputByDtype();

 private:
  class StrideImpl;
  std::unique_ptr<StrideImpl> pimpl_;
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_INCLUDE_STRIDE_H_
