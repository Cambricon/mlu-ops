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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_MUTUAL_INFORMATION_FORWARD_\
MUTUAL_INFORMATION_FORWARD_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_MUTUAL_INFORMATION_FORWARD_\
MUTUAL_INFORMATION_FORWARD_H_

#include "core/tensor.h"
#include "executor.h"
#include "mlu_op.h"

namespace mluoptest {
namespace MutualInformationForward {
class Index3D {
 private:
  int S_ = 0;
  int T_ = 0;

 public:
  Index3D() {}
  explicit Index3D(int S, int T) : S_(S), T_(T) {}
  ~Index3D() {}

  int operator()(int b, int s, int t) {
    return b * T_ * S_ + s * T_ + t;
  }
};
}  // namespace MutualInformationForward

class MutualInformationForwardExecutor : public Executor {
 public:
  MutualInformationForwardExecutor() {}
  ~MutualInformationForwardExecutor() {}

  void workspaceMalloc() override;
  void workspaceFree() override;
  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  void setMiscellaneousParam() override;
  int64_t getTheoryOps() override;

 private:
  void initParam();
  void computeMutualInformation(const int b, const int s_begin, const int s_end,
                                const int t_begin, const int t_end, float *px,
                                float *py, float *p, float *ans);
  float logAdd(float x, float y);

  mluOpTensorDescriptor_t px_desc_ = nullptr;
  mluOpTensorDescriptor_t py_desc_ = nullptr;
  mluOpTensorDescriptor_t opt_boundary_desc_ = nullptr;
  mluOpTensorDescriptor_t p_desc_ = nullptr;
  mluOpTensorDescriptor_t ans_desc_ = nullptr;
  size_t workspace_size_ = 0;

  int B_ = 0;
  int S_ = 0;
  int T_ = 0;
  int64_t theory_ops_ = 0;
  const float min_log_diff_float = -15.9423847198486328125f;

  // max intput num is 4: px, py, opt_boundary, p
  // max output num is 2: p, ans
  const int max_tensor_num_ = 6;

  float *p_in_ = nullptr;

  mluoptest::MutualInformationForward::Index3D px_index_;
  mluoptest::MutualInformationForward::Index3D py_index_;
  mluoptest::MutualInformationForward::Index3D p_index_;
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_MUTUAL_INFORMATION_FORWARD_\
MUTUAL_INFORMATION_FORWARD_H_
