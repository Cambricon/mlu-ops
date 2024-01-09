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
#ifndef TEST_MLU_OP_SRC_ZOO_INDICE_CONVOLUTION_FORWARD_INDICE_CONVOLUTION_FORWARD_H_  // NOLINT
#define TEST_MLU_OP_SRC_ZOO_INDICE_CONVOLUTION_FORWARD_INDICE_CONVOLUTUIN_FORWARD_H_  // NOLINT
#include <string>
#include <vector>

#include "executor.h"

namespace mluoptest {
class IndiceConvolutionForwardExecutor : public Executor {
 public:
  void compute() override;
  void paramCheck() override;
  void cpuCompute() override;
  void workspaceMalloc() override;
  void workspaceFree() override;
  int64_t getTheoryOps() override;
  int64_t getTheoryIoSize() override;

 private:
  void paramInit();
  mluOpTensorDescriptor_t features_desc_ = nullptr;
  mluOpTensorDescriptor_t filters_desc_ = nullptr;
  mluOpTensorDescriptor_t indice_pairs_desc_ = nullptr;
  mluOpTensorDescriptor_t features_out_desc_ = nullptr;
  std::vector<int64_t> indice_num_;
  std::string op_name_ = "mluOpIndiceConvolutionForward";
  size_t workspace_size_ = 0;
  int64_t inverse_ = 0;
  int64_t sub_m_ = 0;
  int64_t num_active_out_ = 0;
};
}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_INDICE_CONVOLUTION_FORWARD_INDICE_CONVOLUTION_FORWARD_H_  // NOLINT
