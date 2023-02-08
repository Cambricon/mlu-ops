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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_GET_INDICE_PAIRS_GET_INDICE_PAIRS_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_GET_INDICE_PAIRS_GET_INDICE_PAIRS_H_
#include <string>
#include <vector>
#include "executor.h"
#include "core/tensor.h"
#include "mlu_op.h"

namespace mluoptest {
class GetIndicePairsExecutor : public Executor {
 public:
  GetIndicePairsExecutor();
  ~GetIndicePairsExecutor();

  void paramCheck() override;
  void compute() override;
  void workspaceMalloc() override;
  void workspaceFree() override;
  // void castOut() override;
  void diffPreprocess() override;
  void castIn() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;
  int64_t getTheoryIoSize() override;
  void *input_host_ = nullptr;

 private:
  void initParam();
  void cpuGetIndicePairs(
      int32_t *indice_in, int32_t *indice_pairs, int32_t *indice_out,
      int32_t *indice_num, int32_t *grid_out,
      mluOpTensorDescriptor_t indice_in_desc, std::vector<int32_t> kernel_size,
      std::vector<int32_t> pad, std::vector<int32_t> stride,
      std::vector<int32_t> dilation, std::vector<int32_t> out_spatail_shape,
      const int32_t dimNb, const int32_t sub_m, const int32_t batch_size);
  int32_t getValidOutPos(int32_t *input_pos, std::vector<int32_t> kernel_size,
                         std::vector<int32_t> pad, std::vector<int32_t> stride,
                         std::vector<int32_t> dilation,
                         std::vector<int32_t> out_spatail_shape, int32_t *out,
                         int NDim);
  int32_t dimNb_;
  int32_t batch_;
  int32_t sub_m_;
  int32_t transpose_ = 0;
  int32_t inverse_ = 0;
  std::vector<int32_t> pad_;
  std::vector<int32_t> stride_;
  std::vector<int32_t> dilation_;
  std::vector<int32_t> input_space_;
  std::vector<int32_t> filter_space_;
  std::vector<int32_t> output_space_;
  size_t workspace_size_ = 0;
  mluOpTensorDescriptor_t indice_in_desc_ = nullptr;
  mluOpTensorDescriptor_t indice_pairs_desc_ = nullptr;
  mluOpTensorDescriptor_t indice_out_desc_ = nullptr;
  mluOpTensorDescriptor_t indice_num_desc_ = nullptr;
  mluOpSparseConvolutionDescriptor_t sparse_conv_desc_ = nullptr;
  std::string op_name_ = std::string("getIndicePairs");
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_GET_INDICE_PAIRS_GET_INDICE_PAIRS_H_
