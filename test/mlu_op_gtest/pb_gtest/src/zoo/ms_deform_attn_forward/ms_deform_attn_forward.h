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
#ifndef TEST_MLUOP_GTEST_PB_GTEST_SRC_ZOO_MS_DEFORM_ATTN_FORWARD_MS_DEFORM_ATTN_FORWARD_H_   // NOLINT
#define TEST_MLUOP_GTEST_PB_GTEST_SRC_ZOO_MS_DEFORM_ATTN_FORWARD_MS_DEFORM_ATTN_FORWARD_H_   // NOLINT
#include "executor.h"

namespace mluoptest {
class MsDeformAttnForwardExecutor : public Executor {
 public:
  MsDeformAttnForwardExecutor() {}
  ~MsDeformAttnForwardExecutor() {}
  void paramCheck();
  void compute();
  void cpuCompute();
  int64_t getTheoryIoSize() override;
  int64_t getTheoryOps() override;
 private:
  float ms_deform_attn_im2col_bilinear(
      const float *&bottom_data, const int &height, const int &width,
      const int &nheads, const int &channels, const float &h, const float &w,
      const int &m, const int &c);
  void cpuMsDeformAttnForward(
      const float *data_value, const float *data_spatial_shapes,
      const float *data_level_start_index, const float *data_sampling_loc,
      const float *data_attn_weight, const int batch_size, const int num_keys,
      const int num_heads, const int channels, const int num_levels,
      const int num_query, const int num_point, float *data_col);
};
}  // namespace mluoptest

#endif  // TEST_MLUOP_GTEST_PB_GTEST_SRC_ZOO_MS_DEFORM_ATTN_FORWARD_MS_DEFORM_ATTN_FORWARD_H_  // NOLINT
