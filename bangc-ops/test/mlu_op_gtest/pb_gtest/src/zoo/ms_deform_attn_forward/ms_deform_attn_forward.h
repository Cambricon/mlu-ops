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
#include <vector>
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
};
}  // namespace mluoptest

mluOpStatus_t MLUOP_WIN_API
mluOpMsDeformAttnForward(
    mluOpHandle_t handle,
    const mluOpTensorDescriptor_t data_value_desc,
    const void *data_value,
    const mluOpTensorDescriptor_t data_spatial_shapes_desc,
    const void *data_spatial_shapes,
    const mluOpTensorDescriptor_t data_level_start_index_desc,
    const void *data_level_start_index,
    const mluOpTensorDescriptor_t data_sampling_loc_desc,
    const void *data_sampling_loc,
    const mluOpTensorDescriptor_t data_attn_weight_desc,
    const void *data_attn_weight,
    const int im2col_step,
    const mluOpTensorDescriptor_t data_col_desc,
    void *data_col);

#endif  // TEST_MLUOP_GTEST_PB_GTEST_SRC_ZOO_MS_DEFORM_ATTN_FORWARD_MS_DEFORM_ATTN_FORWARD_H_  // NOLINT
