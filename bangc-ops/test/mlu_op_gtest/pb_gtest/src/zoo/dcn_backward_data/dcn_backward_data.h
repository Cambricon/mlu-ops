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

#ifndef TEST_MLUOP_GTEST_PBGTEST_SRC_ZOO_DCN_BACKWARD_DATA_DCN_BACKWARD_DATA_H_
#define TEST_MLUOP_GTEST_PBGTEST_SRC_ZOO_DCN_BACKWARD_DATA_DCN_BACKWARD_DATA_H_
#include <vector>
#include "executor.h"

#define MAX_PAD_DIM 6
#define MAX_STRIDE_DIM 3
#define MAX_DILATION_DIM 3

namespace mluoptest {

class DcnBackwardDataExecutor : public Executor {
 public:
  DcnBackwardDataExecutor() {}
  ~DcnBackwardDataExecutor() {}

  void paramCheck();
  void compute();
  void workspaceMalloc();
  void workspaceFree();
  void cpuCompute();
  int64_t getTheoryOps() override;

 private:
  bool use_mask_;
  mluOpDCNDescriptor_t dcn_desc_;
  mluOpTensorDescriptor_t input_desc_, offset_desc_, mask_desc_, weight_desc_,
      grad_output_desc_;
  mluOpTensorDescriptor_t grad_input_desc_, grad_offset_desc_, grad_mask_desc_;
  size_t workspace_size_;
  int pad_[MAX_PAD_DIM] = {0};
  int stride_[MAX_STRIDE_DIM] = {1};
  int dilation_[MAX_DILATION_DIM] = {1};
  int dimNb_ = 4;
  int deformable_group_ = 1;
  int conv_group_ = 1;
  int im2col_step_ = 1;
  int64_t theory_ops_ = 0;
  mluOpDataType_t compute_type_ = MLUOP_DTYPE_FLOAT;
  mluOpDataType_t grad_output_oc_dt_, weight_oc_dt_;
  void *dev_workspace_ = nullptr;
  void hostDCNBackwardData(
      const mluOpTensorDescriptor_t input_desc, float *input,
      const mluOpTensorDescriptor_t offset_desc, float *offset,
      const mluOpTensorDescriptor_t mask_desc, float *mask,
      const mluOpTensorDescriptor_t weight_desc, float *weight,
      const mluOpTensorDescriptor_t grad_output_desc, float *output,
      mluOpTensorDescriptor_t grad_input_desc, float *grad_input,
      mluOpTensorDescriptor_t grad_offset_desc, float *grad_offset,
      mluOpTensorDescriptor_t grad_mask_desc, float *grad_mask);

  void transposeGradOutput(const float *output, const int group, const int ndhw,
                           const int c, float *transpose_output);
  void transposeGradCol(const float *dcol, const int group, const int middle,
                        const int c, float *transpose_dcol);

  void batch_batmul(int batch, int m, int k, int n, float *mat1, float *mat2,
                    float *mat3);

  void col2img4D(float *grad_col, float *input, float *offset, float *mask,
                 float *grad_input, float *grad_offset, float *grad_mask, int n,
                 int hi, int wi, int ci, int co, int kh, int kw, int pad[],
                 int stride[], int dilation[], int deformable_group,
                 int conv_group);

  void im2col_bilinear(float *input, int n_iter, float h_im, float w_im,
                       int deform_iter, int hi, int wi, int deform_group,
                       int ci_per_deform_group, float *bilinear_result);

  void col2img_coordinate(float *input, int n_iter, float h_im, float w_im,
                          int deform_iter, int hi, int wi, int deform_group,
                          int ci_per_deform_group, float *weight_h,
                          float *weight_w);
  int getCoefficientOfLT2CT();
};

}  // namespace mluoptest
#endif  // TEST_MLUOP_GTEST_PBGTEST_SRC_ZOO_DCN_BACKWARD_DATA_DCN_BACKWARD_DATA_H_
