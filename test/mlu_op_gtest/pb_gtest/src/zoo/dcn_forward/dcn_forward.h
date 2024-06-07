/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef TEST_MLUOP_GTEST_SRC_ZOO_DCN_FORWARD_DCN_FORWARD_H_
#define TEST_MLUOP_GTEST_SRC_ZOO_DCN_FORWARD_DCN_FORWARD_H_

#include <vector>
#include "executor.h"

namespace mluoptest {

class DcnForwardExecutor : public Executor {
 public:
  DcnForwardExecutor() {}
  ~DcnForwardExecutor() {}

  void workspaceMalloc();
  void workspaceFree();
  void paramCheck();
  void compute();
  void cpuCompute();
  int64_t getTheoryOps() override;

 private:
  int getCoefficientOfLT2CT();
  void transpose(float *input, float *output, const int dims[],
                 const int dim_num, const int permute[]);
  void computeDCNForwardCPU(
      const int &dg, const int &g, const int &im2col_step,
      const mluOpTensorDescriptor_t input_desc, const void *cpu_input,
      const mluOpTensorDescriptor_t offset_desc, const void *cpu_offset,
      const mluOpTensorDescriptor_t mask_desc, const void *cpu_mask,
      const mluOpTensorDescriptor_t weight_desc, const void *cpu_weight,
      const mluOpTensorDescriptor_t bias_desc, const void *cpu_bias,
      const mluOpTensorDescriptor_t output_desc, const void *cpu_output,
      float *buffer, int pad[], int stride[], int dilation[],
      int64_t &theory_ops);
  mluOpDataType_t input_onchip_dtype;
  mluOpDataType_t weight_onchip_dtype;

  mluOpTensorDescriptor_t input_desc;
  mluOpTensorDescriptor_t offset_desc;
  mluOpTensorDescriptor_t mask_desc = nullptr;  // optional
  mluOpTensorDescriptor_t output_desc;
  mluOpTensorDescriptor_t weight_desc;
  mluOpTensorDescriptor_t bias_desc = nullptr;  // optional

  int dimnb;
  int pad[4];
  int stride[2];
  int dilation[2];
  int dg;
  int g;
  int im2col_step;

  void *input = nullptr;
  void *offset = nullptr;
  void *mask = nullptr;
  void *output = nullptr;
  void *weight = nullptr;
  void *bias = nullptr;

  void *cpu_input = nullptr;
  void *cpu_offset = nullptr;
  void *cpu_mask = nullptr;
  void *cpu_output = nullptr;
  void *cpu_weight = nullptr;
  void *cpu_bias = nullptr;

  void *workspace = nullptr;
  size_t workspace_size = 0;
  int64_t theory_ops = 0;
};

}  // namespace mluoptest
#endif  // TEST_MLUOP_GTEST_SRC_ZOO_DCN_FORWARD_DCN_FORWARD_H_
