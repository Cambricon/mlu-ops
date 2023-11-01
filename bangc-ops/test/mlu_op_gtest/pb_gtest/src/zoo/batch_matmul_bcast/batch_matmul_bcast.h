
/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_BATCH_MATMUL_BCAST_BATCH_MATMUL_BCAST_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_BATCH_MATMUL_BCAST_BATCH_MATMUL_BCAST_H_
#include <vector>
#include "executor.h"
#include "core/type.h"

namespace mluoptest {

class BatchMatmulBcastExecutor : public Executor {
 public:
  BatchMatmulBcastExecutor() {}
  ~BatchMatmulBcastExecutor() {}

  void paramCheck();
  void cpuComputeForCastOutput();
  void baselineOutputMalloc();
  void setOutputQuantParam();
  void compute();
  void cpuCompute();

  void castIn() override;
  void castOut() override;
  void setQuantizedParam() override;
  void getBaselineOutput() override;

  void workspaceMalloc();
  void workspaceFree();
  void float2double(double *dst, float *src, int64_t num);
  void int2double(double *dst, void *src, mluOpDataType_t src_type, int64_t num);
  void qdouble2float(float *dst, double *src, double scale, int64_t num);
  int expandNumAfterFirst(int num);
  bool canBroadCast(std::vector<int> shape0, std::vector<int> shape1);
  void expandComputeCpu(std::vector<int> shape_a,
                        std::vector<int> shape_b,
                        float *input,
                        float *output);
  int64_t getTheoryOps() override;
  int64_t getTheoryIoSize() override;

  void setMiscellaneousParam() override;
 private:
  int pos_               = 0;
  float scale_           = 1.;
  int offset_            = 0;
  mluOpQuantizeMode_t quant_mode_;
  MatMulCastMode cast_mode_;
  mluOpMatMulDescriptor_t bmm_bcast_desc_ = nullptr;
  mluOpMatMulAlgo_t algo_ = nullptr;
  size_t workspace_size_ = 0;
  mluOpMatMulHeuristicResult_t heuristic_result_ = nullptr;
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_BATCH_MATMUL_BCAST_BATCH_MATMUL_BCAST_H_
