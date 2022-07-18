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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_PSROIPOOL_BACKWARD_PSROIPOOL_BACKWARD_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_PSROIPOOL_BACKWARD_PSROIPOOL_BACKWARD_H_
#include "executor.h"

namespace mluoptest {
class PsroipoolBackwardExecutor : public Executor {
 public:
  PsroipoolBackwardExecutor() {}
  ~PsroipoolBackwardExecutor() {}
  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;

 private:
  void initData();
  void transposeNchwToNhwc(const float *in, const uint32_t dim0,
                           const uint32_t dim1, const uint32_t dim2,
                           const uint32_t dim3, float *out);
  void transposeNhwcToNchw(const float *in, const uint32_t dim0,
                           const uint32_t dim1, const uint32_t dim2,
                           const uint32_t dim3, float *out);
  int output_dim_;
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};
}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_PSROIPOOL_BACKWARD_PSROIPOOL_BACKWARD_H_
