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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_PSROIPOOL_FORWARD_PSROIPOOL_FORWARD_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_PSROIPOOL_FORWARD_PSROIPOOL_FORWARD_H_
#include "executor.h"

namespace mluoptest {
class PsroipoolForwardExecutor : public Executor {
 public:
  PsroipoolForwardExecutor() {}
  ~PsroipoolForwardExecutor() {}
  void paramCheck() override;
  void compute() override;
  void cpuCompute() override;
  int64_t getTheoryOps() override;

 private:
  float spatial_scale_;
  int group_size_;
  int64_t theory_ops_ = 0;
};
}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_PSROIPOOL_FORWARD_PSROIPOOL_FORWARD_H_
