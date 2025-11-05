/*************************************************************************
 * Copyright (C) 2021 by Cambricon, Inc. All rights reserved.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_DIV_DIV_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_DIV_DIV_H_
#include <vector>
#include "core/type.h"
#include "executor.h"

namespace mluoptest {

class DivExecutor : public Executor {
 public:
  DivExecutor() {}
  ~DivExecutor() {}

  void paramCheck();
  void compute();
  void cpuCompute();
  int expand_num_after_first(int num);
  bool canBroadCast(std::vector<int> shape0, std::vector<int> shape1);
  void expand_compute_cpu(std::vector<int> shape_a,
                          std::vector<int> shape_b,
                          float *input,
                          float *output);
  int64_t getTheoryOps() override;

 private:
  size_t workspace_size = 0;
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_DIV_DIV_H_
