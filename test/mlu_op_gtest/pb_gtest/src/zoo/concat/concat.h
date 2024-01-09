/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_CONCAT_CONCAT_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_CONCAT_CONCAT_H_
#include <vector>
#include "executor.h"

namespace mluoptest {

class ConcatExecutor : public Executor {
 public:
  ConcatExecutor() {}
  ~ConcatExecutor() {}

  void paramCheck();
  void workspaceMalloc();
  void compute();
  void workspaceFree();
  void cpuCompute();
  void cpuConcat(std::vector<TensorPair> input_desc, std::vector<float *> input,
                 int input_num, int axis_t, float *output);
  int64_t getTheoryOps() override;

 private:
  int axis_;
  int input_num_;
  size_t workspace_size_;
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_CONCAT_CONCAT_H_
