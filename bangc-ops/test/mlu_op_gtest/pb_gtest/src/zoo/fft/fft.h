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
#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_FFT_FFT_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_FFT_FFT_H_
#include <set>
#include <vector>
#include "executor.h"

namespace mluoptest {

class FftExecutor : public Executor {
 public:
  FftExecutor() {}
  ~FftExecutor() {}

  void paramCheck() override;
  void workspaceMalloc() override;
  void compute() override;
  void cpuCompute() override;
  void workspaceFree() override;
  int64_t getTheoryOps() override;
  int64_t getTheoryIoSize() override;
  std::set<Evaluator::Formula> getCriterionsUse() const override;

 private:
  mluOpFFTPlan_t fft_plan_;
  size_t reservespace_size_ = 0, workspace_size_ = 0;
  void *reservespace_addr_ = nullptr;
  void *workspace_addr_ = nullptr;
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_FFT_FFT_H_
