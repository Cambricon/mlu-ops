/*************************************************************************
 * Copyright (C) [2019-2023] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/

#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_SYNC_BATCHNORM_STATS_SYNC_BATCHNORM_STATS_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_SYNC_BATCHNORM_STATS_SYNC_BATCHNORM_STATS_H_

#include "executor.h"

namespace mluoptest {

class SyncBatchnormStatsExecutor : public Executor {
 public:
  SyncBatchnormStatsExecutor() {}
  ~SyncBatchnormStatsExecutor() {}

  void paramCheck();
  void workspaceMalloc();
  void workspaceFree();
  void compute();
  void cpuCompute();
  int64_t getTheoryOps() override;

 private:
  size_t workspace_size_ = 0;
};

}  // namespace mluoptest

#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_SYNC_BATCHNORM_STATS_SYNC_\
BATCHNORM_STATS_H_
