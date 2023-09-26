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

#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_SYNC_BATCHNORM_GATHER_STATS_WITH_COUNTS_\
SYNC_BATCHNORM_GATHER_STATS_WITH_COUNTS_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_SYNC_BATCHNORM_GATHER_STATS_WITH_COUNTS_\
SYNC_BATCHNORM_GATHER_STATS_WITH_COUNTS_H_
#include <set>
#include <vector>

#include "executor.h"

namespace mluoptest {

class SyncBatchnormGatherStatsWithCountsExecutor : public Executor {
 public:
  SyncBatchnormGatherStatsWithCountsExecutor() {}
  ~SyncBatchnormGatherStatsWithCountsExecutor() {}

  void paramCheck();
  void compute();
  void cpuCompute();
  int64_t getTheoryOps() override;
  std::set<Evaluator::Formula> getCriterionsUse() const override;
};

}  // namespace mluoptest

#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_SYNC_BATCHNORM_GATHER_STATS_WITH_COUNTS\
_SYNC_BATCHNORM_GATHER_STATS_WITH_COUNTS_H_
