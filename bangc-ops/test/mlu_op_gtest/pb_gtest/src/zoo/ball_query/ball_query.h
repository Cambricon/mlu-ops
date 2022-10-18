#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_BALL_QUERY_BALL_QUERY_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_BALL_QUERY_BALL_QUERY_H_

#include "executor.h"

namespace mluoptest {
class BallQueryExecutor : public Executor {
 public:
  BallQueryExecutor() {}
  ~BallQueryExecutor() {}

  void paramCheck();
  void compute();
  void cpuCompute();
  int64_t getTheoryOps() override;

 private:
  float min_radius_;
  float max_radius_;
  int nsample_;
};
}  // namespace mluoptest

#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_BALLQUERY_BALLQUERY_H_
