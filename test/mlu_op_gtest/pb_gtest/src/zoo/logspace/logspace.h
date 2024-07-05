#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_LOGSPACE_LOGSPACE_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_LOGSPACE_LOGSPACE_H_
#include "executor.h"

namespace mluoptest {

class LogspaceExecutor : public Executor {
 public:
  LogspaceExecutor() {}
  ~LogspaceExecutor() {}

  void paramCheck();
  void compute();
  void cpuCompute();
  int64_t getTheoryOps() override;


 private:
  void initData();

  float start_num_;
  float end_num_;
  int steps_num_;
  float base_num_;
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_LOGSPACE_LOGSPACE_H_