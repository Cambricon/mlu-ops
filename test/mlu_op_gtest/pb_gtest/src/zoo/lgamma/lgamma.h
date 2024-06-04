#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_LGAMMA_LGAMMA_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_LGAMMA_LGAMMA_H_
#include "executor.h"

namespace mluoptest {

class LgammaExecutor : public Executor {
public:
  LgammaExecutor() {}
  ~LgammaExecutor() {}

  void paramCheck();
  void compute();
  void cpuCompute();
  int64_t getTheoryOps() override;
};

} // namespace mluoptest


#endif  // TEST_MLU_OP_GTEST_SRC_ZOO_LGAMMA_LGAMMA_H_