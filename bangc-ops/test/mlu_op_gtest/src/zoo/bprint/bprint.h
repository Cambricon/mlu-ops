#ifndef TEST_MLU_OP_GTEST_SRC_ZOO_BPRINT_BPRINT_H_
#define TEST_MLU_OP_GTEST_SRC_ZOO_BPRINT_BPRINT_H_
#include "executor.h"

namespace mluoptest {

class BprintExecutor : public Executor {
  public:
    BprintExecutor() {}
    ~BprintExecutor() {}

    void paramCheck();
    void compute();
    void cpuCompute();
    int64_t getTheoryOps() override;
};

} // namespace mluoptest
#endif // TEST_MLU_OP_GTEST_SRC_ZOO_BPRINT_BPRINT_H_

