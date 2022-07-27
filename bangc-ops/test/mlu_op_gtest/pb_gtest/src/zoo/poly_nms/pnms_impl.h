#ifndef TEST_MLU_OP_GTEST_PB_GTEST_SRC_ZOO_POLY_NMS_PNMS_IMPL_H_
#define TEST_MLU_OP_GTEST_PB_GTEST_SRC_ZOO_POLY_NMS_PNMS_IMPL_H_

#include <vector>
using namespace std;
namespace PNMS {
vector<int> PolyNmsImpl(vector<vector<float>> &p, const float thresh);
}
#endif  // TEST_MLU_OP_GTEST_PB_GTEST_SRC_ZOO_POLY_NMS_PNMS_IMPL_H_
