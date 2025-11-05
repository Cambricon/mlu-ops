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

/************************************************************************
 *
 *  @runtime_tools_test.cc
 *
 *
 **************************************************************************/
#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <algorithm>
#include <cstdlib>
#include "variable.h"
#include "mlu_op_gtest.h"
#include "case_collector.h"

using namespace ::testing;
TEST_P(TestSuite, mluOp) {
  Run();
}

// register op testcase here.
// AUTO GENERATE START
INSTANTIATE_TEST_CASE_P(add_tensor, TestSuite, ::testing::ValuesIn(list_case("add_tensor")));
INSTANTIATE_TEST_CASE_P(convbpfilter, TestSuite, ::testing::ValuesIn(list_case("convbpfilter")));
// AUTO GENERATE END
