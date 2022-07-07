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
INSTANTIATE_TEST_CASE_P(abs, TestSuite, Combine(Values("abs"), Range(size_t(0), Collector("abs").num())));
INSTANTIATE_TEST_CASE_P(div, TestSuite, Combine(Values("div"), Range(size_t(0), Collector("div").num())));
INSTANTIATE_TEST_CASE_P(log, TestSuite, Combine(Values("log"), Range(size_t(0), Collector("log").num())));
INSTANTIATE_TEST_CASE_P(roi_crop_backward, TestSuite, Combine(Values("roi_crop_backward"), Range(size_t(0), Collector("roi_crop_backward").num())));
INSTANTIATE_TEST_CASE_P(roi_crop_forward, TestSuite, Combine(Values("roi_crop_forward"), Range(size_t(0), Collector("roi_crop_forward").num())));
INSTANTIATE_TEST_CASE_P(sqrt, TestSuite, Combine(Values("sqrt"), Range(size_t(0), Collector("sqrt").num())));
INSTANTIATE_TEST_CASE_P(sqrt_backward, TestSuite, Combine(Values("sqrt_backward"), Range(size_t(0), Collector("sqrt_backward").num())));
// AUTO GENERATE END
