/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include <iostream>
#include <vector>
#include <string>
#include <tuple>

#include "gtest/gtest.h"
#include "mlu_op.h"
#include "core/logging.h"
#include "api_test_tools.h"

namespace mluopapitest {
TEST(fft_plan_descriptor, BAD_PARAM_DestroyDesc_null) {
  try {
    mluOpFFTPlan_t fft_plan = nullptr;
    mluOpStatus_t status = mluOpDestroyFFTPlan(fft_plan);
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in fft_plan_descriptor"
           << ")";
  }
}
}  // namespace mluopapitest
