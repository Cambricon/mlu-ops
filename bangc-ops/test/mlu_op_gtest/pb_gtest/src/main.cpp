/*************************************************************************
 * Copyright (C) [2022] by Cambricon, Inc.
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

/************************************************************************
 *
 *  @file main.c
 *
 **************************************************************************/

#include <stdlib.h>
#include "gtest/gtest.h"
#include "variable.h"
#include "test_env.h"
#include "mlu_op_test_result_printer.h"
#include "cnrt_test.h"
#include "mlu_op_test.h"
#include "mlu_op_gtest_event_listener.h"
#include "modules_test.h"
#include "src/gtest-internal-inl.h"

using namespace mluoptest;  // NOLINT
int main(int argc, char **argv) {
  global_var.init(argc, argv);
  testing::AddGlobalTestEnvironment(new TestEnvironment);
  testing::InitGoogleTest(&argc, argv);
  testing::TestEventListeners &listeners =
      testing::UnitTest::GetInstance()->listeners();
  delete listeners.Release(listeners.default_xml_generator());
  const std::string &output_format =
      testing::internal::UnitTestOptions::GetOutputFormat();
  if (output_format == "xml") {
    listeners.Append(new xmlPrinter(
        testing::internal::UnitTestOptions::GetAbsolutePathToOutputFile()
            .c_str()));
  } else if (output_format == "json") {
    listeners.Append(new JsonPrinter(
        testing::internal::UnitTestOptions::GetAbsolutePathToOutputFile()
            .c_str()));
  } else if (output_format != "") {
    GTEST_LOG_(WARNING) << "WARNING: unrecognized output format \""
                        << output_format << "\" ignored.";
  }
  if (global_var.enable_gtest_internal_perf) {
    testing::UnitTest::GetInstance()->listeners().Append(
        new MLUOPGtestInternalPerfEventListener);
  }
  return RUN_ALL_TESTS();
}
