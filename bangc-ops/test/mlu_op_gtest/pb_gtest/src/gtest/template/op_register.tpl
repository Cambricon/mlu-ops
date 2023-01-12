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
#ifndef TEST_MLU_OP_GTEST_SRC_GTEST_OP_REGISTER_H_
#define TEST_MLU_OP_GTEST_SRC_GTEST_OP_REGISTER_H_

#include <string>
// AUTO GENERATE HEADER START
#include "../zoo/add_tensor/add_tensor.h"
#include "../zoo/convbpfilter/convbpfilter.h"
// AUTO GENERATE HEADER END

std::shared_ptr<mluoptest::Executor> getOpExecutor(std::string op_name) {
  if (false) {
// AUTO GENERATE START
  } else if (op_name == "add_tensor") {
    return std::make_shared<mluoptest::add_tensor_executor();
  } else if (op_name == "convbpfilter") {
    return new mluoptest::convbpfilter_executor();
// AUTO GENERATE END
  } else {
    LOG(ERROR) << "UnKnown op: " << op_name;
    exit(1);
  }
}

#endif  // TEST_MLU_OP_GTEST_SRC_GTEST_OP_REGISTER_H_
