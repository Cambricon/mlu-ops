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
#ifndef TEST_MLU_OP_GTEST__PB_GTEST_TESTS_MODULES_TEST_H_
#define TEST_MLU_OP_GTEST__PB_GTEST_TESTS_MODULES_TEST_H_

#include <memory>
#include <string>
#include "gtest/gtest.h"
#include "parser.h"

#define TAKE_TIME(func)                                                      \
  {                                                                          \
    auto start = std::chrono::system_clock::now();                           \
    func;                                                                    \
    auto stop = std::chrono::system_clock::now();                            \
    auto duration =                                                          \
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start); \
    std::cout << #func " takes " << duration.count() << " us\n";             \
  }

TEST(DISABLED_GTEST_PARSER, parse) {
  auto test_path =
      "../../test/mlu_op_gtest/pb_gtest/tests/parser_test.prototxt";
  std::ifstream fin(test_path);
  if (!fin.good()) {
    std::cout << "DISABLED_GTEST_PARSER.parse: miss " << test_path
              << ", and skip this tests.\n";
    return;
  }

  auto parse = [](std::string test_path) {
    auto parser = std::make_shared<mluoptest::Parser>();
    TAKE_TIME(parser->parse(test_path));
    ASSERT_EQ(mluoptest::CPU, parser->device());
    ASSERT_EQ(2, parser->criterions().size());
    ASSERT_EQ(1, parser->getInputNum());
    ASSERT_FALSE(parser->inputIsNull(0));
    ASSERT_EQ(MLUOP_LAYOUT_ARRAY, parser->getInputLayout(0));
    ASSERT_EQ(MLUOP_DTYPE_FLOAT, parser->getInputDataType(0));
    ASSERT_EQ(MLUOP_DTYPE_INVALID, parser->getInputOnchipDataType(0));
    ASSERT_EQ(10000, parser->getInputDataCount(0));
    ASSERT_EQ(0, parser->getInputPosition(0));
    ASSERT_EQ(1.0f, parser->getInputScale(0));

    size_t dim_size = parser->getInputDimSize(0);
    ASSERT_EQ(4, dim_size);

    int *dims = new int[dim_size];
    parser->getInputDims(0, dim_size, dims);
    for (size_t i = 0; i < dim_size; ++i) {
      ASSERT_EQ(10, dims[i]);
    }
    delete[] dims;
  };

  ASSERT_NO_THROW(parse(test_path));
}

#endif  // TEST_MLU_OP_GTEST__PB_GTEST_TESTS_MODULES_TEST_H_
