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
#ifndef TEST_MLU_OP_GTEST_INCLUDE_TOOLS_H_
#define TEST_MLU_OP_GTEST_INCLUDE_TOOLS_H_

#include <string>

#include "gtest/gtest.h"
#include "mlu_op.h"

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define GTEST_CHECK(condition, ...)                                            \
  if (unlikely(!(condition))) {                                                \
    ADD_FAILURE() << "Check failed: " #condition ". " #__VA_ARGS__;            \
    throw std::invalid_argument(std::string(__FILE__) + " +" +                 \
                                std::to_string(__LINE__));                     \
  }

#define GTEST_WARNING(condition, ...)                                          \
  if (unlikely(!(condition))) {                                                \
    LOG(WARNING) << "Check failed: " #condition ". " #__VA_ARGS__;             \
  }

#endif // TEST_MLU_OP_GTEST_INCLUDE_TOOLS_H_
