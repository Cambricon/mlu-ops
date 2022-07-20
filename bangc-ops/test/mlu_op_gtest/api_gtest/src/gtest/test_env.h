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
#ifndef TEST_MLU_OP_GTEST_API_GTEST_SRC_GTEST_TEST_ENV_H_
#define TEST_MLU_OP_GTEST_API_GTEST_SRC_GTEST_TEST_ENV_H_

#include "core/logging.h"
#include "gtest/gtest.h"
#include "mlu_op.h"

class TestEnvironment : public testing::Environment {
 public:
  virtual void SetUp();
  virtual void TearDown();
};

#endif  // TEST_MLU_OP_GTEST_API_GTEST_SRC_GTEST_TEST_ENV_H_
