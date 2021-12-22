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
#ifndef TEST_MLU_OP_GTEST_SRC_GTEST_OP_REGISTER_H_
#define TEST_MLU_OP_GTEST_SRC_GTEST_OP_REGISTER_H_

#include <string>
// AUTO GENERATE HEADER START
#include "../zoo/abs/abs.h"
#include "../zoo/div/div.h"
#include "../zoo/log/log.h"
#include "../zoo/sqrt/sqrt.h"
#include "../zoo/sqrt_backward/sqrt_backward.h"
// AUTO GENERATE HEADER END

std::shared_ptr<mluoptest::Executor> getOpExecutor(std::string op_name) {
  if (false) {
// AUTO GENERATE START
  } else if (op_name == "abs") {
    return std::make_shared<mluoptest::AbsExecutor>();
  } else if (op_name == "div") {
    return std::make_shared<mluoptest::DivExecutor>();
  } else if (op_name == "log") {
    return std::make_shared<mluoptest::LogExecutor>();
  } else if (op_name == "sqrt") {
    return std::make_shared<mluoptest::SqrtExecutor>();
  } else if (op_name == "sqrt_backward") {
    return std::make_shared<mluoptest::SqrtBackwardExecutor>();
// AUTO GENERATE END
  } else {
    LOG(ERROR) << "UnKnown op: " << op_name;
    exit(1);
  }
}

#endif  // TEST_MLU_OP_GTEST_SRC_GTEST_OP_REGISTER_H_
