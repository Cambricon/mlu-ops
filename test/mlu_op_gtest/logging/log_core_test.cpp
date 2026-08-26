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
//
// 新引擎（spdlog 后端）gtest。
//
// 直接调用 mluop::log 命名空间下的 LogEngine / Severity / severityEnabled /
// parseMaxLogLevelValue，验证纯逻辑：枚举取值、MAX 阈值过滤、环境变量值解析、
// set/get 往返、越界钳制，以及回调 functionName（暂固定传空指针，见 §12.17）。
//
// 本测试不依赖 cnrt 初始化：LogEngine 构造只解析环境变量与布尔环境变量，不触碰
// 设备。唯一触发设备调用的是 CallbackFunctionName 用例（Emit head 无条件调
// cnrtGetDevice），无设备初始化时其返回 dev=0，安全，无需初始化。
//
#include <cstdio>
#include <string>

#include "core/log_core/log_engine.h"
#include "gtest/gtest.h"

namespace mluop {
namespace log {
namespace {

// 回调捕获的 functionName（匿名命名空间静态字符数组，测试进程内唯一）。
char g_captured_function_name[64] = {0};

void CaptureCallback(int /*severity*/, const char* function_name,
                     const char* /*message*/) {
  snprintf(g_captured_function_name, sizeof(g_captured_function_name), "%s",
           function_name != nullptr ? function_name : "");
}

TEST(LogEngineTest, SeverityValues) {
  EXPECT_EQ(static_cast<int>(Severity::kError), 0);
  EXPECT_EQ(static_cast<int>(Severity::kWarning), 1);
  EXPECT_EQ(static_cast<int>(Severity::kCnpapi), 2);
  EXPECT_EQ(static_cast<int>(Severity::kInfo), 3);
  EXPECT_EQ(static_cast<int>(Severity::kDebug1), 4);
  EXPECT_EQ(static_cast<int>(Severity::kDebug2), 5);
  EXPECT_EQ(static_cast<int>(Severity::kDebug3), 6);
  EXPECT_EQ(static_cast<int>(Severity::kDebug4), 7);
  // EXPECT_EQ(static_cast<int>(Severity::kFatal), static_cast<int>(Severity::kError));
  EXPECT_EQ(static_cast<int>(Severity::kVlog), static_cast<int>(Severity::kDebug1));
}

TEST(LogEngineTest, SeverityEnabledFiltering) {
  // MAX 阈值语义：severity <= max_level 才允许打印。
  EXPECT_TRUE(severityEnabled(0, 0));
  EXPECT_FALSE(severityEnabled(1, 0));
  EXPECT_TRUE(severityEnabled(2, 7));
  EXPECT_FALSE(severityEnabled(7, 6));
  EXPECT_TRUE(severityEnabled(7, 7));
}

TEST(LogEngineTest, ParseMaxLogLevelValue) {
  // 数字映射：0..7。
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("0", -1), 0);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("1", -1), 1);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("2", -1), 2);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("3", -1), 3);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("4", -1), 4);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("5", -1), 5);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("6", -1), 6);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("7", -1), 7);
  // 命名映射（大小写不敏感；DEBUG 别名：DEBUG=DEBUG1，DEBUG-n/DEBUG_n 与 DEBUGn 同义）。
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("error", -1), 0);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("WARNING", -1), 1);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("api_trace", -1), 2);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("INFO", -1), 3);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("debug", 9), 4);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("DEBUG1", -1), 4);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("debug-1", -1), 4);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("debug_2", -1), 5);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("DEBUG3", -1), 6);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("debug-4", -1), 7);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("debug_4", -1), 7);
  // 非法值回落默认。
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("nonsense", -1), -1);
  EXPECT_EQ(LogEngine::parseMaxLogLevelValue("", 42), 42);
}

TEST(LogEngineTest, SetGetRoundTrip) {
  LogEngine& engine = LogEngine::instance();
  engine.setMaxLogLevel(static_cast<int>(Severity::kInfo));
  EXPECT_EQ(engine.maxLogLevel(), 3);
  EXPECT_TRUE(engine.shouldLog(Severity::kInfo));
  EXPECT_FALSE(engine.shouldLog(Severity::kDebug1));
  // 还原，避免影响后续用例。
  engine.setMaxLogLevel(static_cast<int>(Severity::kError));
}

TEST(LogEngineTest, ClampOutOfRange) {
  LogEngine& engine = LogEngine::instance();
  engine.setMaxLogLevel(100);
  EXPECT_EQ(engine.maxLogLevel(), kMaxSeverity);
  engine.setMaxLogLevel(-5);
  EXPECT_EQ(engine.maxLogLevel(), kMinSeverity);
  // 还原。
  engine.setMaxLogLevel(static_cast<int>(Severity::kError));
}

TEST(LogEngineTest, CallbackFunctionName) {
  LogEngine& engine = LogEngine::instance();
  // 确保 kError（severity 0）放行，触发真实 Emit → callback sink。
  engine.setMaxLogLevel(static_cast<int>(Severity::kError));

  g_captured_function_name[0] = '\0';
  engine.setLogCallback(CaptureCallback);

  // 构造 LogMessage 触发真实 Emit → callback sink。
  LogMessage(__FILE__, __LINE__, Severity::kError).stream()
      << "callback functionName probe";

  // 清理回调，避免影响后续用例。
  engine.setLogCallback(nullptr);

  // functionName 暂固定传空指针（栈回溯解析暂缓，见 §12.17），回调收到空串。
  EXPECT_STREQ(g_captured_function_name, "")
      << "actual functionName=[" << g_captured_function_name << "]";
}

}  // namespace
}  // namespace log
}  // namespace mluop
