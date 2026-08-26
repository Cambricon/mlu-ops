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
// golden 快照单测：验证新引擎（mluop::log，spdlog 后端）的屏幕输出逐字节复现
// 【冻结的 golden 基线】（该基线由历史输出派生，锁定输出格式不随改动漂移）。
// golden 中运行时可变字段用哨兵归一化：timestamp→<TS>、pid→<PID>、card→<CARD>、
// tail pid→<TID>。
//
// 实现要点：
//  1. 只 include 新引擎头 core/log_core/log_engine.h。
//  2. 行首时间戳 [YYYY-MM-DD HH:MM:SS.ffffff] 归一化为 <TS>；pid/card/tail 同理。
//  3. NDEBUG（release）下无 tail（NDEBUG 不输出 tail），golden 用 #ifndef NDEBUG
//     区分 debug/release 两种基线。
//  4. 测试前 setenv MLUOP_LOG_COLOR_PRINT=0 强制无色（颜色缓存于单例首次构造，
//     必须先于任何日志）；MLUOP_LOG_ONLY_SHOW=1 保证不落盘。
//  5. 新引擎 head 无条件调用 cnrtGetDevice，故 main 里先做 cnrt 初始化。
//
#include <cstdlib>
#include <regex>
#include <string>

#include "cnrt.h"
#include "core/log_core/log_engine.h"
#include "gtest/gtest.h"

namespace {

constexpr const char* kFile = "log_engine_diff_test.cpp";
constexpr int kLine = 42;

// 行首时间戳：[YYYY-MM-DD HH:MM:SS.ffffff]。用 [0-9] 而非 \d，规避 std::regex
// 在各实现下的差异。
const std::regex& TimestampRegex() {
  static const std::regex re(
      R"(\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}\])");
  return re;
}

// 归一化（顺序与实现一致）：先剥离行尾 '\n'（endl 写入），再替换行首
// [datetime]-><TS>、[pid]->[<PID>]、Card:N->Card:<CARD>、tail pid:NNN->pid:<TID>。
// 行首非时间戳则原样返回（后续比较必不等，便于定位）。
std::string NormalizeGolden(const std::string& s) {
  std::string out = s;
  if (!out.empty() && out.back() == '\n') {
    out.pop_back();
  }
  std::smatch m;
  if (std::regex_search(out, m, TimestampRegex()) && m.position() == 0) {
    out = "<TS>" + out.substr(m.position() + m.length());
  } else {
    return out;
  }
  // head pid: [pid]
  std::regex pid_re(R"(\[\d+\])");
  std::regex card_re(R"(Card:\d+)");
  std::regex tid_re(R"(pid:\d+)");
  out = std::regex_replace(out, pid_re, "[<PID>]");
  out = std::regex_replace(out, card_re, "Card:<CARD>");
  out = std::regex_replace(out, tid_re, "pid:<TID>");
  return out;
}

bool HasLeadingTimestamp(const std::string& s) {
  std::smatch m;
  return std::regex_search(s, m, TimestampRegex()) && m.position() == 0;
}

// golden 基线：冻结的期望输出，经 NormalizeGolden 归一化。release（NDEBUG）下
// 无 tail；debug 下有 tail（file:line pid）。severity 名按塌缩后 spdlog level 显示
// ERROR/WARNING/TRACE/INFO/DEBUG（kCnpapi→TRACE，DEBUG1..4 同显 DEBUG，设计意图）。
#if defined(NDEBUG)
constexpr const char* kGolden[8] = {
    "<TS>[MLU-OPS][ERROR][<PID>][Card:<CARD>]: diff_msg_sev=0",
    "<TS>[MLU-OPS][WARNING][<PID>][Card:<CARD>]: diff_msg_sev=1",
    "<TS>[MLU-OPS][TRACE][<PID>][Card:<CARD>]: diff_msg_sev=2",
    "<TS>[MLU-OPS][INFO][<PID>][Card:<CARD>]: diff_msg_sev=3",
    "<TS>[MLU-OPS][DEBUG][<PID>][Card:<CARD>]: diff_msg_sev=4",
    "<TS>[MLU-OPS][DEBUG][<PID>][Card:<CARD>]: diff_msg_sev=5",
    "<TS>[MLU-OPS][DEBUG][<PID>][Card:<CARD>]: diff_msg_sev=6",
    "<TS>[MLU-OPS][DEBUG][<PID>][Card:<CARD>]: diff_msg_sev=7",
};
#else
constexpr const char* kGolden[8] = {
    "<TS>[MLU-OPS][ERROR][<PID>][Card:<CARD>]: diff_msg_sev=0  [ log_engine_diff_test.cpp:42  pid:<TID>]",
    "<TS>[MLU-OPS][WARNING][<PID>][Card:<CARD>]: diff_msg_sev=1  [ log_engine_diff_test.cpp:42  pid:<TID>]",
    "<TS>[MLU-OPS][TRACE][<PID>][Card:<CARD>]: diff_msg_sev=2  [ log_engine_diff_test.cpp:42  pid:<TID>]",
    "<TS>[MLU-OPS][INFO][<PID>][Card:<CARD>]: diff_msg_sev=3  [ log_engine_diff_test.cpp:42  pid:<TID>]",
    "<TS>[MLU-OPS][DEBUG][<PID>][Card:<CARD>]: diff_msg_sev=4  [ log_engine_diff_test.cpp:42  pid:<TID>]",
    "<TS>[MLU-OPS][DEBUG][<PID>][Card:<CARD>]: diff_msg_sev=5  [ log_engine_diff_test.cpp:42  pid:<TID>]",
    "<TS>[MLU-OPS][DEBUG][<PID>][Card:<CARD>]: diff_msg_sev=6  [ log_engine_diff_test.cpp:42  pid:<TID>]",
    "<TS>[MLU-OPS][DEBUG][<PID>][Card:<CARD>]: diff_msg_sev=7  [ log_engine_diff_test.cpp:42  pid:<TID>]",
};
#endif

}  // namespace

TEST(LogEngineGolden, ByteForByteBaseline) {
  // 放行全部级别（0..7），确保每种 severity 都真正输出。
  mluop::log::LogEngine::instance().setMaxLogLevel(
      static_cast<int>(mluop::log::Severity::kDebug4));

  for (int sev = 0; sev <= 7; ++sev) {
    const std::string msg = "diff_msg_sev=" + std::to_string(sev);

    testing::internal::CaptureStdout();
    mluop::log::LogMessage(kFile, kLine,
                           static_cast<mluop::log::Severity>(sev))
            .stream()
        << msg;
    const std::string new_out = testing::internal::GetCapturedStdout();

    // 结构校验：原始输出行首必须命中时间戳正则。
    EXPECT_TRUE(HasLeadingTimestamp(new_out)) << "sev=" << sev;

    // 归一化后与冻结基线逐字节相等。
    EXPECT_EQ(NormalizeGolden(new_out), std::string(kGolden[sev]))
        << "sev=" << sev << "\nactual=" << new_out;
  }
}

int main(int argc, char** argv) {
  // 必须先于任何日志 / 单例构造：强制无色、只显示不落盘、最高级别 7。
  setenv("MLUOP_LOG_COLOR_PRINT", "0", 1);
  setenv("MLUOP_LOG_ONLY_SHOW", "1", 1);
  setenv("MLUOP_LOG_MAX_LEVEL", "7", 1);

  unsigned int dev_num = 0;
  cnrtGetDeviceCount(&dev_num);
  if (dev_num > 0) {
    cnrtSetDevice(0);
  }

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
