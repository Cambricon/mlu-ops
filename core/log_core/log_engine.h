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
#ifndef CORE_LOG_CORE_LOG_ENGINE_H_
#define CORE_LOG_CORE_LOG_ENGINE_H_

#include <atomic>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include "core/log_core/log_callback_sink.h"
#include "core/log_core/log_project.h"

namespace spdlog {
class logger;
}

namespace mluop {
namespace log {

// 新引擎日志等级：值越小越严重/越基础，值越大日志越丰富（0..7）。kFatal / kVlog
// 为继承别名（kFatal==kError，kVlog==kDebug1），供宏层沿用旧等级命名。
enum class Severity : int {
  kError = 0,
  kWarning = 1,
  kCnpapi = 2,
  kInfo = 3,
  kDebug1 = 4,
  kDebug2 = 5,
  kDebug3 = 6,
  kDebug4 = 7,
  // kFatal = kError,
  kVlog = kDebug1,
};

inline constexpr int kMinSeverity = static_cast<int>(Severity::kError);
inline constexpr int kMaxSeverity = static_cast<int>(Severity::kDebug4);

// 上限阈值过滤（MAX 语义）：severity <= max_level 时允许打印。独立命名避免被
// LogEngine::shouldLog 隐藏。
constexpr bool severityEnabled(int severity, int max_level) {
  return severity <= max_level;
}

// 新引擎日志消息 RAII：构造时记录 file/line/severity，析构时经 LogEngine::Emit
// 统一装配输出。宏层 `LOG(severity)` 映射到
// `LogMessage(__FILE__, __LINE__, Severity::SEV_##severity).stream()`。
class LogMessage {
 public:
  LogMessage(std::string file, int line, Severity severity);
  ~LogMessage();
  std::stringstream& stream();

 private:
  std::string file_;
  int line_;
  Severity severity_;
  std::stringstream stream_;
};

// 新引擎门面：单例承载日志状态（阈值/落盘/颜色/回调），LogMessage 析构经 Emit
// 装配并输出。
class LogEngine {
 public:
  static LogEngine& instance();

  LogEngine(const LogEngine&) = delete;
  LogEngine& operator=(const LogEngine&) = delete;

  int maxLogLevel() const;
  void setMaxLogLevel(int level);

  bool shouldLog(Severity severity) const;

  // 装配并输出一条日志：屏幕 = head + body + tail，文件 = body + tail（无 head）。
  // severity 门控在装配之前（Emit 开头对被抑制级别直接返回，不装配、不输出）。
  void Emit(Severity severity, const std::string& file, int line,
            const std::string& message);

  bool onlyShow() const;
  void setOnlyShow(bool only_show);

  bool colorPrint() const;
  void setColorPrint(bool color_print);

  // 返回 log_file_ 的副本（锁内拷贝，避免并发 setLogFile 下返回内部引用外泄）。
  std::string logFile() const;
  void setLogFile(const std::string& file_name);

  void setLogCallback(LogCallback callback);

  // 解析 max-level 环境变量的【值】（非变量名）：0..7 或 ERROR/WARNING/API_TRACE/
  // INFO/DEBUG(1..4)（DEBUG-n/DEBUG_n 与 DEBUGn 同义；裸 DEBUG=DEBUG1）；非法值
  // 回落 default_para。
  static int parseMaxLogLevelValue(const std::string& value, int default_para);

 private:
  LogEngine();
  ~LogEngine();

  // 线程安全：日志发射（任意线程）与运行时 C API（mluOpSet*，任意线程）可能
  // 跨线程并发读写下列状态，故一律用 atomic（relaxed 即可：配置类值读到稍旧值
  // 无害，只需避免数据竞争 UB）或互斥锁（std::string 不可原子化）。
  std::atomic<int> max_log_level_;
  std::atomic<bool> only_show_;
  std::atomic<bool> color_print_;
  // log_file_ 为 std::string，无法用 std::atomic，改由 log_file_mutex_ 保护。
  std::string log_file_;
  mutable std::mutex log_file_mutex_;

  std::ofstream log_file_stream_;
  std::shared_ptr<spdlog::logger> logger_;
  std::shared_ptr<spdlog::logger> file_logger_;
  std::shared_ptr<LogCallbackSinkMt> callback_sink_;
};

}  // namespace log
}  // namespace mluop

#endif  // CORE_LOG_CORE_LOG_ENGINE_H_
