/*************************************************************************
 * Copyright (C) [2026] by Cambricon, Inc.
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
#include <cstdlib>
#include <memory>
#include <spdlog/logger.h>
#include <sstream>
#include <string>
#include <utility>

#include "core/log_core/log_project.h"

namespace spdlog {
class logger;
}

BEGIN_LOG_NAMESPACE


class FileSink;

// 日志等级：值越小越严重，越大越丰富（0..7）。
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

constexpr bool severityEnabled(int severity, int max_level) {
  return severity <= max_level;
}

// 全局日志阈值。sentinel -1 表示尚未从 env 初始化；LogEngine 构造时仅在仍是
// sentinel 时才从 env 设置，避免覆盖构造前 mluOpSetMaxLogLevel 已设的值。
inline constexpr int kMaxLogLevelUnset = -1;
inline std::atomic<int> g_max_log_level{kMaxLogLevelUnset};

// 引擎存活标志：~LogEngine 时置 0，此后各日志入口不再发射。
inline constexpr int kLogEngineAliveMagic = 1;
inline std::atomic<int> g_log_engine_alive_flag{kLogEngineAliveMagic};

inline bool logEngineAlive() {
  return g_log_engine_alive_flag.load(std::memory_order_relaxed) ==
         kLogEngineAliveMagic;
}

// 用户回调签名：`severity` 为原始 Severity 数值；`function_name` 暂固定传空指针；
// `message` 为格式化后的消息正文（body，无 head）。与 C API 的 MluOpLogCallback
// 布局一致（void(int,const char*,const char*) 函数指针），可安全 reinterpret_cast
// 互转。约定：回调内不得调用 log_core 日志接口（会死锁/递归）。
using LogCallback = void (*)(int severity, const char* function_name,
                             const char* message);

class LogMessage {
public:
  LogMessage(std::string file, int line, Severity severity);
  ~LogMessage();
  std::stringstream &stream();

private:
  std::string file_;
  int line_;
  Severity severity_;
  std::stringstream stream_;
};

// 日志引擎单例
class LogEngine {
public:
  static LogEngine &instance();

  LogEngine(const LogEngine &) = delete;
  LogEngine &operator=(const LogEngine &) = delete;

  static inline void initMaxLogLevel() {
    if (LOG_PREDICT_TRUE(g_max_log_level.load(std::memory_order_relaxed) !=
                           kMaxLogLevelUnset)) {
      return;
    }
    const char *env_raw_ptr = std::getenv(kEnvMaxLogLevel);
    int level =
        env_raw_ptr ? parseMaxLogLevelValue(std::string(env_raw_ptr), 0) : 0;
    g_max_log_level.store(level, std::memory_order_relaxed);
  }

  static inline int maxLogLevel() {
    [[maybe_unused]] static bool inited = (initMaxLogLevel(), true);
    return g_max_log_level.load(std::memory_order_relaxed);
  }

  static inline void setMaxLogLevel(int level) {
    if (LOG_PREDICT_FALSE(level < kMinSeverity)) {
      level = kMinSeverity;
    } else if (LOG_PREDICT_FALSE(level > kMaxSeverity)) {
      level = kMaxSeverity;
    }
    g_max_log_level.store(level, std::memory_order_relaxed);
  }

  static inline bool shouldLog(Severity severity) {
    return severityEnabled(static_cast<int>(severity), maxLogLevel());
  }

  // 流式日志入口（LOG()/VLOG() 走这里）
  void Emit(Severity severity, const std::string &file, int line,
            const std::string &message);

  // 函数式门控日志（LOG_* 系列走这里）：门控后把 fmt+args 直交 spdlog，由 spdlog
  // 内部 fmt 格式化 body（仅一次），head 由各 sink 的 pattern formatter 装配。
  // 被过滤时不构造、不格式化、不输出。有回调时，body 格式化一次后回调（传原始
  // severity）并复用交 spdlog。
  template <typename... Args>
  void EmitFmt(Severity severity, const char *file, int line,
               fmt::string_view fmt, Args &&... args) {
    // 引擎已析构（退出期）时不发射，避免访问已销毁单例成员。
    if (LOG_PREDICT_FALSE(!logEngineAlive())) {
      return;
    }
    if (!LogEngine::shouldLog(severity)) {
      return;
    }
    const spdlog::source_loc loc{file, line, nullptr};
    const auto lvl = cvtSeverityToSpdlog(severity);
    LogCallback callback = callback_.load(std::memory_order_relaxed);
    if (LOG_PREDICT_FALSE(callback != nullptr)) {
      // 有回调：body 手动格式化一次，既给回调又给 spdlog（string 版不重格式化）。
      // fmt::format 包 try/catch 防坏格式串跨 C ABI 逃逸。
      std::string body;
      try {
        body = fmt::format(fmt, args...);
      } catch (...) {
        return;
      }
      callback(static_cast<int>(severity), /*function_name=*/nullptr,
               body.c_str());
      logger_->log(loc, lvl,
                   spdlog::string_view_t(body.data(), body.size()));
      return;
    }
    // 无回调：参数以裸左值传递（不 std::forward）
    logger_->log(loc, lvl, fmt, args...);
  }

  bool colorPrint() const;
  void setColorPrint(bool color_print);

  std::string logFile() const;
  void setLogFile(const std::string &file_name);
  void disableLogFile();

  void setLogCallback(LogCallback callback);

  static int parseMaxLogLevelValue(const std::string &value, int default_para);

private:
  LogEngine();
  ~LogEngine();

  // severity -> spdlog level（仅 spdlog 语义用，显示不走 %L）。
  static spdlog::level::level_enum cvtSeverityToSpdlog(Severity severity);

  std::atomic<bool> color_print_;
  std::shared_ptr<spdlog::logger> logger_;
  std::shared_ptr<FileSink> file_sink_;
  std::atomic<LogCallback> callback_{nullptr};
};

END_LOG_NAMESPACE

#endif // CORE_LOG_CORE_LOG_ENGINE_H_
