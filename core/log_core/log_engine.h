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
#include <cstdlib>
#include <memory>
#include <spdlog/logger.h>
#include <sstream>
#include <string>
#include <utility>

#include "core/log_core/log_project.h"
#include "core/macros.h"

namespace spdlog {
class logger;
}

namespace mluop {
namespace log {

// 文件 sink 定义在 log_engine.cpp（自管日志流/文件名/开关），此处仅前向声明。
class FileSink;

// 日志等级：值越小越严重，越大越丰富（0..7）。kVlog 为 kDebug1 的旧名别名。
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

// 用户回调签名：`severity` 为原始 Severity 数值；`function_name` 暂固定传空指针；
// `message` 为格式化后的消息正文（body，无 head）。与 C API 的 MluOpLogCallback
// 布局一致（void(int,const char*,const char*) 函数指针），可安全 reinterpret_cast
// 互转。
using LogCallback = void (*)(int severity, const char* function_name,
                             const char* message);

// 日志消息 RAII：构造记录 file/line/severity，析构经 LogEngine::Emit 输出。
// `LOG(severity)` 宏映射到 `LogMessage(__FILE__, __LINE__, SEV_##severity).stream()`。
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

// 日志引擎单例：承载阈值/落盘/颜色/回调状态，日志经 Emit / EmitFmt 输出。
class LogEngine {
public:
  static LogEngine &instance();

  LogEngine(const LogEngine &) = delete;
  LogEngine &operator=(const LogEngine &) = delete;

  static inline void initMaxLogLevel() {
    if (MLUOP_PREDICT_TRUE(g_max_log_level.load(std::memory_order_relaxed) !=
                           kMaxLogLevelUnset)) {
      return;
    }
    const char *env_raw_ptr = std::getenv(kEnvMaxLogLevel);
    int level =
        env_raw_ptr ? parseMaxLogLevelValue(std::string(env_raw_ptr), 0) : 0;
    g_max_log_level.store(level, std::memory_order_relaxed);
  }

  static inline int maxLogLevel() {
    static bool inited = (initMaxLogLevel(), true);
    return g_max_log_level.load(std::memory_order_relaxed);
  }

  static inline void setMaxLogLevel(int level) {
    if (MLUOP_PREDICT_FALSE(level < kMinSeverity)) {
      level = kMinSeverity;
    } else if (MLUOP_PREDICT_FALSE(level > kMaxSeverity)) {
      level = kMaxSeverity;
    }
    g_max_log_level.store(level, std::memory_order_relaxed);
  }

  static inline bool shouldLog(Severity severity) {
    return severityEnabled(static_cast<int>(severity), maxLogLevel());
  }

  // 流式日志入口（LOG()/LOG_*_STREAM 走这里）
  void Emit(Severity severity, const std::string &file, int line,
            const std::string &message);

  // 函数式门控日志（LOG_*_FMT 走这里）：门控后把 fmt+args 直交 spdlog，由 spdlog
  // 内部 fmt 格式化 body（仅一次），head 由各 sink 的 pattern formatter 装配。
  // 被过滤时不构造、不格式化、不输出。有回调时，body 格式化一次后回调（传原始
  // severity）并复用交 spdlog。
  template <typename... Args>
  void EmitFmt(Severity severity, const char *file, int line,
               fmt::string_view fmt, Args &&... args) {
    if (!mluop::log::LogEngine::shouldLog(severity)) {
      return;
    }
    const spdlog::source_loc loc{file, line, nullptr};
    const auto lvl = cvtSeverityToSpdlog(severity);
    LogCallback callback = callback_.load(std::memory_order_relaxed);
    if (MLUOP_PREDICT_FALSE(callback != nullptr)) {
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
    // 无回调（主路径）：参数以裸左值传递（不 std::forward）
    logger_->log(loc, lvl, fmt, args...);
  }

  bool colorPrint() const;
  void setColorPrint(bool color_print);

  std::string logFile() const;
  // 启用落盘并切换到 file_name。
  void setLogFile(const std::string &file_name);
  // 关闭落盘（再次 setLogFile 可恢复）。
  void disableLogFile();

  void setLogCallback(LogCallback callback);

  // 解析 max-level 环境变量的值（0..7 或 ERROR/WARNING/... 名称）；
  // 非法值回落default_para。
  static int parseMaxLogLevelValue(const std::string &value, int default_para);

private:
  LogEngine();
  ~LogEngine();

  // severity -> spdlog level（仅 spdlog 语义用，显示不走 %L）。
  static spdlog::level::level_enum cvtSeverityToSpdlog(Severity severity);

  // 颜色开关供 ConsoleSink 的 Module/Level flag 查询；文件状态整体在 FileSink 内；
  // 回调以原子函数指针存储，零锁、无 sinks() 修改。
  std::atomic<bool> color_print_;
  std::shared_ptr<spdlog::logger> logger_;
  std::shared_ptr<FileSink> file_sink_;
  std::atomic<LogCallback> callback_{nullptr};
};

} // namespace log
} // namespace mluop

#endif // CORE_LOG_CORE_LOG_ENGINE_H_
