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
#include "core/log_core/log_engine.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include "core/tool.h"
#include "mlu_op.h"

#if defined(WINDOWS) || defined(WIN32)  // used in windows
#include <windows.h>
#include <process.h>
#include <processthreadsapi.h>
#include <io.h>
#else
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <fmt/chrono.h>
#include <fmt/color.h>
#include <spdlog/sinks/base_sink.h>
#include <spdlog/pattern_formatter.h>
#include <iterator>
#include <spdlog/spdlog.h>

namespace mluop {
namespace log {

namespace {

// 引擎存活标志：默认存活，仅 ~LogEngine 置 0。LogMessage 析构据此在引擎已销毁时
// 跳过发射（不触碰 instance()），避免退出期后台线程/全局析构打日志的 use-after-free。
constexpr int kLogEngineAliveMagic = 1;
std::atomic<int> g_log_engine_alive_flag = kLogEngineAliveMagic;

int maxLogLevelFromEnv(const char* env_name, int default_para) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr) {
    return default_para;
  }
  return LogEngine::parseMaxLogLevelValue(std::string(raw), default_para);
}

bool isPrintToScreen() { return isatty(fileno(stdout)); }

void clearEnter(std::string* ss) {
  for (std::string::iterator it = (*ss).begin(); it != (*ss).end(); it++) {
    if (*it == '\n') {
      *it = ' ';
    }
  }
}

// 判两个路径是否指向同一文件：字符串快比较优先，不同则用 weakly_canonical
// 规范化（解析相对/绝对、.、..、多余分隔符及存在部分的符号链接）再比。
bool isSameFilePath(const std::string& a, const std::string& b) {
  if (a == b) {
    return true;
  }
  try {
    return std::filesystem::weakly_canonical(a) ==
           std::filesystem::weakly_canonical(b);
  } catch (...) {
    return false;  // 规范化失败（无效路径）按不同处理，交由 reopen 报错。
  }
}

class ConsoleSink final : public spdlog::sinks::base_sink<std::mutex> {
 protected:
  void sink_it_(const spdlog::details::log_msg& msg) override {
    spdlog::memory_buf_t formatted;
    formatter_->format(msg, formatted);
    // formatter 已追加 eol（\n），这里只写、不再 put('\n')，否则双换行。
    std::cout.write(formatted.data(),
                    static_cast<std::streamsize>(formatted.size()));
    std::cout.flush();
  }
  void flush_() override { std::cout.flush(); }
};

// 自定义 pattern flag %k：Card 卡号（cnrtGetDevice）。
class CardFlag final : public spdlog::custom_flag_formatter {
 public:
  void format(const spdlog::details::log_msg&, const std::tm&,
              spdlog::memory_buf_t& dest) override {
    int dev_index = -1;
    cnrtGetDevice(&dev_index);
    auto sv = std::to_string(dev_index);
    dest.append(sv.data(), sv.data() + sv.size());
  }
  std::unique_ptr<spdlog::custom_flag_formatter> clone() const override {
    return std::make_unique<CardFlag>();
  }
};

// 自定义 pattern flag %K：模块名（kProjectDisplayName），color_print_ 时加粗黄。
class ModuleFlag final : public spdlog::custom_flag_formatter {
 public:
  void format(const spdlog::details::log_msg&, const std::tm&,
              spdlog::memory_buf_t& dest) override {
    const char* module = kProjectDisplayName;
    if (mluop::log::LogEngine::instance().colorPrint()) {
      // memory_buf_t 属 bundled fmt，外层 fmt 写不进去：先在栈 buffer 格式化再拷贝。
      fmt::memory_buffer tmp;
      fmt::format_to(std::back_inserter(tmp), "{}",
                     fmt::styled(module, fmt::emphasis::bold |
                                              fmt::fg(fmt::terminal_color::yellow)));
      dest.append(tmp.data(), tmp.data() + tmp.size());
    } else {
      dest.append(module, module + strlen(module));
    }
  }
  std::unique_ptr<spdlog::custom_flag_formatter> clone() const override {
    return std::make_unique<ModuleFlag>();
  }
};

// 自定义 pattern flag %c：等级名（按 cvtSeverityToSpdlog 塌缩到 spdlog 级），
// color_print_ 时按等级配色。
class LevelFlag final : public spdlog::custom_flag_formatter {
 public:
  void format(const spdlog::details::log_msg& msg, const std::tm&,
              spdlog::memory_buf_t& dest) override {
    // 等级名大写。
    const char* name = "UNKNOWN";
    switch (msg.level) {
      case spdlog::level::err: {
        name = "ERROR";
        break;
      }
      case spdlog::level::warn: {
        name = "WARNING";
        break;
      }
      case spdlog::level::info: {
        name = "INFO";
        break;
      }
      case spdlog::level::trace: {
        name = "TRACE";
        break;
      }
      case spdlog::level::debug: {
        name = "DEBUG";
        break;
      }
      case spdlog::level::critical: {
        name = "CRITICAL";
        break;
      }
      default: {
        break;
      }
    }
    if (mluop::log::LogEngine::instance().colorPrint()) {
      auto color = fmt::terminal_color::white;
      switch (msg.level) {
        case spdlog::level::err: {
          color = fmt::terminal_color::red;
          break;
        }
        case spdlog::level::warn: {
          color = fmt::terminal_color::magenta;
          break;
        }
        case spdlog::level::info: {
          color = fmt::terminal_color::green;
          break;
        }
        case spdlog::level::trace: {
          color = fmt::terminal_color::blue;
          break;
        }
        case spdlog::level::debug: {
          color = fmt::terminal_color::cyan;
          break;
        }
        default: {
          break;
        }
      }
      fmt::memory_buffer tmp;
      fmt::format_to(std::back_inserter(tmp), "{}",
                     fmt::styled(name, fmt::fg(color) | fmt::emphasis::bold));
      dest.append(tmp.data(), tmp.data() + tmp.size());
    } else {
      dest.append(name, name + strlen(name));
    }
  }
  std::unique_ptr<spdlog::custom_flag_formatter> clone() const override {
    return std::make_unique<LevelFlag>();
  }
};

// 屏幕 / 回调共用的带 head pattern formatter（时间/模块/等级/pid/Card，Debug 下
// 追加 file:line tail）。
std::unique_ptr<spdlog::pattern_formatter> makeConsoleFormatter() {
  auto formatter = std::make_unique<spdlog::pattern_formatter>();
  formatter->add_flag<CardFlag>('k');
  formatter->add_flag<ModuleFlag>('K');
  formatter->add_flag<LevelFlag>('c');
#ifdef NDEBUG
  formatter->set_pattern(
      "[%Y-%m-%d %H:%M:%S.%f][%K][%c][%P][Card:%k]: %v");
#else
  formatter->set_pattern(
      "[%Y-%m-%d %H:%M:%S.%f][%K][%c][%P][Card:%k]: %v  [%s:%#  pid:%P]");
#endif
  return formatter;
}

}  // namespace

// 文件 sink：pattern 为 %v（无 head），写到自己持有的 ofstream。
// - 自管流与文件名：setLogFile 在锁内关旧流、开新流（trunc），运行期切换生效。
// - only_show 映射到 sink level（off/trace）：off 时 logger 分发直接跳过，零落盘
//   开销；禁用不关流，重新启用不丢已写内容。
// - sink_it_ 在 base_sink::log 的 mutex_ 内执行，与 setEnabled/setLogFile 互斥。
class FileSink final : public spdlog::sinks::base_sink<std::mutex> {
 public:
  explicit FileSink(std::string file_name, bool enabled)
      : file_name_(std::move(file_name)) {
    if (enabled) {
      reopen();
    }
    this->level_.store(enabled ? spdlog::level::trace : spdlog::level::off,
                       std::memory_order_relaxed);
  }

  bool enabled() const {
    return this->level_.load(std::memory_order_relaxed) != spdlog::level::off;
  }

  // only_show 开关：true 停写（level=off，分发跳过），false 恢复（level=trace）。
  void setEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    if (enabled && !stream_.is_open()) {
      reopen();
    }
    this->level_.store(enabled ? spdlog::level::trace : spdlog::level::off,
                       std::memory_order_relaxed);
  }

  std::string logFile() {
    std::lock_guard<std::mutex> lock(this->mutex_);
    return file_name_;
  }

  void setLogFile(std::string file_name) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    if (stream_.is_open() && isSameFilePath(file_name_, file_name)) {
      return;  // 同一文件已开（含不同路径写法）：跳过 reopen，避免 trunc 清空。
    }
    file_name_ = std::move(file_name);
    // 流开着才换文件；禁用且未开时只记名字，等 setEnabled(true) 时再 reopen。
    if (stream_.is_open()) {
      reopen();
    }
  }

 protected:
  void sink_it_(const spdlog::details::log_msg& msg) override {
    if (!stream_.is_open()) {
      return;
    }
    spdlog::memory_buf_t formatted;
    formatter_->format(msg, formatted);
    stream_.write(formatted.data(),
                  static_cast<std::streamsize>(formatted.size()));
    stream_.flush();
  }
  void flush_() override {
    if (stream_.is_open()) {
      stream_.flush();
    }
  }

 private:
  void reopen() {
    stream_.close();
    stream_.open(file_name_, std::ios::out | std::ios::trunc);
    if (!stream_.is_open()) {
      std::cerr << "can't init Log, open file failed! " << file_name_
                << std::endl;
    }
  }

  std::string file_name_;
  std::ofstream stream_;
};

LogEngine& LogEngine::instance() {
  static LogEngine engine;
  return engine;
}

LogEngine::LogEngine()
    : color_print_(mluop::getBoolEnvVar(kEnvColorPrint, true)) {
  const bool only_show = mluop::getBoolEnvVar(kEnvOnlyShow, true);
  if (color_print_.load(std::memory_order_relaxed)) {
    color_print_.store(isPrintToScreen(), std::memory_order_relaxed);
  }

  // 单 logger_ 挂 Console/File 两 sink，各自独立 formatter；Callback sink 在首次
  // setLogCallback 时才惰性挂载。级别门控已在 shouldLog / Emit 开头完成，logger_
  // 仅 set_level(trace) 放行全部；FileSink 例外——它的 level 承载 only_show 开关。
  auto console_sink = std::make_shared<ConsoleSink>();
  auto file_sink = std::make_shared<FileSink>(kLogFileName, !only_show);
  file_sink_ = file_sink;
  logger_ = std::make_shared<spdlog::logger>(
      kProjectDisplayName,
      spdlog::sinks_init_list{console_sink, file_sink});
  logger_->set_level(spdlog::level::trace);

  // 逐 sink 设 formatter：logger_->set_formatter 会把同一实例 clone 给所有 sink，
  // 覆盖文件的无 head pattern。
  console_sink->set_formatter(makeConsoleFormatter());
  file_sink->set_formatter(std::make_unique<spdlog::pattern_formatter>("%v"));

  // 构造完成，置存活标志，允许日志发射。
  g_log_engine_alive_flag.store(kLogEngineAliveMagic, std::memory_order_relaxed);
}

LogEngine::~LogEngine() {
  // 先清存活标志：此后 LogMessage 析构不再触碰已销毁的引擎成员。
  g_log_engine_alive_flag.store(0, std::memory_order_relaxed);
}

bool LogEngine::onlyShow() const {
  // only_show 语义与 FileSink 开关相反：only_show=true → 不落盘。
  return file_sink_ == nullptr || !file_sink_->enabled();
}

void LogEngine::setOnlyShow(bool only_show) {
  if (file_sink_ != nullptr) {
    file_sink_->setEnabled(!only_show);
  }
}

bool LogEngine::colorPrint() const {
  return color_print_.load(std::memory_order_relaxed);
}

void LogEngine::setColorPrint(bool color_print) {
  color_print_.store(color_print, std::memory_order_relaxed);
}

std::string LogEngine::logFile() const {
  return file_sink_ != nullptr ? file_sink_->logFile() : kLogFileName;
}

void LogEngine::setLogFile(const std::string& file_name) {
  if (file_sink_ != nullptr) {
    file_sink_->setLogFile(file_name);
  }
}

void LogEngine::setLogCallback(LogCallback callback) {
  // 原子函数指针 store：零锁，与 EmitFmt/Emit 的 relaxed load 无数据竞争。
  callback_.store(callback, std::memory_order_relaxed);
}

int LogEngine::parseMaxLogLevelValue(const std::string& value,
                                     int default_para) {
  std::string env_var = value;
  // 转大写后匹配。
  std::transform(env_var.begin(), env_var.end(), env_var.begin(),
                 [](unsigned char c) { return static_cast<char>(::toupper(c)); });
  if (env_var == "0" || env_var == "ERROR") {
    return 0;
  } else if (env_var == "1" || env_var == "WARNING") {
    return 1;
  } else if (env_var == "2" || env_var == "API_TRACE") {
    return 2;
  } else if (env_var == "3" || env_var == "INFO") {
    return 3;
  } else if (env_var == "4" || env_var == "DEBUG" || env_var == "DEBUG1" ||
             env_var == "DEBUG-1" || env_var == "DEBUG_1") {
    return 4;
  } else if (env_var == "5" || env_var == "DEBUG2" || env_var == "DEBUG-2" ||
             env_var == "DEBUG_2") {
    return 5;
  } else if (env_var == "6" || env_var == "DEBUG3" || env_var == "DEBUG-3" ||
             env_var == "DEBUG_3") {
    return 6;
  } else if (env_var == "7" || env_var == "DEBUG4" || env_var == "DEBUG-4" ||
             env_var == "DEBUG_4") {
    return 7;
  }
  return default_para;
}

LogMessage::LogMessage(std::string file, int line, Severity severity)
    : file_(file), line_(line), severity_(severity) {}

LogMessage::~LogMessage() {
  // 引擎未构造完成或已析构时不发射，避免访问已销毁单例。
  if (g_log_engine_alive_flag.load(std::memory_order_relaxed) !=
      kLogEngineAliveMagic) {
    return;
  }
  LogEngine::instance().Emit(severity_, file_, line_, stream_.str());
}

std::stringstream& LogMessage::stream() { return stream_; }

// 流式日志入口（LOG()/LOG_*_STREAM 走这里）：body 单行化后经 logger_ 分发到各
// sink（屏幕带 head、文件仅 body）。severity 门控在开头，被抑制级别直接返回，
// 不装配、不触发 cnrtGetDevice / getTime 副作用。有回调时先回调（传原始 severity
// 和 body）再分发。
void LogEngine::Emit(Severity severity, const std::string& file, int line,
                     const std::string& message) {
  static std::mutex log_mutex;

  if (!mluop::log::LogEngine::shouldLog(severity)) {
    return;
  }

  std::string body = message;
  clearEnter(&body);

  std::lock_guard<std::mutex> lock(log_mutex);
  const auto lvl = cvtSeverityToSpdlog(severity);
  const spdlog::source_loc loc{file.c_str(), line, nullptr};
  LogCallback callback = callback_.load(std::memory_order_relaxed);
  if (MLUOP_PREDICT_FALSE(callback != nullptr)) {
    callback(static_cast<int>(severity), /*function_name=*/nullptr, body.c_str());
  }
  logger_->log(loc, lvl, spdlog::string_view_t(body.data(), body.size()));
}
}  // namespace log
}  // namespace mluop

// severity -> spdlog level。显示不走 %L（head 里已有名字），仅 spdlog 语义用。
spdlog::level::level_enum mluop::log::LogEngine::cvtSeverityToSpdlog(mluop::log::Severity severity) {
  switch (severity) {
    case mluop::log::Severity::kError: {
      return spdlog::level::err;
    }
    case mluop::log::Severity::kWarning: {
      return spdlog::level::warn;
    }
    case mluop::log::Severity::kCnpapi: {
      return spdlog::level::trace;
    }
    case mluop::log::Severity::kInfo: {
      return spdlog::level::info;
    }
    case mluop::log::Severity::kDebug1:
    case mluop::log::Severity::kDebug2:
    case mluop::log::Severity::kDebug3:
    case mluop::log::Severity::kDebug4: {
      return spdlog::level::debug;
    }
  }
  return spdlog::level::debug;
}


extern "C" {

mluOpStatus_t MLUOP_WIN_API mluOpSetMaxLogLevel(MluOpLogLevel level) {
  mluop::log::LogEngine::setMaxLogLevel(static_cast<int>(level));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetLogOnlyShow(int only_show) {
  mluop::log::LogEngine::instance().setOnlyShow(only_show != 0);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetLogColorPrint(int color) {
  mluop::log::LogEngine::instance().setColorPrint(color != 0);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetLogFile(const char* file_name) {
  if (file_name == nullptr) {
    return MLUOP_STATUS_BAD_PARAM;
  }
  mluop::log::LogEngine::instance().setLogFile(std::string(file_name));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetLogCallback(MluOpLogCallback callback) {
  mluop::log::LogEngine::instance().setLogCallback(
      reinterpret_cast<mluop::log::LogCallback>(callback));
  return MLUOP_STATUS_SUCCESS;
}

}  // extern "C"
