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

// 引擎存活标志（对照旧引擎 cnlogSingletonInitFlag_）：默认=存活（1），仅 ~LogEngine
// 置 0（已析构）。文件级静态原子，进程退出前可安全读取；LogMessage 析构据此在引擎
// 已销毁时跳过发射（不触碰 instance()），避免退出期后台线程/全局析构打日志的
// use-after-free。默认=存活使首条日志经 instance() 按需构造引擎，开箱即有日志。
constexpr int kLogEngineAliveMagic = 1;
std::atomic<int> g_log_engine_alive_flag = kLogEngineAliveMagic;

// 读环境变量并解析为最高可打印级别；未设置或非法回落 default_para。
int maxLogLevelFromEnv(const char* env_name, int default_para) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr) {
    return default_para;
  }
  return LogEngine::parseMaxLogLevelValue(std::string(raw), default_para);
}

// 是否打印到屏幕：恒写 std::cout，故只判 stdout 是否为 tty。
bool isPrintToScreen() { return isatty(fileno(stdout)); }


// 把消息中的换行替换为空格，保证单行输出。
void clearEnter(std::string* ss) {
  for (std::string::iterator it = (*ss).begin(); it != (*ss).end(); it++) {
    if (*it == '\n') {
      *it = ' ';
    }
  }
}

// 屏幕 sink：逐字节写 msg.payload 到 std::cout（endl = '\n' + flush）。payload
// 非持有，sink 同步链保证在本次 log() 调用内写完。
class ConsoleSink final : public spdlog::sinks::base_sink<std::mutex> {
 protected:
  void sink_it_(const spdlog::details::log_msg& msg) override {
    // 标准 spdlog 契约：sink 先用 formatter_ 格式化（pattern head + body），再写出。
    spdlog::memory_buf_t formatted;
    formatter_->format(msg, formatted);
    // pattern_formatter 已追加 eol（\n），这里只写、不再 put('\n')，否则双换行。
    std::cout.write(formatted.data(),
                    static_cast<std::streamsize>(formatted.size()));
    std::cout.flush();
  }
  void flush_() override { std::cout.flush(); }
};

// 文件 sink：逐字节写 msg.payload 到 log_file_stream_（endl = '\n' + flush）。
// 以引用持有流，is_open() 守卫防止流未打开 / 已关闭时写出错。
class FileSink final : public spdlog::sinks::base_sink<std::mutex> {
 public:
  explicit FileSink(std::ofstream& stream) : stream_(stream) {}

 protected:
  void sink_it_(const spdlog::details::log_msg& msg) override {
    if (stream_.is_open()) {
      spdlog::memory_buf_t formatted;
      formatter_->format(msg, formatted);
      // 同 ConsoleSink：formatter 已带 eol，不再 put('\n')。
      stream_.write(formatted.data(),
                    static_cast<std::streamsize>(formatted.size()));
      stream_.flush();
    }
  }
  void flush_() override {
    if (stream_.is_open()) {
      stream_.flush();
    }
  }

 private:
  std::ofstream& stream_;
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
      // spdlog 的 memory_buf_t 是 bundled fmt，外层 fmt::format_to 写不进去；
      // 用外层 fmt 栈 buffer 格式化（短串零堆分配），再拷贝到 dest。
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

// 自定义 pattern flag %c：等级名（小写 spdlog 名，按 cvtSeverityToSpdlog 塌缩），
// color_print_ 时按旧 per-severity 配色（经塌缩映射到 fmt terminal_color）。
class LevelFlag final : public spdlog::custom_flag_formatter {
 public:
  void format(const spdlog::details::log_msg& msg, const std::tm&,
              spdlog::memory_buf_t& dest) override {
    // 等级名大写（按 cvtSeverityToSpdlog 塌缩后的 spdlog 级）。
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

}  // namespace

LogEngine& LogEngine::instance() {
  static LogEngine engine;
  return engine;
}

LogEngine::LogEngine()
    : max_log_level_(maxLogLevelFromEnv(kEnvMaxLogLevel, 0)),
      only_show_(mluop::getBoolEnvVar(kEnvOnlyShow, true)),
      color_print_(mluop::getBoolEnvVar(kEnvColorPrint, true)),
      log_file_(kLogFileName) {
  // 仅当非 only_show 时打开日志文件；打开失败打印提示。随后按 stdout 是否
  // tty 决定是否保留彩色。构造期为单线程初始化，直接读 atomic 值即可。
  if (!only_show_.load(std::memory_order_relaxed)) {
    log_file_stream_.open(log_file_);
    if (!log_file_stream_.is_open()) {
      std::cerr << "can't init Log, open file failed!" << std::endl;
    }
  }
  if (color_print_.load(std::memory_order_relaxed)) {
    color_print_.store(isPrintToScreen(), std::memory_order_relaxed);
  }

  // 接线 spdlog：logger_（屏幕）挂 ConsoleSink + callback_sink_，file_logger_
  // （文件）挂 FileSink（引用 log_file_stream_）。两 logger 均 set_level(trace)
  // —— spdlog logger 默认 info(2) 会滤掉 ERROR/WARNING（severity 0/1），必须放行。
  // Emit 里已按 max_log_level_ 在函数开头门控，此处只需不额外过滤。
  callback_sink_ = std::make_shared<LogCallbackSinkMt>();
  // 默认 logger_ 只挂 ConsoleSink（未设回调时日志仅打屏，不付回调 sink 的开销）；
  // 调 setLogCallback 时才把 callback_sink_ 动态加入 logger_->sinks()。
  logger_ = std::make_shared<spdlog::logger>(
      kProjectDisplayName,
      spdlog::sinks_init_list{std::make_shared<ConsoleSink>()});
  logger_->set_level(spdlog::level::trace);

  file_logger_ = std::make_shared<spdlog::logger>(
      kProjectDisplayName, std::make_shared<FileSink>(log_file_stream_));
  file_logger_->set_level(spdlog::level::trace);

  // 屏幕 pattern：head 全部由 spdlog 装配（时间 %Y-%m-%d %H:%M:%S.%f、模块 %n、等级 %L
  // 按 cvtSeverityToSpdlog 塌缩为小写、pid %P、Card 自定义 flag %k），body 为 %v。
  // 文件 sink 只 %v（无 head）。Debug 下屏幕追加 tail（file:line pid）。
  auto console_formatter = std::make_unique<spdlog::pattern_formatter>();
  console_formatter->add_flag<CardFlag>('k');
  console_formatter->add_flag<ModuleFlag>('K');
  console_formatter->add_flag<LevelFlag>('c');
#ifdef NDEBUG
  console_formatter->set_pattern(
      "[%Y-%m-%d %H:%M:%S.%f][%K][%c][%P][Card:%k]: %v");
#else
  console_formatter->set_pattern(
      "[%Y-%m-%d %H:%M:%S.%f][%K][%c][%P][Card:%k]: %v  [%s:%#  pid:%P]");
#endif
  logger_->set_formatter(std::move(console_formatter));
  file_logger_->set_pattern("%v");

  // 构造完成，置存活标志，允许日志发射。
  g_log_engine_alive_flag.store(kLogEngineAliveMagic, std::memory_order_relaxed);
}

LogEngine::~LogEngine() {
  // 析构开始即清存活标志：此后日志发射在 LogMessage 析构入口被拦下，
  // 不会访问已销毁的成员。
  g_log_engine_alive_flag.store(0, std::memory_order_relaxed);
}

int LogEngine::maxLogLevel() const {
  return max_log_level_.load(std::memory_order_relaxed);
}

void LogEngine::setMaxLogLevel(int level) {
  if (level < kMinSeverity) {
    level = kMinSeverity;
  } else if (level > kMaxSeverity) {
    level = kMaxSeverity;
  }
  max_log_level_.store(level, std::memory_order_relaxed);
}

bool LogEngine::shouldLog(Severity severity) const {
  return severityEnabled(static_cast<int>(severity),
                         max_log_level_.load(std::memory_order_relaxed));
}

bool LogEngine::onlyShow() const {
  return only_show_.load(std::memory_order_relaxed);
}

void LogEngine::setOnlyShow(bool only_show) {
  only_show_.store(only_show, std::memory_order_relaxed);
}

bool LogEngine::colorPrint() const {
  return color_print_.load(std::memory_order_relaxed);
}

void LogEngine::setColorPrint(bool color_print) {
  color_print_.store(color_print, std::memory_order_relaxed);
}

std::string LogEngine::logFile() const {
  std::lock_guard<std::mutex> lock(log_file_mutex_);
  return log_file_;
}

void LogEngine::setLogFile(const std::string& file_name) {
  std::lock_guard<std::mutex> lock(log_file_mutex_);
  log_file_ = file_name;
}

void LogEngine::setLogCallback(LogCallback callback) {
  // 动态接线：回调收到完整屏幕行（head + body + tail）。callback 存于
  // callback_sink_（atomic）。设非空回调时把该 sink 加入 logger_，设空回调时移除，
  // 保证默认（无回调）状态下 logger_ 只有 ConsoleSink、不付额外开销。
  if (callback_sink_ == nullptr) {
    return;
  }
  callback_sink_->set_callback(callback);
  auto& sinks = logger_->sinks();
  if (callback != nullptr) {
    if (std::find(sinks.begin(), sinks.end(), callback_sink_) == sinks.end()) {
      sinks.push_back(callback_sink_);
    }
  } else {
    sinks.erase(std::remove(sinks.begin(), sinks.end(), callback_sink_),
                sinks.end());
  }
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
  // 引擎未存活（未构造完成或已析构）时不发射，避免访问已销毁单例。
  if (g_log_engine_alive_flag.load(std::memory_order_relaxed) !=
      kLogEngineAliveMagic) {
    return;
  }
  LogEngine::instance().Emit(severity_, file_, line_, stream_.str());
}

std::stringstream& LogMessage::stream() { return stream_; }

// 装配并输出一条日志。屏幕 = head + body + tail；文件 = body + tail（无 head）。
// severity 门控提到函数开头：被抑制级别的日志直接返回，不做 head/body/tail 装配，
// 也不触发 cnrtGetDevice / getTime 副作用。NDEBUG 下不输出 tail。
void LogEngine::Emit(Severity severity, const std::string& file, int line,
                     const std::string& message) {
  static std::mutex log_mutex;

  // 提前门控：被抑制级别不输出。
  if (static_cast<int>(severity) >
      max_log_level_.load(std::memory_order_relaxed)) {
    return;
  }

  // 屏幕/文件均只传 body；head 由 spdlog pattern 装配（时间/模块/%L/pid/Card）。
  std::string body = message;
  clearEnter(&body);

  std::lock_guard<std::mutex> lock(log_mutex);
  const auto lvl = cvtSeverityToSpdlog(severity);
  const spdlog::source_loc loc{file.c_str(), line, nullptr};
  logger_->log(loc, lvl, spdlog::string_view_t(body.data(), body.size()));
  if (!only_show_.load(std::memory_order_relaxed)) {
    file_logger_->log(loc, lvl, spdlog::string_view_t(body.data(), body.size()));
  }
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
  return spdlog::level::trace;
}


extern "C" {

mluOpStatus_t MLUOP_WIN_API mluOpSetMaxLogLevel(MluOpLogLevel level) {
  mluop::log::LogEngine::instance().setMaxLogLevel(static_cast<int>(level));
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
