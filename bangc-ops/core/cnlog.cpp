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
#include "cnlog.hpp"

#include <string>
#include <mutex>  // NOLINT
#include <algorithm>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>

#include "tool.h"

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

#ifdef ANDROID_LOG  // used in android

#include <jni.h>
#include "android/log.h"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "camb", __VA_ARGS__)
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, "camb", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "camb", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, "camb", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "camb", __VA_ARGS__)

#endif

#define BASE_YEAR 1900
#define BASE_MONTH 1
#define SECONDS_PER_HOUR 3600
#define HOURS_DIFFERENCE 8

#define CNLOG_INIT_MAGIC_NUM 1

namespace mluop {
namespace logging {

static int getLevelEnvVar(const std::string& str, int default_para = false);

class cnlogSingleton {
 public:
  static inline cnlogSingleton& get() {
    static cnlogSingleton _log;  // = new cnlogSingleton;
    return _log;
  }
  static inline bool is_only_show() { return get().is_only_show_; }
  static int init() {
    get().initCNLog();
    return 0;
  }
  static inline const std::map<std::string, bool>& module_print_map() {
    return get().module_print_map_;
  }
  static inline int logLevel() { return get().logLevel_; }
  static inline bool g_color_print() { return get().g_color_print_; }
  static inline std::ofstream& logFile() { return get().logFile_; }
  static inline std::ostream& userStream() { return get().userStream_; }

  bool isPrintToScreen() {
    if (userStream_.rdbuf() == std::cout.rdbuf()) {
      return isatty(fileno(stdout));
    }
    if (userStream_.rdbuf() == std::cerr.rdbuf()) {
      return isatty(fileno(stderr));
    }
    return false;
  }
  ~cnlogSingleton() { cnlogSingletonInitFlag_ = 0; }
  static int cnlogSingletonInitFlag_;

 private:
  void initCNLog() {
    if (is_only_show_) {
      initLogOnlyShow();
    } else {
      initLog("mlu_op_auto_log");
    }
    cnlogSingletonInitFlag_ = CNLOG_INIT_MAGIC_NUM;
  }

  void initLogOnlyShow() {
    if (is_open_log_) {
      return;
    }
    is_open_log_ = true;
  }

  /*
   * @brief: initialing log system, make sure use it before any log
   */
  void initLog(std::string file) {
#ifndef ANDROID_LOG
    if (is_open_log_) {
      return;
    }
    logFile_.open(file);
    if (!logFile_.is_open()) {
      std::cerr << "can't init Log, open file failed!" << std::endl;
    } else {
      is_open_log_ = true;
    }
#else
    is_open_log_ = true;
#endif
  }

  cnlogSingleton() {
    initCNLog();
    if (g_color_print_) {
      g_color_print_ = isPrintToScreen();
    }
  }
  std::ostream userStream_{std::cout.rdbuf()};  // stream to print on screen.
  bool is_open_log_ = false;
  std::ofstream logFile_;  // which file to save log message.
  int logLevel_ = getLevelEnvVar("MLUOP_MIN_LOG_LEVEL",
                                 0);  // only the level GE this will be print.
  bool is_only_show_ = mluop::getBoolEnvVar(
      "MLUOP_LOG_ONLY_SHOW", true);  // whether only show on screen.
  bool g_color_print_ = mluop::getBoolEnvVar("MLUOP_LOG_COLOR_PRINT",
                                             true);  // whether print with color
  const std::map<std::string, bool> module_print_map_{
      {"MLUOP", mluop::getBoolEnvVar("MLUOP_LOG_PRINT", true)}};
};

static uint64_t warningCnt = 0;  // counts of warning
static uint64_t errorCnt = 0;    // counts of error
static int64_t fatalCnt = 0;     // counts of fatal

int cnlogSingleton::cnlogSingletonInitFlag_ = 0;

static bool releasePrint(std::string module_name) {
  if (cnlogSingleton::cnlogSingletonInitFlag_ != CNLOG_INIT_MAGIC_NUM)
    return false;
  bool module_print = false;
  if (cnlogSingleton::module_print_map().count(module_name) > 0) {
    module_print = cnlogSingleton::module_print_map().at(module_name);
  }
  return module_print;
}

LogMessage::LogMessage(std::string file, int line, int module, int severity,
                       std::string module_name, bool is_print_head,
                       bool is_print_tail, bool is_clear_endl,
                       bool release_can_print)
    : logInfoFile_(file),
      logInfoLine_(line),
      log_module_(module),
      logSeverity_(severity),
      module_name_(module_name),
      is_print_head_(is_print_head),
      is_print_tail_(is_print_tail),
      is_clear_endl_(is_clear_endl),
      release_can_print_(release_can_print) {
#ifndef ANDROID_LOG
  if (is_print_head_) {
    if ((log_module_ == LOG_SHOW_ONLY) || (log_module_ == LOG_SAVE_AND_SHOW)) {
      if (cnlogSingleton::g_color_print()) {
        printHead(true);  // print head colored.
      } else {
        printHead(false);  // print head no colored.
      }
    } else {
      printHead(false);  // print head no colored.
    }
  }
#else
  printHead(false);  // print head no colored.
#endif
}

/**
 * @brief: used by interface macro, use << to get the string content.
 */
std::stringstream& LogMessage::stream() { return contex_str_; }

/**
 * @brief: clear endl in param string.
 */
void clearEnter(std::string* ss) {
  for (std::string::iterator it = (*ss).begin(); it != (*ss).end(); it++) {
    if (*it == '\n') {
      *it = ' ';
    }
  }
}

/*
 * @brief: the destructor that output the string to the file or screen.
 */
LogMessage::~LogMessage() {
  static std::mutex log_mutex;  // to protect write to file.
  int log_level = LOG_INFO;
#ifdef NDEBUG
  if (!releasePrint(module_name_)) {
    return;
  }
  log_level = cnlogSingleton::logLevel();
  is_print_tail_ = false;
#else
  if (!releasePrint(module_name_)) {
    return;
  }
  log_level = cnlogSingleton::logLevel();
  is_print_tail_ = true;
#endif
  file_str_ << contex_str_.str();
  cout_str_ << contex_str_.str();
  if (is_print_tail_) {
#ifndef ANDROID_LOG
    if ((log_module_ == LOG_SHOW_ONLY) || (log_module_ == LOG_SAVE_AND_SHOW)) {
      if (cnlogSingleton::g_color_print()) {
        printTail(true);
      } else {
        printTail(false);
      }
    } else {
      printTail(false);
    }
#else
    printTail(false);
#endif
  }
  std::string file_ss = file_str_.str();
  std::string cout_ss = cout_str_.str();
  if (is_clear_endl_) {
    clearEnter(&file_ss);
    clearEnter(&cout_ss);
  }
  if (logSeverity_ >= log_level) {
    std::lock_guard<std::mutex> lock(log_mutex);
    switch (logSeverity_) {
      case LOG_WARNING: {
        warningCnt++;
        break;
      }
      case LOG_ERROR: {
        errorCnt++;
        break;
      }
      case LOG_FATAL: {
        fatalCnt++;
        break;
      }
      default: {
        break;
      }
    }
#ifndef ANDROID_LOG
    if ((log_module_ == LOG_SAVE_ONLY) || (log_module_ == LOG_SAVE_AND_SHOW)) {
      if (!cnlogSingleton::is_only_show()) {
        cnlogSingleton::logFile() << file_ss;
        if (is_clear_endl_) {
          cnlogSingleton::logFile() << std::endl;
        }
      }
    }
    if ((log_module_ == LOG_SHOW_ONLY) || (log_module_ == LOG_SAVE_AND_SHOW)) {
      cnlogSingleton::userStream() << cout_ss;
      if (is_clear_endl_) {
        cnlogSingleton::userStream() << std::endl;
      }
    }
#else
    switch (logSeverity_) {
      case LOG_INFO: {
        LOGI("%s", file_ss.c_str());
        break;
      }
      case LOG_WARNING: {
        LOGW("%s", file_ss.c_str());
        break;
      }
      case LOG_ERROR: {
      }
      case LOG_FATAL: {
        LOGE("%s", file_ss.c_str());
        break;
      }
      case LOG_VLOG: {
        LOGD("%s", file_ss.c_str());
        break;
      }
      case LOG_CNPAPI: {
        LOGD("%s", file_ss.c_str());
        break;
      }
      default: {
        break;
      }
    }
#endif
  }
}

/**
 * @brief: get system time.
 * @return: time stamp in string format.
 */
std::string LogMessage::getTime() {
#ifdef ANDROID_LOG
  return "";
#else
#if defined(WINDOWS) || defined(WIN32)
  SYSTEMTIME systime;
  GetLocalTime(&systime);
  std::string year = std::to_string(systime.wYear);
  std::string month = std::to_string(systime.wMonth);
  std::string day = std::to_string(systime.wDay);
  std::string hour = std::to_string(systime.wHour);
  std::string min = std::to_string(systime.wMinute);
  std::string second = std::to_string(systime.wSecond);
  std::string time_stamp = "[" + year + "-" + month + "-" + day + " " + hour +
                           ":" + min + ":" + second + "] ";
  return time_stamp;
#else
  time_t g_time;
  time(&g_time);
  g_time = g_time + HOURS_DIFFERENCE * SECONDS_PER_HOUR;
  tm general_time;
  if (NULL == gmtime_r(&g_time, &general_time)) {
    return "";
  }
  std::string year = std::to_string(general_time.tm_year + BASE_YEAR);
  std::string month = std::to_string(general_time.tm_mon + BASE_MONTH);
  std::string day = std::to_string(general_time.tm_mday);
  std::string hour = std::to_string(general_time.tm_hour);
  std::string min = std::to_string(general_time.tm_min);
  std::string second = std::to_string(general_time.tm_sec);
  std::string time_stamp = "[" + year + "-" + month + "-" + day + " " + hour +
                           ":" + min + ":" + second + "] ";
  return time_stamp;
#endif
#endif
}

/**
 * @brief: print log message head to string stream.
 * @param: switch whether print log message colored to cout_str_.
 */
void LogMessage::printHead(bool is_colored) {
#ifndef NDEBUG
  if (cnlogSingleton::logLevel() == 5) {
    return;
  }
#endif
  file_str_ << getTime();
  file_str_ << "[" << module_name_ << "] ";
  switch (logSeverity_) {
    case LOG_INFO: {
      file_str_ << "[Info]: ";
      break;
    }
    case LOG_WARNING: {
      file_str_ << "[Warning]: ";
      break;
    }
    case LOG_ERROR: {
      file_str_ << "[Error]: ";
      break;
    }
    case LOG_FATAL: {
      file_str_ << "[Fatal]: ";
      break;
    }
    case LOG_VLOG: {
      file_str_ << "[Vlog]: ";
      break;
    }
    case LOG_CNPAPI: {
      file_str_ << "[Cnpapi]: ";
      break;
    }
    default: {
      break;
    };
  }
  if (is_colored) {
    cout_str_ << getTime();
    cout_str_ << HIGHLIGHT << YELLOW << "[" << module_name_ << "] ";
    switch (logSeverity_) {
      case LOG_INFO: {
        cout_str_ << HIGHLIGHT << GREEN << "[Info]:" << RESET;
        break;
      }
      case LOG_WARNING: {
        cout_str_ << HIGHLIGHT << MAGENTA << "[Warning]:" << RESET;
        break;
      }
      case LOG_ERROR: {
        cout_str_ << HIGHLIGHT << RED << "[Error]:" << RESET;
        break;
      }
      case LOG_FATAL: {
        cout_str_ << HIGHLIGHT << RED << "[Fatal]:" << RESET;
        break;
      }
      case LOG_VLOG: {
        cout_str_ << HIGHLIGHT << BLUE << "[Vlog]:" << RESET;
        break;
      }
      case LOG_CNPAPI: {
        cout_str_ << HIGHLIGHT << BLUE << "[Cnpapi]:" << RESET;
        break;
      }
      default: {
        break;
      }
    }
  } else {
    cout_str_ << file_str_.str();
  }
}

/**
 * @brief: print log message tail to string stream.
 * @param: switch whether print log message colored to cout_str_.
 */
void LogMessage::printTail(bool is_colored) {
#ifndef NDEBUG
  if (!releasePrint(module_name_)) {
    return;
  }
#if defined(WINDOWS) || defined(WIN32)
  int pid = _getpid();
  int tid = GetCurrentThreadId();
#else
  int pid = getpid();
  int tid = syscall(SYS_gettid);
#endif
  std::stringstream realId;
  if (pid == tid) {
    realId << pid;
  } else {
    realId << "{" << tid << "}";
  }
  file_str_ << "  "
            << "[ " << logInfoFile_ << ":" << logInfoLine_
            << "  pid:" << realId.str() << "]";
  if (is_colored) {
    cout_str_ << "  " << GREEN << "[ " << logInfoFile_ << ":" << logInfoLine_
              << "  pid:" << realId.str() << "]" << RESET;
  } else {
    cout_str_ << "  "
              << "[ " << logInfoFile_ << ":" << logInfoLine_
              << "  pid:" << realId.str() << "]";
  }
#else
  if (releasePrint(module_name_)) {
#if defined(WINDOWS) || defined(WIN32)
    int pid = _getpid();
    int tid = GetCurrentThreadId();
#else
    int pid = getpid();
    int tid = syscall(SYS_gettid);
#endif
    std::stringstream realId;
    if (pid == tid) {
      realId << pid;
    } else {
      realId << "{" << tid << "}";
    }
    file_str_ << "  "
              << "[ " << logInfoFile_ << ":" << logInfoLine_
              << "  pid:" << realId.str() << "]";
    if (is_colored) {
      cout_str_ << "  " << GREEN << "[ " << logInfoFile_ << ":" << logInfoLine_
                << "  pid:" << realId.str() << "]" << RESET;
    } else {
      cout_str_ << "  "
                << "[ " << logInfoFile_ << ":" << logInfoLine_
                << "  pid:" << realId.str() << "]";
    }
  }
#endif
}

int getLevelEnvVar(const std::string& str, int default_para) {
  const char* env_raw_ptr = std::getenv(str.c_str());
  if (env_raw_ptr == nullptr) {
    return default_para;
  }
  std::string env_var = std::string(env_raw_ptr);
  /// to up case
  std::transform(env_var.begin(), env_var.end(), env_var.begin(), ::toupper);
  if (env_var == "0" || env_var == "INFO") {
    return 0;
  } else if (env_var == "1" || env_var == "WARNING") {
    return 1;
  } else if (env_var == "2" || env_var == "ERROR") {
    return 2;
  } else if (env_var == "3" || env_var == "FATAL") {
    return 3;
  }
  return default_para;
}

}  // namespace logging
}  // namespace mluop

#if 0  // TODO(None): extend lifetime
using mluop::logging::cnlogSingleton;
static cnlogSingleton *log_ctx = nullptr;

static void __attribute__((destructor(101))) mluOpLoggingLibDestructor() {
  std::cout << __func__ << std::endl;
}
#endif
