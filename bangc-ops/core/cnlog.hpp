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
#ifndef CNLOG_HPP_
#define CNLOG_HPP_

#include <sstream>
#include <iostream>
#include <fstream>

namespace mluop {
namespace logging {

/**
 * @brief: define save in file or not, show on screen or not.
 */
#define LOG_OFF_ALL 1
#define LOG_SAVE_ONLY 2
#define LOG_SHOW_ONLY 3
#define LOG_SAVE_AND_SHOW 4

// use this macro to witch whether print on screen in color or not
// #define COLOR_SILENCE

/**
 * @brief: define colors used to print word on the screen.
 */
#if !defined(WIN32) && !defined(COLOR_SILENCE)

#define RESET "\033[0m"
#define HIGHLIGHT "\033[1m"
#define UNDERLINE "\033[4m"
#define BLACK "\033[30m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"

#else

#define RESET ""
#define HIGHLIGHT ""
#define UNDERLINE ""
#define BLACK ""
#define RED ""
#define GREEN ""
#define YELLOW ""
#define BLUE ""
#define MAGENTA ""

#endif

/**
 * @brief: define the six levels of the log.
 */
#define LOG_INFO 0
#define LOG_WARNING 1
#define LOG_ERROR 2
#define LOG_FATAL 3
#define LOG_VLOG 4
#define LOG_CNPAPI 5

/**
 * @brief: define the interface of the log system.
 */
#define CLOG(module, severity)                                               \
  LogMessage(__FILE__, __LINE__, LOG_SAVE_AND_SHOW, LOG_##severity, #module, \
             true, true, true, true)                                         \
      .stream()

#define DCLOG(module, severity)                                              \
  LogMessage(__FILE__, __LINE__, LOG_SAVE_AND_SHOW, LOG_##severity, #module, \
             true, true, true, false)                                        \
      .stream()

#define PLOG(module, severity)                                                \
  LogMessage("", 0, LOG_SAVE_AND_SHOW, LOG_##severity, #module, false, false, \
             true, true)                                                      \
      .stream()

#define DPLOG(module, severity)                                               \
  LogMessage("", 0, LOG_SAVE_AND_SHOW, LOG_##severity, #module, false, false, \
             true, false)                                                     \
      .stream()

#define SCOUT(module, severity)                                               \
  LogMessage("", 0, LOG_SAVE_AND_SHOW, LOG_##severity, #module, false, false, \
             false, true)                                                     \
      .stream()

#define DSCOUT(module, severity)                                              \
  LogMessage("", 0, LOG_SAVE_AND_SHOW, LOG_##severity, #module, false, false, \
             false, false)                                                    \
      .stream()

/**
 * @brief: the log class to realize the log system.
 */
class LogMessage {
 public:
  LogMessage(std::string file, int line, int module, int severity,
             std::string module_name, bool is_print_head, bool is_print_tail,
             bool is_clear_endl, bool release_can_print);
  ~LogMessage();

  /**
   * @brief: used by interface macro, use << to get the string content.
   */
  std::stringstream& stream();

 private:
  std::string logInfoFile_;  // the file that the log message code belongs to
  int logInfoLine_;          // the line that the log message code in
  int log_module_;   // describe the log message whether to save or to show
  int logSeverity_;  // the level of the message
  std::string module_name_;       // used to print module name stamp
  bool is_print_head_;            // whether print log head or not
  bool is_print_tail_;            // whether print log tail or not
  bool is_clear_endl_;            // whether clear endl int the string contex
  bool release_can_print_;        // whether can print in release mode
  std::stringstream contex_str_;  // the context behind "<<"
  std::stringstream cout_str_;    // the context to show in the screen
  std::stringstream file_str_;    // the context to save in the file

  /**
   * @brief: get system time.
   * @return: time stamp in string format.
   */
  std::string getTime();

  /**
   * @brief: print log message head to string stream.
   * @param: switch whether print log message colored to cout_str_.
   */
  void printHead(bool is_colored);

  /**
   * @brief: print log message tail to string stream.
   * @param: switch whether print log message colored to cout_str_.
   */
  void printTail(bool is_colored);
};

}  // namespace logging
}  // namespace mluop

#endif  // CNLOG_HPP_
