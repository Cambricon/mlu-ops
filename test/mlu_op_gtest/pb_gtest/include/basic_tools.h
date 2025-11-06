/*************************************************************************
 * Copyright (C) [2025] by Cambricon, Inc.
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
#pragma once
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>  //NOLINT
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>

// TODO(niewenchang): 拆分自tools.h，不依赖proto
namespace mluoptest {
static std::time_t getCurrentTimeT() {
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  return std::chrono::system_clock::to_time_t(now);
}

inline static std::string getCurrentTimeStr() {
  std::time_t current_time = getCurrentTimeT();
  std::tm *local_time = std::localtime(&current_time);
  char buffer[80];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", local_time);
  return std::string(buffer);
}

inline bool getEnv(const std::string &env, bool default_ret) {
  char *env_temp = getenv(env.c_str());
  if (env_temp != NULL) {
    if (strcmp(env_temp, "ON") == 0 || strcmp(env_temp, "1") == 0) {
      return true;
    } else if (strcmp(env_temp, "OFF") == 0 || strcmp(env_temp, "0") == 0) {
      return false;
    } else {
      return default_ret;
    }
  } else {
    return default_ret;
  }
}

inline int getEnvInt(const std::string &env, int default_ret) {
  char *env_temp = std::getenv(env.c_str());
  if (env_temp) {
    return std::atoi(env_temp);
  }

  return default_ret;
}

inline static unsigned long getFileSize(const std::string &filename) {  // NOLINT
  // XXX I want C++17 and std::file_system ...
  struct stat file_stat;
  int fd = open(filename.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd == -1) {
    std::cerr << "File open failed: " << filename << ". Reason: " << errno
              << "-" << strerror(errno);
    return 0;
  }

  int ret_stat = fstat(fd, &file_stat);
  if (ret_stat == -1) {
    std::cerr << "File stat failed: " << filename << ". Reason: " << errno
              << "-" << strerror(errno);
    close(fd);
    return 0;
  }

  close(fd);
  return file_stat.st_size;
}

}  // namespace mluoptest
