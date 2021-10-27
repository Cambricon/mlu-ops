/*************************************************************************
 * Copyright (C) 2021 by Cambricon, Inc. All rights reserved.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include "core/platform/env_time.h"
#include "core/logging.h"

namespace {

int ParseInteger(const char *str, size_t size) {
  // Ideally we would use env_var / safe_strto64, but it is
  // hard to use here without pulling in a lot of dependencies,
  // so we use std:istringstream instead
  std::string integer_str(str, size);
  std::istringstream ss(integer_str);
  int level = 0;
  ss >> level;
  return level;
}

// Parse log level (int64) from environment variable (char*)
int64_t LogLevelStrToInt(const char *mlu_op_env_var_val) {
  if (mlu_op_env_var_val == nullptr) {
    return 0;
  }
  return ParseInteger(mlu_op_env_var_val, strlen(mlu_op_env_var_val));
}

// Using StringPiece breaks Windows build.
struct StringData {
  struct Hasher {
    size_t operator()(const StringData &sdata) const {
      // For dependency reasons, we cannot use hash.h here. Use DBJHash instead.
      size_t hash = 5381;
      const char *data = sdata.data;
      for (const char *top = data + sdata.size; data < top; ++data) {
        hash = ((hash << 5) + hash) + (*data);
      }
      return hash;
    }
  };

  StringData() = default;
  StringData(const char *data, size_t size) : data(data), size(size) {}

  bool operator==(const StringData &rhs) const {
    return size == rhs.size && memcmp(data, rhs.data, size) == 0;
  }

  const char *data = nullptr;
  size_t size = 0;
};

using VmoduleMap = std::unordered_map<StringData, int, StringData::Hasher>;

// Returns a mapping from module name to VLOG level, derived from the
// MLUOP_CPP_VMOUDLE environment variable; ownership is transferred to the caller.
VmoduleMap *VmodulesMapFromEnv() {
  // The value of the env var is supposed to be of the form:
  //    "foo=1,bar=2,baz=3"
  const char *env = getenv("MLUOP_CPP_VMODULE");
  if (env == nullptr) {
    // If there is no MLUOP_CPP_VMODULE configuration (most common case), return
    // nullptr so that the ShouldVlogModule() API can fast bail out of it.
    return nullptr;
  }
  // The memory returned by getenv() can be invalidated by following getenv() or
  // setenv() calls. And since we keep references to it in the VmoduleMap in
  // form of StringData objects, make a copy of it.
  const char *env_data = strdup(env);
  VmoduleMap *result = new VmoduleMap();
  while (true) {
    const char *eq = strchr(env_data, '=');
    if (eq == nullptr) {
      break;
    }
    const char *after_eq = eq + 1;

    // Comma either points at the next comma delimiter, or at a null terminator.
    // We check that the integer we parse ends at this delimiter.
    const char *comma = strchr(after_eq, ',');
    const char *new_env_data;
    if (comma == nullptr) {
      comma = strchr(after_eq, '\0');
      new_env_data = comma;
    } else {
      new_env_data = comma + 1;
    }
    (*result)[StringData(env_data, eq - env_data)] = ParseInteger(after_eq, comma - after_eq);
    env_data = new_env_data;
  }
  return result;
}

}  // namespace

namespace mluop {
namespace internal {

int64_t MinLogLevelFromEnv() {
  const char *mlu_op_env_var_val = std::getenv("MLUOP_MIN_LOG_LEVEL");
  return LogLevelStrToInt(mlu_op_env_var_val);
}

int64_t MinVLogLevelFromEnv() {
  const char *mlu_op_env_var_val = std::getenv("MLUOP_MIN_VLOG_LEVEL");
  return LogLevelStrToInt(mlu_op_env_var_val);
}

int64_t LogMessage::MinVLogLevel() {
  static int64_t min_vlog_level = MinVLogLevelFromEnv();
  return min_vlog_level;
}

void LogMessage::GenerateLogMessage() {
  static platform::EnvTime *env_time = platform::EnvTime::Default();
  uint64_t now_micros = env_time->NowMicros();
  time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
  int32_t micros_remainder = static_cast<int32_t>(now_micros % 1000000);
  const size_t time_buffer_size = 30;
  char time_buffer[time_buffer_size];  // NOLINT

  strftime(time_buffer, time_buffer_size, "%Y-%m-%d %H:%M:%S", localtime(&now_seconds));
  fprintf(stderr, "%s.%06d: %c %s:%d] %s\n", time_buffer, micros_remainder, "IWEF"[severity_],
          fname_, line_, str().c_str());
}

LogMessage::LogMessage(const char *fname, int line, int severity)
    : fname_(fname), line_(line), severity_(severity) {}

LogMessage::~LogMessage() {
  // Read the min log level once during the first call to logging.
  static int64_t min_log_level = MinLogLevelFromEnv();
  if (severity_ >= min_log_level) {
    GenerateLogMessage();
  }
}

bool LogMessage::VmoduleActivated(const char *fname, int level) {
  if (level <= MinVLogLevel()) {
    return true;
  }
  static VmoduleMap *vmodules = VmodulesMapFromEnv();
  if (MLUOP_PREDICT_TRUE(vmodules == nullptr)) {
    return false;
  }
  const char *last_slash = strrchr(fname, '/');
  const char *module_start = last_slash == nullptr ? fname : last_slash + 1;
  const char *dot_after = strchr(module_start, '.');
  const char *module_limit = dot_after == nullptr ? strchr(fname, '\0') : dot_after;
  StringData module(module_start, module_limit - module_start);
  auto it = vmodules->find(module);
  return it != vmodules->end() && it->second >= level;
}

void LogString(const char *fname, int line, int severity, const std::string &message) {
  LogMessage(fname, line, severity) << message;
}

template <>
void MakeCheckOpValueString(std::ostream *os, const char &v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "char value " << static_cast<short>(v);  // NOLINT
  }
}

template <>
void MakeCheckOpValueString(std::ostream *os, const signed char &v) {  // NOLINT
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "signed char value " << static_cast<short>(v);  // NOLINT
  }
}

template <>
void MakeCheckOpValueString(std::ostream *os, const unsigned char &v) {  // NOLINT
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "unsigned char value " << static_cast<unsigned short>(v);  // NOLINT
  }
}

#if LANG_CXX11
template <>
void MakeCheckOpValueString(std::ostream *os, const std::nullptr_t &p) {
  (*os) << "nullptr";
}
#endif

CheckOpMessageBuilder::CheckOpMessageBuilder(const char *exprtext)
    : stream_(new std::ostringstream) {
  *stream_ << "Check failed: " << exprtext << " (";
}

CheckOpMessageBuilder::~CheckOpMessageBuilder() {
  delete stream_;
}

std::ostream *CheckOpMessageBuilder::ForVar2() {
  *stream_ << " vs. ";
  return stream_;
}

std::string *CheckOpMessageBuilder::NewString() {
  *stream_ << ")";
  return new std::string(stream_->str());
}

}  // namespace internal
}  // namespace mluop

void mluOpCheck(mluOpStatus_t result,
                char const *const func,
                const char *const file,
                int const line) {
  if (result) {
    std::string error =
        "\"" + std::string(mluOpGetErrorString(result)) + " in " + std::string(func) + "\"";
    LOG(ERROR) << error;
    throw std::runtime_error(error);
  }
}
