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
#pragma once

#include <type_traits>

#include "core/tool.h"
#include "core/preprocessor.h"

#ifdef STR
#undef STR
#endif
#define STR(e) #e

namespace mluop {
namespace cfg {

#define MLUOP_CONFIG_ENV_TYPE_LIST                                       \
  MLUOP_TRACE_ENABLE, MLUOP_TRACE_ENABLE_API, MLUOP_TRACE_ENABLE_KERNEL, \
      MLUOP_TRACE_DATA_DIR, MLUOP_DEBUG_KERNEL_TRACING,                  \
      MLUOP_EVENT_ENABLE_API, MLUOP_EVENT_ENABLE_KERNEL, MLUOP_DUMP_API_COUNT

#define ENUM_CASE_CONFIG_ENV_TYPE(e) \
  case ConfigEnvType::e: {           \
    return STR(e);                   \
  }
#define CFG_ENUM_TO_STR(e) ConfigEnvTypeReflection(ConfigEnvType::e)

// wrap env into single file
enum ConfigEnvType {
  MLUOP_TRACE_ENABLE,
  MLUOP_TRACE_ENABLE_API,
  MLUOP_TRACE_ENABLE_KERNEL,
  MLUOP_TRACE_DATA_DIR,
  MLUOP_DEBUG_KERNEL_TRACING,
  MLUOP_EVENT_ENABLE_API,
  MLUOP_EVENT_ENABLE_KERNEL,
  MLUOP_DUMP_API_COUNT,
};

static const char* ConfigEnvTypeReflection(enum ConfigEnvType evt) {
  switch (evt) {
    MLUOP_PP_MAP(ENUM_CASE_CONFIG_ENV_TYPE, (MLUOP_CONFIG_ENV_TYPE_LIST));
  }
  return NULL;
}

class Config {
 public:
  static Config& instance() {
    static Config cfg;
    return cfg;
  }

  template <
      enum ConfigEnvType T,
      typename std::enable_if<T == ConfigEnvType::MLUOP_EVENT_ENABLE_API ||
                                  T == ConfigEnvType::MLUOP_EVENT_ENABLE_KERNEL,
                              bool>::type = true>
  static void set_event(bool enable) {
    switch (T) {
      case ConfigEnvType::MLUOP_EVENT_ENABLE_API:
        instance().event_enable_api_ = enable;
        break;
      case ConfigEnvType::MLUOP_EVENT_ENABLE_KERNEL:
        instance().event_enable_kernel_ = enable;
        break;
      default:
        __builtin_unreachable();
    }
  }

  template <
      enum ConfigEnvType T,
      typename std::enable_if<T == ConfigEnvType::MLUOP_EVENT_ENABLE_API ||
                                  T == ConfigEnvType::MLUOP_EVENT_ENABLE_KERNEL,
                              bool>::type = true>
  static bool get_event() {
    if (T == MLUOP_EVENT_ENABLE_API) {
      return instance().event_enable_api_;
    }
    if (T == MLUOP_EVENT_ENABLE_KERNEL) {
      return instance().event_enable_kernel_;
    }
  }

#if 0
      static bool get_event(ConfigEnvType evt) {
        switch (evt) {
          case ConfigEnvType::MLUOP_EVENT_ENABLE_API:
            return instance().event_enable_api_;
          case ConfigEnvType::MLUOP_EVENT_ENABLE_KERNEL:
            return instance().event_enable_kernel_;
          default:
            // unsupported event type
            return false;
        }
      }
#endif

 private:
  Config() = default;
  // TODO(NONE): expose event enable/disable interface (not just env)
  // TODO(NONE): should be refined: add direct mapping which could avoid
  // 'if-else'/'switch-case' pattern
  bool event_enable_api_ =
      mluop::getBoolEnvVar(CFG_ENUM_TO_STR(MLUOP_EVENT_ENABLE_API), false);
  bool event_enable_kernel_ =
      mluop::getBoolEnvVar(CFG_ENUM_TO_STR(MLUOP_EVENT_ENABLE_KERNEL), true);
};

}  // namespace cfg
}  // namespace mluop
