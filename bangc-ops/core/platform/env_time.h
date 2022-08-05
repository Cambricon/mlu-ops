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
#ifndef CORE_PLATFORM_ENV_TIME_H_
#define CORE_PLATFORM_ENV_TIME_H_

#include <utility>
#include <string>
#include <limits>
#include <sstream>
#include "core/macros.h"

namespace mluop {
namespace platform {

// An interface used by the implementation to access timer related operations.
class EnvTime {
 public:
  static constexpr uint64_t kMicrosToPicos   = 1000ULL * 1000ULL;
  static constexpr uint64_t kMicrosToNanos   = 1000ULL;
  static constexpr uint64_t kMillisToMicros  = 1000ULL;
  static constexpr uint64_t kMillisToNanos   = 1000ULL * 1000ULL;
  static constexpr uint64_t kSecondsToMillis = 1000ULL;
  static constexpr uint64_t kSecondsToMicros = 1000ULL * 1000ULL;
  static constexpr uint64_t kSecondsToNanos  = 1000ULL * 1000ULL * 1000ULL;

  EnvTime() {}
  virtual ~EnvTime() = default;

  // Returns a default impl suitable for the current operating system.
  // The result of Default() belongs to this library and must never be deleted.
  static EnvTime *Default();

  // Returns the number of nano-seconds since the Unix epoch.
  virtual uint64_t NowNanos() const = 0;

  // Returns the number of micro-seconds since the Unix epoch.
  virtual uint64_t NowMicros() const { return NowNanos() / kMicrosToNanos; }

  // Returns the number of seconds since the Unix epoch.
  virtual uint64_t NowSeconds() const { return NowNanos() / kSecondsToNanos; }
};

}  // namespace platform
}  // namespace mluop

#endif  // CORE_PLATFORM_ENV_TIME_H_
