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
#include <utility>
#include <string>
#include <limits>
#include <sstream>
#include "core/platform/env_time.h"

namespace mluop {
namespace platform {

class PosixEnvTime : public EnvTime {
 public:
  PosixEnvTime() {}

  uint64_t NowNanos() const override {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (static_cast<uint64_t>(ts.tv_sec) * kSecondsToNanos + static_cast<uint64_t>(ts.tv_nsec));
  }
};

#if defined(PLATFORM_POSIX)
EnvTime *EnvTime::Default() {
  static EnvTime *default_env_time = new PosixEnvTime;
  return default_env_time;
}
#endif

}  // namespace platform
}  // namespace mluop
