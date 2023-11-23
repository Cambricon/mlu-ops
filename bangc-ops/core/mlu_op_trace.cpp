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
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <ctime>
#include <chrono>  // NOLINT
#include <fstream>
#include <string>
#include <atomic>
#include <vector>
#include <set>
#include <mutex>  // NOLINT
#include <iomanip>
#include <sstream>
#include "core/logging.h"
#include "core/tool.h"
#include "core/config_env.h"
#include "core/mlu_op_internal_api.h"

#define TRACE_RAW_DATA_FILE_NAME std::string("mlu_op_trace_raw_data")
#define TRACE_RAW_DATA_DIR_DEFAULT std::string(".")
#define TRACE_RAW_DATA_DIR load_config_from_env_mluop_trace_data_dir()
#define API_FILE_NAME std::string("mlu_op_api.csv")
#define KERNEL_FILE_NAME std::string("mlu_op_kernel.csv")

using mluop::cfg::Config;
using mluop::cfg::ConfigEnvType;
using mluop::cfg::ConfigEnvTypeReflection;

inline static std::string load_config_from_env_mluop_trace_data_dir() {
  static std::string mluop_trace_dir =
      mluop::getStringEnvVar(CFG_ENUM_TO_STR(MLUOP_TRACE_DATA_DIR),
                             TRACE_RAW_DATA_DIR_DEFAULT) +
      "/" + TRACE_RAW_DATA_FILE_NAME;
  return mluop_trace_dir;
}

#define TRACE_API 0x01u     // 0b0001
#define TRACE_KERNEL 0x02u  // 0b0010
#define TRACE_MASK 0x03u    // 0b0011

inline static uint32_t load_config_from_env_mluop_trace() {
  if (!mluop::getBoolEnvVar(CFG_ENUM_TO_STR(MLUOP_DEBUG_KERNEL_TRACING),
                            true)) {
    LOG(INFO) << "MLUOP_DEBUG_KERNEL_TRACING is setup, tracing is disabled "
                 "completely";
    return 0;
  }
  uint32_t trace_bit_config = 0x0;
  /* - if env 'MLUOP_TRACE_ENABLE' is false, env 'MLUOP_TRACE_ENABLE_XXX' (XXX
   * is API/KERNEL) will be used (env value will be 'OR'ed)
   * - if env 'MLUOP_TRACE_ENABLE' is true, env 'MLUOP_TRACE_ENABLE_XXX'
   *   will be used to uncheck specific option (env value will be 'AND'ed)
   */
  if (mluop::getBoolEnvVar(CFG_ENUM_TO_STR(MLUOP_TRACE_ENABLE), false)) {
    trace_bit_config |= TRACE_MASK;
    if (!mluop::getBoolEnvVar(CFG_ENUM_TO_STR(MLUOP_TRACE_ENABLE_API), true)) {
      trace_bit_config &= (~TRACE_API);
    }
    if (!mluop::getBoolEnvVar(CFG_ENUM_TO_STR(MLUOP_TRACE_ENABLE_KERNEL),
                              true)) {
      trace_bit_config &= (~TRACE_KERNEL);
    }
  } else {
    if (mluop::getBoolEnvVar(CFG_ENUM_TO_STR(MLUOP_TRACE_ENABLE_API), false)) {
      trace_bit_config |= TRACE_API;
    }
    if (mluop::getBoolEnvVar(CFG_ENUM_TO_STR(MLUOP_TRACE_ENABLE_KERNEL),
                             false)) {
      trace_bit_config |= TRACE_KERNEL;
    }
  }
  if (trace_bit_config & TRACE_API) {
    Config::set_event<ConfigEnvType::MLUOP_EVENT_ENABLE_API>(true);
  }
  if (trace_bit_config & TRACE_KERNEL) {
    Config::set_event<ConfigEnvType::MLUOP_EVENT_ENABLE_KERNEL>(true);
  }
  return trace_bit_config & TRACE_MASK;
}

static bool filterApiNameComputeOnly(const char *name) {
  if (std::strstr(name, "Descriptor")) return false;
  if (std::strstr(name, "Workspace")) return false;
  if (std::strstr(name, "Create")) return false;
  if (std::strstr(name, "Destroy")) return false;
  if (std::strstr(name, "Get")) return false;
  if (std::strstr(name, "Set")) return false;
  return true;
}

static void traceKernel(const void *param, void *);

static void traceApi(const int *api_idx, void *);

namespace {

class mluOpTrace {
 private:
  int mkdirIfNotExist(const char *pathname) {
    struct stat dir_stat = {};
    if (stat(pathname, &dir_stat) != 0) {
      if (mkdir(pathname, 0777) != 0) {
        return errno;
      }
      return 0;
    } else if (!S_ISDIR(dir_stat.st_mode)) {
      return ENOTDIR;
    }
    return 0;
  }

  int mkdirRecursive(const char *pathname) {
    // let caller ensure pathname is not null
    const char path_token = '/';
    size_t pos = 0;
    const std::string pathname_view(pathname);
    while (pos < pathname_view.size()) {
      auto find_path_token = pathname_view.find(path_token, pos);
      if (find_path_token == std::string::npos) {
        return mkdirIfNotExist(pathname_view.c_str());
      }
      int ret =
          mkdirIfNotExist(pathname_view.substr(0, find_path_token + 1).c_str());
      if (ret) return ret;
      pos = find_path_token + 1;
    }
    return 0;
  }

  void serializeLine(std::ofstream &case_file, int, const std::string &s) {
    case_file << s << "\n";
  }

  void serializeLine(std::ofstream &case_file, int idx,
                     const std::atomic_int &hit) {
    if (mluOpTrace::flag_dump_api()) {
      if (hit.load()) {
        LOG(INFO) << "[MLUOP_API_CNT] " << mluOpInternalGetApiNameById(idx)
                  << ",\t" << hit.load();
      }
    }
    case_file << mluOpInternalGetApiNameById(idx) << ",\t" << hit.load()
              << "\n";
  }

  template <int policy, class Iterable>
  void dumpToFile(const std::string &filename, Iterable &data) {
    std::string filepath = raw_data_dir_ + "/" + filename;
    std::ofstream case_file;
    if (!case_file.is_open()) {
      case_file.open(filepath.c_str(), std::ios::app);
      if (case_file) {
        int line_cnt = 0;
        for (const auto &s : data) {
          serializeLine(case_file, line_cnt++, s);
        }
      } else {
        LOG(ERROR) << __func__ << ": failed to write file: " << filename
                   << " !";
      }
      case_file.close();
    }
  }

  void dumpTraceData() {
    if (mkdirRecursive(raw_data_dir_.c_str()) != 0) {
      LOG(ERROR) << __func__ << ": failed to create folder: " << raw_data_dir_
                 << " ! (" << errno << ": " << strerror(errno) << ")"
                 << std::endl;
      return;
    }
    if (getInstance().trace_api_enabled) {
      getInstance().dumpToFile<TRACE_API>(api_filename_, api_counter_);
    }
    if (getInstance().trace_kernel_enabled) {
      getInstance().dumpToFile<TRACE_KERNEL>(kernel_filename_, kernel_list_);
    }
  }

  static std::string stripKernelNameParam(const std::string &name) {
    size_t pos = name.find(">(");
    if (pos == std::string::npos) {
      // normal kernel name, find param location '('
      pos = name.find("(");
    } else {
      // templated kernel name, remove string after xxxx<...>
      pos++;
    }
    return name.substr(0, pos);
  }

 public:
  static mluOpTrace &getInstance() {
    static mluOpTrace mluop_trace;
    return mluop_trace;
  }

  //  static void addApi(const std::string &api) {
  //    const std::lock_guard<std::mutex> lock(getInstance().mtx_trace_);
  //    getInstance().api_list_.insert(api);
  //  }

  static void addApi(int api_idx) {
    getInstance().api_counter_[api_idx]++;
    if (getInstance().dump_api_count_) {
      const char *fname = mluOpInternalGetApiNameById(api_idx);
      if (filterApiNameComputeOnly(fname)) {
        LOG(CNPAPI) << "[MLUOP_API_CNT] logged api " << fname
                    << " (hit: " << getInstance().api_counter_[api_idx] << ")";
      }
    }
  }

  static void addKernel(const std::string &kernel) {
    const std::lock_guard<std::mutex> lock(getInstance().mtx_trace_);
    std::string name = stripKernelNameParam(kernel);
    getInstance().kernel_list_.insert(name);
  }

  ~mluOpTrace() {
    dumpTraceData();
    mluOpInternalUnsubscribe(kernel_ctx_);
    mluOpInternalUnsubscribe(api_ctx_);
  }

  void subscribeTraceKernel() {
    mluOpInternalSubscribe(MLUOP_EVENT_CNRT_INVOKE_KERNEL,
                           (mluOpInternalHandler_t)traceKernel, nullptr,
                           &kernel_ctx_);
  }

  void subscribeTraceApi() {
    mluOpInternalSubscribe(MLUOP_EVENT_MLUOP_API,
                           (mluOpInternalHandler_t)traceApi, nullptr,
                           &api_ctx_);
  }

  static int enable(uint32_t config) {
    if (mluop::getBoolEnvVar(CFG_ENUM_TO_STR(MLUOP_DUMP_API_COUNT), false)) {
      // if (flag_dump_api()) {
      // FIXME this is workaround to trigger cnlog init, and I could not init
      // mluOpTrace before cnlog
      LOG(INFO) << "MLUOP Trace enabled";
    } else {
      VLOG(7) << "MLUOP Trace enabled";
    }
    if (config) {
      getInstance().subscribeTraceKernel();
      getInstance().subscribeTraceApi();
    }
    if (config & TRACE_KERNEL) {
      getInstance().trace_kernel_enabled = true;
    }
    if (config & TRACE_API) {
      getInstance().trace_api_enabled = true;
    }
    return 0;
  }

  std::string getRawDataDirName() {
    auto const now = std::chrono::system_clock::now();
    std::time_t newt = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << TRACE_RAW_DATA_DIR << "/trace_data_"
        << std::put_time(std::localtime(&newt), "%m%d_%H%M%S") << "_"
        << getpid();
    return oss.str();
  }

  static inline bool flag_dump_api() { return getInstance().dump_api_count_; }

 private:
  mluOpTrace()
      : api_counter_(
            std::vector<std::atomic_int>(mluOpInternalGetApiNameNumber())) {
#if DEBUG
    printf("mluOpTrace singleten init\n");
#endif
  }
  const std::string raw_data_dir_ = getRawDataDirName();
  const std::string api_filename_ = API_FILE_NAME;
  const std::string kernel_filename_ = KERNEL_FILE_NAME;
  std::vector<std::atomic_int> api_counter_;
  std::atomic_bool dump_api_count_{
      mluop::getBoolEnvVar(CFG_ENUM_TO_STR(MLUOP_DUMP_API_COUNT), false)};
  std::set<std::string> kernel_list_;
  mluOpSubscriber_t kernel_ctx_;
  mluOpSubscriber_t api_ctx_;
  std::mutex mtx_trace_;
  bool trace_api_enabled = false;
  bool trace_kernel_enabled = false;
};

}  // namespace

static void traceKernel(const void *param, void *) {
  const void *kernel =
      static_cast<const struct mluOpEventParamCnrtInvokeKernel *>(param)
          ->kernel;
  const char *name = nullptr;
  mluOpInternalGetKernelName(kernel, &name, nullptr);
  mluOpTrace::addKernel(name);
}

static void traceApi(const int *api_idx, void *) {
  mluOpTrace::addApi(*api_idx);
}

// For debug purpose
#if 0
static void __attribute__((constructor)) _my_init() {
  printf("====== mluop_trace constructor\n");
  mluOpTrace::enable(load_config_from_env_mluop_trace());
}
#endif
static int __attribute__((unused)) enable_flag =
    mluOpTrace::enable(load_config_from_env_mluop_trace());

static void __attribute__((destructor(105))) mluOpTraceLibDestructor() {
#if DEBUG
  std::cout << "\033[1;33m mluop_trace fini \033[0m" << std::endl;
#endif
}
