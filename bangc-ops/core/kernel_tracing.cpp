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

#include <cxxabi.h>
#include <dlfcn.h>
#include <pthread.h>  // If we support c++17, we could use std::shared_mutex instead
#include <stdio.h>
#include <stdlib.h>

#include <tuple>
#include <vector>
#include <map>
#include <mutex>  // NOLINT

#include "config_env.h"
#include "cnrt.h"
#include "logging.h"
#include "macros.h"
#include "mlu_op_internal_api.h"
#include "subscriber.hpp"
#include "tool.h"

#if MLUOP_TRACE_WITH_DLOPEN
#define WRAP_FUNC(func) func
#else
#define WRAP_FUNC(func) \
  MLUOP_ATTRIBUTE_WEAK MLUOP_ATTRIBUTE_VISIBILITY_HIDDEN __wrap_##func
#endif

#ifdef DBG_LOG
#undef DBG_LOG
#endif
#if MLUOP_TRACE_WITH_DLOPEN
#define DBG_LOG VLOG(0) << "[KERNEL_HOOK_DLOPEN] "
#else
#define DBG_LOG VLOG(8) << "[KERNEL_HOOK] "
#endif

#if MLUOP_TRACE_WITH_DLOPEN
#include <sstream>

#include <errno.h>        // NOLINT
#include <dlfcn.h>        // NOLINT
#include <sys/types.h>    // NOLINT
#include <sys/syscall.h>  // NOLINT
#include <unistd.h>       // NOLINT
#define gettid() ((pid_t)syscall(SYS_gettid))
#define DLOPEN_TRACE_LOG(msg)                                     \
  do {                                                            \
    std::ostringstream oss;                                       \
    oss << "[KERNEL_HOOK_DLOPEN] sid: " << getsid(0)              \
        << ", pgid: " << getpgid(0) << ", ppid: " << getppid()    \
        << ", pid: " << getpid() << ", tid: " << gettid() << ", " \
        << program_invocation_name << ": " << msg << ";";         \
    fprintf(stderr, "%s\n", oss.str().c_str());                   \
  } while (0)
#else
#define DLOPEN_TRACE_LOG(...)
#endif

using mluop::cfg::Config;
using mluop::cfg::ConfigEnvType;

inline static bool load_config_from_env_kernel_tracing() {
  // NOTE should be true by default
  //      only set false for special debug purpose, that will disable tracing /
  //      crop_tool completely
  static bool flag =
      mluop::getBoolEnvVar(CFG_ENUM_TO_STR(MLUOP_DEBUG_KERNEL_TRACING), true);
  return flag;
}

namespace {

class kernelMapping {
 public:
  static kernelMapping &instance() {
    static kernelMapping kernel_mapping;
    return kernel_mapping;
  }
  static void addKernel(const void *key, const char *symbol) {
    WriteLock lock(instance().rwlock_);
    instance().kernel_mapping_raw_[key] = symbol;
    int status = 0;
    char *name = abi::__cxa_demangle(symbol, NULL, NULL, &status);
    if (status != 0) {
      LOG(ERROR) << "demangle kernel symbol failed for " << symbol
                 << ", status is " << status;
      instance().kernel_mapping_[key] = symbol;
    } else {
      DBG_LOG << " add device kernel function: " << name;
      instance().kernel_mapping_[key] = name;
      free(name);
    }
  }
  static inline const char *getKernelName(const void *key) {
    ReadLock lock(instance().rwlock_);
    return instance().kernel_mapping_[key].c_str();
  }
  ~kernelMapping() { pthread_rwlock_destroy(&rwlock_); }

 private:
  kernelMapping() = default;
  std::map<const void *, std::string> kernel_mapping_raw_;
  std::map<const void *, std::string> kernel_mapping_;
  pthread_rwlock_t rwlock_ = PTHREAD_RWLOCK_INITIALIZER;
};
}  // namespace

extern "C" {
MLUOP_WIN_API mluOpStatus_t mluOpInternalGetKernelName(const void *kernel,
                                                       const char **name,
                                                       int *) {
  *name = kernelMapping::getKernelName(kernel);
  return MLUOP_STATUS_SUCCESS;
}
}

static void addKernel(const mluop::pubsub::ParamBangRegisterFunction *param,
                      void *) {
  const char *symbol = param->deviceName;
  // TODO(NONE): if we are under LD_PRELOAD, we could consider calling addKernel
  // inside mluOp.so
  kernelMapping::addKernel(param->hostFunc, symbol);
}

static int initTracing(bool enable) {
#if MLUOP_TRACE_WITH_DLOPEN
  DLOPEN_TRACE_LOG("initTracing");
#else
  DBG_LOG << "initTracing " << (int)enable;
#endif
  Config::set_event<ConfigEnvType::MLUOP_EVENT_ENABLE_KERNEL>(enable);
  if (!enable) return -1;
  // TODO(NONE): call once
  auto idx = mluop::pubsub::Publisher::subscribe(
      mluop::pubsub::EventType::BANG_REGISTER_FUNCTION,
      (void (*)(const void *, void *))addKernel, nullptr);
  mluop::pubsub::Publisher::save_internal_subscriber(
      mluop::pubsub::EventType::BANG_REGISTER_FUNCTION, idx);
  return 0;
}

// just a demo method, to log kernel invoked, could be func for kernel tracing
// crop
static void logCnrtInvokeKernel(
    const struct mluOpEventParamCnrtInvokeKernel *param, void *) {
  const char *name = kernelMapping::getKernelName(param->kernel);
  DBG_LOG << "invoking " << name;
}

static void __attribute__((constructor)) _my_init() {
  DBG_LOG << __FILE__ << " ctor";
  if (VLOG_IS_ON(7)) {
    auto idx = mluop::pubsub::Publisher::subscribe(
        mluop::pubsub::EventType::CNRT_INVOKE_KERNEL,
        (void (*)(const void *, void *))logCnrtInvokeKernel, nullptr);
    mluop::pubsub::Publisher::save_internal_subscriber(
        mluop::pubsub::EventType::CNRT_INVOKE_KERNEL, idx);
  }
}

#if MLUOP_TRACE_WITH_DLOPEN
// workaround for linaro gcc 6 which does not support `constexpr if` and type
// trait _v suffix
#if __cpp_if_constexpr >= 201606
#define IMPL_HOOK_WITH_IF_CONSTEXPR 1
#else
#define IMPL_HOOK_WITH_IF_CONSTEXPR 0
#endif

namespace {
#if IMPL_HOOK_WITH_IF_CONSTEXPR == 0
template <typename RetType>
RetType on_failed() {
  return (RetType)0;
}
template <>
void on_failed() {}
template <>
cnrtRet_t on_failed() {
  return cnrtErrorInvalidSymbol;
}
#endif

#ifdef DEF_LIBCNRT_WRAPPER
#undef DEF_LIBCNRT_WRAPPER
#endif
#define DEF_LIBCNRT_WRAPPER(RetType, Func)                \
  template <typename... ArgType>                          \
  static RetType Func(ArgType... args) {                  \
    return dispatch<RetType, ArgType...>(#Func, args...); \
  }

class libcnrt {
 public:
  DEF_LIBCNRT_WRAPPER(void, __bangRegisterFunction);
  DEF_LIBCNRT_WRAPPER(cnrtRet_t, cnrtInvokeKernel);

  template <typename RetType, typename... ArgType>
  static RetType dispatch(const char fname[], ArgType... args) {
    static auto fptr = getInstance().getFuncImpl<RetType, ArgType...>(fname);
    return fptr(args...);
  }

  template <typename RetType, typename... ArgType>
  static RetType dispatch_failed(ArgType...) {
#if IMPL_HOOK_WITH_IF_CONSTEXPR
    if constexpr (std::is_same_v<RetType, void>) {
      return;
    } else if constexpr (std::is_same_v<RetType, cnrtRet_t>) {
      return cnrtErrorInvalidSymbol;
    }
    return (RetType)0;
#else
    return on_failed<RetType>();
#endif
  }
  const std::string name() const {
    // XXX just a backdoor, to dlopen libcnrt.so with another name
    const char *cnrt_name = std::getenv("LIBCNRT_NAME");
    if (cnrt_name != nullptr && strlen(cnrt_name)) {
      return cnrt_name;
    }
    return "libcnrt.so";
  }
  static libcnrt &getInstance() {
    static libcnrt lib;
    return lib;
  }
  static void *getHandle() {
    static void *handle = getInstance().getHandleImpl();
    return handle;
  }

 private:
  template <typename RetType, typename... ArgType>
  using F = RetType (*)(ArgType...);

  template <typename RetType, typename... ArgType>
  F<RetType, ArgType...> getFuncImpl(const char fname[]) {
    DBG_LOG << "dlsym " << fname;
    void *symbol = dlsym(getHandle(), fname);
    if (symbol == nullptr) {
      LOG(ERROR) << "dlsym " << fname << " failed, reason: " << dlerror();
      return dispatch_failed<RetType, ArgType...>;
    }
    return (F<RetType, ArgType...>)symbol;
  }

  void *getHandleImpl() {
    void *handle = dlopen(name().c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
      std::ostringstream err;
      err << "dlopen " << name() << " failed, reason: " << dlerror();
      LOG(ERROR) << err.str();
    }
    return handle;
  }
  libcnrt() = default;
};  // class libcnrt
}  // namespace
#endif  // MLUOP_TRACE_WITH_DLOPEN

extern "C" {

#if MLUOP_TRACE_WITH_DLOPEN
static void __real___bangRegisterFunction(void **module, const char *hostFunc,
                                          char *deviceFunc,
                                          const char *deviceName,
                                          int unionLimit, cnrtDim3_t *taskId,
                                          cnrtDim3_t *taskDim, int *wSize) {
  return libcnrt::__bangRegisterFunction(module, hostFunc, deviceFunc,
                                         deviceName, unionLimit, taskId,
                                         taskDim, wSize);
}
static cnrtRet_t __real_cnrtInvokeKernel(const void *kernel, cnrtDim3_t dim,
                                         cnrtFunctionType_t ktype, void **args,
                                         size_t reserved, cnrtQueue_t queue) {
  return libcnrt::cnrtInvokeKernel(kernel, dim, ktype, args, reserved, queue);
}
#else  // MLUOP_TRACE_WITH_DLOPEN
#if !defined(MLUOP_EXPERIMENTAL_TRACE)
__attribute__((weak))
#endif
extern void
__real___bangRegisterFunction(void **module, const char *hostFunc,
                              char *deviceFunc, const char *deviceName,
                              int unionLimit, cnrtDim3_t *taskId,
                              cnrtDim3_t *taskDim, int *wSize);

#if !defined(MLUOP_EXPERIMENTAL_TRACE)
__attribute__((weak))
#endif
extern cnrtRet_t
__real_cnrtInvokeKernel(const void *kernel, cnrtDim3_t dim,
                        cnrtFunctionType_t ktype, void **args, size_t reserved,
                        cnrtQueue_t queue);
#endif  // MLUOP_TRACE_WITH_DLOPEN

void WRAP_FUNC(__bangRegisterFunction)(void **module, const char *hostFunc,
                                       char *deviceFunc, const char *deviceName,
                                       int unionLimit, cnrtDim3_t *taskId,
                                       cnrtDim3_t *taskDim, int *wSize) {
#if MLUOP_TRACE_WITH_DLOPEN
  DBG_LOG << __func__ << ": " << deviceName;
#endif
  // this function is invoked before global static data initialization and .ctor
  static bool flag_hook = load_config_from_env_kernel_tracing();
  static int __attribute__((unused)) init_tracing = initTracing(flag_hook);
  if (flag_hook) {
    mluop::pubsub::ParamBangRegisterFunction params{module, hostFunc,
                                                    deviceName, wSize};
    mluop::pubsub::Publisher::publish(
        mluop::pubsub::EventType::BANG_REGISTER_FUNCTION, &params);
  }
  return __real___bangRegisterFunction(module, hostFunc, deviceFunc, deviceName,
                                       unionLimit, taskId, taskDim, wSize);
}

// could be used together with '-Wl,--wrap' linker flag, could override
// cnrtInvokeKernel
cnrtRet_t WRAP_FUNC(cnrtInvokeKernel)(const void *kernel, cnrtDim3_t dim,
                                      cnrtFunctionType_t ktype, void **args,
                                      size_t reserved, cnrtQueue_t queue) {
  // TODO(NONE): add flag to disable event publish (like cnpapi's enable
  // callback)
#if MLUOP_TRACE_WITH_DLOPEN
  DBG_LOG << __func__ << ": " << kernelMapping::getKernelName(kernel);
#endif
  if (Config::get_event<ConfigEnvType::MLUOP_EVENT_ENABLE_KERNEL>()) {
    mluOpEventParamCnrtInvokeKernel params{kernel, dim, ktype, args};
    mluop::pubsub::Publisher::publish(
        mluop::pubsub::EventType::CNRT_INVOKE_KERNEL, &params);
  }
  return __real_cnrtInvokeKernel(kernel, dim, ktype, args, reserved, queue);
}
}
