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

/*
 * NOTE: Different implementation for kernel tracing:
 *  - use cnpapi, which is simple to implement, but have could not be used
 * together with cnperf-cli
 *  - hook `cnrtInvokeKernel` and `__bangRegisterFunction`
 *
 * Will use hook instead of cnpapi
 */

#include <functional>
#include <memory>
#include <type_traits>
#include <string>
#include <sstream>
#include <unordered_map>

// #include "cn_api.h"
#include "cnrt.h"
#include "cnpapi.h"
#include "cnpapi_generated_cndrv_params.h"

namespace mluoptest {

class kernelTracingCtx;
static void pub_event_kernel_tracing(kernelTracingCtx *ctx, cnrtFunctionType_t,
                                     cnrtDim3_t, const char *name);

#define KERNEL_TRACING_DISABLED "disabled"
class kernelTracingCtx {
 public:
  friend void pub_event_kernel_tracing(kernelTracingCtx *, cnrtFunctionType_t,
                                       cnrtDim3_t, const char *);
  virtual ~kernelTracingCtx() {}
  virtual void init() = 0;
  virtual void release() = 0;
  virtual void enableKernelTracingImpl() = 0;  // enable tracing itself
  virtual void disableKernelTracingImpl() = 0;
  using kernel_traced_cb_t =
      std::function<void(cnrtFunctionType_t, cnrtDim3_t, const char *)>;
  inline void setCallbackKernelTraced(kernel_traced_cb_t &&cb) {
    kernel_traced_cb_ = cb;
  }
  virtual bool is_initialized() const = 0;
  static std::shared_ptr<kernelTracingCtx> instance(const char *type) {
    return ctxManager::get_ctx_manager().at(type);
  }
  // add implementation for kernel tracing
  template <class T>
  static bool register_method(const char *name) {
    static_assert(std::is_base_of<kernelTracingCtx, T>::value);
    ctxManager::get_ctx_manager()[name] = std::shared_ptr<T>(new T);
    return true;
  }
  static bool is_method_available(std::string &name) {
    if (ctxManager::get_ctx_manager().find(name) ==
        ctxManager::get_ctx_manager().end()) {
      return false;
    }
    return true;
  }
  static std::string getSupportedMethodNames() {
    std::ostringstream oss;
    for (const auto &item : ctxManager::get_ctx_manager()) {
      oss << item.first << ",";
    }
    return oss.str();
  }

 protected:
  void onKernelTraced(cnrtFunctionType_t ktype, cnrtDim3_t dim,
                      const char *name) {
    if (kernel_traced_cb_) {
      kernel_traced_cb_(ktype, dim, name);
    }
  }

 private:
  kernel_traced_cb_t kernel_traced_cb_ = nullptr;

  // save different policy for hooking kernel launch, and ensure initialization
  // order safe
  class ctxManager {
   public:
    static ctxManager &instance() {
      static ctxManager manager;
      return manager;
    }
    static std::unordered_map<std::string, std::shared_ptr<kernelTracingCtx> >
        &get_ctx_manager() {
      return instance().ctx_manager_;
    }

   private:
    ctxManager() = default;
    std::unordered_map<std::string, std::shared_ptr<kernelTracingCtx> >
        ctx_manager_{
            {KERNEL_TRACING_DISABLED, nullptr},
        };
  };
};

#define exportKernelTracingMethod(cls, alias)                \
  static __attribute__((used)) bool kernel_tracing_##alias = \
      kernelTracingCtx::register_method<cls>(#alias)

static void pub_event_kernel_tracing(kernelTracingCtx *ctx,
                                     cnrtFunctionType_t ktype, cnrtDim3_t dim,
                                     const char *name) {
  if (ctx == nullptr) return;
  ctx->onKernelTraced(ktype, dim, name);
}

}  // namespace mluoptest

extern "C" {
#if 0
CNresult __wrap_cnInvokeKernel(CNkernel hkernel, unsigned int dimx,
    unsigned int dimy, unsigned int dimz, KernelClass c, unsigned int reserve,
    CNqueue hqueue, void **kernelParams, void **extra);
#endif

// cnrtRet_t __wrap_cnrtInvokeKernel(const void *kernel, cnrtDim3_t dim,
//    cnrtFunctionType_t ktype, void **args,
//    size_t reserved, cnrtQueue_t queue);
}
