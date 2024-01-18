/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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

#include <stdint.h>

#include <functional>
#include <list>
#include <map>
#include <utility>
#include <tuple>
#include <unordered_map>
#include <memory>
#include <mutex>

#include <pthread.h>

#include "cnrt.h"

#include "core/macros.h"

// c++17 shared_mutex/unique_mutex alternative, may need to move into separate
// file
template <class LockType, bool exclusive>
inline void Lock(LockType *lck_);

template <class LockType, bool exclusive>
inline void UnLock(LockType *lck_);

template <>
inline void Lock<pthread_rwlock_t, true>(pthread_rwlock_t *lck_) {
  pthread_rwlock_wrlock(lck_);
}

template <>
inline void Lock<pthread_rwlock_t, false>(pthread_rwlock_t *lck_) {
  pthread_rwlock_rdlock(lck_);
}

template <>
inline void UnLock<pthread_rwlock_t, true>(pthread_rwlock_t *lck_) {
  pthread_rwlock_unlock(lck_);
}

template <>
inline void UnLock<pthread_rwlock_t, false>(pthread_rwlock_t *lck_) {
  pthread_rwlock_unlock(lck_);
}

template <class LockType, bool exclusive>
class LockGuard {
 public:
  explicit LockGuard(LockType &lck) : lck_(&lck) {
    Lock<LockType, exclusive>(lck_);
  }
  ~LockGuard() { UnLock<LockType, exclusive>(lck_); }

 private:
  LockType *lck_;
};

using ReadLock = LockGuard<pthread_rwlock_t, false>;
using WriteLock = LockGuard<pthread_rwlock_t, true>;

namespace mluop {
namespace pubsub {

enum EventType : uint32_t {
  UNINITIALIZED = 0,
  BANG_REGISTER_FUNCTION = 0x1,
  CNRT_INVOKE_KERNEL = 0x2,
  MLUOP_API = 0x1000,  // for all mluOp api
  // MLUOP_API + offset is for specific mluOp api
  ALL = INT32_MAX,
};

struct ParamBangRegisterFunction {
  void **module;
  const char *hostFunc;
  const char *deviceName;
  int *wSize;
};

class Publisher {
 public:
  using EventHandler =
      std::pair<std::function<void(const void *, void *)>, void *>;
  static Publisher &instance() {
    static Publisher publisher;
    return publisher;
  }
  static void publish(EventType event, const void *params) {
    if (MLUOP_PREDICT_FALSE(delete_flag)) return;
    // TODO handle event type ALL
    const auto &handlers = instance().subscriber_manager_[event];
    ReadLock lock(instance().mtx_pubsub_);
    for (const auto &handler : handlers) {
      handler.second.first(params, handler.second.second);
      // handler.first(params, handler.second);
    }
  }
  static size_t subscribe(EventType event,
                          std::function<void(const void *, void *)> handler,
                          void *usr);
  static void unsubscribe(EventType event, size_t idx);
  static void save_internal_subscriber(EventType event, size_t idx);
  ~Publisher();

 private:
  explicit Publisher() = default;
  Publisher(const Publisher &) = delete;
  Publisher &operator=(const Publisher &) = delete;
  Publisher(Publisher &&) = delete;
  std::unordered_map<EventType, std::map<std::shared_ptr<char>, EventHandler>>
      subscriber_manager_{
          {EventType::BANG_REGISTER_FUNCTION, {}},
          {EventType::CNRT_INVOKE_KERNEL, {}},
          {EventType::MLUOP_API, {}},
      };
  std::map<size_t, std::shared_ptr<char>> ugly_key_store_;

  // TODO consider different lock for different event type
  pthread_rwlock_t mtx_pubsub_ = PTHREAD_RWLOCK_INITIALIZER;

  std::list<std::tuple<EventType, size_t>> internal_subscribers_;

  // XXX ugly workaround to avoid ASan's 'Use After Free' when mluOp is called
  // by `dlopen`
  static bool delete_flag;
};

}  // namespace pubsub
}  // namespace mluop
