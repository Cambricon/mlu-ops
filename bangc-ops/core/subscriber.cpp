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

#include "mlu_op_internal_api.h"

#include "macros.h"
#include "logging.h"
#include "subscriber.hpp"

namespace mluop {
namespace pubsub {

size_t Publisher::subscribe(EventType event,
                            std::function<void(const void *, void *)> handler,
                            void *usr) {
#if 0
  instance().subscriber_manager[event].emplace_back(handler, usr);
  return instance().subscriber_manager[event].size();
#endif
  std::shared_ptr<char> key(new char);
  WriteLock lock(instance().mtx_pubsub_);
  instance().subscriber_manager_[event][key] = {handler, usr};
  size_t key_idx = reinterpret_cast<size_t>(key.get());
  instance().ugly_key_store_[key_idx] = key;
  return key_idx;
}

void Publisher::unsubscribe(EventType event, size_t idx) {
  // TODO(NONE): return type should be status enum
  // TODO(NONE): check idx existence, check key existence
  WriteLock lock(instance().mtx_pubsub_);
  auto kv_key = instance().ugly_key_store_.find(idx);
  if (kv_key == instance().ugly_key_store_.end()) return;
  if (kv_key->second.use_count() > 0) {
    instance().subscriber_manager_[event].erase(kv_key->second);
    instance().ugly_key_store_.erase(idx);
  }
}

// save ::subscribe called internally (which has no corresponding ::unsubscribe)
void Publisher::save_internal_subscriber(EventType event, size_t idx) {
  static std::mutex mtx;
  std::lock_guard<std::mutex> lck(mtx);
  instance().internal_subscribers_.emplace_back(event, idx);
}

bool Publisher::delete_flag = false;

Publisher::~Publisher() {
  for (auto &sub : internal_subscribers_) {
    unsubscribe(std::get<0>(sub), std::get<1>(sub));
  }
  ReadLock lock(instance().mtx_pubsub_);
  if (instance().ugly_key_store_.size()) {
    LOG(WARNING) << "forgot unsubscribe mluOp event or unsubscribe will be "
                    "called after this destructor";
  }
  Publisher::delete_flag = true;
}

}  // namespace pubsub
}  // namespace mluop

MLUOP_WIN_API mluOpStatus_t mluOpInternalSubscribe(
    mluOpInternalEventType event_type, mluOpInternalHandler_t handler,
    void *usr, mluOpSubscriber_t *subscriber) {
  // TODO(NONE): param check event_type
  static_assert((uint32_t)mluop::pubsub::EventType::CNRT_INVOKE_KERNEL ==
                (uint32_t)MLUOP_EVENT_CNRT_INVOKE_KERNEL);
  static_assert((uint32_t)mluop::pubsub::EventType::MLUOP_API ==
                (uint32_t)MLUOP_EVENT_MLUOP_API);
  PARAM_CHECK("[mluOpInternalUnsubscribe]", subscriber != NULL);
  size_t idx_ = mluop::pubsub::Publisher::subscribe(
      (mluop::pubsub::EventType)event_type, handler, usr);
  *((size_t *)(subscriber->idx)) = idx_;
  subscriber->event_type = event_type;
  return MLUOP_STATUS_SUCCESS;
}

MLUOP_WIN_API mluOpStatus_t
mluOpInternalUnsubscribe(mluOpSubscriber_t subscriber) {
  size_t idx = *((size_t *)(subscriber.idx));
  mluop::pubsub::Publisher::unsubscribe(
      (mluop::pubsub::EventType)(subscriber.event_type), idx);
  return MLUOP_STATUS_SUCCESS;
}
