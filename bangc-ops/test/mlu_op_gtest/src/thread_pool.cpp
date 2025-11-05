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
#include <memory>
#include <utility>
#include "thread_pool.h"

namespace mluoptest {

ThreadPool::ThreadPool(size_t thread_num) {
  ctx_ = std::make_shared<Context>();

  auto work = [=] {
    auto ctx = ctx_;
    std::unique_lock<std::mutex> lk(ctx->mtx);
    for (;;) {
      if (!ctx->tasks.empty()) {
        auto task = std::move(ctx->tasks.front());
        ctx->tasks.pop();
        lk.unlock();
        task();
        lk.lock();
      } else if (ctx->is_shutdown) {
        break;
      } else {
        ctx->cond.wait(lk);
      }
    }
  };

  for (size_t i = 0; i < thread_num; ++i) {
    ctx_->workers.emplace_back(work);
  }
}

ThreadPool::~ThreadPool() {
  if (ctx_ == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> lk(ctx_->mtx);
    ctx_->is_shutdown = true;
  }
  ctx_->cond.notify_all();
  for (std::thread &worker : ctx_->workers) {
    worker.join();
  }
}

}  // namespace mluoptest
