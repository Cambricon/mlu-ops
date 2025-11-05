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
#ifndef TEST_MLU_OP_GTEST_INCLUDE_THREAD_POOL_H_
#define TEST_MLU_OP_GTEST_INCLUDE_THREAD_POOL_H_

#include <atomic>
#include <condition_variable>  // NOLINT
#include <functional>
#include <future>  // NOLINT
#include <iostream>
#include <memory>
#include <mutex>  // NOLINT
#include <queue>
#include <thread>   // NOLINT
#include <utility>  // NOLINT
#include <vector>

#include "tools.h"

namespace mluoptest {

class ThreadPool {
 public:
  ThreadPool() = default;
  ThreadPool(ThreadPool&&) = default;
  explicit ThreadPool(size_t thread_num);
  ~ThreadPool();

  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    if (!ctx_) {
      throw std::runtime_error("ThreadPool not initialized.");
    }

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();

    {
      std::unique_lock<std::mutex> lock(ctx_->mtx);
      if (ctx_->is_shutdown)
        throw std::runtime_error("enqueue on stopped ThreadPool");
      ctx_->tasks.emplace([task]() { (*task)(); });
    }
    ctx_->cond.notify_one();
    return res;
  }

 private:
  struct Context {
    std::mutex mtx;
    std::condition_variable cond;
    bool is_shutdown = false;
    std::queue<std::function<void()>> tasks;
    std::vector<std::thread> workers;
  };
  std::shared_ptr<Context> ctx_ = nullptr;
};

}  // namespace mluoptest

#endif  // TEST_MLU_OP_GTEST_INCLUDE_THREAD_POOL_H_
