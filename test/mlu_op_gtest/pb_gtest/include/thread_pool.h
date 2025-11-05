#ifndef TEST_MLU_OP_GTEST_INCLUDE_THREAD_POOL_H_
#define TEST_MLU_OP_GTEST_INCLUDE_THREAD_POOL_H_

#include <mutex>               // NOLINT
#include <condition_variable>  // NOLINT
#include <future>              // NOLINT
#include <thread>              // NOLINT
#include <utility>             // NOLINT
#include <functional>
#include <queue>
#include <memory>
#include <vector>
#include <iostream>
#include <atomic>
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
