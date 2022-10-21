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
#ifndef TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_MEMORY_POOL_H_
#define TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_MEMORY_POOL_H_

#include <iostream>
#include <set>
#include <memory>
#include <string>
#include <vector>
#include <list>
#include <mutex>               //NOLINT
#include <condition_variable>  //NOLINT
#include "cnrt.h"
#include "gtest/gtest.h"
#include "pb_test_tools.h"

namespace mluoptest {

class MemoryPool {
 public:
  struct Chunk {
    Chunk(size_t as, size_t rs, void *p)
        : allocated_size(as), requested_size(rs), ptr(p) {}
    uint64_t allocated_size = 0;
    uint64_t requested_size = 0;
    void *ptr = nullptr;
  };

  MemoryPool() { ctx_ = std::make_shared<Context>(); }
  virtual ~MemoryPool() { destroy(); }

  virtual void *allocate(size_t num_bytes, const std::string &name = "") {
    return nullptr;
  }
  virtual void deallocate(void *ptr) {}
  virtual void destroy() {}
  virtual void clear() {}  // free obj of 1 thread.

 protected:
  struct Context {
    std::list<Chunk> chunks;
    uint64_t total_allocated_size = 0;
  };
  std::shared_ptr<Context> ctx_ = nullptr;
};

class CPUMemoryPool : public MemoryPool {
 public:
  void *allocate(size_t num_bytes, const std::string &name = "");
  void deallocate(void *ptr);
  void destroy();
  void clear() {}  // free all obj of 1 thread.
};

class MLUMemoryPool : public MemoryPool {
 public:
  void *allocate(size_t num_bytes, const std::string &name = "");
  void deallocate(void *ptr);
  void destroy();
  void clear() {}
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_MEMORY_POOL_H_
