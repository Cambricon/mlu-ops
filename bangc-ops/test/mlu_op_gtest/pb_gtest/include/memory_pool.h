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
#ifndef TEST_MLU_OP_GTEST_INCLUDE_MEMORY_POOL_H_
#define TEST_MLU_OP_GTEST_INCLUDE_MEMORY_POOL_H_

#include <iostream>
#include <set>
#include <memory>
#include <string>
#include <vector>
#include <list>
#include <utility>
#include <unordered_set>
#include <mutex>               //NOLINT
#include <condition_variable>  //NOLINT
#include "gtest/gtest.h"
#include "cnrt.h"
#include "tools.h"

namespace mluoptest {

using ADDR_BYTES_PAIR =
    std::pair<size_t, size_t>;  // mini_block_pair<addr, bytes>

}  // namespace mluoptest

namespace std {

template <>
class hash<std::list<mluoptest::ADDR_BYTES_PAIR>> {
 public:
  size_t operator()(
      std::list<mluoptest::ADDR_BYTES_PAIR> &mini_block_list) const {
    std::string str = "";
    for (auto &addr_size : mini_block_list) {
      str += (std::to_string(addr_size.first) + ",");
    }
    // undefined behavior if str is empty
    if (!str.empty()) {
      str.pop_back();
    }
    return std::hash<std::string>{}(str);
  }
};

}  // namespace std

namespace mluoptest {

class RandomSpaceWithinChunk {
 public:
  RandomSpaceWithinChunk(size_t total_size, void *ptr)
      : total_size(total_size),
        head_addr(*(reinterpret_cast<size_t *>(&ptr))),
        random_number(0, total_size - 1) {
    GTEST_WARNING(
        total_size >= 1,
        "Memory blocks in memory pool should have size no smaller than 1.");
  }
  std::pair<bool, size_t> allocate(size_t bytes_needed);
  void bookKeep() { allocation_hash_table.insert(curr_allocations_hash); }
  void clearOneIter();
  void clearBookKeep() { allocation_hash_table.clear(); }
  bool gotUniqueAllocations();
  size_t getCurrAllocationsHash() {
    return std::hash<std::list<ADDR_BYTES_PAIR>>{}(mini_block_list);
  }

 private:
  const size_t head_addr;
  const size_t total_size;
  std::list<ADDR_BYTES_PAIR> mini_block_list;  // each pair<addr, bytes>
  std::unordered_set<size_t> allocation_hash_table;
  RandomUniformNumber random_number;
  std::pair<size_t, std::list<ADDR_BYTES_PAIR>::iterator> determineAddrLocation(
      size_t rand_addr_start);
  size_t curr_allocations_hash;
};

class MemoryPool {
 public:
  struct Chunk {
    Chunk(size_t as, size_t rs, void *p)
        : allocated_size(as),
          requested_size(rs),
          ptr(p),
          random_space(allocated_size, p) {}
    uint64_t allocated_size = 0;
    uint64_t requested_size = 0;
    void *ptr = nullptr;
    RandomSpaceWithinChunk random_space;
  };

  MemoryPool() : ctx_(std::make_shared<Context>()) {}
  virtual ~MemoryPool() { destroy(); }

  virtual void *allocate(size_t num_bytes, const std::string &name = "") {
    return nullptr;
  }
  virtual void deallocate(void *ptr) {}
  virtual void destroy() {}
  virtual void clear() {}  // free obj of 1 thread.
  void clearRandomSpaceBigChunk();
  void bookKeepRandomSpaceBigChunk();
  std::pair<bool, void *> getRandomSpaceBigChunk(size_t size_needed);
  void clearBookKeepRandomSpaceBigChunk();
  bool gotUniqueRandomSpaceAllocations();

 protected:
  struct Context {
    std::list<Chunk> chunks;
    uint64_t total_allocated_size = 0;
  };
  std::shared_ptr<Context> ctx_ = nullptr;
  std::list<MemoryPool::Chunk>::iterator getOnlyOneBigChunk() const;
};

class CPUMemoryPool : public MemoryPool {
 public:
  ~CPUMemoryPool() { destroy(); }
  void *allocate(size_t num_bytes, const std::string &name = "");
  void deallocate(void *ptr);
  void destroy();
  void clear() {}  // free all obj of 1 thread.
};

class MLUMemoryPool : public MemoryPool {
 public:
  ~MLUMemoryPool() { destroy(); }
  void *allocate(size_t num_bytes, const std::string &name = "");
  void deallocate(void *ptr);
  void destroy();
  void clear() {}
};

}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_INCLUDE_MEMORY_POOL_H_
