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
#include <string>
#include <functional>
#include "memory_pool.h"
#include "variable.h"

namespace mluoptest {

// return the proper starting address and the mini block that's just behind it
std::pair<size_t, std::list<ADDR_BYTES_PAIR>::iterator>
RandomSpaceWithinChunk::determineAddrLocation(size_t rand_addr_start) {
  // the very first one
  if (mini_block_list.empty()) {
    return std::make_pair(rand_addr_start, mini_block_list.end());
  }

  // at least one mini block already
  auto it = mini_block_list.begin();
  if (rand_addr_start < it->first) {  // before the first block
    return std::make_pair(rand_addr_start, it);
  }
  // within the first block
  if (it->first <= rand_addr_start &&
      rand_addr_start < it->first + it->second) {
    rand_addr_start = it->first + it->second;
  }

  size_t addr_prev_end = it->first + it->second;
  ++it;
  // at least 2 mini blocks exist
  for (; it != mini_block_list.end(); ++it) {
    // it's inside an allocated mini block
    if (it->first <= rand_addr_start &&
        rand_addr_start < (it->first + it->second)) {
      rand_addr_start = (it->first + it->second);
      // two mini blocks might be adjacent
      while (true) {
        auto temp_it = it;
        if ((++temp_it) != mini_block_list.end()) {
          if (rand_addr_start < (++it)->first) {  // it is at the next block
            break;
          } else {
            rand_addr_start = (it->first + it->second);
          }
        } else {
          ++it;
          break;
        }
      }
      break;
    }
    // outside the mini blocks, but where
    if (addr_prev_end <= rand_addr_start && rand_addr_start < it->first) {
      break;
    } else {
      addr_prev_end = it->first + it->second;
    }
  }
  return std::make_pair(rand_addr_start, it);
}

std::pair<bool, size_t> RandomSpaceWithinChunk::allocate(size_t bytes_needed) {
  size_t rand_addr_start = random_number();
  size_t init_num = rand_addr_start;
  if (rand_addr_start >= total_size) {
    printf("rand_addr_start %lu\n", rand_addr_start);
    GTEST_CHECK(false);
  }
  bool first_round = true;
  size_t space_len_traversed = 0;
  auto block_next = mini_block_list.begin();

  auto padUp = [](size_t init_val, size_t align_val) -> size_t {
    return ((init_val + align_val - 1) / align_val) * align_val;
  };
  // the allocated bytes should be aligned
  // TODO(None): delete
  size_t bytes_needed_del = bytes_needed;
  bytes_needed = padUp(bytes_needed, getSizeAlign(bytes_needed));

  while (first_round) {
    // the absolute address should be aligned accordingly
    rand_addr_start =
        padUp(head_addr + rand_addr_start, getAddrAlign(bytes_needed)) -
        head_addr;
    while (true) {
      rand_addr_start =
          (rand_addr_start >= total_size) ? total_size : rand_addr_start;
      std::tie(rand_addr_start, block_next) =
          determineAddrLocation(rand_addr_start);
      // the new address should be aligned, otherwise do another round
      size_t new_start_aligned =
          padUp(head_addr + rand_addr_start, getAddrAlign(bytes_needed)) -
          head_addr;
      if (rand_addr_start == new_start_aligned ||
          rand_addr_start == total_size) {
        break;
      } else {
        rand_addr_start = new_start_aligned;
      }
    }

    size_t rand_addr_end = rand_addr_start + bytes_needed;
    size_t rand_addr_end_max =
        (block_next != mini_block_list.end()) ? block_next->first : total_size;
    if (rand_addr_end <= rand_addr_end_max) {
      mini_block_list.insert(block_next,
                             std::make_pair(rand_addr_start, bytes_needed));
      break;
    } else {
      size_t rand_addr_start_next;
      if (block_next != mini_block_list.end()) {
        rand_addr_start_next = block_next->first + block_next->second;
        space_len_traversed += (rand_addr_start_next - rand_addr_start);
      } else {
        if (!mini_block_list.empty()) {
          auto first_block = mini_block_list.begin();
          rand_addr_start_next =
              (first_block->first > 0)
                  ? 0
                  : (first_block->first + first_block->second);
        } else {
          rand_addr_start_next = 0;
        }
        space_len_traversed +=
            ((total_size - rand_addr_start) + (rand_addr_start_next - 0));
      }
      rand_addr_start = rand_addr_start_next;
    }
    first_round = (space_len_traversed < total_size) ? true : false;
  }

  if (unlikely(!first_round)) {
    printf("total_size %lu: ", total_size);
    for (auto it = mini_block_list.begin(); it != mini_block_list.end(); ++it) {
      printf("%lu(%lu), ", it->first, it->second);
    }
    printf(
        "failed allocation %lu (aligned %lu), init_random_num %lu, "
        "space_len_traversed %lu\n",
        bytes_needed_del, bytes_needed, init_num, space_len_traversed);
  }

  return std::make_pair(first_round, rand_addr_start);
}

void RandomSpaceWithinChunk::clearOneIter() { mini_block_list.clear(); }

bool RandomSpaceWithinChunk::gotUniqueAllocations() {
  curr_allocations_hash = getCurrAllocationsHash();
  auto it = allocation_hash_table.find(curr_allocations_hash);
  return (it == allocation_hash_table.end()) ? true : false;
}

std::list<MemoryPool::Chunk>::iterator MemoryPool::getOnlyOneBigChunk() const {
  GTEST_CHECK(ctx_->chunks.size() == 1,
              "There should be only one chunk in the MLU memory pool.");
  return ctx_->chunks.begin();
}

bool MemoryPool::gotUniqueRandomSpaceAllocations() {
  auto big_chunk_itr = getOnlyOneBigChunk();
  return big_chunk_itr->random_space.gotUniqueAllocations();
}

void MemoryPool::clearBookKeepRandomSpaceBigChunk() {
  auto big_chunk_itr = getOnlyOneBigChunk();
  big_chunk_itr->random_space.clearBookKeep();
}

void MemoryPool::bookKeepRandomSpaceBigChunk() {
  // after all inputs and outputs tensor have got their spaces, ready for
  // bookkeeping
  auto big_chunk_itr = getOnlyOneBigChunk();
  big_chunk_itr->random_space.bookKeep();
}

void MemoryPool::clearRandomSpaceBigChunk() {
  auto big_chunk_itr = getOnlyOneBigChunk();
  big_chunk_itr->random_space.clearOneIter();
}

std::pair<bool, void *> MemoryPool::getRandomSpaceBigChunk(
    size_t bytes_needed) {
  auto big_chunk_itr = getOnlyOneBigChunk();
  bool found;
  size_t random_offset;
  std::tie(found, random_offset) =
      big_chunk_itr->random_space.allocate(bytes_needed);
  return std::make_pair(found, (char *)big_chunk_itr->ptr + random_offset);
}

void *CPUMemoryPool::allocate(size_t num_bytes, const std::string &name) {
  if (0 == num_bytes) {
    return nullptr;
  } else {
    void *ptr = malloc(num_bytes);
    ctx_->chunks.emplace_back(Chunk(num_bytes, num_bytes, ptr));
    ctx_->total_allocated_size += num_bytes;
    return ptr;
  }
}

void CPUMemoryPool::deallocate(void *ptr) {
  auto it = ctx_->chunks.begin();
  for (; it != ctx_->chunks.end();) {
    if ((*it).ptr == ptr) {
      free(ptr);
      ctx_->total_allocated_size -= (*it).allocated_size;
      it = ctx_->chunks.erase(it);
      return;
    } else {
      ++it;
    }
  }
}

void CPUMemoryPool::destroy() {
  for (auto it = ctx_->chunks.begin(); it != ctx_->chunks.end(); ++it) {
    if (it->ptr != nullptr) {
      free(it->ptr);
      it->ptr = nullptr;
      ctx_->total_allocated_size -= it->allocated_size;
    }
  }
  ctx_->chunks.clear();
}

void *MLUMemoryPool::allocate(size_t num_bytes, const std::string &name) {
  if (0 == num_bytes) {
    return nullptr;
  }

  void *ptr = nullptr;
  CNdev dev;
  int deviceSupportLinear;
  int is_linear = 0;
  GTEST_CHECK(CN_SUCCESS ==
              cnDeviceGetAttribute(&deviceSupportLinear,
                                   CN_DEVICE_ATTRIBUTE_LINEAR_MAPPING_SUPPORTED,
                                   dev));
  if (0 != deviceSupportLinear &&  // arch support linear
      (global_var.exclusive_ || global_var.run_on_jenkins_)) {  // exclusive
    int granularity = 0;
    unsigned long maxFreeSize = 0;  // NOLINT
    unsigned long temp = 0;  // NOLINT
#ifndef ROUND_DOWN
#define ROUND_DOWN(x, y) (((x) / (y)) * (y))
    cnDeviceGetAttribute(&granularity,
                         CN_DEVICE_ATTRIBUTE_LINEAR_RECOMMEND_GRANULARITY, dev);
    cnrtMemGetInfo(&temp, NULL);
    maxFreeSize = num_bytes;
    maxFreeSize = ROUND_DOWN(maxFreeSize, granularity);
#undef ROUND_DOWN
#endif
    while (maxFreeSize > 0) {
      auto result = cnrtMalloc(&ptr, maxFreeSize);
      if (cnrtSuccess == result) {
        cnGetMemAttribute((void *)&is_linear, CN_MEM_ATTRIBUTE_ISLINEAR,
                          (CNaddr)ptr);
        if (is_linear)
          break;
        else
          cnrtFree(ptr);
      }
      maxFreeSize -= granularity;
    }
    num_bytes = maxFreeSize;
  } else {
    GTEST_CHECK(cnrtSuccess == cnrtMalloc(&ptr, num_bytes));
  }

  ctx_->chunks.emplace_back(Chunk(num_bytes, num_bytes, ptr));
  ctx_->total_allocated_size += num_bytes;
  printLinearMemoryMsg(ptr, num_bytes);
  return ptr;
}

void MLUMemoryPool::deallocate(void *ptr) {
  for (auto it = ctx_->chunks.begin(); it != ctx_->chunks.end(); ++it) {
    if ((*it).ptr == ptr) {
      GTEST_CHECK(cnrtSuccess == cnrtFree(ptr));
      ctx_->total_allocated_size -= (*it).allocated_size;
      it = ctx_->chunks.erase(it);
      return;
    }
  }
}

void MLUMemoryPool::destroy() {
  for (auto it = ctx_->chunks.begin(); it != ctx_->chunks.end(); ++it) {
    if (it->ptr != nullptr) {
      GTEST_WARNING(cnrtSuccess == cnrtFree(it->ptr));
      it->ptr = nullptr;
      ctx_->total_allocated_size -= it->allocated_size;
    }
  }
  ctx_->chunks.clear();
}

}  // namespace mluoptest
