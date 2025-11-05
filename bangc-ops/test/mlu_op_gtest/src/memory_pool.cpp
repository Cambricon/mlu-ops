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
#include <string>
#include "memory_pool.h"

namespace mluoptest {

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
  } else {
    void *ptr = nullptr;
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&ptr, num_bytes));
    ctx_->chunks.emplace_back(Chunk(num_bytes, num_bytes, ptr));
    ctx_->total_allocated_size += num_bytes;
    return ptr;
  }
}

void MLUMemoryPool::deallocate(void *ptr) {
  auto it = ctx_->chunks.begin();
  for (; it != ctx_->chunks.end();) {
    if ((*it).ptr == ptr) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(ptr));
      ctx_->total_allocated_size -= (*it).allocated_size;
      it = ctx_->chunks.erase(it);
      return;
    } else {
      ++it;
    }
  }
}

void MLUMemoryPool::destroy() {
  for (auto it = ctx_->chunks.begin(); it != ctx_->chunks.end(); ++it) {
    if (it->ptr != nullptr) {
      auto res = cnrtFree(it->ptr);  // go on free ptr. don't throw exception.
      it->ptr = nullptr;
      ctx_->total_allocated_size -= it->allocated_size;
    }
  }
  ctx_->chunks.clear();
}

}  // namespace mluoptest
