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
#include <memory>
#include <cstdio>
#include <random>
#include <algorithm>
#include <chrono>    // NOLINT
#include "runtime.h"

#ifdef __AVX__
const int AVX_ALIGN = 32;
#endif

namespace mluoptest {

// CPURuntime part
CPURuntime::CPURuntime() {}

CPURuntime::~CPURuntime() {}

// all member variable are shared_ptr.
cnrtRet_t CPURuntime::destroy() { return CNRT_RET_SUCCESS; }

void *CPURuntime::allocate(void *ptr, std::string name) {
  if (ptr == NULL) {
    return NULL;  // can't free NULL, don't push NULL into vector.
  } else {
    memory_blocks_.push_back(
        std::make_shared<MemBlock<void *>>(ptr, free, name));
    return ptr;
  }
}

void *CPURuntime::allocate(size_t num_bytes, std::string name) {
  if (num_bytes == 0) {
    return NULL;
  }

#ifdef __AVX__
  void *ptr = _mm_malloc(num_bytes, AVX_ALIGN);  // avx need align to 32
#else
  void *ptr = malloc(num_bytes);
#endif

  if (ptr != NULL) {
#ifdef __AVX__
    memory_blocks_.push_back(
        std::make_shared<MemBlock<void *>>(ptr, _mm_free, name));
#else
    memory_blocks_.push_back(
        std::make_shared<MemBlock<void *>>(ptr, free, name));
#endif
    return ptr;
  } else {
    LOG(ERROR) << "CPURuntime: Failed to allocate " << num_bytes << " bytes.";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
    return NULL;
  }
}

// MLURuntime part
MLURuntime::MLURuntime() {
  check_enable_ = getEnv("MLUOP_GTEST_OVERWRITTEN_CHECK", true);
  if (true == check_enable_) {
    header_mask_ = std::shared_ptr<char>(new char[mask_bytes_],
                                         [](char *p) { delete[] p; });
    footer_mask_ = std::shared_ptr<char>(new char[mask_bytes_],
                                         [](char *p) { delete[] p; });
    rand_set_mask();

    header_check_ = std::shared_ptr<char>(new char[mask_bytes_],
                                          [](char *p) { delete[] p; });
    footer_check_ = std::shared_ptr<char>(new char[mask_bytes_],
                                          [](char *p) { delete[] p; });
  }
}

MLURuntime::~MLURuntime() {}

bool MLURuntime::checkAndFree(struct MemBlock &mem_block) {
  cnrtRet_t ret = CNRT_RET_SUCCESS;
  bool ok = true;
  char *header = mem_block.header;
  if (true == check_enable_) {
    reset_check();
    void *mlu_addr = (void *)(header + mask_bytes_);
    std::string name = mem_block.name;
    char *footer = header + mem_block.raw_bytes - mask_bytes_ -
                   mem_block.unalign_address_offset;
#ifdef GTEST_DEBUG_LOG
    VLOG(4) << "MLURuntime: checkAndFree get ptr " << (void *)(mlu_addr)
            << " for [" << name << "]";
    VLOG(4) << "MLURuntime: checkAndFree free [" << (void *)(mlu_addr) << ", "
            << (void *)(footer) << ")";
#endif
    ret = cnrtMemcpy((void *)header_check_.get(), header, mask_bytes_,
                     CNRT_MEM_TRANS_DIR_DEV2HOST);
    if (ret != CNRT_RET_SUCCESS) {
      ADD_FAILURE() << "MLURuntime: checkAndFree failed. Addr = "
                    << (void *)header;
      ok = false;
    }
    ret = cnrtMemcpy((void *)footer_check_.get(), footer, mask_bytes_,
                     CNRT_MEM_TRANS_DIR_DEV2HOST);
    if (ret != CNRT_RET_SUCCESS) {
      ADD_FAILURE() << "MLURuntime: checkAndFree failed. Addr = "
                    << (void *)footer;
      ok = false;
    }
#ifdef GTEST_DEBUG_LOG
    VLOG(4) << "MLURuntime: checkAndFree check " << (void *)header << " begin.";
#endif
    if (!check_byte((void *)header_check_.get(), (void *)header_mask_.get(),
                    mask_bytes_)) {
      ADD_FAILURE() << "MLURuntime: Addr " << mlu_addr << "(" << name
                    << ") has been overwritten.";
      ok = false;
    }
#ifdef GTEST_DEBUG_LOG
    VLOG(4) << "MLURuntime: checkAndFree check " << (void *)header << " end.";
#endif
#ifdef GTEST_DEBUG_LOG
    VLOG(4) << "MLURuntime: checkAndFree check " << (void *)footer << " begin.";
#endif
    if (!check_byte((void *)footer_check_.get(), (void *)footer_mask_.get(),
                    mask_bytes_)) {
      ADD_FAILURE() << "MLURuntime: Addr " << mlu_addr << "(" << name
                    << ") has been overwritten.";
      ok = false;
    }
#ifdef GTEST_DEBUG_LOG
    VLOG(4) << "MLURuntime: checkAndFree check " << (void *)footer << " end.";
#endif
  }  // endif (true == check_enable_)
  ret = cnrtFree(header - mem_block.unalign_address_offset);
  if (ret != CNRT_RET_SUCCESS) {
    ADD_FAILURE() << "MLURuntime: checkAndFree failed. Addr = "
                  << (void *)header;
    ok = false;
  }
  allocated_size -= mem_block.raw_bytes;
  return ok;
}

cnrtRet_t MLURuntime::destroy() {
  cnrtRet_t ret = CNRT_RET_SUCCESS;
  bool ok = true;
  for (auto mem_block : memory_blocks_) {
    ok = ok && (checkAndFree(mem_block));
  }
  if (!ok) {
    return CNRT_RET_ERR_INVALID;
  } else {
    return ret;
  }
}

static int getOffsetValue() {
  static bool init = false;
  static std::random_device rd;
  static std::mt19937 eng;
  static std::uniform_int_distribution<int> dist(1, 63);
  if (!init) {
    eng.seed(rd());
    init = true;
  }
  return dist(eng);
}

void *MLURuntime::allocate(size_t num_bytes, std::string name) {
#ifdef GTEST_DEBUG_LOG
  VLOG(4) << "MLURuntime: [allocate] malloc for [" << name << "] " << num_bytes
          << " bytes.";
#endif
  if (num_bytes == 0) {
    return NULL;
  }
  if (global_var.unaligned_mlu_address_random_ &&
      (global_var.unaligned_mlu_address_set_ > 0)) {
    LOG(ERROR) << "MLURuntime: Failed to allocate. "
               << "Please check the command or environment variable. "
               << "Create non-64-type aligned address can only by a may "
               << "that Fixed offset or random offset " << std::endl;
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
    return NULL;
  } else {
    unalign_address_ = global_var.unaligned_mlu_address_random_ ||
                       (global_var.unaligned_mlu_address_set_ > 0);
  }
  size_t unalign_address_offset = 0;
  if (unalign_address_) {
    unalign_address_offset = global_var.unaligned_mlu_address_set_ > 0
                                 ? global_var.unaligned_mlu_address_set_
                                 : getOffsetValue();
    VLOG(4) << "the mlu address is non-64bytes align and offset is:"
            << unalign_address_offset;
  }
  char *raw_addr = NULL;
  size_t raw_bytes = num_bytes + unalign_address_offset;
  if (true == check_enable_) {
    raw_bytes += 2 * mask_bytes_;
  }
  cnrtRet_t ret = CNRT_RET_SUCCESS;
  ret = cnrtMalloc((void **)&raw_addr, raw_bytes);
  if (raw_addr == NULL || ret != CNRT_RET_SUCCESS) {
    LOG(ERROR) << "MLURuntime: Failed to allocate " << num_bytes << " bytes.";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
    return NULL;
  }
  allocated_size += raw_bytes;
  raw_addr += unalign_address_offset;
  if (false == check_enable_) {
    memory_blocks_.push_back(
        MemBlock(raw_bytes, raw_addr, name, unalign_address_offset));
    return raw_addr;
  }
  char *header = raw_addr;
  char *footer = raw_addr + mask_bytes_ + num_bytes;
  char *mlu_addr = header + mask_bytes_;

#ifdef GTEST_DEBUG_LOG
  VLOG(4) << "MLURuntime: [allocate] malloc [" << (void *)mlu_addr << ", "
          << (void *)footer << ")";
#endif

  ret = cnrtMemcpy(header, (void *)header_mask_.get(), mask_bytes_,
                   CNRT_MEM_TRANS_DIR_HOST2DEV);
  if (ret != CNRT_RET_SUCCESS) {
    LOG(ERROR) << "MLURuntime: Failed to copy header " << num_bytes
               << " bytes.";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
    return NULL;
  }

  ret = cnrtMemcpy(footer, (void *)footer_mask_.get(), mask_bytes_,
                   CNRT_MEM_TRANS_DIR_HOST2DEV);
  if (ret != CNRT_RET_SUCCESS) {
    LOG(ERROR) << "MLURuntime: Failed to copy footer " << num_bytes
               << " bytes.";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
    return NULL;
  }

  memory_blocks_.push_back(
      MemBlock(raw_bytes, header, name, unalign_address_offset));

#ifdef GTEST_DEBUG_LOG
  VLOG(4) << "MLURuntime: [allocate] return ptr is " << (void *)(mlu_addr);
#endif
  return mlu_addr;
}

cnrtRet_t MLURuntime::deallocate(void *mlu_addr) {
  if (mlu_addr == NULL) {
    return CNRT_RET_SUCCESS;
  }
  char *header = (char *)mlu_addr;
  if (true == check_enable_) {
    header = header - mask_bytes_;
  }
  cnrtRet_t ret = CNRT_RET_SUCCESS;
  // get header and footer
  auto it = std::find_if(memory_blocks_.begin(), memory_blocks_.end(),
                         [=](MemBlock b) { return b.header == header; });
  if (it == memory_blocks_.end()) {
    LOG(ERROR) << "MLURuntime: Failed to deallocate " << mlu_addr;
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
    return CNRT_RET_ERR_INVALID;
  }
  bool ok = checkAndFree(*it);
  memory_blocks_.erase(it);
  if (!ok) {
    return CNRT_RET_ERR_INVALID;
  } else {
    return ret;
  }
}

bool MLURuntime::check_byte(void *new_mask, void *org_mask, size_t mask_bytes) {
  return (0 == memcmp(new_mask, org_mask, mask_bytes));
}

void MLURuntime::reset_check() {
  memset((void *)header_check_.get(), 0, mask_bytes_);
  memset((void *)footer_check_.get(), 0, mask_bytes_);
}

// set mask to nan/inf due to date
// if date is even, set nan or set inf
void MLURuntime::rand_set_mask() {
  auto now = time(0);
  struct tm now_time;
  auto *ltm = localtime_r(&now, &now_time);
  auto mday = ltm->tm_mday;
  auto mask_value = nan("");
  auto *user_mask_value = getenv("MLUOP_GTEST_SET_GDRAM");
  auto set_mask = [&](float *mask_start) {
    if (!user_mask_value) {
      mask_value = (mday % 2) ? INFINITY : nan("");
    } else if (strcmp(user_mask_value, "NAN") == 0) {
      mask_value = nan("");
    } else if (strcmp(user_mask_value, "INF") == 0) {
      mask_value = INFINITY;
    } else {
      LOG(WARNING) << "env MLUOP_GTEST_SET_GDRAM only supports NAN or INF"
                   << ", now it is set " << user_mask_value;
    }
    std::fill(mask_start, mask_start + (mask_bytes_ / sizeof(float)),
              mask_value);
  };
  set_mask((float *)footer_mask_.get());
  set_mask((float *)header_mask_.get());
#ifdef GTEST_DEBUG_LOG
  VLOG(4) << "MLURuntime: set " << mask_value
          << " before and after input/output gdram.";
#endif
}

}  // namespace mluoptest
