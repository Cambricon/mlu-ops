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
#ifndef TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_RUNTIME_H_
#define TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_RUNTIME_H_

#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <cstring>
#include "cnrt.h"
#include "core/logging.h"
#include "memory_pool.h"
#include "mlu_op.h"
#include "pb_test_tools.h"

namespace mluoptest {

class Runtime {
 public:
  Runtime() {}
  virtual ~Runtime() {}

  // this function will throw exception
  // don't call this function in ctor
  void *allocate(size_t num_bytes, std::string name = "") { return NULL; }

  // this function will throw exception
  cnrtRet_t deallocate(void *ptr) { return CNRT_RET_SUCCESS; }

  // this function won't throw exception
  // so only this function can be called in dtor
  cnrtRet_t destroy() { return CNRT_RET_SUCCESS; }
  // use cnrtRet_t, cuz when call cnrtFree .. can return directly.
};

class CPURuntime : public Runtime {
 public:
  CPURuntime();
  virtual ~CPURuntime();
  void init(std::shared_ptr<CPUMemoryPool> cmp) {}

  // allocate(mluOpCreate(), mluOpDestroy());
  // this function will throw exception
  // don't call this function in ctor
  template <typename T>
  T allocate(mluOpStatus_t (*ctor)(T *), mluOpStatus_t (*dtor)(T),
             std::string name = "") {
    T obj = NULL;
    mluOpStatus_t status = (*ctor)(&obj);
    if (status != MLUOP_STATUS_SUCCESS) {
      LOG(ERROR) << "CPURuntime: Failed to allocate " << name;
      throw std::invalid_argument(std::string(__FILE__) + " +" +
                                  std::to_string(__LINE__));
      return NULL;
    }
    memory_blocks_.push_back(std::make_shared<MemBlock<T>>(obj, dtor, name));
    return obj;
  }

  // allocate(malloc(bytes))
  void *allocate(void *ptr, std::string name = "");

  // allocate(new float[])
  template <typename R>
  R *allocate(R *ptr, std::string name = "") {
    void (*f)(void *) = (operator delete[]);
    memory_blocks_.push_back(std::make_shared<MemBlock<R *>>(ptr, f, name));
    return ptr;
  }

  // allocate(size_in_bytes)
  void *allocate(size_t num_bytes, std::string name = "");

  template <typename T>
  cnrtRet_t deallocate(T object) {
    if (NULL == (void *)object) {
      return CNRT_RET_SUCCESS;
    }
    auto it = std::find_if(memory_blocks_.begin(), memory_blocks_.end(),
                           [=](std::shared_ptr<MemBlockBase> b) {
                             return b->id == (void *)object;
                           });
    if (it == memory_blocks_.end()) {
      LOG(ERROR) << "CPURuntime: Failed to deallocate " << (void *)object
                 << ", double free.";
      throw std::invalid_argument(std::string(__FILE__) + " +" +
                                  std::to_string(__LINE__));
      return CNRT_RET_ERR_INVALID;
    }
    it->reset();
    memory_blocks_.erase(it);
    return CNRT_RET_SUCCESS;
  }

  // so only this function can be called in dtor
  cnrtRet_t destroy();

 private:
  struct MemBlockBase {
    MemBlockBase() {}
    virtual ~MemBlockBase() {}
    void *id = NULL;
  };

  template <typename T>
  struct MemBlock : MemBlockBase {
    MemBlock(T o, mluOpStatus_t (*f)(T), std::string n)
        : obj(o), c_dtor(f), name(n) {
      id = (void *)o;
#ifdef GTEST_DEBUG_LOG
      VLOG(4) << "CPURuntime: [allocate] malloc for [" << name << "] "
              << (void *)obj;
#endif
    }
    MemBlock(T o, void (*f)(void *), std::string n)
        : obj(o), v_dtor(f), name(n) {
      id = (void *)o;
#ifdef GTEST_DEBUG_LOG
      VLOG(4) << "CPURuntime: [allocate] malloc for [" << name << "] "
              << (void *)obj;
#endif
    }
    ~MemBlock() {
#ifdef GTEST_DEBUG_LOG
      VLOG(4) << "CPURuntime: [deallocate] free for [" << name << "] "
              << (void *)obj;
#endif
      if (c_dtor != NULL) {
        (*c_dtor)(obj);
      } else if (v_dtor != NULL) {
        (*v_dtor)(obj);
      }
    }
    T obj;
    // we have 2 kind of dtor
    // * void (*fp) for buildin type
    // * mluOpStatus (*fp) for customized type
    // here put different type together in 1 vector
    // when deallocate, dtor type is unknown(only known obj type)
    // i don't want a map(or something) to find out dtor type by obj type
    //
    // * no matter what type ptr is, delete void*/ free(void*) can release space
    // correctly
    void (*v_dtor)(void *) = NULL;
    mluOpStatus_t (*c_dtor)(T) = NULL;
    // here can't set object as shared_ptr directly.
    // cuz we need put all object (different type) in a vector
    // so declare vector of father struct, but push son struct in it.
    // by inheritance of struct, call son's dtor
    std::string name;
  };

  std::vector<std::shared_ptr<MemBlockBase>> memory_blocks_;
};

class MLURuntime : public Runtime {
 public:
  MLURuntime();
  virtual ~MLURuntime();
  void init(std::shared_ptr<MLUMemoryPool> mmp) {}

  // this function throw exception
  // don't call this function in ctor
  void *allocate(size_t num_bytes, std::string name = "");
  // this function throw exception
  cnrtRet_t deallocate(void *mlu_addr);

  // only destroy can be called in dtor, so don't throw exception.
  // return status
  cnrtRet_t destroy();

 private:
  struct MemBlock {
    MemBlock(size_t s, char *p, std::string n)
        : raw_bytes(s), header(p), name(n) {}
    size_t raw_bytes = 0;   // include mask
    char *header = NULL;    // mlu addr
    bool is_const = false;  // if is const, this buffer shouldn't be modified.
    void *mask = NULL;      // if is const, use host_ptr to check.
    std::string name;       // memory block id
  };

  // each memory has head mask and foot mask
  const int mask_ = 0xa5;
  // when __memcpy size is:
  // [0, 64)    -> head addr should align to 64
  // [128,256)  -> 128
  // [256, 512) -> 256
  // [512, ~)   -> 512
  // for compatibility of most situation, mask_size set as 512.
  const size_t mask_bytes_ = 512;  // iodma best performance size
  std::shared_ptr<char> header_mask_;
  std::shared_ptr<char> footer_mask_;
  std::shared_ptr<char> header_check_;
  std::shared_ptr<char> footer_check_;

  std::vector<MemBlock> memory_blocks_;

  bool check_byte(void *new_mask, void *org_mask, size_t num);
  void reset_check();
  void rand_set_mask();
  bool check_enable_ = true;
};

}  // namespace mluoptest

#endif  // TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_RUNTIME_H_
