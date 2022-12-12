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
#ifndef TEST_MLU_OP_GTEST_PB_GTEST_TESTS_MLU_OP_TEST_H_
#define TEST_MLU_OP_GTEST_PB_GTEST_TESTS_MLU_OP_TEST_H_

#include <limits.h>
#include <functional>
#include <vector>
#include <string>
#include <thread>              // NOLINT
#include <mutex>               // NOLINT
#include <condition_variable>  // NOLINT
#include <memory>
#include <fstream>
#include <list>
#include "gtest/gtest.h"
#include "cnrt.h"
#include "mlu_op.h"
#include "core/tensor.h"
#include "core/type.h"

#define CHECK_VALID(expect, func)                                             \
  {                                                                           \
    auto ret = func;                                                          \
    if (expect == ret) {                                                      \
      LOG(INFO) << "Check passed: " #func " returned "                        \
                << mluOpGetErrorString(ret) << ", same as expected " #expect; \
    } else {                                                                  \
      FAIL() << "Check failed: " #func " returned "                           \
             << mluOpGetErrorString(ret) << ", but expect " #expect;          \
    }                                                                         \
  }

TEST(DISABLED_MLUOP, mluOpSetTensorDescriptor) {
  mluOpTensorDescriptor_t desc = nullptr;
  ASSERT_EQ(MLUOP_STATUS_SUCCESS, mluOpCreateTensorDescriptor(&desc));

  // 1. set null
  // std::vector<int> dims_null;
  // ASSERT_EQ(MLUOP_STATUS_SUCCESS, mluOpSetTensorDescriptor(desc,
  // MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
  //                                                        dims_null.size(),
  //                                                        dims_null.data()));

  // 2. set 8 dims
  std::vector<int> dims_8{1, 2, 3, 4, 5, 6, 7, 8};
  ASSERT_EQ(MLUOP_STATUS_SUCCESS,
            mluOpSetTensorDescriptor(desc, MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                     dims_8.size(), dims_8.data()));

  // 3. set again
  std::vector<int> dims_7{7, 6, 5, 4, 3, 2, 1};
  ASSERT_EQ(MLUOP_STATUS_SUCCESS, mluOpResetTensorDescriptor(desc));
  ASSERT_EQ(MLUOP_STATUS_SUCCESS,
            mluOpSetTensorDescriptor(desc, MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                     dims_7.size(), dims_7.data()));

  // 3. set > 11 dims
  std::vector<int> dims_11{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  ASSERT_EQ(MLUOP_STATUS_SUCCESS, mluOpResetTensorDescriptor(desc));
  ASSERT_EQ(MLUOP_STATUS_SUCCESS,
            mluOpSetTensorDescriptor(desc, MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                     dims_11.size(), dims_11.data()));

  // 3. set > 12 dims (again)
  std::vector<int> dims_12{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  ASSERT_EQ(MLUOP_STATUS_SUCCESS, mluOpResetTensorDescriptor(desc));
  ASSERT_EQ(MLUOP_STATUS_SUCCESS,
            mluOpSetTensorDescriptor(desc, MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                     dims_12.size(), dims_12.data()));

  std::vector<int> dims_0{1, 2, 0, 4};
  ASSERT_EQ(MLUOP_STATUS_SUCCESS, mluOpResetTensorDescriptor(desc));
  ASSERT_EQ(MLUOP_STATUS_SUCCESS,
            mluOpSetTensorDescriptor(desc, MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                                     dims_0.size(), dims_0.data()));

  // foolproof
  ASSERT_EQ(MLUOP_STATUS_SUCCESS, mluOpResetTensorDescriptor(desc));
  CHECK_VALID(
      MLUOP_STATUS_BAD_PARAM,
      mluOpSetTensorDescriptor(nullptr, MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
                               dims_8.size(), dims_8.data()));

  ASSERT_EQ(MLUOP_STATUS_SUCCESS, mluOpResetTensorDescriptor(desc));
  CHECK_VALID(MLUOP_STATUS_BAD_PARAM,
              mluOpSetTensorDescriptor(desc, MLUOP_LAYOUT_NCHW,
                                       MLUOP_DTYPE_FLOAT, 2, nullptr));

  ASSERT_EQ(MLUOP_STATUS_SUCCESS, mluOpDestroyTensorDescriptor(desc));
}

TEST(DISABLED_MLUOP, SINGLE_THREAD_mluOpTensorDescriptor_t) {
  auto run_test = [&](bool by_queue = true) {
    size_t REPEAT_NUM = 1000;
    std::vector<mluOpTensorDescriptor_t> desc_vec(REPEAT_NUM);

    auto a = std::chrono::system_clock::now();
    for (size_t i = 0; i < REPEAT_NUM; ++i) {
      ASSERT_TRUE(MLUOP_STATUS_SUCCESS ==
                  mluOpCreateTensorDescriptor(&(desc_vec[i])));
    }
    auto b = std::chrono::system_clock::now();
    auto dur1 = std::chrono::duration_cast<std::chrono::nanoseconds>(b - a);
    auto print1 = "mluOpCreateTensorDescriptor * " + std::to_string(REPEAT_NUM);
    LOG(INFO) << std::left << std::setw(40) << print1 << " takes "
              << dur1.count() << " ns\n";

    std::vector<int> dims{1, 2, 3, 4, 5, 6, 7, 8};
    auto c = std::chrono::system_clock::now();
    for (size_t i = 0; i < REPEAT_NUM; ++i) {
      ASSERT_TRUE(MLUOP_STATUS_SUCCESS ==
                  mluOpSetTensorDescriptor(desc_vec[i], MLUOP_LAYOUT_NCHW,
                                           MLUOP_DTYPE_FLOAT, dims.size(),
                                           dims.data()));
    }
    auto d = std::chrono::system_clock::now();
    auto dur2 = std::chrono::duration_cast<std::chrono::nanoseconds>(d - c);
    auto print2 = "mluOpSetTensorDescriptor * " + std::to_string(REPEAT_NUM);
    LOG(INFO) << std::left << std::setw(40) << print2 << " takes "
              << dur2.count() << " ns\n";

    auto e = std::chrono::system_clock::now();
    for (size_t i = 0; i < REPEAT_NUM; ++i) {
      ASSERT_TRUE(MLUOP_STATUS_SUCCESS ==
                  mluOpDestroyTensorDescriptor(desc_vec[i]));
    }
    auto f = std::chrono::system_clock::now();
    auto dur3 = std::chrono::duration_cast<std::chrono::nanoseconds>(f - e);
    auto print3 =
        "mluOpDestroyTensorDescriptor * " + std::to_string(REPEAT_NUM);
    LOG(INFO) << std::left << std::setw(40) << print3 << " takes "
              << dur3.count() << " ns\n";
  };

  LOG(INFO) << "by tensor queue:\n";
  run_test(true);
}

TEST(DISABLED_MLUOP, SINGLE_THREAD_group_mluOpTensorDescriptor_t) {
  auto run_test = [&](bool by_queue = true) {
    size_t REPEAT_NUM = 1000;
    mluOpTensorDescriptor_t desc_vect[REPEAT_NUM];
    mluOpTensorDescriptor_t *desc_vec[REPEAT_NUM];
    for (int i = 0; i < REPEAT_NUM; ++i) {
      desc_vec[i] = &desc_vect[i];
    }

    auto a = std::chrono::system_clock::now();
    ASSERT_TRUE(MLUOP_STATUS_SUCCESS ==
                mluOpCreateGroupTensorDescriptors(desc_vec, REPEAT_NUM));
    auto b = std::chrono::system_clock::now();
    auto dur1 = std::chrono::duration_cast<std::chrono::nanoseconds>(b - a);
    auto print1 =
        "mluOpCreateGroupTensorDescriptors * " + std::to_string(REPEAT_NUM);
    LOG(INFO) << std::left << std::setw(40) << print1 << " takes "
              << dur1.count() << " ns\n";

    std::vector<int> dims{2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<mluOpTensorLayout_t> layout_array(REPEAT_NUM,
                                                  MLUOP_LAYOUT_NCHW);
    std::vector<mluOpDataType_t> dtype_array(REPEAT_NUM, MLUOP_DTYPE_FLOAT);
    std::vector<int> dimNb_array(REPEAT_NUM, 8);
    std::vector<int> dimSize_array(REPEAT_NUM * 8, 2);
    auto c = std::chrono::system_clock::now();
    ASSERT_TRUE(MLUOP_STATUS_SUCCESS ==
                mluOpSetGroupTensorDescriptors(
                    desc_vec, layout_array.data(), dtype_array.data(),
                    dimNb_array.data(), dimSize_array.data(), REPEAT_NUM));

    auto d = std::chrono::system_clock::now();
    auto dur2 = std::chrono::duration_cast<std::chrono::nanoseconds>(d - c);
    auto print2 =
        "mluOpSetGroupTensorDescriptors * " + std::to_string(REPEAT_NUM);
    LOG(INFO) << std::left << std::setw(40) << print2 << " takes "
              << dur2.count() << " ns\n";

    auto e = std::chrono::system_clock::now();
    ASSERT_TRUE(MLUOP_STATUS_SUCCESS ==
                mluOpDestroyGroupTensorDescriptors(desc_vec, REPEAT_NUM));
    auto f = std::chrono::system_clock::now();
    auto dur3 = std::chrono::duration_cast<std::chrono::nanoseconds>(f - e);
    auto print3 =
        "mluOpDestroyGroupTensorDescriptors * " + std::to_string(REPEAT_NUM);
    LOG(INFO) << std::left << std::setw(40) << print3 << " takes "
              << dur3.count() << " ns\n";
  };

  LOG(INFO) << "by tensor queue:\n";
  run_test(true);
}

//
//                      | mluOpCreateTensorDescriptorQueue
//                      v
//  ---------------------------------------------
//  |     mluOpTensorDescriptorQueueHandle_t     |
//  ---------------------------------------------
//    ^ ^ ^                               | | |
//    | | | push threads:                 | | | pop threads:
//    | | |  return back tensor to        | | |  pop tensor from
//    mluOpTensorQueue | | |  mluOpTensorQueue.             | | |  call
//    mluOpSetTensor* and mluOpGetTensor* | | |  if external pool is empty    |
//    | |  to check if tensor is available | | |  block this thread.           |
//    | |  if mluOpTensorQueue is empty mluOpPopTensor* will | | | | | |  create
//    more tensors and return back 1 tensor. | | | | | |  and mluOpTensorQueue
//    won't be full. | | |                               v v v
//  ---------------------------------------------
//  |           external pool(user)             |
//  ---------------------------------------------
//                      | mluOpDestroyTensorDescriptorQueue
//                      v
//
TEST(DISABLED_MLUOP, MULTI_THREAD_mluOpTensorDescriptor_t) {
  // get random value
  auto random = [=](int lower, int upper) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dis(lower, upper);  // [lower, upper)
    return dis(mt);
  };

  // external pool (user)
  struct External {
    // FILE *fp = NULL;
    External() { /*fp = fopen("log", "w+");*/
    }
    ~External() { /*fclose(fp);*/
    }

    std::list<mluOpTensorDescriptor_t> user_pool;
    std::mutex mtx;
    std::condition_variable cond;
    // here block pop() when pool is empty to make 2 threads more balanced.
    // if not, the thread which called pop() will quit quickly.
    // but if all thread called push() are done. just mark shut_down as true.
    // don't call pop() anymore, it will block thread.
    bool shut_down = false;
    void push(mluOpTensorDescriptor_t desc) {
      std::lock_guard<std::mutex> lk(mtx);
      user_pool.emplace_back(desc);
      cond.notify_one();
    }

    mluOpTensorDescriptor_t pop() {
      std::unique_lock<std::mutex> lk(mtx);
      cond.wait(lk, [=] { return !user_pool.empty() || shut_down; });
      if (shut_down && user_pool.empty()) {
        return nullptr;
      }

      mluOpTensorDescriptor_t desc = user_pool.front();
      user_pool.pop_front();
      return desc;
    }
  };

  auto pop = [=](std::shared_ptr<External> external) {
    // size_t pop_num = random(0, 500);  // i need [0, 500) tensor desc in each
    // thread
    size_t pop_num = 1000;
    int *ret_dims = new int[MLUOP_DIM_MAX];
    for (size_t i = 0; i < pop_num; ++i) {
      // get 1 desc
      mluOpTensorDescriptor_t desc = nullptr;
      ASSERT_TRUE(MLUOP_STATUS_SUCCESS == mluOpCreateTensorDescriptor(&desc));
      ASSERT_NE(desc, nullptr);

      // set param
      // random range is not important
      // just make sure the value is random
      // and when mluOpSetTensor* and mluOpGet they are equal.
      mluOpTensorLayout_t layout =
          (mluOpTensorLayout_t)(random(0, 5 + 1));  // layout is [0, 5]
      mluOpDataType_t dtype =
          (mluOpDataType_t)(random(0, 8 + 1));  // dtype is [0, 8]

      int dimNb = random(1, 8);  // [1, 8] dimNb
      std::vector<int> dims(dimNb);
      for (int d = 0; d < dims.size(); ++d) {
        dims[d] = random(0, 10);
      }

      ASSERT_TRUE(MLUOP_STATUS_SUCCESS ==
                  mluOpSetTensorDescriptor(desc, layout, dtype, dims.size(),
                                           dims.data()));

      // get param
      mluOpDataType_t ret_dtype;
      mluOpTensorLayout_t ret_layout;
      int ret_dimNb = 0;
      ASSERT_TRUE(MLUOP_STATUS_SUCCESS ==
                  mluOpGetTensorDescriptor(desc, &ret_layout, &ret_dtype,
                                           &ret_dimNb, ret_dims));

      // check params
      ASSERT_EQ(ret_dtype, dtype);
      ASSERT_EQ(ret_layout, layout);
      ASSERT_EQ(dimNb, ret_dimNb);
      for (int d = 0; d < ret_dimNb; ++d) {
        ASSERT_EQ(ret_dims[d], dims[d]);
      }

      external->push(
          desc);  // save desc in external pool, means user is using this desc
    }
    delete[] ret_dims;
  };

  auto push = [=](std::shared_ptr<External> external) {
    // size_t push_num = random(0, 500);  // return back [0, 500) tensor desc
    size_t push_num = 1000;
    for (size_t i = 0; i < push_num; ++i) {
      auto desc = external->pop();  // user return back 1 desc
      if (external->shut_down && external->user_pool.empty()) {
        return;  // all pop() thread done, and pool is empty, just quit.
      }
      ASSERT_TRUE(MLUOP_STATUS_SUCCESS == mluOpDestroyTensorDescriptor(desc));
    }
  };

  auto run_test = [=]() {
    auto external_pool = std::make_shared<External>();
    std::vector<std::thread> pop_threads;
    std::vector<std::thread> push_threads;
    for (size_t t = 0; t < 40; ++t) {
      pop_threads.emplace_back(std::thread(pop, external_pool));
    }
    for (size_t t = 0; t < 40; ++t) {
      push_threads.emplace_back(std::thread(push, external_pool));
    }

    for (size_t t = 0; t < pop_threads.size(); ++t) {
      pop_threads[t].join();
    }

    {
      std::lock_guard<std::mutex> lk(external_pool->mtx);
      external_pool->shut_down = true;
    }
    external_pool->cond.notify_all();
    for (size_t t = 0; t < push_threads.size(); ++t) {
      push_threads[t].join();
    }
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < 4; ++i) {
    threads.emplace_back(run_test);
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}

#endif  // TEST_MLU_OP_GTEST_PB_GTEST_TESTS_MLU_OP_TEST_H_
