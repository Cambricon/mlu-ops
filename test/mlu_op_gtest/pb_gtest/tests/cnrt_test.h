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
#ifndef TEST_MLU_OP_GTEST_PB_GTEST_TESTS_CNRT_TEST_H_
#define TEST_MLU_OP_GTEST_PB_GTEST_TESTS_CNRT_TEST_H_

#include <thread>  // NOLINT
#include <vector>
#include <string>
#include "gtest/gtest.h"
#include "cnrt.h"
#include "mlu_op.h"

#define BUS_ID_LINE_LENGTH 60
#define DEVICE_NAME_BUFFER_SIZE 64

// when run daily test, only 1 device is visible to cnrt
// but all device is visible to lspci
// so remove this test.
// TEST(DISABLED_CNRT, cnrtGetDeviceCount) {
//
//   int dev        = 0;  // not ptr, it's long unsigned int
//   unsigned int dev_num = 0;
//   ASSERT_TRUE(CNRT_RET_SUCCESS == cnrtGetDeviceCount(&dev_num));
//   ASSERT_GT(dev_num, 0);  // dev_num > 0
//
//   ASSERT_TRUE(cnrtSuccess == cnrtGetDevice(&dev));  // use device: 0
//   ASSERT_TRUE(cnrtSuccess == cnrtSetDevice(dev));
//
//   // platform
//   CNdev mlu_dev;
//   ASSERT_TRUE(CN_SUCCESS == cnCtxGetDevice(&mlu_dev));
//
//   char device_name[DEVICE_NAME_BUFFER_SIZE] = "";
//   ASSERT_TRUE(CN_SUCCESS == cnDeviceGetName(device_name,
//   DEVICE_NAME_BUFFER_SIZE, mlu_dev));
//
//   auto GetBusId = [](std::string platform) -> std::vector<std::string> {
//     std::vector<std::string> res;
//     std::string cmd = "lspci -d:" + platform;
//     FILE *fp        = popen(cmd.c_str(), "r");
//     char buffer[BUS_ID_LINE_LENGTH];
//     while (fgets(buffer, sizeof(buffer), fp) != NULL) {
//       res.emplace_back(std::string(buffer));
//     }
//     pclose(fp);
//     return res;
//   };
//
//   if (strcmp("MLU270", device_name) == 0) {
//     ASSERT_EQ(GetBusId("0270").size(), dev_num)
//         << "The result of cnrtGetDeviceCount() is different with lspci
//         -d:270";
//   } else if (strcmp("MLU290", device_name) == 0) {
//     ASSERT_EQ(GetBusId("0290").size(), dev_num)
//         << "The result of cnrtGetDeviceCount() is different with lspci
//         -d:290";
//   } else if (strcmp("MLU220", device_name) == 0) {
//     ASSERT_EQ(GetBusId("0220").size(), dev_num)
//         << "The result of cnrtGetDeviceCount() is different with lspci
//         -d:220";
//   }
// }

TEST(DISABLED_CNRT, cnrtMemGetInfo) {
  unsigned int dev_num = 0;
  ASSERT_TRUE(CNRT_RET_SUCCESS == cnrtGetDeviceCount(&dev_num));
  ASSERT_GT(dev_num, 0);  // dev_num > 0

  int dev_id;
  ASSERT_TRUE(cnrtSuccess == cnrtGetDevice(&dev_id));  // use device: 0
  ASSERT_TRUE(cnrtSuccess == cnrtSetDevice(dev_id));

  size_t total_bytes = 0, free_bytes = 0;
  ASSERT_TRUE(cnrtSuccess == cnrtMemGetInfo(&free_bytes, &total_bytes));
  std::cout << "The mlu free memory size(of all channels) is " << free_bytes
            << " bytes. \n";
  std::cout << "The mlu total memory size(of all channels) is " << total_bytes
            << " bytes. \n";
}

// example,
// this test is useless
// test cnrtNotifier:
// 1.create notifier
// 2.place notifier
// 3.get duration
// 4.destroy notifier
// multi-thread create and place notifier to 1 queue.
// and multi-thread destroy these notifier
TEST(DISABLED_CNRT, cnrtNotifier) {
  const size_t thread_num = 4;

  unsigned int dev_num = 0;
  ASSERT_TRUE(CNRT_RET_SUCCESS == cnrtGetDeviceCount(&dev_num));
  ASSERT_GT(dev_num, 0);

  int dev_id;
  ASSERT_TRUE(cnrtSuccess == cnrtGetDevice(&dev_id));
  ASSERT_TRUE(cnrtSuccess == cnrtSetDevice(dev_id));

  cnrtQueue_t queue = nullptr;
  ASSERT_TRUE(cnrtSuccess == cnrtQueueCreate(&queue));

  struct Context {
    cnrtNotifier_t na;
    cnrtNotifier_t nb;
  };
  std::vector<Context> ctxs(thread_num);

  auto task_part1 = [&queue, &ctxs](int idx) {
    ASSERT_TRUE(cnrtSuccess == cnrtNotifierCreate(&(ctxs.at(idx).na)));
    ASSERT_TRUE(cnrtSuccess == cnrtNotifierCreate(&(ctxs.at(idx).nb)));
    ASSERT_TRUE(CNRT_RET_SUCCESS == cnrtPlaceNotifier(ctxs.at(idx).na, queue));
    ASSERT_TRUE(CNRT_RET_SUCCESS == cnrtPlaceNotifier(ctxs.at(idx).nb, queue));
  };

  auto task_part2 = [&queue, &ctxs](int idx) {
    ASSERT_TRUE(cnrtSuccess == cnrtQueueSync(queue));

    float hwt = -1.0f;
    ASSERT_TRUE(CNRT_RET_SUCCESS ==
                cnrtNotifierDuration(ctxs.at(idx).na, ctxs.at(idx).nb, &hwt));
    ASSERT_EQ(0.0f, hwt);

    ASSERT_TRUE(cnrtSuccess == cnrtNotifierDestroy(ctxs.at(idx).na));
    ASSERT_TRUE(cnrtSuccess == cnrtNotifierDestroy(ctxs.at(idx).nb));
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < thread_num; ++i) {
    threads.push_back(std::thread(task_part1, i));
  }
  for (int i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }

  for (int i = 0; i < thread_num; ++i) {
    threads[i] = std::thread(task_part2, i);
  }
  for (int i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }

  ASSERT_TRUE(cnrtSuccess == cnrtQueueDestroy(queue));
}

#endif  // TEST_MLU_OP_GTEST_PB_GTEST_TESTS_CNRT_TEST_H_
