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
#include "test_env.h"

extern mluoptest::GlobalVar global_var;
void TestEnvironment::SetUp() {
  // 1. set up cnrt env
  VLOG(4) << "SetUp CNRT environment.";

  // 2. get device num
  unsigned int dev_num = 0;
  ASSERT_EQ(cnrtGetDeviceCount(&dev_num), CNRT_RET_SUCCESS);
  if (dev_num <= 0) {  // dev_num_ should > 0
    FAIL() << "Can't find device";
  } else {
    VLOG(4) << "Found " << dev_num << " devices.";
  }

  // 3. random device id [0, dev_num)
  // [a, b] => (rand() % (b - a + 1)) + a
  unsigned int seed = time(0);
  global_var.dev_id_ = (rand_r(&seed) % (dev_num - 1 - 0 + 1)) + 0;

  // cnrt set current device using CNRT_DEFAULT_DEVICE
  // in cnrtGetDeviceHandle() CNRT_DEFAULT_DEVICE > id
  VLOG(4) << "Set current device as device: " << global_var.dev_id_;
  if (global_var.thread_num_ == 1) {
    // if single thread is 1, set current device.
    // if multi thread set current in each thread.
    ASSERT_EQ(cnrtSetDevice(global_var.dev_id_), CNRT_RET_SUCCESS);
  }
}

void TestEnvironment::TearDown() {
  VLOG(4) << "TearDown CNRT environment.";

  auto summary = global_var.summary_;
  std::cout << "[ SUMMARY  ] "
            << "Total " << summary.case_count << " cases of "
            << summary.suite_count << " op(s).\n";
  if (summary.failed_list.empty()) {
    std::cout << "ALL PASSED.\n";
  } else {
    auto case_list = summary.failed_list;
    std::cout << case_list.size() << " CASES FAILED:\n";
    for (auto it = case_list.begin(); it != case_list.end(); ++it) {
      std::cout << "Failed: " << (*it) << "\n";
    }
  }
}
