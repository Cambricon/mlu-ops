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

void TestEnvironment::SetUp() {
  // 1. set up cnrt env
  VLOG(4) << "SetUp CNRT environment.";

  // 2. get device num
  cnrtDev_t dev = 0;  // not ptr, it's long unsigned int
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
  unsigned int dev_id = (rand_r(&seed) % (dev_num - 1 - 0 + 1)) + 0;

  // cnrt set current device using CNRT_DEFAULT_DEVICE
  // in cnrtGetDeviceHandle() CNRT_DEFAULT_DEVICE > id
  VLOG(4) << "Set current device as device: " << dev_id;
  ASSERT_EQ(cnrtGetDeviceHandle(&dev, dev_id), CNRT_RET_SUCCESS);
  ASSERT_EQ(cnrtSetCurrentDevice(dev), CNRT_RET_SUCCESS);
}

void TestEnvironment::TearDown() {
  // destroy cnrt env
  VLOG(4) << "TearDown CNRT environment.";
}
