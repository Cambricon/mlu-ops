/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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
#include <iostream>
#include <vector>
#include <string>
#include <tuple>

#include "gtest/gtest.h"
#include "mlu_op.h"
#include "core/context.h"
#include "core/logging.h"
#include "api_test_tools.h"

namespace mluopapitest {
class fft_SetFFTReserveArea : public testing::Test {
 public:
  void setParam(bool handle, bool fft_plan, bool reservespace) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (fft_plan) {
      MLUOP_CHECK(mluOpCreateFFTPlan(&fft_plan_));
    }

    if (reservespace) {
      GTEST_CHECK(cnrtSuccess == cnrtMalloc(&reservespace_, reservespace_size));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status =
        mluOpSetFFTReserveArea(handle_, fft_plan_, reservespace_);

    destroy();
    return status;
  }

 protected:
  virtual void SetUp() {
    handle_ = nullptr;
    fft_plan_ = nullptr;
    reservespace_ = nullptr;
  }

  void destroy() {
    try {
      if (handle_) {
        CNRT_CHECK(cnrtQueueSync(handle_->queue));
        VLOG(4) << "Destroy handle_";
        MLUOP_CHECK(mluOpDestroy(handle_));
        handle_ = nullptr;
      }
      if (fft_plan_) {
        VLOG(4) << "Destroy fft_plan_";
        MLUOP_CHECK(mluOpDestroyFFTPlan(fft_plan_));
        fft_plan_ = nullptr;
      }
      if (reservespace_) {
        VLOG(4) << "Destroy reservespace_";
        GTEST_CHECK(cnrtSuccess == cnrtFree(reservespace_));
        reservespace_ = nullptr;
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in fft_SetFFTReserveArea";
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpFFTPlan_t fft_plan_ = nullptr;
  void *reservespace_ = nullptr;
  size_t reservespace_size = 64;
};

TEST_F(fft_SetFFTReserveArea, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in fft_SetFFTReserveArea";
  }
}

TEST_F(fft_SetFFTReserveArea, BAD_PARAM_fft_plan_null) {
  try {
    setParam(true, false, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in fft_SetFFTReserveArea";
  }
}

TEST_F(fft_SetFFTReserveArea, BAD_PARAM_reservespace_null) {
  try {
    setParam(true, true, false);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in fft_SetFFTReserveArea";
  }
}

}  // namespace mluopapitest
