/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
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
#include <vector>
#include <string>
#include <tuple>

#include "api_test_tools.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "gtest/gtest.h"
#include "mlu_op.h"
#include "core/context.h"

namespace mluopapitest {
class tin_shift_backward : public testing::Test {
 public:
  void setParam(bool handle, bool grad_output_desc, bool grad_output,
                bool shifts_desc, bool shifts, bool grad_input_desc,
                bool grad_input) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (grad_output_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_output_desc_));
      std::vector<int> dim_size = {4, 4, 4, 4};
      MLUOP_CHECK(
          mluOpSetTensorDescriptor(grad_output_desc_, MLUOP_LAYOUT_ARRAY,
                                   MLUOP_DTYPE_FLOAT, 4, dim_size.data()));
    }
    if (grad_output) {
      if (grad_output_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&grad_output_,
                               MLUOP_DTYPE_FLOAT * mluOpGetTensorElementNum(
                                                       grad_output_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&grad_output_, MLUOP_DTYPE_FLOAT * 2));
      }
    }
    if (shifts_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&shifts_desc_));
      std::vector<int> dim_size = {4, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(shifts_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 2,
                                           dim_size.data()));
    }
    if (shifts) {
      if (shifts_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&shifts_, MLUOP_DTYPE_INT32 *
                                     mluOpGetTensorElementNum(shifts_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&shifts_, MLUOP_DTYPE_INT32 * 2));
      }
    }
    if (grad_input_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_input_desc_));
      std::vector<int> dim_size = {4, 4, 4, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(grad_input_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           dim_size.data()));
    }
    if (grad_input) {
      if (grad_input_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&grad_input_,
                               MLUOP_DTYPE_FLOAT *
                                   mluOpGetTensorElementNum(grad_input_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&grad_input_, MLUOP_DTYPE_FLOAT * 2));
      }
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpTinShiftBackward(
        handle_, grad_output_desc_, grad_output_, shifts_desc_, shifts_,
        grad_input_desc_, grad_input_);
    destroy();
    return status;
  }

 protected:
  void destroy() {
    VLOG(4) << "Destroy parameters.";
    if (handle_) {
      CNRT_CHECK(cnrtQueueSync(handle_->queue));
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = NULL;
    }
    if (grad_output_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_output_desc_));
      grad_output_desc_ = NULL;
    }
    if (shifts_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(shifts_desc_));
      shifts_desc_ = NULL;
    }
    if (grad_input_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_input_desc_));
      grad_input_desc_ = NULL;
    }
    if (grad_output_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_output_));
      grad_output_ = NULL;
    }
    if (shifts_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(shifts_));
      shifts_ = NULL;
    }
    if (grad_input_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_input_));
      grad_input_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t grad_output_desc_ = NULL;
  mluOpTensorDescriptor_t shifts_desc_ = NULL;
  mluOpTensorDescriptor_t grad_input_desc_ = NULL;
  void* grad_output_ = NULL;
  void* shifts_ = NULL;
  void* grad_input_ = NULL;
};

TEST_F(tin_shift_backward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in tin_shift_backward";
  }
}

TEST_F(tin_shift_backward, BAD_PARAM_grad_output_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in tin_shift_backward";
  }
}

TEST_F(tin_shift_backward, BAD_PARAM_grad_output_null) {
  try {
    setParam(true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in tin_shift_backward";
  }
}

TEST_F(tin_shift_backward, BAD_PARAM_shifts_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in tin_shift_backward";
  }
}
TEST_F(tin_shift_backward, BAD_PARAM_shifts_null) {
  try {
    setParam(true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in tin_shift_backward";
  }
}

TEST_F(tin_shift_backward, BAD_PARAM_grad_input_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in tin_shift_backward";
  }
}

TEST_F(tin_shift_backward, BAD_PARAM_grad_input_null) {
  try {
    setParam(true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in tin_shift_backward";
  }
}
}  // namespace mluopapitest
