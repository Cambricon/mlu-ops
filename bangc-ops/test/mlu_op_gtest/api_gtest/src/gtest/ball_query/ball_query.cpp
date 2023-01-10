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
#include <iostream>
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
class ball_query : public testing::Test {
 public:
  void setParam(bool handle, bool new_xyz_desc, bool new_xyz, bool xyz_desc,
                bool xyz, bool idx_desc, bool idx) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (new_xyz_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&new_xyz_desc_));
      std::vector<int> new_xyz_dims = {2, 16, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(new_xyz_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           new_xyz_dims.data()));
    }
    if (new_xyz) {
      size_t new_xyz_ele_num = 2 * 16 * 3;
      size_t new_xyz_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t new_xyz_bytes = new_xyz_ele_num * new_xyz_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&new_xyz_, new_xyz_bytes));
    }
    if (xyz_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&xyz_desc_));
      std::vector<int> xyz_dims = {2, 4, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(xyz_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           xyz_dims.data()));
    }
    if (xyz) {
      size_t xyz_ele_num = 2 * 4 * 3;
      size_t xyz_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t xyz_bytes = xyz_ele_num * xyz_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&xyz_, xyz_bytes));
    }
    if (idx_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&idx_desc_));
      std::vector<int> idx_dims = {2, 4, 32};
      MLUOP_CHECK(mluOpSetTensorDescriptor(idx_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 3,
                                           idx_dims.data()));
    }
    if (idx) {
      size_t idx_ele_num = 2 * 4 * 32;
      size_t idx_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      size_t idx_bytes = idx_ele_num * idx_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&idx_, idx_bytes));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status =
        mluOpBallQuery(handle_, new_xyz_desc_, new_xyz_, xyz_desc_, xyz_,
                       min_radius_, max_radius_, nsample_, idx_desc_, idx_);
    destroy();
    return status;
  }

 protected:
  void destroy() {
    if (handle_) {
      CNRT_CHECK(cnrtQueueSync(handle_->queue));
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = NULL;
    }
    if (new_xyz_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(new_xyz_desc_));
      new_xyz_desc_ = NULL;
    }
    if (new_xyz_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(new_xyz_));
      new_xyz_ = NULL;
    }
    if (xyz_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(xyz_desc_));
      xyz_desc_ = NULL;
    }
    if (xyz_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(xyz_));
      xyz_ = NULL;
    }
    if (idx_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(idx_desc_));
      idx_desc_ = NULL;
    }
    if (idx_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(idx_));
      idx_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  float min_radius_ = 0;
  float max_radius_ = 0.2;
  int nsample_ = 32;
  mluOpTensorDescriptor_t new_xyz_desc_ = NULL;
  void* new_xyz_ = NULL;
  mluOpTensorDescriptor_t xyz_desc_ = NULL;
  void* xyz_ = NULL;
  mluOpTensorDescriptor_t idx_desc_ = NULL;
  void* idx_ = NULL;
};

TEST_F(ball_query, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in ball_query";
  }
}

TEST_F(ball_query, BAD_PARAM_new_xyz_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in ball_query";
  }
}

TEST_F(ball_query, BAD_PARAM_new_xyz_null) {
  try {
    setParam(true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in ball_query";
  }
}

TEST_F(ball_query, BAD_PARAM_xyz_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in ball_query";
  }
}

TEST_F(ball_query, BAD_PARAM_xyz_null) {
  try {
    setParam(true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in ball_query";
  }
}

TEST_F(ball_query, BAD_PARAM_idx_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in ball_query";
  }
}

TEST_F(ball_query, BAD_PARAM_idx_null) {
  try {
    setParam(true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in ball_query";
  }
}
}  // namespace mluopapitest
