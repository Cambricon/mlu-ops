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
class roi_crop_forward : public testing::Test {
 public:
  void setParam(bool handle, bool input_desc, bool input, bool grid_desc,
                bool grid, bool output_desc, bool output) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (input_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
      std::vector<int> dim_size = {1, 7, 7, 9};
      MLUOP_CHECK(mluOpSetTensorDescriptor(input_desc_, MLUOP_LAYOUT_NHWC,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           dim_size.data()));
    }
    if (grid_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grid_desc_));
      std::vector<int> dim_size = {1, 7, 7, 2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(grid_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           dim_size.data()));
    }
    if (output_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));
      std::vector<int> dim_size = {1, 7, 7, 9};
      MLUOP_CHECK(mluOpSetTensorDescriptor(output_desc_, MLUOP_LAYOUT_NHWC,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           dim_size.data()));
    }
    if (input) {
      size_t i_ele_num = 1 * 7 * 7 * 9;
      size_t i_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t i_bytes = i_ele_num * i_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&input_, i_bytes));
    }
    if (grid) {
      size_t g_ele_num = 1 * 7 * 7 * 2;
      size_t g_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t g_bytes = g_ele_num * g_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&grid_, g_bytes));
    }
    if (output) {
      size_t o_ele_num = 1 * 7 * 7 * 9;
      size_t o_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t o_bytes = o_ele_num * o_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&output_, o_bytes));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpRoiCropForward(
        handle_, input_desc_, input_, grid_desc_, grid_, output_desc_, output_);
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
    if (input_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_desc_));
      input_desc_ = NULL;
    }
    if (grid_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grid_desc_));
      grid_desc_ = NULL;
    }
    if (output_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
      output_desc_ = NULL;
    }
    if (input_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_));
      input_ = NULL;
    }
    if (grid_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grid_));
      grid_ = NULL;
    }
    if (output_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_));
      output_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t input_desc_ = NULL;
  mluOpTensorDescriptor_t grid_desc_ = NULL;
  mluOpTensorDescriptor_t output_desc_ = NULL;
  void* input_ = NULL;
  void* grid_ = NULL;
  void* output_ = NULL;
};

TEST_F(roi_crop_forward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in roi_crop_forward";
  }
}

TEST_F(roi_crop_forward, BAD_PARAM_input_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in roi_crop_forward";
  }
}

TEST_F(roi_crop_forward, BAD_PARAM_input_null) {
  try {
    setParam(true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in roi_crop_forward";
  }
}

TEST_F(roi_crop_forward, BAD_PARAM_grid_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in roi_crop_forward";
  }
}
TEST_F(roi_crop_forward, BAD_PARAM_grid_null) {
  try {
    setParam(true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in roi_crop_forward";
  }
}

TEST_F(roi_crop_forward, BAD_PARAM_output_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in roi_crop_forward";
  }
}

TEST_F(roi_crop_forward, BAD_PARAM_output_null) {
  try {
    setParam(true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in roi_crop_forward";
  }
}
}  // namespace mluopapitest
