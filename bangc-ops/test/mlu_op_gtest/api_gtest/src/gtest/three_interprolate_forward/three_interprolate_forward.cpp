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
class three_interprolate_forward : public testing::Test {
 public:
  void setParam(bool handle, bool features_desc, bool features,
                bool indices_desc, bool indices, bool weight_desc, bool weight,
                bool output_desc, bool output) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (features_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&features_desc_));
      std::vector<int> features_shape = {1, 2, 5};
      MLUOP_CHECK(mluOpSetTensorDescriptor(features_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           features_shape.data()));
    }
    if (indices_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&indices_desc_));
      std::vector<int> indices_shape = {1, 4, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(indices_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 3,
                                           indices_shape.data()));
    }
    if (weight_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&weight_desc_));
      std::vector<int> weight_shape = {1, 4, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(weight_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           weight_shape.data()));
    }
    if (output_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));
      std::vector<int> output_shape = {1, 2, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(output_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           output_shape.data()));
    }
    if (features) {
      size_t f_ele_num = 1 * 2 * 5;
      size_t f_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t f_bytes = f_ele_num * f_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&features_, f_bytes));
    }
    if (indices) {
      size_t i_ele_num = 1 * 4 * 3;
      size_t i_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      size_t i_bytes = i_ele_num * i_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&indices_, i_bytes));
    }
    if (weight) {
      size_t w_ele_num = 1 * 4 * 3;
      size_t w_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      size_t w_bytes = w_ele_num * w_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&weight_, w_bytes));
    }
    if (output) {
      size_t o_ele_num = 1 * 2 * 4;
      size_t o_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      size_t o_bytes = o_ele_num * o_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&output_, o_bytes));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpThreeInterpolateForward(
        handle_, features_desc_, features_, indices_desc_, indices_,
        weight_desc_, weight_, output_desc_, output_);
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
    if (features_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(features_desc_));
      features_desc_ = NULL;
    }
    if (indices_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indices_desc_));
      indices_desc_ = NULL;
    }
    if (weight_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(weight_desc_));
      weight_desc_ = NULL;
    }
    if (output_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
      output_desc_ = NULL;
    }
    if (features_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(features_));
      features_ = NULL;
    }
    if (indices_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(indices_));
      indices_ = NULL;
    }
    if (weight_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(weight_));
      weight_ = NULL;
    }
    if (output_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_));
      output_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t features_desc_ = NULL;
  mluOpTensorDescriptor_t indices_desc_ = NULL;
  mluOpTensorDescriptor_t weight_desc_ = NULL;
  mluOpTensorDescriptor_t output_desc_ = NULL;
  void* features_ = NULL;
  void* indices_ = NULL;
  void* weight_ = NULL;
  void* output_ = NULL;
};

TEST_F(three_interprolate_forward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_forward";
  }
}

TEST_F(three_interprolate_forward, BAD_PARAM_features_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_forward";
  }
}

TEST_F(three_interprolate_forward, BAD_PARAM_features_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_forward";
  }
}

TEST_F(three_interprolate_forward, BAD_PARAM_indices_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_forward";
  }
}

TEST_F(three_interprolate_forward, BAD_PARAM_indices_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_forward";
  }
}

TEST_F(three_interprolate_forward, BAD_PARAM_weight_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_forward";
  }
}

TEST_F(three_interprolate_forward, BAD_PARAM_weight_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_forward";
  }
}

TEST_F(three_interprolate_forward, BAD_PARAM_output_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_forward";
  }
}

TEST_F(three_interprolate_forward, BAD_PARAM_output_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_forward";
  }
}
}  // namespace mluopapitest
