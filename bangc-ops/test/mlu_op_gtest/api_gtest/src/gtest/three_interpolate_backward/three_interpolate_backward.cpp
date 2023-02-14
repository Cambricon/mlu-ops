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
class three_interprolate_backward : public testing::Test {
 public:
  void setParam(bool handle, bool grad_output_desc, bool grad_output,
                bool indices_desc, bool indices, bool weight_desc, bool weight,
                bool grad_features_desc, bool grad_features) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (grad_output_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_output_desc_));
      std::vector<int> grad_output_shape = {1, 2, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_output_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 3,
          grad_output_shape.data()));
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
    if (grad_features_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_features_desc_));
      std::vector<int> grad_features_shape = {1, 2, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_features_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 3,
          grad_features_shape.data()));
    }
    if (grad_output) {
      size_t grad_output_ele_num = 1 * 2 * 4;
      size_t grad_output_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t grad_output_bytes = grad_output_ele_num * grad_output_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&grad_output_, grad_output_bytes));
    }
    if (indices) {
      size_t indices_ele_num = 1 * 4 * 3;
      size_t indices_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      size_t indices_bytes = indices_ele_num * indices_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&indices_, indices_bytes));
    }
    if (weight) {
      size_t weight_ele_num = 1 * 4 * 3;
      size_t weight_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      size_t weight_bytes = weight_ele_num * weight_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&weight_, weight_bytes));
    }
    if (grad_features) {
      size_t grad_features_ele_num = 1 * 2 * 4;
      size_t grad_features_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      size_t grad_features_bytes =
          grad_features_ele_num * grad_features_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&grad_features_, grad_features_bytes));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpThreeInterpolateBackward(
        handle_, grad_output_desc_, grad_output_, indices_desc_, indices_,
        weight_desc_, weight_, grad_features_desc_, grad_features_);
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
    if (indices_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indices_desc_));
      indices_desc_ = NULL;
    }
    if (weight_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(weight_desc_));
      weight_desc_ = NULL;
    }
    if (grad_features_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_features_desc_));
      grad_features_desc_ = NULL;
    }
    if (grad_output_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_output_));
      grad_output_ = NULL;
    }
    if (indices_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(indices_));
      indices_ = NULL;
    }
    if (weight_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(weight_));
      weight_ = NULL;
    }
    if (grad_features_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_features_));
      grad_features_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t grad_output_desc_ = NULL;
  mluOpTensorDescriptor_t indices_desc_ = NULL;
  mluOpTensorDescriptor_t weight_desc_ = NULL;
  mluOpTensorDescriptor_t grad_features_desc_ = NULL;
  void* grad_output_ = NULL;
  void* indices_ = NULL;
  void* weight_ = NULL;
  void* grad_features_ = NULL;
};

TEST_F(three_interprolate_backward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_backward";
  }
}

TEST_F(three_interprolate_backward, BAD_PARAM_grad_output_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_backward";
  }
}

TEST_F(three_interprolate_backward, BAD_PARAM_grad_output_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_backward";
  }
}

TEST_F(three_interprolate_backward, BAD_PARAM_indices_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_backward";
  }
}

TEST_F(three_interprolate_backward, BAD_PARAM_indices_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_backward";
  }
}

TEST_F(three_interprolate_backward, BAD_PARAM_weight_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_backward";
  }
}

TEST_F(three_interprolate_backward, BAD_PARAM_weight_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_backward";
  }
}

TEST_F(three_interprolate_backward, BAD_PARAM_grad_features_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_backward";
  }
}

TEST_F(three_interprolate_backward, BAD_PARAM_grad_features_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in three_interprolate_backward";
  }
}
}  // namespace mluopapitest
