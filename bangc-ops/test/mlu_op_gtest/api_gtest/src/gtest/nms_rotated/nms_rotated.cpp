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
#include "core/context.h"
#include "core/tensor.h"
#include "core/logging.h"
#include "gtest/gtest.h"
#include "mlu_op.h"

namespace mluopapitest {
class nms_rotated : public testing::Test {
 public:
  void setParam(bool handle, bool boxes_desc, bool boxes, bool scores_desc,
                bool scores, bool workspace, bool output_desc, bool output,
                bool result_num) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (boxes_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&boxes_desc_));
      std::vector<int> dim_size = {2, 5};
      MLUOP_CHECK(mluOpSetTensorDescriptor(boxes_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           dim_size.data()));
    }
    if (boxes) {
      size_t b_ele_num = 2 * 5;
      size_t b_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t b_bytes = b_ele_num * b_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&boxes_, b_bytes));
    }
    if (scores_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&scores_desc_));
      std::vector<int> dim_size = {2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(scores_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 1,
                                           dim_size.data()));
    }
    if (scores) {
      size_t s_ele_num = 2;
      size_t s_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t s_bytes = s_ele_num * s_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&scores_, s_bytes));
    }
    if (workspace) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_));
    }
    if (output_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));
      std::vector<int> dim_size = {2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(output_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 1,
                                           dim_size.data()));
    }
    if (output) {
      size_t o_ele_num = 2;
      size_t o_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      size_t o_bytes = o_ele_num * o_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&output_, o_bytes));
    }
    if (result_num) {
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(&result_num_, mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status =
        mluOpNmsRotated(handle_, iou_threshold_, boxes_desc_, boxes_,
                        scores_desc_, scores_, workspace_, workspace_size_,
                        output_desc_, output_, (int32_t*)result_num_);
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
    if (boxes_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(boxes_desc_));
      boxes_desc_ = NULL;
    }
    if (boxes_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(boxes_));
      boxes_ = NULL;
    }
    if (scores_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(scores_desc_));
      scores_desc_ = NULL;
    }
    if (scores_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(scores_));
      scores_ = NULL;
    }
    if (workspace_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = NULL;
    }
    if (output_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
      output_desc_ = NULL;
    }
    if (output_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_));
      output_ = NULL;
    }
    if (result_num_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(result_num_));
      result_num_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  size_t workspace_size_ = 10;
  mluOpTensorDescriptor_t boxes_desc_ = NULL;
  void* boxes_ = NULL;
  mluOpTensorDescriptor_t scores_desc_ = NULL;
  void* scores_ = NULL;
  float iou_threshold_ = 0.5;
  void* workspace_ = NULL;
  mluOpTensorDescriptor_t output_desc_ = NULL;
  void* output_ = NULL;
  void* result_num_ = NULL;
};

TEST_F(nms_rotated, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in nms_rotated";
  }
}

TEST_F(nms_rotated, BAD_PARAM_boxes_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in nms_rotated";
  }
}

TEST_F(nms_rotated, BAD_PARAM_boxes_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in nms_rotated";
  }
}

TEST_F(nms_rotated, BAD_PARAM_scores_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in nms_rotated";
  }
}

TEST_F(nms_rotated, BAD_PARAM_scores_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in nms_rotated";
  }
}

TEST_F(nms_rotated, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in nms_rotated";
  }
}

TEST_F(nms_rotated, BAD_PARAM_output_desc_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in nms_rotated";
  }
}

TEST_F(nms_rotated, BAD_PARAM_output_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in nms_rotated";
  }
}

TEST_F(nms_rotated, BAD_PARAM_result_num_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in nms_rotated";
  }
}
}  // namespace mluopapitest
