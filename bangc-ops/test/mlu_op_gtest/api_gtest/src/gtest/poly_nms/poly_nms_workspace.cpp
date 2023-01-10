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
class poly_nms_workspace : public testing::Test {
 public:
  void setParam(bool handle, bool boxes_desc, bool size) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (boxes_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&boxes_desc_));
      std::vector<int> dim_size = {2, 9};
      MLUOP_CHECK(mluOpSetTensorDescriptor(boxes_desc_, MLUOP_LAYOUT_NHWC,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           dim_size.data()));
    }
    if (size) {
      size_t size_temp = 0;
      size_ = &size_temp;
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status =
        mluOpGetPolyNmsWorkspaceSize(handle_, boxes_desc_, size_);
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
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t boxes_desc_ = NULL;
  size_t* size_ = NULL;
};

TEST_F(poly_nms_workspace, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in poly_nms";
  }
}

TEST_F(poly_nms_workspace, BAD_PARAM_boxes_desc_null) {
  try {
    setParam(true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in poly_nms";
  }
}

TEST_F(poly_nms_workspace, BAD_PARAM_size_null) {
  try {
    setParam(true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in poly_nms";
  }
}
}  // namespace mluopapitest
