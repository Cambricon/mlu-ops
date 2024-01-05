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
class three_nn_forward_workspace : public testing::Test {
 public:
  void setParam(bool handle, bool known_desc, bool workspace_size) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (known_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&known_desc_));
      std::vector<int> known_dims = {1, 10, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(known_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           known_dims.data()));
    }
    if (workspace_size) {
      size_t size_temp = 0;
      workspace_size_ = &size_temp;
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpGetThreeNNForwardWorkspaceSize(
        handle_, known_desc_, workspace_size_);
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
    if (known_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(known_desc_));
      known_desc_ = NULL;
    }
    if (workspace_size_) {
      workspace_size_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t known_desc_ = NULL;
  size_t *workspace_size_ = NULL;
};

TEST_F(three_nn_forward_workspace, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in three_nn_forward";
  }
}

TEST_F(three_nn_forward_workspace, BAD_PARAM_known_desc_null) {
  try {
    setParam(true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in three_nn_forward";
  }
}

TEST_F(three_nn_forward_workspace, BAD_PARAM_workspace_size_null) {
  try {
    setParam(true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in three_nn_forward";
  }
}
}  // namespace mluopapitest
