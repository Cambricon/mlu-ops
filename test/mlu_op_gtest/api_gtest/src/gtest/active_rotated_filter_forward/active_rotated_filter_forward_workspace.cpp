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
class active_rotated_filter_forward_workspace : public testing::Test {
 public:
  void setParam(bool handle, bool input_desc, bool workspace_size) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (input_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
      std::vector<int> input_dims{256, 200};
      MLUOP_CHECK(mluOpSetTensorDescriptor(input_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, input_dims.size(),
                                           input_dims.data()));
    }

    if (workspace_size) {
      size_t size_temp;
      workspace_size_ = &size_temp;
    }
  }
  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpGetActiveRotatedFilterForwardWorkspaceSize(
        handle_, input_desc_, workspace_size_);
    destroy();
    return status;
  }

  void destroy() {
    try {
      VLOG(4) << "Destroy parameters";
      if (handle_) {
        CNRT_CHECK(cnrtQueueSync(handle_->queue));
        VLOG(4) << "Destroy handle_";
        MLUOP_CHECK(mluOpDestroy(handle_));
        handle_ = nullptr;
      }
      if (input_desc_) {
        VLOG(4) << "Destroy input_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_desc_));
        input_desc_ = nullptr;
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in active_rotated_filter_forward_workspace";
    }
  }

  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t input_desc_ = nullptr;
  size_t *workspace_size_ = nullptr;
};

TEST_F(active_rotated_filter_forward_workspace, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in active_rotated_filter_forward_workspace";
  }
}

TEST_F(active_rotated_filter_forward_workspace, BAD_PARAM_input_desc_null) {
  try {
    setParam(true, false, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in active_rotated_filter_forward_workspace";
  }
}

TEST_F(active_rotated_filter_forward_workspace, BAD_PARAM_workspace_size_null) {
  try {
    setParam(true, true, false);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in active_rotated_filter_forward_workspace";
  }
}
}  // namespace mluopapitest
