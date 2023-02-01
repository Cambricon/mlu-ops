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
class roiaware_pool3d_forward_workspace : public testing::Test {
 public:
  void setParam(bool handle, bool pts_feature_desc, bool workspace_size) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (pts_feature_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&pts_feature_desc_));
      std::vector<int> pts_feature_dims{3, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          pts_feature_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
          pts_feature_dims.data()));
    }
    if (workspace_size) {
      size_t size_temp;
      workspace_size_ = &size_temp;
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpGetRoiawarePool3dForwardWorkspaceSize(
        handle_, rois_desc_, pts_desc_, pts_feature_desc_, workspace_size_);
    destroy();
    return status;
  }

 protected:
  void destroy() {
    if (handle_) {
      CNRT_CHECK(cnrtQueueSync(handle_->queue));
      VLOG(4) << "Destroy handle";
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = nullptr;
    }

    if (rois_desc_) {
      VLOG(4) << "Destroy rois_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(rois_desc_));
      rois_desc_ = nullptr;
    }

    if (pts_desc_) {
      VLOG(4) << "Destroy pts_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(pts_desc_));
      pts_desc_ = nullptr;
    }

    if (pts_feature_desc_) {
      VLOG(4) << "Destroy pts_feature_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(pts_feature_desc_));
      pts_feature_desc_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t rois_desc_ = nullptr;
  mluOpTensorDescriptor_t pts_desc_ = nullptr;
  size_t *workspace_size_ = nullptr;
  mluOpTensorDescriptor_t pts_feature_desc_ = nullptr;
};

TEST_F(roiaware_pool3d_forward_workspace, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward_workspace";
  }
}

TEST_F(roiaware_pool3d_forward_workspace, BAD_PARAM_pts_feature_desc_null) {
  try {
    setParam(true, false, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward_workspace";
  }
}

TEST_F(roiaware_pool3d_forward_workspace, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, false);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward_workspace";
  }
}

}  // namespace mluopapitest
