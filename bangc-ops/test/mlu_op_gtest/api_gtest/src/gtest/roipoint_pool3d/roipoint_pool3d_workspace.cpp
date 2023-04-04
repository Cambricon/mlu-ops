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
class roipoint_pool3d_workspace : public testing::Test {
 public:
  void setParam(bool handle, bool points_desc, bool point_features_desc,
                bool boxes3d_desc, bool pooled_features_desc,
                bool pooled_empty_flag_desc, bool size) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (points_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&points_desc_));
      std::vector<int> points_dims{1, 1, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(points_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           points_dims.data()));
    }

    if (point_features_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&point_features_desc_));
      std::vector<int> point_features_dims{1, 1, 2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          point_features_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 3,
          point_features_dims.data()));
    }

    if (boxes3d_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&boxes3d_desc_));
      std::vector<int> boxes3d_dims{1, 1, 7};
      MLUOP_CHECK(mluOpSetTensorDescriptor(boxes3d_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           boxes3d_dims.data()));
    }

    if (pooled_features_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&pooled_features_desc_));
      std::vector<int> pooled_features_dims{1, 1, 1, 5};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          pooled_features_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 4,
          pooled_features_dims.data()));
    }

    if (pooled_empty_flag_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&pooled_empty_flag_desc_));
      std::vector<int> pooled_empty_flag_dims{1, 1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          pooled_empty_flag_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 2,
          pooled_empty_flag_dims.data()));
    }

    if (size) {
      size_t size_temp;
      size_ = &size_temp;
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpGetRoiPointPool3dWorkspaceSize(
        handle_, batch_size_, pts_num_, boxes_num_, feature_in_len_,
        sampled_pts_num_, points_desc_, point_features_desc_, boxes3d_desc_,
        pooled_features_desc_, pooled_empty_flag_desc_, size_);
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
    if (points_desc_) {
      VLOG(4) << "Destroy points_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(points_desc_));
      points_desc_ = nullptr;
    }
    if (point_features_desc_) {
      VLOG(4) << "Destroy point_features_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(point_features_desc_));
      point_features_desc_ = nullptr;
    }
    if (boxes3d_desc_) {
      VLOG(4) << "Destroy boxes3d_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(boxes3d_desc_));
      boxes3d_desc_ = nullptr;
    }
    if (pooled_features_desc_) {
      VLOG(4) << "Destroy pooled_features_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(pooled_features_desc_));
      pooled_features_desc_ = nullptr;
    }
    if (pooled_empty_flag_desc_) {
      VLOG(4) << "Destroy pooled_empty_flag_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(pooled_empty_flag_desc_));
      pooled_empty_flag_desc_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  int batch_size_ = 1;
  int pts_num_ = 1;
  int boxes_num_ = 1;
  int feature_in_len_ = 1;
  int sampled_pts_num_ = 1;
  mluOpTensorDescriptor_t points_desc_ = nullptr;
  mluOpTensorDescriptor_t point_features_desc_ = nullptr;
  mluOpTensorDescriptor_t boxes3d_desc_ = nullptr;
  mluOpTensorDescriptor_t pooled_features_desc_ = nullptr;
  mluOpTensorDescriptor_t pooled_empty_flag_desc_ = nullptr;
  size_t *size_ = nullptr;
};

TEST_F(roipoint_pool3d_workspace, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roipoint_pool3d_workspace";
  }
}

TEST_F(roipoint_pool3d_workspace, BAD_PARAM_points_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roipoint_pool3d_workspace";
  }
}

TEST_F(roipoint_pool3d_workspace, BAD_PARAM_point_features_desc_null) {
  try {
    setParam(true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roipoint_pool3d_workspace";
  }
}

TEST_F(roipoint_pool3d_workspace, BAD_PARAM_boxes3d_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roipoint_pool3d_workspace";
  }
}

TEST_F(roipoint_pool3d_workspace, BAD_PARAM_pooled_features_desc_null) {
  try {
    setParam(true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roipoint_pool3d_workspace";
  }
}

TEST_F(roipoint_pool3d_workspace, BAD_PARAM_pooled_empty_flag_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roipoint_pool3d_workspace";
  }
}

TEST_F(roipoint_pool3d_workspace, BAD_PARAM_size_null) {
  try {
    setParam(true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roipoint_pool3d_workspace";
  }
}
}  // namespace mluopapitest
