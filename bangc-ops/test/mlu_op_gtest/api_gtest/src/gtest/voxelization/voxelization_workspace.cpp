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

namespace mluopapitest {
class voxelization_workspace : public testing::Test {
 public:
  void setParam(bool handle, bool points_desc, bool voxel_size_desc,
                bool coors_range_desc, bool voxels_desc, bool coors_desc,
                bool num_points_per_voxel_desc, bool voxel_num_desc,
                bool size) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (points_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&points_desc_));
      std::vector<int> points_dims = {1, 2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(points_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           points_dims.data()));
    }
    if (voxel_size_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&voxel_size_desc_));
      std::vector<int> voxel_size_dims = {3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(voxel_size_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 1,
                                           voxel_size_dims.data()));
    }
    if (coors_range_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&coors_range_desc_));
      std::vector<int> coors_range_dims = {6};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          coors_range_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 1,
          coors_range_dims.data()));
    }
    if (voxels_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&voxels_desc_));
      std::vector<int> voxels_dims = {5, 4, 2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(voxels_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           voxels_dims.data()));
    }
    if (coors_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&coors_desc_));
      std::vector<int> coors_dims = {5, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(coors_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           coors_dims.data()));
    }
    if (num_points_per_voxel_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&num_points_per_voxel_desc_));
      std::vector<int> num_points_per_voxel_dims = {5};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          num_points_per_voxel_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 1,
          num_points_per_voxel_dims.data()));
    }
    if (voxel_num_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&voxel_num_desc_));
      std::vector<int> voxel_num_dims = {1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(voxel_num_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 1,
                                           voxel_num_dims.data()));
    }
    if (size) {
      size_t size_temp = 0;
      size_ = &size_temp;
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpGetVoxelizationWorkspaceSize(
        handle_, points_desc_, voxel_size_desc_, coors_range_desc_, max_points_,
        max_voxels_, NDim_, deterministic_, voxels_desc_, coors_desc_,
        num_points_per_voxel_desc_, voxel_num_desc_, size_);
    destroy();
    return status;
  }

 protected:
  void destroy() {
    if (handle_) {
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = NULL;
    }
    if (points_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(points_desc_));
      points_desc_ = NULL;
    }
    if (voxel_size_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(voxel_size_desc_));
      voxel_size_desc_ = NULL;
    }
    if (coors_range_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(coors_range_desc_));
      coors_range_desc_ = NULL;
    }
    if (voxels_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(voxels_desc_));
      voxels_desc_ = NULL;
    }
    if (coors_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(coors_desc_));
      coors_desc_ = NULL;
    }
    if (num_points_per_voxel_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(num_points_per_voxel_desc_));
      num_points_per_voxel_desc_ = NULL;
    }
    if (voxel_num_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(voxel_num_desc_));
      voxel_num_desc_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t points_desc_ = NULL;
  mluOpTensorDescriptor_t voxel_size_desc_ = NULL;
  mluOpTensorDescriptor_t coors_range_desc_ = NULL;
  int max_points_ = 4;
  int max_voxels_ = 5;
  int NDim_ = 3;
  bool deterministic_ = true;
  mluOpTensorDescriptor_t voxels_desc_ = NULL;
  mluOpTensorDescriptor_t coors_desc_ = NULL;
  mluOpTensorDescriptor_t num_points_per_voxel_desc_ = NULL;
  mluOpTensorDescriptor_t voxel_num_desc_ = NULL;
  size_t* size_ = NULL;
};

TEST_F(voxelization_workspace, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in voxelization";
  }
}

TEST_F(voxelization_workspace, BAD_PARAM_points_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in voxelization";
  }
}

TEST_F(voxelization_workspace, BAD_PARAM_voxel_size_desc_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in voxelization";
  }
}

TEST_F(voxelization_workspace, BAD_PARAM_coors_range_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in voxelization";
  }
}

TEST_F(voxelization_workspace, BAD_PARAM_voxels_desc_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in voxelization";
  }
}

TEST_F(voxelization_workspace, BAD_PARAM_coors_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in voxelization";
  }
}

TEST_F(voxelization_workspace, BAD_PARAM_num_points_per_voxel_desc_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in voxelization";
  }
}

TEST_F(voxelization_workspace, BAD_PARAM_voxel_num_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in voxelization";
  }
}

TEST_F(voxelization_workspace, BAD_PARAM_size_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in voxelization";
  }
}
}  // namespace mluopapitest
