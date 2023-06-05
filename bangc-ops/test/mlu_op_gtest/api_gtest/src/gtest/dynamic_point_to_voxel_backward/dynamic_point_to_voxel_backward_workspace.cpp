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
class dynamic_point_to_voxel_backward_workspace : public testing::Test {
 public:
  void setParam(bool handle, bool grad_voxel_feats_desc, 
		bool feats_desc, bool voxel_feats_desc,
		bool point2voxel_map_desc, bool voxel_points_count_desc,
                bool voxel_num_desc, bool workspace_size) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (grad_voxel_feats_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_voxel_feats_desc_));
      std::vector<int> grad_voxel_feats_desc_dims{4, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(feats_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           feats_desc_dims.data()));
    }

    if (feats_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&feats_desc_));
      std::vector<int> feats_desc_dims{4, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(feats_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           feats_desc_dims.data()));
    }

    if (voxel_feats_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&voxel_feats_desc_));
      std::vector<int> voxel_feats_desc_dims{4, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(voxel_feats_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           voxel_feats_desc_dims.data()));
    }

    if (point2voxel_map_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&point2voxel_map_desc_));
      std::vector<int> point2voxel_map_desc_dims{4, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(point2voxel_map_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           point2voxel_map_desc_dims.data()));
    }
    if (voxel_points_count_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&voxel_points_count_desc_));
      std::vector<int> voxel_points_count_desc_dims{4, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(voxel_points_count_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           voxel_points_count_desc_dims.data()));
    }
    if (voxel_num_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&voxel_num_desc_));
      std::vector<int> voxel_num_desc_dims{4, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(voxel_num_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           voxel_num_desc_dims.data()));
    }
    if (workspace_size) {
      workspace_size_ = &workspace_size__;
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpGetDynamicPointToVoxelBackwardWorkspaceSize(
        handle_, reduce_type_, grad_voxel_feats_desc_, feats_desc_, 
	voxel_feats_desc_, point2voxel_map_desc_, voxel_points_count_desc_,
	voxel_num_desc_, voxel_num_desc_, workspace_size_);
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

    if (grad_voxel_feats_desc_) {
      VLOG(4) << "Destroy grad_voxel_feats_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_voxel_feats_desc_));
      grad_voxel_feats_desc_ = nullptr;
    }

    if (feats_desc_) {
      VLOG(4) << "Destroy feats_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(feats_desc_));
      feats_desc_ = nullptr;
    }

    if (voxel_feats_desc_) {
      VLOG(4) << "Destroy voxel_feats_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(voxel_feats_desc_));
      voxel_feats_desc_ = nullptr;
    }

    if (point2voxel_map_desc_) {
      VLOG(4) << "Destroy point2voxel_map_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(point2voxel_map_desc_));
      point2voxel_map_desc_ = nullptr;
    }

    if (voxel_points_count_desc_) {
      VLOG(4) << "Destroy voxel_points_count_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(voxel_points_count_desc_));
      voxel_points_count_desc_ = nullptr;
    }

    if (voxel_num_desc_) {
      VLOG(4) << "Destroy voxel_num_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(voxel_num_desc_));
      voxel_num_desc_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpReduceMode_t reduce_type_ = MLUOP_REDUCE_DMAX;
  mluOpTensorDescriptor_t grad_voxel_feats_desc_ = nullptr;
  mluOpTensorDescriptor_t feats_desc_ = nullptr;
  mluOpTensorDescriptor_t voxel_feats_desc_ = nullptr;
  mluOpTensorDescriptor_t point2voxel_map_desc_ = nullptr;
  mluOpTensorDescriptor_t voxel_points_count_desc_ = nullptr;
  mluOpTensorDescriptor_t voxel_num_desc_ = nullptr;
  size_t workspace_size_ = 10;
  size_t *workspace_size_ = nullptr;
};

TEST_F(dynamic_point_to_voxel_backward_workspace, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_backward_workspace";
  }
}

TEST_F(dynamic_point_to_voxel_backward_workspace, BAD_PARAM_grad_voxel_feats_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_backward_workspace";
  }
}

TEST_F(dynamic_point_to_voxel_backward_workspace, BAD_PARAM_feats_desc_null) {
  try {
    setParam(true, true, false, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_backward_workspace";
  }
}

TEST_F(dynamic_point_to_voxel_backward_workspace, BAD_PARAM_voxel_feats_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_backward_workspace";
  }
}

TEST_F(dynamic_point_to_voxel_backward_workspace, BAD_PARAM_point2voxel_map_desc_null) {
  try {
    setParam(true, true, true, true, false, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_backward_workspace";
  }
}

TEST_F(dynamic_point_to_voxel_backward_workspace, BAD_PARAM_voxel_points_count_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_backward_workspace";
  }
}

TEST_F(dynamic_point_to_voxel_backward_workspace, BAD_PARAM_voxel_num_desc_null) {
  try {
    setParam(true, true, true, true, true, true, false, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_backward_workspace";
  }
}

TEST_F(dynamic_point_to_voxel_backward_workspace,
       BAD_PARAM_workspace_size_null) {
  try {
    setParam(true, true, true, true, true, true, true, false);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_backward_workspace";
  }
}
}  // namespace mluopapitest
