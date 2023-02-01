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
class roiaware_pool3d_forward : public testing::Test {
 public:
  void setParam(bool handle, bool rois_desc, bool rois, bool pts_desc, bool pts,
                bool pts_feature_desc, bool pts_feature, bool worksapce,
                bool argmax_desc, bool argmax, bool pts_idx_of_voxels_desc,
                bool pts_idx_of_voxels, bool pooled_features_desc,
                bool pooled_features) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (rois_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&rois_desc_));
      std::vector<int> rois_dims{2, 7};
      MLUOP_CHECK(mluOpSetTensorDescriptor(rois_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           rois_dims.data()));
    }

    if (rois) {
      if (rois_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&rois_, mluOpGetTensorElementNum(rois_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&rois_, 64 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    if (pts_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&pts_desc_));
      std::vector<int> pts_dims{3, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(pts_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           pts_dims.data()));
    }

    if (pts) {
      if (pts_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&pts_, mluOpGetTensorElementNum(pts_desc_) *
                                  mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&pts_, 64 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    if (pts_feature_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&pts_feature_desc_));
      std::vector<int> pts_feature_dims{3, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          pts_feature_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
          pts_feature_dims.data()));
    }

    if (pts_feature) {
      if (pts_feature_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&pts_feature_,
                               mluOpGetTensorElementNum(pts_feature_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&pts_feature_,
                               64 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    if (pooled_features_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&pooled_features_desc_));
      std::vector<int> pooled_features_dims{2, 1, 2, 3, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          pooled_features_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 5,
          pooled_features_dims.data()));
    }

    if (pooled_features) {
      if (pooled_features_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&pooled_features_,
                               mluOpGetTensorElementNum(pooled_features_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&pooled_features_,
                               64 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    if (argmax_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&argmax_desc_));
      std::vector<int> argmax_dims{2, 1, 2, 3, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(argmax_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 5,
                                           argmax_dims.data()));
    }

    if (argmax) {
      if (argmax_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&argmax_, mluOpGetTensorElementNum(argmax_desc_) *
                                     mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&argmax_, 64 * mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      }
    }

    if (pts_idx_of_voxels_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&pts_idx_of_voxels_desc_));
      std::vector<int> pts_idx_of_voxels_dims{2, 1, 2, 3, 5};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          pts_idx_of_voxels_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 5,
          pts_idx_of_voxels_dims.data()));
    }

    if (pts_idx_of_voxels) {
      if (pts_idx_of_voxels_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&pts_idx_of_voxels_,
                       mluOpGetTensorElementNum(pts_idx_of_voxels_desc_) *
                           mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&pts_idx_of_voxels_,
                               64 * mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      }
    }

    if (worksapce) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpRoiawarePool3dForward(
        handle_, pool_method_, boxes_num_, pts_num_, channels_, rois_desc_,
        rois_, pts_desc_, pts_, pts_feature_desc_, pts_feature_, workspace_,
        workspace_size_, max_pts_each_voxel_, out_x_, out_y_, out_z_,
        argmax_desc_, argmax_, pts_idx_of_voxels_desc_, pts_idx_of_voxels_,
        pooled_features_desc_, pooled_features_);
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

    if (rois_) {
      VLOG(4) << "Destroy rois";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(rois_));
      rois_ = nullptr;
    }

    if (pts_desc_) {
      VLOG(4) << "Destroy pts_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(pts_desc_));
      pts_desc_ = nullptr;
    }

    if (pts_) {
      VLOG(4) << "Destroy pts";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(pts_));
      pts_ = nullptr;
    }

    if (pts_feature_desc_) {
      VLOG(4) << "Destroy pts_feature_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(pts_feature_desc_));
      pts_feature_desc_ = nullptr;
    }

    if (pts_feature_) {
      VLOG(4) << "Destroy pts_feature";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(pts_feature_));
      pts_feature_ = nullptr;
    }

    if (workspace_) {
      VLOG(4) << "Destroy workspace";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = nullptr;
    }

    if (argmax_desc_) {
      VLOG(4) << "Destroy argmax_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(argmax_desc_));
      argmax_desc_ = nullptr;
    }

    if (argmax_) {
      VLOG(4) << "Destroy argmax";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(argmax_));
      argmax_ = nullptr;
    }

    if (pts_idx_of_voxels_desc_) {
      VLOG(4) << "Destroy pts_idx_of_voxels_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(pts_idx_of_voxels_desc_));
      pts_idx_of_voxels_desc_ = nullptr;
    }

    if (pts_idx_of_voxels_) {
      VLOG(4) << "Destroy pts_idx_of_voxels";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(pts_idx_of_voxels_));
      pts_idx_of_voxels_ = nullptr;
    }

    if (pooled_features_desc_) {
      VLOG(4) << "Destroy pooled_features_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(pooled_features_desc_));
      pooled_features_desc_ = nullptr;
    }

    if (pooled_features_) {
      VLOG(4) << "Destroy pooled_features";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(pooled_features_));
      pooled_features_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  int pool_method_ = 0;
  int boxes_num_ = 2;
  int pts_num_ = 3;
  int channels_ = 4;
  mluOpTensorDescriptor_t rois_desc_ = nullptr;
  void *rois_ = nullptr;
  mluOpTensorDescriptor_t pts_desc_ = nullptr;
  void *pts_ = nullptr;
  mluOpTensorDescriptor_t pts_feature_desc_ = nullptr;
  void *pts_feature_ = nullptr;
  void *workspace_ = nullptr;
  size_t workspace_size_ = 64;
  int max_pts_each_voxel_ = 5;
  int out_x_ = 1;
  int out_y_ = 2;
  int out_z_ = 3;
  mluOpTensorDescriptor_t argmax_desc_ = nullptr;
  void *argmax_ = nullptr;
  mluOpTensorDescriptor_t pts_idx_of_voxels_desc_ = nullptr;
  void *pts_idx_of_voxels_ = nullptr;
  mluOpTensorDescriptor_t pooled_features_desc_ = nullptr;
  void *pooled_features_ = nullptr;
};

TEST_F(roiaware_pool3d_forward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true, true,
             true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward";
  }
}

TEST_F(roiaware_pool3d_forward, BAD_PARAM_rois_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true, true,
             true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward";
  }
}

TEST_F(roiaware_pool3d_forward, BAD_PARAM_rois_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true, true,
             true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward";
  }
}

TEST_F(roiaware_pool3d_forward, BAD_PARAM_pts_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true, true,
             true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward";
  }
}

TEST_F(roiaware_pool3d_forward, BAD_PARAM_pts_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true, true,
             true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward";
  }
}

TEST_F(roiaware_pool3d_forward, BAD_PARAM_pts_feature_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true, true,
             true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward";
  }
}

TEST_F(roiaware_pool3d_forward, BAD_PARAM_pts_feature_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true, true,
             true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward";
  }
}

TEST_F(roiaware_pool3d_forward, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true, true,
             true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward";
  }
}

TEST_F(roiaware_pool3d_forward, BAD_PARAM_argmax_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true, true,
             true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward";
  }
}

TEST_F(roiaware_pool3d_forward, BAD_PARAM_argmax_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false, true,
             true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward";
  }
}

TEST_F(roiaware_pool3d_forward, BAD_PARAM_pts_idx_of_voxels_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, false,
             true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward";
  }
}

TEST_F(roiaware_pool3d_forward, BAD_PARAM_pts_idx_of_voxels_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             false, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward";
  }
}

TEST_F(roiaware_pool3d_forward, BAD_PARAM_pooled_features_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, false, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward";
  }
}

TEST_F(roiaware_pool3d_forward, BAD_PARAM_pooled_features_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, false);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in roiaware_pool3d_forward";
  }
}
}  // namespace mluopapitest
