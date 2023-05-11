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
class dynamic_point_to_voxel_forward : public testing::Test {
 public:
  void setParam(bool handle, bool feats_desc, bool feats, bool coors_desc,
                bool coors, bool voxel_feats_desc, bool voxel_feats,
                bool voxel_coors_desc, bool voxel_coors,
                bool point2voxel_map_desc, bool point2voxel_map,
                bool voxel_points_count_desc, bool voxel_points_count,
                bool voxel_num_desc, bool voxel_num, bool workspace) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (feats_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&feats_desc_));
      std::vector<int> feats_desc_dims{4, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(feats_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           feats_desc_dims.data()));
    }

    if (feats) {
      if (feats_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&feats_, MLUOP_DTYPE_FLOAT *
                                    mluOpGetTensorElementNum(feats_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&feats_, MLUOP_DTYPE_FLOAT * 2));
      }
    }

    if (coors_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&coors_desc_));
      std::vector<int> coors_desc_dims{4, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(coors_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 2,
                                           coors_desc_dims.data()));
    }

    if (coors) {
      if (coors_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&coors_, MLUOP_DTYPE_INT32 *
                                    mluOpGetTensorElementNum(coors_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&coors_, MLUOP_DTYPE_INT32 * 2));
      }
    }

    if (voxel_feats_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&voxel_feats_desc_));
      std::vector<int> voxel_feats_desc_dims{4, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          voxel_feats_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
          voxel_feats_desc_dims.data()));
    }

    if (voxel_feats) {
      if (voxel_feats_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&voxel_feats_,
                               MLUOP_DTYPE_FLOAT * mluOpGetTensorElementNum(
                                                       voxel_feats_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&voxel_feats_, MLUOP_DTYPE_FLOAT * 2));
      }
    }

    if (voxel_coors_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&voxel_coors_desc_));
      std::vector<int> voxel_coors_desc_dims{4, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          voxel_coors_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 2,
          voxel_coors_desc_dims.data()));
    }

    if (voxel_coors) {
      if (voxel_coors_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&voxel_coors_,
                               MLUOP_DTYPE_INT32 * mluOpGetTensorElementNum(
                                                       voxel_coors_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&voxel_coors_, MLUOP_DTYPE_INT32 * 2));
      }
    }

    if (point2voxel_map_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&point2voxel_map_desc_));
      std::vector<int> point2voxel_map_desc_dims{4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          point2voxel_map_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 1,
          point2voxel_map_desc_dims.data()));
    }

    if (point2voxel_map) {
      if (point2voxel_map_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&point2voxel_map_,
                               MLUOP_DTYPE_INT32 * mluOpGetTensorElementNum(
                                                       point2voxel_map_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&point2voxel_map_, MLUOP_DTYPE_INT32 * 2));
      }
    }

    if (voxel_points_count_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&voxel_points_count_desc_));
      std::vector<int> voxel_points_count_desc_dims{4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          voxel_points_count_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 1,
          voxel_points_count_desc_dims.data()));
    }

    if (voxel_points_count) {
      if (voxel_points_count_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&voxel_points_count_,
                       MLUOP_DTYPE_INT32 *
                           mluOpGetTensorElementNum(voxel_points_count_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&voxel_points_count_, MLUOP_DTYPE_INT32 * 2));
      }
    }

    if (voxel_num_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&voxel_num_desc_));
      std::vector<int> fvoxel_num_desc_dims{1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(voxel_num_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 1,
                                           fvoxel_num_desc_dims.data()));
    }

    if (voxel_num) {
      if (voxel_num_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&voxel_num_,
                               MLUOP_DTYPE_INT32 *
                                   mluOpGetTensorElementNum(voxel_num_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&voxel_num_, MLUOP_DTYPE_INT32 * 1));
      }
    }

    if (workspace) {
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&workspace_, MLUOP_DTYPE_INT32 * workspace_size_));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpDynamicPointToVoxelForward(
        handle_, reduce_type_, feats_desc_, feats_, coors_desc_, coors_,
        workspace_, workspace_size_, voxel_feats_desc_, voxel_feats_,
        voxel_coors_desc_, voxel_coors_, point2voxel_map_desc_,
        point2voxel_map_, voxel_points_count_desc_, voxel_points_count_,
        voxel_num_desc_, voxel_num_);
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

    if (feats_desc_) {
      VLOG(4) << "Destroy feats_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(feats_desc_));
      feats_desc_ = nullptr;
    }

    if (feats_) {
      VLOG(4) << "Destroy feats_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(feats_));
      feats_ = nullptr;
    }

    if (coors_desc_) {
      VLOG(4) << "Destroy coors_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(coors_desc_));
      coors_desc_ = nullptr;
    }

    if (coors_) {
      VLOG(4) << "Destroy coors_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(coors_));
      coors_ = nullptr;
    }

    if (voxel_feats_desc_) {
      VLOG(4) << "Destroy voxel_feats_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(voxel_feats_desc_));
      voxel_feats_desc_ = nullptr;
    }

    if (voxel_feats_) {
      VLOG(4) << "Destroy voxel_feats_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(voxel_feats_));
      voxel_feats_ = nullptr;
    }

    if (voxel_coors_desc_) {
      VLOG(4) << "Destroy voxel_coors_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(voxel_coors_desc_));
      voxel_coors_desc_ = nullptr;
    }

    if (voxel_coors_) {
      VLOG(4) << "Destroy voxel_coors_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(voxel_coors_));
      voxel_coors_ = nullptr;
    }

    if (point2voxel_map_desc_) {
      VLOG(4) << "Destroy point2voxel_map_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(point2voxel_map_desc_));
      point2voxel_map_desc_ = nullptr;
    }

    if (point2voxel_map_) {
      VLOG(4) << "Destroy point2voxel_map_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(point2voxel_map_));
      point2voxel_map_ = nullptr;
    }

    if (voxel_points_count_desc_) {
      VLOG(4) << "Destroy data_col_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(voxel_points_count_desc_));
      voxel_points_count_desc_ = nullptr;
    }

    if (voxel_points_count_) {
      VLOG(4) << "Destroy voxel_points_count_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(voxel_points_count_));
      voxel_points_count_ = nullptr;
    }

    if (voxel_num_desc_) {
      VLOG(4) << "Destroy voxel_num_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(voxel_num_desc_));
      voxel_num_desc_ = nullptr;
    }

    if (voxel_num_) {
      VLOG(4) << "Destroy voxel_num_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(voxel_num_));
      voxel_num_ = nullptr;
    }

    if (workspace_) {
      VLOG(4) << "Destroy workspace_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t feats_desc_ = nullptr;
  void *feats_ = nullptr;
  mluOpTensorDescriptor_t coors_desc_ = nullptr;
  void *coors_ = nullptr;
  mluOpTensorDescriptor_t voxel_feats_desc_ = nullptr;
  void *voxel_feats_ = nullptr;
  mluOpTensorDescriptor_t voxel_coors_desc_ = nullptr;
  void *voxel_coors_ = nullptr;
  mluOpTensorDescriptor_t point2voxel_map_desc_ = nullptr;
  void *point2voxel_map_ = nullptr;
  mluOpTensorDescriptor_t voxel_points_count_desc_ = nullptr;
  void *voxel_points_count_ = nullptr;
  mluOpTensorDescriptor_t voxel_num_desc_ = nullptr;
  void *voxel_num_ = nullptr;
  size_t workspace_size_ = 10;
  void *workspace_ = nullptr;
  mluOpReduceMode_t reduce_type_ = MLUOP_REDUCE_DMEAN;
};

TEST_F(dynamic_point_to_voxel_forward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_forward";
  }
}

TEST_F(dynamic_point_to_voxel_forward, BAD_PARAM_feats_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_forward";
  }
}

TEST_F(dynamic_point_to_voxel_forward, BAD_PARAM_feats_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_forward";
  }
}

TEST_F(dynamic_point_to_voxel_forward, BAD_PARAM_coors_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_forward";
  }
}

TEST_F(dynamic_point_to_voxel_forward, BAD_PARAM_coors_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_forward";
  }
}

TEST_F(dynamic_point_to_voxel_forward, BAD_PARAM_voxel_feats_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_forward";
  }
}

TEST_F(dynamic_point_to_voxel_forward, BAD_PARAM_voxel_feats_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_forward";
  }
}

TEST_F(dynamic_point_to_voxel_forward, BAD_PARAM_voxel_coors_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_forward";
  }
}

TEST_F(dynamic_point_to_voxel_forward, BAD_PARAM_voxel_coors_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_forward";
  }
}

TEST_F(dynamic_point_to_voxel_forward, BAD_PARAM_point2voxel_map_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_forward";
  }
}

TEST_F(dynamic_point_to_voxel_forward, BAD_PARAM_point2voxel_map_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, false,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_forward";
  }
}

TEST_F(dynamic_point_to_voxel_forward, BAD_PARAM_voxel_points_count_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             false, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_forward";
  }
}

TEST_F(dynamic_point_to_voxel_forward, BAD_PARAM_voxel_points_count_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, false, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_forward";
  }
}

TEST_F(dynamic_point_to_voxel_forward, BAD_PARAM_voxel_num_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, false, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_forward";
  }
}

TEST_F(dynamic_point_to_voxel_forward, BAD_PARAM_voxel_num_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, true, false, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_forward";
  }
}

TEST_F(dynamic_point_to_voxel_forward, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, true, true, false);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in dynamic_point_to_voxel_forward";
  }
}
}  // namespace mluopapitest
