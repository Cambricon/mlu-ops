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
class voxel_pooling_forward : public testing::Test {
 public:
  void setParam(bool handle, bool geom_xyz_desc, bool geom_xyz,
                bool input_features_desc, bool input_features,
                bool output_features_desc, bool output_features,
                bool pos_memo_desc, bool pos_memo) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (geom_xyz_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&geom_xyz_desc_));
      std::vector<int> dim_size = {2, 4, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(geom_xyz_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 3,
                                           dim_size.data()));
    }
    if (input_features_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_features_desc_));
      std::vector<int> dim_size = {2, 4, 10};
      MLUOP_CHECK(
          mluOpSetTensorDescriptor(input_features_desc_, MLUOP_LAYOUT_ARRAY,
                                   MLUOP_DTYPE_FLOAT, 3, dim_size.data()));
    }
    if (output_features_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_features_desc_));
      std::vector<int> dim_size = {2, 5, 6, 10};
      MLUOP_CHECK(
          mluOpSetTensorDescriptor(output_features_desc_, MLUOP_LAYOUT_ARRAY,
                                   MLUOP_DTYPE_FLOAT, 4, dim_size.data()));
    }
    if (pos_memo_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&pos_memo_desc_));
      std::vector<int> dim_size = {2, 4, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(pos_memo_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 3,
                                           dim_size.data()));
    }
    if (geom_xyz) {
      size_t g_ele_num = 2 * 4 * 3;
      size_t g_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      size_t g_bytes = g_ele_num * g_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&geom_xyz_, g_bytes));
    }
    if (input_features) {
      size_t i_ele_num = 2 * 4 * 10;
      size_t i_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t i_bytes = i_ele_num * i_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&input_features_, i_bytes));
    }
    if (output_features) {
      size_t o_ele_num = 2 * 5 * 6 * 10;
      size_t o_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t o_bytes = o_ele_num * o_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&output_features_, o_bytes));
    }
    if (pos_memo) {
      size_t p_ele_num = 2 * 4 * 3;
      size_t p_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      size_t p_bytes = p_ele_num * p_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&pos_memo_, p_bytes));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpVoxelPoolingForward(
        handle_, batch_size_, num_points_, num_channels_, num_voxel_x_,
        num_voxel_y_, num_voxel_z_, geom_xyz_desc_, geom_xyz_,
        input_features_desc_, input_features_, output_features_desc_,
        output_features_, pos_memo_desc_, pos_memo_);
    destroy();
    return status;
  }

 protected:
  void destroy() {
    VLOG(4) << "Destroy parameters.";
    if (handle_) {
      CNRT_CHECK(cnrtQueueSync(handle_->queue));
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = NULL;
    }
    if (geom_xyz_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(geom_xyz_desc_));
      geom_xyz_desc_ = NULL;
    }
    if (input_features_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_features_desc_));
      input_features_desc_ = NULL;
    }
    if (output_features_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_features_desc_));
      output_features_desc_ = NULL;
    }
    if (pos_memo_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(pos_memo_desc_));
      pos_memo_desc_ = NULL;
    }
    if (geom_xyz_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(geom_xyz_));
      geom_xyz_ = NULL;
    }
    if (input_features_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_features_));
      input_features_ = NULL;
    }
    if (output_features_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_features_));
      output_features_ = NULL;
    }
    if (pos_memo_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(pos_memo_));
      output_features_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t geom_xyz_desc_ = NULL;
  mluOpTensorDescriptor_t input_features_desc_ = NULL;
  mluOpTensorDescriptor_t output_features_desc_ = NULL;
  mluOpTensorDescriptor_t pos_memo_desc_ = NULL;
  void* geom_xyz_ = NULL;
  void* input_features_ = NULL;
  void* output_features_ = NULL;
  void* pos_memo_ = NULL;
  int batch_size_ = 2;
  int num_points_ = 4;
  int num_channels_ = 10;
  int num_voxel_x_ = 6;
  int num_voxel_y_ = 5;
  int num_voxel_z_ = 1;
};

TEST_F(voxel_pooling_forward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in voxel_pooling_forward";
  }
}

TEST_F(voxel_pooling_forward, BAD_PARAM_geom_xyz_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in voxel_pooling_forward";
  }
}

TEST_F(voxel_pooling_forward, BAD_PARAM_geom_xyz_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in voxel_pooling_forward";
  }
}

TEST_F(voxel_pooling_forward, BAD_PARAM_input_features_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in voxel_pooling_forward";
  }
}
TEST_F(voxel_pooling_forward, BAD_PARAM_input_features_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in voxel_pooling_forward";
  }
}

TEST_F(voxel_pooling_forward, BAD_PARAM_output_features_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in voxel_pooling_forward";
  }
}

TEST_F(voxel_pooling_forward, BAD_PARAM_output_features_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in voxel_pooling_forward";
  }
}

TEST_F(voxel_pooling_forward, BAD_PARAM_pos_memo_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in voxel_pooling_forward";
  }
}

TEST_F(voxel_pooling_forward, BAD_PARAM_pos_memo_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in voxel_pooling_forward";
  }
}
}  // namespace mluopapitest
