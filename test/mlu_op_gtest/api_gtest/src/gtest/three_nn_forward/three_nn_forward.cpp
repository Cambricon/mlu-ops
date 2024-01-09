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
class three_nn_forward : public testing::Test {
 public:
  void setParam(bool handle, bool unknown_desc, bool unknown, bool known_desc,
                bool known, bool workspace, bool dist2_desc, bool dist2,
                bool idx_desc, bool idx) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (unknown_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&unknown_desc_));
      std::vector<int> unknown_dims = {1, 2, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(unknown_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           unknown_dims.data()));
    }
    if (unknown) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&unknown_, 6 * 4))
    }
    if (known_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&known_desc_));
      std::vector<int> known_dims = {1, 10, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(known_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           known_dims.data()));
    }
    if (known) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&known_, 30 * 4))
    }
    if (workspace) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_))
    }
    if (dist2_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&dist2_desc_));
      std::vector<int> dist2_dims = {1, 2, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(dist2_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           dist2_dims.data()));
    }
    if (dist2) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&dist2_, 6 * 4))
    }
    if (idx_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&idx_desc_));
      std::vector<int> idx_dims = {1, 2, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(idx_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           idx_dims.data()));
    }
    if (idx) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&idx_, 6 * 4))
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpThreeNNForward(
        handle_, unknown_desc_, unknown_, known_desc_, known_, workspace_,
        workspace_size_, dist2_desc_, dist2_, idx_desc_, idx_);
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
    if (unknown_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(unknown_desc_));
      unknown_desc_ = NULL;
    }
    if (unknown_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(unknown_));
      unknown_ = NULL;
    }
    if (known_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(known_desc_));
      known_desc_ = NULL;
    }
    if (known_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(known_));
      known_ = NULL;
    }
    if (workspace_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = NULL;
    }
    if (dist2_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(dist2_desc_));
      dist2_desc_ = NULL;
    }
    if (dist2_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(dist2_))
      dist2_ = NULL;
    }
    if (idx_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(idx_desc_));
      idx_desc_ = NULL;
    }
    if (idx_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(idx_));
      idx_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t unknown_desc_ = NULL;
  mluOpTensorDescriptor_t known_desc_ = NULL;
  mluOpTensorDescriptor_t dist2_desc_ = NULL;
  mluOpTensorDescriptor_t idx_desc_ = NULL;
  void *unknown_ = NULL;
  void *known_ = NULL;
  void *dist2_ = NULL;
  void *idx_ = NULL;
  size_t workspace_size_ = 100;
  void *workspace_ = NULL;
};

TEST_F(three_nn_forward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in three_nn_forward";
  }
}

TEST_F(three_nn_forward, BAD_PARAM_unknown_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in three_nn_forward";
  }
}

TEST_F(three_nn_forward, BAD_PARAM_unknown_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in three_nn_forward";
  }
}

TEST_F(three_nn_forward, BAD_PARAM_known_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in three_nn_forward";
  }
}

TEST_F(three_nn_forward, BAD_PARAM_known_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in three_nn_forward";
  }
}

TEST_F(three_nn_forward, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in three_nn_forward";
  }
}

TEST_F(three_nn_forward, BAD_PARAM_dist2_desc_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in three_nn_forward";
  }
}

TEST_F(three_nn_forward, BAD_PARAM_dist2_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in three_nn_forward";
  }
}

TEST_F(three_nn_forward, BAD_PARAM_dix_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in three_nn_forward";
  }
}

TEST_F(three_nn_forward, BAD_PARAM_idx_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in three_nn_forward";
  }
}
}  // namespace mluopapitest
