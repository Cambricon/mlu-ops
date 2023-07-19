/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
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
class masked_im2col_forward : public testing::Test {
 public:
  void setParam(bool handle, bool feature_desc, bool feature,
                bool mask_h_idx_desc, bool mask_h_idx, bool mask_w_idx_desc,
                bool mask_w_idx, bool workspace, bool data_col_desc,
                bool data_col) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (feature_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&feature_desc_));
      std::vector<int> feature_desc_dims{1, 36, 37, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(feature_desc_, MLUOP_LAYOUT_NCHW,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           feature_desc_dims.data()));
    }

    if (feature) {
      if (feature_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&feature_, MLUOP_DTYPE_FLOAT *
                                      mluOpGetTensorElementNum(feature_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&feature_, MLUOP_DTYPE_FLOAT * 2));
      }
    }

    if (mask_h_idx_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&mask_h_idx_desc_));
      std::vector<int> mask_h_idx_desc_dims{88};
      MLUOP_CHECK(mluOpSetTensorDescriptor(mask_h_idx_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 1,
                                           mask_h_idx_desc_dims.data()));
    }

    if (mask_h_idx) {
      if (mask_h_idx_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&mask_h_idx_,
                               MLUOP_DTYPE_INT32 *
                                   mluOpGetTensorElementNum(mask_h_idx_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&mask_h_idx_, MLUOP_DTYPE_INT32 * 2));
      }
    }

    if (mask_w_idx_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&mask_w_idx_desc_));
      std::vector<int> mask_w_idx_desc_dims{88};
      MLUOP_CHECK(mluOpSetTensorDescriptor(mask_w_idx_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 1,
                                           mask_w_idx_desc_dims.data()));
    }

    if (mask_w_idx) {
      if (mask_w_idx_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&mask_w_idx_,
                               MLUOP_DTYPE_INT32 *
                                   mluOpGetTensorElementNum(mask_w_idx_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&mask_w_idx_, MLUOP_DTYPE_INT32 * 2));
      }
    }

    if (data_col_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_col_desc_));
      std::vector<int> data_col_desc_dims{144, 88};
      MLUOP_CHECK(mluOpSetTensorDescriptor(data_col_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           data_col_desc_dims.data()));
    }

    if (data_col) {
      if (data_col_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&data_col_,
                               MLUOP_DTYPE_FLOAT *
                                   mluOpGetTensorElementNum(data_col_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&data_col_, MLUOP_DTYPE_FLOAT * 2));
      }
    }

    if (workspace) {
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&workspace_, MLUOP_DTYPE_FLOAT * workspace_size_));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpMaskedIm2colForward(
        handle_, feature_desc_, feature_, mask_h_idx_desc_, mask_h_idx_,
        mask_w_idx_desc_, mask_w_idx_, kernel_h_, kernel_w_, pad_h_, pad_w_,
        workspace_, workspace_size_, data_col_desc_, data_col_);
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

    if (feature_desc_) {
      VLOG(4) << "Destroy feature_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(feature_desc_));
      feature_desc_ = nullptr;
    }

    if (feature_) {
      VLOG(4) << "Destroy feature_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(feature_));
      feature_ = nullptr;
    }

    if (mask_h_idx_desc_) {
      VLOG(4) << "Destroy mask_h_idx_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(mask_h_idx_desc_));
      mask_h_idx_desc_ = nullptr;
    }

    if (mask_h_idx_) {
      VLOG(4) << "Destroy mask_h_idx_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(mask_h_idx_));
      mask_h_idx_ = nullptr;
    }

    if (mask_w_idx_desc_) {
      VLOG(4) << "Destroy mask_w_idx_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(mask_w_idx_desc_));
      mask_w_idx_desc_ = nullptr;
    }

    if (mask_w_idx_) {
      VLOG(4) << "Destroy mask_w_idx_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(mask_w_idx_));
      mask_w_idx_ = nullptr;
    }

    if (data_col_desc_) {
      VLOG(4) << "Destroy data_col_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(data_col_desc_));
      data_col_desc_ = nullptr;
    }

    if (data_col_) {
      VLOG(4) << "Destroy data_col_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(data_col_));
      data_col_ = nullptr;
    }

    if (workspace_) {
      VLOG(4) << "Destroy workspace_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t feature_desc_ = nullptr;
  void *feature_ = nullptr;
  mluOpTensorDescriptor_t mask_h_idx_desc_ = nullptr;
  void *mask_h_idx_ = nullptr;
  mluOpTensorDescriptor_t mask_w_idx_desc_ = nullptr;
  void *mask_w_idx_ = nullptr;
  mluOpTensorDescriptor_t data_col_desc_ = nullptr;
  void *data_col_ = nullptr;
  int kernel_w_ = 2;
  int kernel_h_ = 2;
  int pad_w_ = 2;
  int pad_h_ = 1;
  size_t workspace_size_ = 10;
  void *workspace_ = nullptr;
};

TEST_F(masked_im2col_forward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_im2col_forward";
  }
}

TEST_F(masked_im2col_forward, BAD_PARAM_feature_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_im2col_forward";
  }
}

TEST_F(masked_im2col_forward, BAD_PARAM_feature_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_im2col_forward";
  }
}

TEST_F(masked_im2col_forward, BAD_PARAM_mask_h_idx_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_im2col_forward";
  }
}

TEST_F(masked_im2col_forward, BAD_PARAM_mask_h_idx_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_im2col_forward";
  }
}

TEST_F(masked_im2col_forward, BAD_PARAM_mask_w_idx_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_im2col_forward";
  }
}

TEST_F(masked_im2col_forward, BAD_PARAM_mask_w_idx_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_im2col_forward";
  }
}

TEST_F(masked_im2col_forward, BAD_PARAM_data_col_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_im2col_forward";
  }
}

TEST_F(masked_im2col_forward, BAD_PARAM_data_col_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_im2col_forward";
  }
}

TEST_F(masked_im2col_forward, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_im2col_forward";
  }
}
}  // namespace mluopapitest
