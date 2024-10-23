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
class masked_col2im_forward : public testing::Test {
 public:
  void setParam(bool handle, bool col_desc, bool col, bool mask_h_idx_desc,
                bool mask_h_idx, bool mask_w_idx_desc, bool mask_w_idx,
                bool workspace, bool im_desc, bool im) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (col_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&col_desc_));
      std::vector<int> col_desc_dims{36, 88};
      MLUOP_CHECK(mluOpSetTensorDescriptor(col_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           col_desc_dims.data()));
    }

    if (col) {
      if (col_desc) {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&col_, MLUOP_DTYPE_FLOAT *
                                          mluOpGetTensorElementNum(col_desc_)));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&col_, MLUOP_DTYPE_FLOAT * 2));
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
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&mask_h_idx_,
                               MLUOP_DTYPE_INT32 *
                                   mluOpGetTensorElementNum(mask_h_idx_desc_)));
      } else {
        GTEST_CHECK(cnrtSuccess ==
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
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&mask_w_idx_,
                               MLUOP_DTYPE_INT32 *
                                   mluOpGetTensorElementNum(mask_w_idx_desc_)));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&mask_w_idx_, MLUOP_DTYPE_INT32 * 2));
      }
    }

    if (im_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&im_desc_));
      std::vector<int> im_desc_dims{1, 36, 37, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(im_desc_, MLUOP_LAYOUT_NCHW,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           im_desc_dims.data()));
    }

    if (im) {
      if (im_desc) {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&im_, MLUOP_DTYPE_FLOAT *
                                         mluOpGetTensorElementNum(im_desc_)));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&im_, MLUOP_DTYPE_FLOAT * 2));
      }
    }

    if (workspace) {
      GTEST_CHECK(cnrtSuccess ==
                  cnrtMalloc(&workspace_, MLUOP_DTYPE_FLOAT * workspace_size_));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status =
        mluOpMaskedCol2imForward(handle_, col_desc_, col_, mask_h_idx_desc_,
                                 mask_h_idx_, mask_w_idx_desc_, mask_w_idx_,
                                 workspace_size_, workspace_, im_desc_, im_);
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

    if (col_desc_) {
      VLOG(4) << "Destroy col_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(col_desc_));
      col_desc_ = nullptr;
    }

    if (col_) {
      VLOG(4) << "Destroy col_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(col_));
      col_ = nullptr;
    }

    if (mask_h_idx_desc_) {
      VLOG(4) << "Destroy mask_h_idx_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(mask_h_idx_desc_));
      mask_h_idx_desc_ = nullptr;
    }

    if (mask_h_idx_) {
      VLOG(4) << "Destroy mask_h_idx_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(mask_h_idx_));
      mask_h_idx_ = nullptr;
    }

    if (mask_w_idx_desc_) {
      VLOG(4) << "Destroy mask_w_idx_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(mask_w_idx_desc_));
      mask_w_idx_desc_ = nullptr;
    }

    if (mask_w_idx_) {
      VLOG(4) << "Destroy mask_w_idx_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(mask_w_idx_));
      mask_w_idx_ = nullptr;
    }

    if (im_desc_) {
      VLOG(4) << "Destroy im_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(im_desc_));
      im_desc_ = nullptr;
    }

    if (im_) {
      VLOG(4) << "Destroy im_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(im_));
      im_ = nullptr;
    }

    if (workspace_) {
      VLOG(4) << "Destroy workspace_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(workspace_));
      workspace_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t col_desc_ = nullptr;
  void *col_ = nullptr;
  mluOpTensorDescriptor_t mask_h_idx_desc_ = nullptr;
  void *mask_h_idx_ = nullptr;
  mluOpTensorDescriptor_t mask_w_idx_desc_ = nullptr;
  void *mask_w_idx_ = nullptr;
  mluOpTensorDescriptor_t im_desc_ = nullptr;
  void *im_ = nullptr;
  size_t workspace_size_ = 10;
  void *workspace_ = nullptr;
};

TEST_F(masked_col2im_forward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_col2im_forward";
  }
}

TEST_F(masked_col2im_forward, BAD_PARAM_col_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_col2im_forward";
  }
}

TEST_F(masked_col2im_forward, BAD_PARAM_col_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_col2im_forward";
  }
}

TEST_F(masked_col2im_forward, BAD_PARAM_mask_h_idx_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_col2im_forward";
  }
}

TEST_F(masked_col2im_forward, BAD_PARAM_mask_h_idx_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_col2im_forward";
  }
}

TEST_F(masked_col2im_forward, BAD_PARAM_mask_w_idx_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_col2im_forward";
  }
}

TEST_F(masked_col2im_forward, BAD_PARAM_mask_w_idx_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_col2im_forward";
  }
}

TEST_F(masked_col2im_forward, BAD_PARAM_im_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_col2im_forward";
  }
}

TEST_F(masked_col2im_forward, BAD_PARAM_im_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_col2im_forward";
  }
}

TEST_F(masked_col2im_forward, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in masked_col2im_forward";
  }
}
}  // namespace mluopapitest
