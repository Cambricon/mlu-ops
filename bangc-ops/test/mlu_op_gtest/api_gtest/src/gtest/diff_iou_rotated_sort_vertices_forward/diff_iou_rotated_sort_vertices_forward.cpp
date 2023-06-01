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
class diff_iou_rotated_sort_vertices_forward : public testing::Test {
 public:
  void setParam(bool handle,
                bool vertices_desc,
                bool vertices,
                bool mask_desc,
                bool mask,
                bool num_valid_desc,
                bool num_valid,
                bool idx_desc,
                bool idx) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (vertices_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&vertices_desc_));
      std::vector<int> vertices_desc_dims{4, 16, 24, 2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(vertices_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           vertices_desc_dims.data()));
    }

    if (vertices) {
      if (vertices_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&vertices_,MLUOP_DTYPE_FLOAT *
                                         mluOpGetTensorElementNum(
                                             vertices_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&vertices_, MLUOP_DTYPE_FLOAT * 2));
      }
    }

    if (mask_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&mask_desc_));
      std::vector<int> mask_desc_dims{4, 16, 24};
      MLUOP_CHECK(mluOpSetTensorDescriptor(mask_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_BOOL, 3,
                                           mask_desc_dims.data()));
    }

    if (mask) {
      if (mask_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&mask_, MLUOP_DTYPE_BOOL *
                                         mluOpGetTensorElementNum(mask_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&mask_, MLUOP_DTYPE_BOOL * 2));
      }
    }

    if (num_valid_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&num_valid_desc_));
      std::vector<int> num_valid_desc_dims{4, 16};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          num_valid_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 2,
          num_valid_desc_dims.data()));
    }

    if (num_valid) {
      if (num_valid_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&num_valid_,
                               MLUOP_DTYPE_INT32 * mluOpGetTensorElementNum(
                                                       num_valid_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&num_valid_, MLUOP_DTYPE_INT32 * 2));
      }
    }

    if (idx_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&idx_desc_));
      std::vector<int> idx_desc_dims{4, 16, 9};
      MLUOP_CHECK(mluOpSetTensorDescriptor(idx_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 3,
                                           idx_desc_dims.data()));
    }

    if (idx) {
      if (idx_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&idx_, MLUOP_DTYPE_INT32 *
                                        mluOpGetTensorElementNum(idx_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS == 
                    cnrtMalloc(&idx_, MLUOP_DTYPE_INT32 * 2));
      }
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpDiffIouRotatedSortVerticesForward(
        handle_, vertices_desc_, vertices_, mask_desc_, mask_, num_valid_desc_,
        num_valid_, idx_desc_, idx_);
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

    if (vertices_desc_) {
      VLOG(4) << "Destroy vertices_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(vertices_desc_));
      vertices_desc_ = nullptr;
    }

    if (vertices_) {
      VLOG(4) << "Destroy vertices_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(vertices_));
      vertices_ = nullptr;
    }

    if (mask_desc_) {
      VLOG(4) << "Destroy mask_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(mask_desc_));
      mask_desc_ = nullptr;
    }

    if (mask_) {
      VLOG(4) << "Destroy mask_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(mask_));
      mask_ = nullptr;
    }

    if (num_valid_desc_) {
      VLOG(4) << "Destroy num_valid_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(num_valid_desc_));
      num_valid_desc_ = nullptr;
    }

    if (num_valid_) {
      VLOG(4) << "Destroy num_valid_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(num_valid_));
      num_valid_ = nullptr;
    }

    if (idx_desc_) {
      VLOG(4) << "Destroy idx_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(idx_desc_));
      idx_desc_ = nullptr;
    }

    if (idx_) {
      VLOG(4) << "Destroy idx_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(idx_));
      idx_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t vertices_desc_ = nullptr;
  void *vertices_ = nullptr;
  mluOpTensorDescriptor_t mask_desc_ = nullptr;
  void *mask_ = nullptr;
  mluOpTensorDescriptor_t num_valid_desc_ = nullptr;
  void *num_valid_ = nullptr;
  mluOpTensorDescriptor_t idx_desc_ = nullptr;
  void *idx_ = nullptr;
};

TEST_F(diff_iou_rotated_sort_vertices_forward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in diff_iou_rotated_sort_vertices_forward";
  }
}

TEST_F(diff_iou_rotated_sort_vertices_forward, BAD_PARAM_vertices_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in diff_iou_rotated_sort_vertices_forward";
  }
}

TEST_F(diff_iou_rotated_sort_vertices_forward, BAD_PARAM_vertices_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in diff_iou_rotated_sort_vertices_forward";
  }
}

TEST_F(diff_iou_rotated_sort_vertices_forward, BAD_PARAM_mask_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in diff_iou_rotated_sort_vertices_forward";
  }
}

TEST_F(diff_iou_rotated_sort_vertices_forward, BAD_PARAM_mask_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in diff_iou_rotated_sort_vertices_forward";
  }
}

TEST_F(diff_iou_rotated_sort_vertices_forward, BAD_PARAM_num_valid_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in diff_iou_rotated_sort_vertices_forward";
  }
}

TEST_F(diff_iou_rotated_sort_vertices_forward, BAD_PARAM_num_valid_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in diff_iou_rotated_sort_vertices_forward";
  }
}

TEST_F(diff_iou_rotated_sort_vertices_forward, BAD_PARAM_idx_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in diff_iou_rotated_sort_vertices_forward";
  }
}

TEST_F(diff_iou_rotated_sort_vertices_forward, BAD_PARAM_idx_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in diff_iou_rotated_sort_vertices_forward";
  }
}
}  // namespace mluopapitest
