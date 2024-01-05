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
#include <string>
#include <tuple>
#include <vector>

#include "api_test_tools.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "gtest/gtest.h"
#include "mlu_op.h"
#include "core/context.h"

namespace mluopapitest {
class psroipool_backward : public testing::Test {
 public:
  void setParam(bool handle, bool bottom_grad_desc, bool rois_desc,
                bool top_grad_desc, bool mapping_channel_desc, bool bottom_grad,
                bool rois, bool top_grad, bool mapping_channel) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (bottom_grad_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&bottom_grad_desc_));
      std::vector<int> b_dims = {1, 5, 5, 9};
      MLUOP_CHECK(mluOpSetTensorDescriptor(bottom_grad_desc_, MLUOP_LAYOUT_NHWC,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           b_dims.data()));
    }
    if (bottom_grad) {
      size_t b_ele_num = 1 * 5 * 5 * 9;
      size_t b_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t b_bytes = b_ele_num * b_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&bottom_grad_, b_bytes));
    }
    if (rois_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&rois_desc_));
      std::vector<int> r_dims = {1, 5};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          rois_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2, r_dims.data()));
    }
    if (rois) {
      size_t r_ele_num = 1 * 5;
      size_t r_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t r_bytes = r_ele_num * r_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&rois_, r_bytes));
    }
    if (top_grad_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&top_grad_desc_));
      std::vector<int> o_dims = {1, 3, 3, 1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(top_grad_desc_, MLUOP_LAYOUT_NHWC,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           o_dims.data()));
    }
    if (top_grad) {
      size_t o_ele_num = 1 * 3 * 3 * 1;
      size_t o_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t o_bytes = o_ele_num * o_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&top_grad_, o_bytes));
    }
    if (mapping_channel_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&mapping_channel_desc_));
      std::vector<int> m_dims = {1, 3, 3, 1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(mapping_channel_desc_,
                                           MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
                                           4, m_dims.data()));
    }
    if (mapping_channel) {
      size_t m_ele_num = 1 * 3 * 3 * 1;
      size_t m_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      size_t m_bytes = m_ele_num * m_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&mapping_channel_, m_bytes));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpPsRoiPoolBackward(
        handle_, pooled_height_, pooled_width_, spatial_scale_, output_dim_,
        top_grad_desc_, top_grad_, rois_desc_, rois_, mapping_channel_desc_,
        mapping_channel_, bottom_grad_desc_, bottom_grad_);
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
    if (bottom_grad_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(bottom_grad_desc_));
      bottom_grad_desc_ = NULL;
    }
    if (bottom_grad_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(bottom_grad_));
      bottom_grad_ = NULL;
    }
    if (rois_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(rois_desc_));
      rois_desc_ = NULL;
    }
    if (rois_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(rois_));
      rois_ = NULL;
    }
    if (top_grad_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(top_grad_desc_));
      top_grad_desc_ = NULL;
    }
    if (top_grad_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(top_grad_));
      top_grad_ = NULL;
    }
    if (mapping_channel_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(mapping_channel_desc_));
      mapping_channel_desc_ = NULL;
    }
    if (mapping_channel_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(mapping_channel_));
      mapping_channel_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t bottom_grad_desc_ = NULL;
  mluOpTensorDescriptor_t rois_desc_ = NULL;
  mluOpTensorDescriptor_t top_grad_desc_ = NULL;
  mluOpTensorDescriptor_t mapping_channel_desc_ = NULL;
  void* bottom_grad_ = NULL;
  void* rois_ = NULL;
  void* top_grad_ = NULL;
  void* mapping_channel_ = NULL;
  int pooled_height_ = 3;
  int pooled_width_ = 3;
  float spatial_scale_ = 0.25;
  int output_dim_ = 1;
};

TEST_F(psroipool_backward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_backward";
  }
}

TEST_F(psroipool_backward, BAD_PARAM_bottom_grad_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_backward";
  }
}

TEST_F(psroipool_backward, BAD_PARAM_rois_desc_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_backward";
  }
}

TEST_F(psroipool_backward, BAD_PARAM_top_grad_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_backward";
  }
}

TEST_F(psroipool_backward, BAD_PARAM_mapping_desc_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_backward";
  }
}

TEST_F(psroipool_backward, BAD_PARAM_bottom_grad_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_backward";
  }
}

TEST_F(psroipool_backward, BAD_PARAM_rois_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_backward";
  }
}

TEST_F(psroipool_backward, BAD_PARAM_top_grad_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_backward";
  }
}

TEST_F(psroipool_backward, BAD_PARAM_mapping_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_backward";
  }
}
}  // namespace mluopapitest
