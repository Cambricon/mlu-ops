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
#include "core/context.h"

namespace mluopapitest {
class psroipool_forward : public testing::Test {
 public:
  void setParam(bool handle, bool input_desc, bool rois_desc, bool output_desc,
                bool mapping_channel_desc, bool input, bool rois, bool output,
                bool mapping_channel) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (input_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
      std::vector<int> i_dims = {1, 5, 5, 9};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          input_desc_, MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT, 4, i_dims.data()));
    }
    if (input) {
      size_t i_ele_num = 1 * 5 * 5 * 9;
      size_t i_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t i_bytes = i_ele_num * i_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&input_, i_bytes));
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
    if (output_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));
      std::vector<int> o_dims = {1, 3, 3, 1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(output_desc_, MLUOP_LAYOUT_NHWC,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           o_dims.data()));
    }
    if (output) {
      size_t o_ele_num = 1 * 3 * 3 * 1;
      size_t o_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t o_bytes = o_ele_num * o_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&output_, o_bytes));
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
    mluOpStatus_t status = mluOpPsRoiPoolForward(
        handle_, pooled_height_, pooled_width_, spatial_scale_, group_size_,
        output_dim_, input_desc_, input_, rois_desc_, rois_, output_desc_,
        output_, mapping_channel_desc_, mapping_channel_);
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
    if (input_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_desc_));
      input_desc_ = NULL;
    }
    if (input_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_));
      input_ = NULL;
    }
    if (rois_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(rois_desc_));
      rois_desc_ = NULL;
    }
    if (rois_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(rois_));
      rois_ = NULL;
    }
    if (output_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
      output_desc_ = NULL;
    }
    if (output_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_));
      output_ = NULL;
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
  mluOpTensorDescriptor_t input_desc_ = NULL;
  mluOpTensorDescriptor_t rois_desc_ = NULL;
  mluOpTensorDescriptor_t output_desc_ = NULL;
  mluOpTensorDescriptor_t mapping_channel_desc_ = NULL;
  void* input_ = NULL;
  void* rois_ = NULL;
  void* output_ = NULL;
  void* mapping_channel_ = NULL;
  int pooled_height_ = 3;
  int pooled_width_ = 3;
  float spatial_scale_ = 0.25;
  int group_size_ = 3;
  int output_dim_ = 1;
};

TEST_F(psroipool_forward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_forward";
  }
}

TEST_F(psroipool_forward, BAD_PARAM_input_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_forward";
  }
}

TEST_F(psroipool_forward, BAD_PARAM_rois_desc_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_forward";
  }
}

TEST_F(psroipool_forward, BAD_PARAM_output_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_forward";
  }
}

TEST_F(psroipool_forward, BAD_PARAM_mapping_desc_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_forward";
  }
}

TEST_F(psroipool_forward, BAD_PARAM_input_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_forward";
  }
}

TEST_F(psroipool_forward, BAD_PARAM_rois_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_forward";
  }
}

TEST_F(psroipool_forward, BAD_PARAM_output_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_forward";
  }
}

TEST_F(psroipool_forward, BAD_PARAM_mapping_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_forward";
  }
}
}  // namespace mluopapitest
