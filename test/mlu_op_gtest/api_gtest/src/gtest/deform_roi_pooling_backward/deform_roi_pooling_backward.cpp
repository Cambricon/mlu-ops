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
class deform_roi_pooling_backward : public testing::Test {
 public:
  void setParam(bool handle, bool grad_output_desc, bool grad_output,
                bool input_desc, bool input, bool rois_desc, bool rois,
                bool offset_desc, bool offset, bool grad_input_desc,
                bool grad_input, bool grad_offset_desc, bool grad_offset) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (grad_output_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_output_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_output_desc_, MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT, 4,
          std::vector<int>({3, 3, 3, 1}).data()));
    }
    if (grad_output) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&grad_output_, 10));
    }
    if (input_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          input_desc_, MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT, 4,
          std::vector<int>({1, 5, 5, 1}).data()));
    }
    if (input) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&input_, 10));
    }
    if (rois_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&rois_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(rois_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           std::vector<int>({3, 5}).data()));
    }
    if (rois) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&rois_, 5));
    }
    if (offset_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&offset_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          offset_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 4,
          std::vector<int>({3, 2, 3, 3}).data()));
    }
    if (offset) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&offset_, 5));
    }
    if (grad_input_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_input_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_input_desc_, MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT, 4,
          std::vector<int>({1, 5, 5, 1}).data()));
    }
    if (grad_input_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&grad_input_, 10));
    }
    if (grad_offset_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_offset_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_offset_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 4,
          std::vector<int>({3, 2, 3, 3}).data()));
    }
    if (grad_offset) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&grad_offset_, 10));
    }
  }
  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpDeformRoiPoolBackward(
        handle_, grad_output_desc_, grad_output_, input_desc_, input_,
        rois_desc_, rois_, offset_desc_, offset_, pooled_height_, pooled_width_,
        spatial_scale_, sampling_ratio_, gamma_, grad_input_desc_, grad_input_,
        grad_offset_desc_, grad_offset_);
    destroy();
    return status;
  }

 protected:
  virtual void SetUp() {
    handle_ = nullptr;
    grad_output_desc_ = nullptr;
    grad_output_ = nullptr;
    input_desc_ = nullptr;
    input_ = nullptr;
    rois_desc_ = nullptr;
    rois_ = nullptr;
    offset_desc_ = nullptr;
    offset_ = nullptr;
    grad_input_desc_ = nullptr;
    grad_input_ = nullptr;
    grad_offset_desc_ = nullptr;
    grad_offset_ = nullptr;
  }

  void destroy() {
    if (handle_) {
      CNRT_CHECK(cnrtQueueSync(handle_->queue));
      VLOG(4) << "Destroy handle";
      MLUOP_CHECK(mluOpDestroy(handle_));
    }
    if (grad_output_desc_) {
      VLOG(4) << "Destroy grad_output_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_output_desc_));
    }
    if (grad_output_) {
      VLOG(4) << "Destroy grad_output";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_output_));
      grad_output_ = nullptr;
    }
    if (input_desc_) {
      VLOG(4) << "Destroy input_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_desc_));
    }
    if (input_) {
      VLOG(4) << "Destroy input";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_));
      input_ = nullptr;
    }
    if (rois_desc_) {
      VLOG(4) << "Destroy rois_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(rois_desc_));
    }
    if (rois_) {
      VLOG(4) << "Destroy rois";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(rois_));
      rois_ = nullptr;
    }
    if (offset_desc_) {
      VLOG(4) << "Destroy offset_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(offset_desc_));
    }
    if (offset_) {
      VLOG(4) << "Destroy offset";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(offset_));
      offset_ = nullptr;
    }
    if (grad_input_desc_) {
      VLOG(4) << "Destroy grad_input_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_input_desc_));
    }
    if (grad_input_) {
      VLOG(4) << "Destroy grad_input";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_input_));
      grad_input_ = nullptr;
    }
    if (grad_offset_desc_) {
      VLOG(4) << "Destroy grad_offsest_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_offset_desc_));
    }
    if (grad_offset_) {
      VLOG(4) << "Destroy grad_offset";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_offset_));
      grad_offset_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t grad_output_desc_ = nullptr;
  void *grad_output_ = nullptr;
  mluOpTensorDescriptor_t input_desc_ = nullptr;
  void *input_ = nullptr;
  mluOpTensorDescriptor_t rois_desc_ = nullptr;
  void *rois_ = nullptr;
  mluOpTensorDescriptor_t offset_desc_ = nullptr;
  void *offset_ = nullptr;
  mluOpTensorDescriptor_t grad_input_desc_ = nullptr;
  void *grad_input_ = nullptr;
  mluOpTensorDescriptor_t grad_offset_desc_ = nullptr;
  void *grad_offset_ = nullptr;
  int pooled_height_ = 3;
  int pooled_width_ = 3;
  float spatial_scale_ = 0.5;
  int sampling_ratio_ = 1;
  float gamma_ = 0.1;
  // what's the difference between defining here and in setUp?
};

TEST_F(deform_roi_pooling_backward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true, true,
             true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in deform_roi_pooling_backward";
  }
}

TEST_F(deform_roi_pooling_backward, BAD_PARAM_grad_output_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true, true,
             true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in deform_roi_pooling_backward";
  }
}

TEST_F(deform_roi_pooling_backward, BAD_PARAM_grad_output_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true, true,
             true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in deform_roi_pooling_backward";
  }
}

TEST_F(deform_roi_pooling_backward, BAD_PARAM_input_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true, true,
             true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in deform_roi_pooling_backward";
  }
}

TEST_F(deform_roi_pooling_backward, BAD_PARAM_input_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true, true,
             true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in deform_roi_pooling_backward";
  }
}

TEST_F(deform_roi_pooling_backward, BAD_PARAM_rois_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true, true,
             true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in deform_roi_pooling_backward";
  }
}

TEST_F(deform_roi_pooling_backward, BAD_PARAM_rois_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true, true,
             true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in deform_roi_pooling_backward";
  }
}

TEST_F(deform_roi_pooling_backward, BAD_PARAM_offset_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true, true,
             true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in deform_roi_pooling_backward";
  }
}

TEST_F(deform_roi_pooling_backward, BAD_PARAM_offset_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true, true,
             true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in deform_roi_pooling_backward";
  }
}

TEST_F(deform_roi_pooling_backward, BAD_PARAM_grad_input_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false, true,
             true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in deform_roi_pooling_backward";
  }
}

TEST_F(deform_roi_pooling_backward, BAD_PARAM_grad_input_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, false,
             true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in deform_roi_pooling_backward";
  }
}

TEST_F(deform_roi_pooling_backward, BAD_PARAM_grad_offset_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             false, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in deform_roi_pooling_backward";
  }
}

TEST_F(deform_roi_pooling_backward, BAD_PARAM_grad_offset_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, false);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in deform_roi_pooling_backward";
  }
}
}  // namespace mluopapitest
