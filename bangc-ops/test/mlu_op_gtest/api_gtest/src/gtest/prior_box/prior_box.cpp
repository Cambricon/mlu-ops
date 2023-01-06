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
class prior_box : public testing::Test {
 public:
  void setParam(bool handle, bool min_desc, bool min, bool aspect_desc,
                bool aspect, bool variance_desc, bool variance, bool max_desc,
                bool max, bool output_desc, bool output, bool var_desc,
                bool var) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (min_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&min_desc_));
      std::vector<int> dim_size = {1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(min_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 1,
                                           dim_size.data()));
    }
    if (min) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&min_, 8));
    }
    if (aspect_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&aspect_ratios_desc_));
      std::vector<int> dim_size = {1};
      MLUOP_CHECK(
          mluOpSetTensorDescriptor(aspect_ratios_desc_, MLUOP_LAYOUT_ARRAY,
                                   MLUOP_DTYPE_FLOAT, 1, dim_size.data()));
    }
    if (aspect) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&aspect_ratios_, 8));
    }
    if (variance_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&variance_desc_));
      std::vector<int> dim_size = {4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(variance_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 1,
                                           dim_size.data()));
    }
    if (variance) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&variance_, 8));
    }
    if (max_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&max_desc_));
      std::vector<int> dim_size = {1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(max_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 1,
                                           dim_size.data()));
    }
    if (max) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&max_, 8));
    }
    if (output_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));
      std::vector<int> dim_size = {3, 3, 2, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(output_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           dim_size.data()));
    }
    if (output) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&output_, 8));
    }
    if (var_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&var_desc_));
      std::vector<int> dim_size = {3, 3, 2, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(var_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           dim_size.data()));
    }
    if (var) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&var_, 8));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpPriorBox(
        handle_, min_desc_, min_, aspect_ratios_desc_, aspect_ratios_,
        variance_desc_, variance_, max_desc_, max_, height_, width_, im_height_,
        im_width_, step_h_, step_w_, offset_, is_clip_, min_max_aspect_order_,
        output_desc_, output_, var_desc_, var_);
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
    if (min_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(min_desc_));
      min_desc_ = NULL;
    }
    if (aspect_ratios_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(aspect_ratios_desc_));
      aspect_ratios_desc_ = NULL;
    }
    if (variance_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(variance_desc_));
      variance_desc_ = NULL;
    }
    if (max_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(max_desc_));
      max_desc_ = NULL;
    }
    if (output_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
      output_desc_ = NULL;
    }
    if (var_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(var_desc_));
      var_desc_ = NULL;
    }
    if (min_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(min_));
      min_ = NULL;
    }
    if (aspect_ratios_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(aspect_ratios_));
      aspect_ratios_ = NULL;
    }
    if (variance_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(variance_));
      variance_ = NULL;
    }
    if (max_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(max_));
      max_ = NULL;
    }
    if (output_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_));
      output_ = NULL;
    }
    if (var_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(var_));
      var_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t min_desc_ = NULL;
  mluOpTensorDescriptor_t aspect_ratios_desc_ = NULL;
  mluOpTensorDescriptor_t variance_desc_ = NULL;
  mluOpTensorDescriptor_t max_desc_ = NULL;
  mluOpTensorDescriptor_t output_desc_ = NULL;
  mluOpTensorDescriptor_t var_desc_ = NULL;
  void* min_ = NULL;
  void* aspect_ratios_ = NULL;
  void* variance_ = NULL;
  void* max_ = NULL;
  void* output_ = NULL;
  void* var_ = NULL;
  int height_ = 3;
  int width_ = 3;
  int im_height_ = 9;
  int im_width_ = 9;
  float step_w_ = 3.0;
  float step_h_ = 3.0;
  float offset_ = 0.5;
  bool is_clip_ = false;
  bool min_max_aspect_order_ = false;
};

TEST_F(prior_box, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true, true,
             true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in prior_box";
  }
}

TEST_F(prior_box, BAD_PARAM_min_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true, true,
             true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in prior_box";
  }
}

TEST_F(prior_box, BAD_PARAM_min_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true, true,
             true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in prior_box";
  }
}

TEST_F(prior_box, BAD_PARAM_aspect_ratios_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true, true,
             true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in prior_box";
  }
}

TEST_F(prior_box, BAD_PARAM_aspect_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true, true,
             true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in prior_box";
  }
}

TEST_F(prior_box, BAD_PARAM_variance_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true, true,
             true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in prior_box";
  }
}

TEST_F(prior_box, BAD_PARAM_variance_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true, true,
             true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in prior_box";
  }
}

TEST_F(prior_box, BAD_PARAM_max_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true, true,
             true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in prior_box";
  }
}

TEST_F(prior_box, BAD_PARAM_max_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true, true,
             true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in prior_box";
  }
}

TEST_F(prior_box, BAD_PARAM_output_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false, true,
             true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in prior_box";
  }
}

TEST_F(prior_box, BAD_PARAM_output_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, false,
             true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in prior_box";
  }
}

TEST_F(prior_box, BAD_PARAM_var_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in prior_box";
  }
}

TEST_F(prior_box, BAD_PARAM_var_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in prior_box";
  }
}
}  // namespace mluopapitest
