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
class indice_convolution_backward_filter : public testing::Test {
 public:
  void setParam(bool handle, bool features_desc, bool features,
                bool output_grad_desc, bool output_grad, bool indice_pairs_desc,
                bool indice_pairs, bool filters_grad_desc, bool filters_grad,
                bool indice_num, bool worksapce) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (features_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&features_desc_));
      std::vector<int> features_dims{3, 5};
      MLUOP_CHECK(mluOpSetTensorDescriptor(features_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           features_dims.data()));
    }

    if (features) {
      if (features_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&features_, mluOpGetTensorElementNum(features_desc_) *
                                       mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&features_, 64 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    if (output_grad_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_grad_desc_));
      std::vector<int> output_grad_dims{3, 7};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          output_grad_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
          output_grad_dims.data()));
    }

    if (output_grad) {
      if (output_grad_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&output_grad_,
                               mluOpGetTensorElementNum(output_grad_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&output_grad_,
                               64 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    if (indice_pairs_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&indice_pairs_desc_));
      std::vector<int> indice_pairs_dims{9, 2, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          indice_pairs_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 3,
          indice_pairs_dims.data()));
    }

    if (indice_pairs) {
      if (indice_pairs_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&indice_pairs_,
                               mluOpGetTensorElementNum(indice_pairs_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&indice_pairs_,
                               64 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    if (filters_grad_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&filters_grad_desc_));
      std::vector<int> filters_grad_dims{3, 3, 5, 7};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          filters_grad_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 4,
          filters_grad_dims.data()));
    }

    if (filters_grad) {
      if (filters_grad_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&filters_grad_,
                               mluOpGetTensorElementNum(filters_grad_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&filters_grad_,
                               64 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    std::vector<int> num = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 0; i < num.size(); i++) {
      if (indice_num) {
        indice_num_.push_back(num[i]);
      }
    }

    if (worksapce) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_));
    }
  }
  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpIndiceConvolutionBackwardFilter(
        handle_, features_desc_, features_, output_grad_desc_, output_grad_,
        indice_pairs_desc_, indice_pairs_, indice_num_.data(), inverse_, sub_m_,
        workspace_, workspace_size_, filters_grad_desc_, filters_grad_);
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

    if (features_desc_) {
      VLOG(4) << "Destroy features_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(features_desc_));
      features_desc_ = nullptr;
    }

    if (features_) {
      VLOG(4) << "Destroy features";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(features_));
      features_ = nullptr;
    }

    if (output_grad_desc_) {
      VLOG(4) << "Destroy output_grad_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_grad_desc_));
      output_grad_desc_ = nullptr;
    }

    if (output_grad_) {
      VLOG(4) << "Destroy output_grad";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_grad_));
      output_grad_ = nullptr;
    }

    if (indice_pairs_desc_) {
      VLOG(4) << "Destroy indice_pairs_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indice_pairs_desc_));
      indice_pairs_desc_ = nullptr;
    }

    if (indice_pairs_) {
      VLOG(4) << "Destroy indice_pairs";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(indice_pairs_));
      indice_pairs_ = nullptr;
    }

    if (workspace_) {
      VLOG(4) << "Destroy workspace";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = nullptr;
    }

    if (filters_grad_desc_) {
      VLOG(4) << "Destroy filters_grad_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(filters_grad_desc_));
      filters_grad_desc_ = nullptr;
    }

    if (filters_grad_) {
      VLOG(4) << "Destroy filters_grad";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(filters_grad_));
      filters_grad_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t features_desc_ = nullptr;
  void *features_ = nullptr;
  mluOpTensorDescriptor_t output_grad_desc_ = nullptr;
  void *output_grad_ = nullptr;
  mluOpTensorDescriptor_t indice_pairs_desc_ = nullptr;
  void *indice_pairs_ = nullptr;
  std::vector<int64_t> indice_num_;
  int64_t inverse_ = 0;
  int64_t sub_m_ = 0;
  void *workspace_ = nullptr;
  size_t workspace_size_ = 64;
  mluOpTensorDescriptor_t filters_grad_desc_ = nullptr;
  void *filters_grad_ = nullptr;
};

TEST_F(indice_convolution_backward_filter, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_backward_filter";
  }
}

TEST_F(indice_convolution_backward_filter, BAD_PARAM_features_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_backward_filter";
  }
}

TEST_F(indice_convolution_backward_filter, BAD_PARAM_features_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_backward_filter";
  }
}

TEST_F(indice_convolution_backward_filter, BAD_PARAM_output_grad_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_backward_filter";
  }
}

TEST_F(indice_convolution_backward_filter, BAD_PARAM_output_grad_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_backward_filter";
  }
}

TEST_F(indice_convolution_backward_filter, BAD_PARAM_indice_pairs_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_backward_filter";
  }
}

TEST_F(indice_convolution_backward_filter, BAD_PARAM_indice_pairs_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_backward_filter";
  }
}

TEST_F(indice_convolution_backward_filter, BAD_PARAM_filters_grad_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_backward_filter";
  }
}

TEST_F(indice_convolution_backward_filter, BAD_PARAM_filters_grad_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_backward_filter";
  }
}

TEST_F(indice_convolution_backward_filter, BAD_PARAM_indice_num_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_backward_filter";
  }
}

TEST_F(indice_convolution_backward_filter, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_backward_filter";
  }
}

}  // namespace mluopapitest
