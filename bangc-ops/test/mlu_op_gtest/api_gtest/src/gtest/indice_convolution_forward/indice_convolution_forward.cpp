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
class indice_convolution_forward : public testing::Test {
 public:
  void setParam(bool handle, bool features_desc, bool features,
                bool filters_desc, bool filters, bool indice_pairs_desc,
                bool indice_pairs, bool features_out_desc, bool features_out,
                bool indice_num, bool worksapce) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (features_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&features_desc_));
      std::vector<int> features_dims{2, 7};
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

    if (filters_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&filters_desc_));
      std::vector<int> filters_dims{9, 2, 2, 2, 7};
      MLUOP_CHECK(mluOpSetTensorDescriptor(filters_desc_, MLUOP_LAYOUT_NDHWC,
                                           MLUOP_DTYPE_FLOAT, 5,
                                           filters_dims.data()));
    }

    if (filters) {
      if (filters_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&filters_, mluOpGetTensorElementNum(filters_desc_) *
                                      mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&filters_, 64 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    if (indice_pairs_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&indice_pairs_desc_));
      std::vector<int> indice_pairs_dims{8, 2, 2};
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

    if (features_out_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&features_out_desc_));
      std::vector<int> features_out_dims{10, 9};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          features_out_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
          features_out_dims.data()));
    }

    if (features_out) {
      if (features_out_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&features_out_,
                               mluOpGetTensorElementNum(features_out_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&features_out_,
                               64 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    std::vector<int> num = {1, 1, 1, 1, 1, 1, 1, 1};
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
    mluOpStatus_t status = mluOpIndiceConvolutionForward(
        handle_, features_desc_, features_, filters_desc_, filters_,
        indice_pairs_desc_, indice_pairs_, indice_num_.data(), num_act_out_,
        inverse_, sub_m_, workspace_, workspace_size_, features_out_desc_,
        features_out_);
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

    if (filters_desc_) {
      VLOG(4) << "Destroy filters_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(filters_desc_));
      filters_desc_ = nullptr;
    }

    if (filters_) {
      VLOG(4) << "Destroy filters";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(filters_));
      filters_ = nullptr;
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

    if (features_out_desc_) {
      VLOG(4) << "Destroy features_out_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(features_out_desc_));
      features_out_desc_ = nullptr;
    }

    if (features_out_) {
      VLOG(4) << "Destroy features_out";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(features_out_));
      features_out_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t features_desc_ = nullptr;
  void *features_ = nullptr;
  mluOpTensorDescriptor_t filters_desc_ = nullptr;
  void *filters_ = nullptr;
  mluOpTensorDescriptor_t indice_pairs_desc_ = nullptr;
  void *indice_pairs_ = nullptr;
  std::vector<int64_t> indice_num_;
  int64_t num_act_out_ = 10;
  int64_t inverse_ = 0;
  int64_t sub_m_ = 0;
  void *workspace_ = nullptr;
  size_t workspace_size_ = 64;
  mluOpTensorDescriptor_t features_out_desc_ = nullptr;
  void *features_out_ = nullptr;
};

TEST_F(indice_convolution_forward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_forward";
  }
}

TEST_F(indice_convolution_forward, BAD_PARAM_features_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_forward";
  }
}

TEST_F(indice_convolution_forward, BAD_PARAM_features_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_forward";
  }
}

TEST_F(indice_convolution_forward, BAD_PARAM_filters_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_forward";
  }
}

TEST_F(indice_convolution_forward, BAD_PARAM_filters_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_forward";
  }
}

TEST_F(indice_convolution_forward, BAD_PARAM_indice_pairs_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_forward";
  }
}

TEST_F(indice_convolution_forward, BAD_PARAM_indice_pairs_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_forward";
  }
}

TEST_F(indice_convolution_forward, BAD_PARAM_features_out_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_forward";
  }
}

TEST_F(indice_convolution_forward, BAD_PARAM_features_out_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_forward";
  }
}

TEST_F(indice_convolution_forward, BAD_PARAM_indice_num_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_forward";
  }
}

TEST_F(indice_convolution_forward, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indice_convolution_forward";
  }
}

}  // namespace mluopapitest
