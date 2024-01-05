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
class indice_convolution_backward_data : public testing::Test {
 public:
  void setParam(bool handle, bool output_grad_desc, bool output_grad,
                bool filters_desc, bool filters, bool indice_pairs_desc,
                bool indice_pairs, bool workspace, bool input_grad_desc,
                bool input_grad) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (output_grad_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_grad_desc_));
      std::vector<int> output_grad_shape = {10, 10};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          output_grad_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
          output_grad_shape.data()));
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
                               100 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }
    if (filters_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&filters_desc_));
      std::vector<int> filters_shape = {3, 3, 21, 10};
      MLUOP_CHECK(mluOpSetTensorDescriptor(filters_desc_, MLUOP_LAYOUT_HWCN,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           filters_shape.data()));
    }
    if (filters) {
      if (filters_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&filters_, mluOpGetTensorElementNum(filters_desc_) *
                                      mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&filters_,
                               1890 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }
    if (indice_pairs_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&indice_pairs_desc_));
      std::vector<int> indice_pairs_shape = {9, 2, 10};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          indice_pairs_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 3,
          indice_pairs_shape.data()));
    }
    if (indice_pairs) {
      if (indice_pairs_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&indice_pairs_,
                               mluOpGetTensorElementNum(indice_pairs_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&indice_pairs_,
                               180 * mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      }
    }
    if (input_grad_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_grad_desc_));
      std::vector<int> input_grad_shape = {10, 21};
      MLUOP_CHECK(mluOpSetTensorDescriptor(input_grad_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           input_grad_shape.data()));
    }
    if (input_grad) {
      if (input_grad_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&input_grad_,
                               mluOpGetTensorElementNum(input_grad_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&input_grad_,
                               210 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }
    if (workspace) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpIndiceConvolutionBackwardData(
        handle_, output_grad_desc_, output_grad_, filters_desc_, filters_,
        indice_pairs_desc_, indice_pairs_, indice_num_, inverse_, sub_m_,
        workspace_, workspace_size_, input_grad_desc_, input_grad_);
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
    if (output_grad_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_grad_desc_));
      output_grad_desc_ = NULL;
    }
    if (output_grad_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_grad_));
      output_grad_ = NULL;
    }
    if (filters_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(filters_desc_));
      filters_desc_ = NULL;
    }
    if (filters_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(filters_));
      filters_ = NULL;
    }
    if (indice_pairs_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indice_pairs_desc_));
      indice_pairs_desc_ = NULL;
    }
    if (indice_pairs_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(indice_pairs_));
      indice_pairs_ = NULL;
    }
    if (workspace_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = nullptr;
    }
    if (input_grad_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_grad_desc_));
      input_grad_desc_ = NULL;
    }
    if (input_grad_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_grad_));
      input_grad_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t output_grad_desc_ = NULL;
  void* output_grad_ = NULL;
  mluOpTensorDescriptor_t filters_desc_ = NULL;
  void* filters_ = NULL;
  mluOpTensorDescriptor_t indice_pairs_desc_ = NULL;
  void* indice_pairs_ = NULL;
  int64_t indice_num_[9] = {10};
  int64_t inverse_ = 0;
  int64_t sub_m_ = 0;
  void* workspace_ = NULL;
  size_t workspace_size_ = 64;
  mluOpTensorDescriptor_t input_grad_desc_ = NULL;
  void* input_grad_ = NULL;
};

TEST_F(indice_convolution_backward_data, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in indices_convolution_backward_data";
  }
}

TEST_F(indice_convolution_backward_data, BAD_PARAM_output_grad_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in indices_convolution_backward_data";
  }
}

TEST_F(indice_convolution_backward_data, BAD_PARAM_output_grad_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in indices_convolution_backward_data";
  }
}

TEST_F(indice_convolution_backward_data, BAD_PARAM_filters_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in indices_convolution_backward_data";
  }
}

TEST_F(indice_convolution_backward_data, BAD_PARAM_filters_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in indices_convolution_backward_data";
  }
}

TEST_F(indice_convolution_backward_data, BAD_PARAM_indice_pairs_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in indices_convolution_backward_data";
  }
}

TEST_F(indice_convolution_backward_data, BAD_PARAM_indice_pairs_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in indices_convolution_backward_data";
  }
}

TEST_F(indice_convolution_backward_data, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in indices_convolution_backward_data";
  }
}

TEST_F(indice_convolution_backward_data, BAD_PARAM_input_grad_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in indices_convolution_backward_data";
  }
}

TEST_F(indice_convolution_backward_data, BAD_PARAM_input_grad_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in indices_convolution_backward_data";
  }
}
}  // namespace mluopapitest
