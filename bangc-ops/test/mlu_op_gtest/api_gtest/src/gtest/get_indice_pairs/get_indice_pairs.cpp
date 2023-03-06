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
class get_indice_pairs : public testing::Test {
 public:
  void setParam(bool handle, bool sparse_conv_desc, bool indices_desc,
                bool indices, bool workspace, bool indice_pairs_desc,
                bool indice_pairs, bool out_indices_desc, bool out_indices,
                bool indice_num_desc, bool indice_num) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (sparse_conv_desc) {
      MLUOP_CHECK(mluOpCreateSparseConvolutionDescriptor(&sparse_conv_desc_));
      std::vector<int> pad{1, 1, 1};
      std::vector<int> stride{1, 1, 1};
      std::vector<int> dilation{1, 1, 1};
      std::vector<int> input_space{1, 1, 1};
      std::vector<int> filter_space{3, 3, 3};
      std::vector<int> output_spcae{3, 3, 3};
      MLUOP_CHECK(mluOpSetSparseConvolutionDescriptor(
          sparse_conv_desc_, 5, 1, pad.data(), stride.data(), dilation.data(),
          input_space.data(), filter_space.data(), output_spcae.data(), 0, 0,
          0));
    }
    if (indices_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&indices_desc_));
      std::vector<int> indices_shape = {1, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(indices_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 2,
                                           indices_shape.data()));
    }
    if (indices) {
      if (indices_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&indices_, mluOpGetTensorElementNum(indices_desc_) *
                                      mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&indices_, 4 * mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      }
    }
    if (workspace) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_));
    }
    if (indice_pairs_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&indice_pairs_desc_));
      std::vector<int> indice_pairs_shape = {27, 2, 1};
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
                               54 * mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      }
    }
    if (out_indices_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&out_indices_desc_));
      std::vector<int> out_indices_shape = {27, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          out_indices_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 2,
          out_indices_shape.data()));
    }
    if (out_indices) {
      if (out_indices_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&out_indices_,
                               mluOpGetTensorElementNum(out_indices_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&out_indices_,
                               180 * mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      }
    }
    if (indice_num_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&indice_num_desc_));
      std::vector<int> indice_num_shape = {27};
      MLUOP_CHECK(mluOpSetTensorDescriptor(indice_num_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 1,
                                           indice_num_shape.data()));
    }
    if (indice_num) {
      if (indice_num_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&indice_num_,
                               mluOpGetTensorElementNum(indice_num_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&indice_num_,
                               27 * mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      }
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpGetIndicePairs(
        handle_, sparse_conv_desc_, indices_desc_, indices_, workspace_,
        workspace_size_, indice_pairs_desc_, indice_pairs_, out_indices_desc_,
        out_indices_, indice_num_desc_, indice_num_);
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
    if (sparse_conv_desc_) {
      MLUOP_CHECK(mluOpDestroySparseConvolutionDescriptor(sparse_conv_desc_));
      sparse_conv_desc_ = nullptr;
    }
    if (indices_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indices_desc_));
      indices_desc_ = NULL;
    }
    if (indices_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(indices_));
      indices_ = NULL;
    }
    if (workspace_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = nullptr;
    }
    if (indice_pairs_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indice_pairs_desc_));
      indice_pairs_desc_ = NULL;
    }
    if (indice_pairs_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(indice_pairs_));
      indice_pairs_ = NULL;
    }
    if (out_indices_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(out_indices_desc_));
      out_indices_desc_ = NULL;
    }
    if (out_indices_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(out_indices_));
      out_indices_ = NULL;
    }
    if (indice_num_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indice_num_desc_));
      indice_num_desc_ = NULL;
    }
    if (indice_num_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(indice_num_));
      indice_num_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpSparseConvolutionDescriptor_t sparse_conv_desc_ = NULL;
  mluOpTensorDescriptor_t indices_desc_ = NULL;
  void* indices_ = NULL;
  void* workspace_ = NULL;
  size_t workspace_size_ = 64;
  mluOpTensorDescriptor_t indice_pairs_desc_ = NULL;
  void* indice_pairs_ = NULL;
  mluOpTensorDescriptor_t out_indices_desc_ = NULL;
  void* out_indices_ = NULL;
  mluOpTensorDescriptor_t indice_num_desc_ = NULL;
  void* indice_num_ = NULL;
};

TEST_F(get_indice_pairs, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in get_indice_pairs";
  }
}

TEST_F(get_indice_pairs, BAD_PARAM_sparse_conv_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in get_indice_pairs";
  }
}

TEST_F(get_indice_pairs, BAD_PARAM_indices_desc_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in get_indice_pairs";
  }
}

TEST_F(get_indice_pairs, BAD_PARAM_indices_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in get_indice_pairs";
  }
}

TEST_F(get_indice_pairs, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in get_indice_pairs";
  }
}

TEST_F(get_indice_pairs, BAD_PARAM_indice_pairs_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in get_indice_pairs";
  }
}

TEST_F(get_indice_pairs, BAD_PARAM_indice_pairs_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in get_indice_pairs";
  }
}

TEST_F(get_indice_pairs, BAD_PARAM_out_indices_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in get_indice_pairs";
  }
}

TEST_F(get_indice_pairs, BAD_PARAM_out_indices_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in get_indice_pairs";
  }
}

TEST_F(get_indice_pairs, BAD_PARAM_indice_num_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in get_indice_pairs";
  }
}

TEST_F(get_indice_pairs, BAD_PARAM_indice_num_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in get_indice_pairs";
  }
}
}  // namespace mluopapitest

