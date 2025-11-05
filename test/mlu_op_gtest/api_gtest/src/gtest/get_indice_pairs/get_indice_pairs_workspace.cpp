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
class get_indice_pairs_workspace : public testing::Test {
 public:
  void setParam(bool handle, bool sparse_conv_desc, bool indices_desc,
                bool indice_pairs_desc, bool out_indices_desc,
                bool indice_num_desc, bool workspace_size) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (sparse_conv_desc) {
      MLUOP_CHECK(mluOpCreateSparseConvolutionDescriptor(&sparse_conv_desc_));
      std::vector<int> pad{1, 1, 1};
      std::vector<int> stride{1, 1, 1};
      std::vector<int> dilation{1, 1, 1};
      std::vector<int> input_space{1, 1, 1};
      std::vector<int> filter_space{1, 1, 1};
      std::vector<int> output_space{3, 3, 3};
      MLUOP_CHECK(mluOpSetSparseConvolutionDescriptor(
          sparse_conv_desc_, 5, 1, pad.data(), stride.data(), dilation.data(),
          input_space.data(), filter_space.data(), output_space.data(), 0, 0,
          0));
    }
    if (indices_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&indices_desc_));
      std::vector<int> indices_dims{1, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(indices_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 2,
                                           indices_dims.data()));
    }
    if (indice_pairs_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&indice_pairs_desc_));
      std::vector<int> indice_pairs_dims{1, 2, 1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          indice_pairs_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 3,
          indice_pairs_dims.data()));
    }
    if (out_indices_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&out_indices_desc_));
      std::vector<int> out_indices_dims{27, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          out_indices_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 2,
          out_indices_dims.data()));
    }
    if (indice_num_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&indice_num_desc_));
      std::vector<int> indice_num_dims{1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(indice_num_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 1,
                                           indice_num_dims.data()));
    }
    if (workspace_size) {
      size_t size;
      workspace_size_ = &size;
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpGetIndicePairsWorkspaceSize(
        handle_, sparse_conv_desc_, indices_desc_, indice_pairs_desc_,
        out_indices_desc_, indice_num_desc_, workspace_size_);
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
    if (sparse_conv_desc_) {
      VLOG(4) << "Destroy sparse_conv_desc";
      MLUOP_CHECK(mluOpDestroySparseConvolutionDescriptor(sparse_conv_desc_));
      sparse_conv_desc_ = nullptr;
    }
    if (indices_desc_) {
      VLOG(4) << "Destroy indices_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indices_desc_));
      indices_desc_ = nullptr;
    }
    if (indice_pairs_desc_) {
      VLOG(4) << "Destroy indice_pairs_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indice_pairs_desc_));
      indice_pairs_desc_ = nullptr;
    }
    if (out_indices_desc_) {
      VLOG(4) << "Destroy out_indices_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(out_indices_desc_));
      out_indices_desc_ = nullptr;
    }
    if (indice_num_desc_) {
      VLOG(4) << "Destroy indice_num_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indice_num_desc_));
      indice_num_desc_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpSparseConvolutionDescriptor_t sparse_conv_desc_ = nullptr;
  mluOpTensorDescriptor_t indices_desc_ = nullptr;
  mluOpTensorDescriptor_t indice_pairs_desc_ = nullptr;
  mluOpTensorDescriptor_t out_indices_desc_ = nullptr;
  mluOpTensorDescriptor_t indice_num_desc_ = nullptr;
  size_t *workspace_size_ = nullptr;
};

TEST_F(get_indice_pairs_workspace, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in get_indice_pairs_workspace";
  }
}

TEST_F(get_indice_pairs_workspace, BAD_PARAM_sparse_conv_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in get_indice_pairs_workspace";
  }
}

TEST_F(get_indice_pairs_workspace, BAD_PARAM_indices_desc_null) {
  try {
    setParam(true, true, false, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in get_indice_pairs_workspace";
  }
}

TEST_F(get_indice_pairs_workspace, BAD_PARAM_indice_pairs_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in get_indice_pairs_workspace";
  }
}

TEST_F(get_indice_pairs_workspace, BAD_PARAM_out_indices_desc_null) {
  try {
    setParam(true, true, true, true, false, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in get_indice_pairs_workspace";
  }
}

TEST_F(get_indice_pairs_workspace, BAD_PARAM_indice_num_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in get_indice_pairs_workspace";
  }
}

TEST_F(get_indice_pairs_workspace, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, true, true, true, true, false);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in get_indice_pairs_workspace";
  }
}

}  // namespace mluopapitest

