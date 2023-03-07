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
class indice_convolution_backward_data_workspace : public testing::Test {
 public:
  void setParam(bool handle, bool output_grad_desc, bool filters_desc,
                bool indice_pairs_desc, bool input_grad_desc,
                bool workspace_size) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (output_grad_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_grad_desc_));
      std::vector<int> output_grad_dims{10, 10};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          output_grad_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
          output_grad_dims.data()));
    }
    if (filters_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&filters_desc_));
      std::vector<int> filters_dims{3, 3, 21, 10};
      MLUOP_CHECK(mluOpSetTensorDescriptor(filters_desc_, MLUOP_LAYOUT_HWCN,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           filters_dims.data()));
    }
    if (indice_pairs_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&indice_pairs_desc_));
      std::vector<int> indice_pairs_dims{9, 2, 10};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          indice_pairs_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 3,
          indice_pairs_dims.data()));
    }
    if (input_grad_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_grad_desc_));
      std::vector<int> input_grad_dims{10, 21};
      MLUOP_CHECK(mluOpSetTensorDescriptor(input_grad_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           input_grad_dims.data()));
    }
    if (workspace_size) {
      size_t size;
      workspace_size_ = &size;
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpGetIndiceConvolutionBackwardDataWorkspaceSize(
        handle_, output_grad_desc_, filters_desc_, indice_pairs_desc_,
        input_grad_desc_, indice_num_, inverse_, workspace_size_);
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

    if (output_grad_desc_) {
      VLOG(4) << "Destroy output_grad_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_grad_desc_));
      output_grad_desc_ = nullptr;
    }

    if (filters_desc_) {
      VLOG(4) << "Destroy filters_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(filters_desc_));
      filters_desc_ = nullptr;
    }

    if (indice_pairs_desc_) {
      VLOG(4) << "Destroy indice_pairs_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indice_pairs_desc_));
      indice_pairs_desc_ = nullptr;
    }

    if (input_grad_desc_) {
      VLOG(4) << "Destroy input_grad_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_grad_desc_));
      input_grad_desc_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t output_grad_desc_ = nullptr;
  mluOpTensorDescriptor_t filters_desc_ = nullptr;
  mluOpTensorDescriptor_t indice_pairs_desc_ = nullptr;
  mluOpTensorDescriptor_t input_grad_desc_ = nullptr;
  int64_t indice_num_[9] = {10};
  int64_t inverse_ = 0;
  size_t *workspace_size_ = nullptr;
};

TEST_F(indice_convolution_backward_data_workspace, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indices_convolution_backward_data_workspace";
  }
}

TEST_F(indice_convolution_backward_data_workspace,
       BAD_PARAM_output_grad_desc_null) {
  try {
    setParam(true, false, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indices_convolution_backward_data_workspace";
  }
}

TEST_F(indice_convolution_backward_data_workspace,
       BAD_PARAM_filters_desc_null) {
  try {
    setParam(true, true, false, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indices_convolution_backward_data_workspace";
  }
}

TEST_F(indice_convolution_backward_data_workspace,
       BAD_PARAM_indice_pairs_desc_null) {
  try {
    setParam(true, true, true, false, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indices_convolution_backward_data_workspace";
  }
}

TEST_F(indice_convolution_backward_data_workspace,
       BAD_PARAM_input_grad_desc_null) {
  try {
    setParam(true, true, true, true, false, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indices_convolution_backward_data_workspace";
  }
}

TEST_F(indice_convolution_backward_data_workspace, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, true, true, true, false);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in indices_convolution_backward_data_workspace";
  }
}

}  // namespace mluopapitest
