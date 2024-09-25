/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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

#include "gtest/gtest.h"
#include "mlu_op.h"
#include "api_test_tools.h"
#include "core/context.h"
#include "core/logging.h"
#include "core/tensor.h"

namespace mluopapitest {
class fft_ExecFFT : public testing::Test {
 public:
  void setParam(bool handle, bool fft_plan, bool input, bool workspace,
                bool output) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (fft_plan) {
      MLUOP_CHECK(mluOpCreateFFTPlan(&fft_plan_));
    }

    if (input) {
      size_t i_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&input_, i_bytes));
    }

    if (workspace) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_));
    }

    if (output) {
      size_t o_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_COMPLEX_FLOAT);
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&output_, o_bytes));
    }
  }

  mluOpStatus_t compute() {
    mluOpDataType_t input_data_type = MLUOP_DTYPE_FLOAT;
    mluOpDataType_t output_data_type = MLUOP_DTYPE_COMPLEX_FLOAT;
    mluOpDataType_t execution_dtype = MLUOP_DTYPE_FLOAT;
    const int rank = 1;
    const int batch = 2000;
    const int n[rank] = {400};
    const int ndim = rank + 1;
    const int64_t input_dim_size[ndim] = {batch, n[0] / 2 + 1};
    const int64_t input_dim_stride[ndim] = {n[0] / 2 + 1, 1};

    const int64_t output_dim_size[ndim] = {batch, n[0] / 2 + 1};
    const int64_t output_dim_stride[ndim] = {n[0] / 2 + 1, 1};

    mluOpCreateTensorDescriptor(&input_desc_);
    mluOpCreateTensorDescriptor(&output_desc_);
    mluOpSetTensorDescriptorEx_v2(input_desc_, MLUOP_LAYOUT_ARRAY,
                                  input_data_type, ndim, input_dim_size,
                                  input_dim_stride);
    mluOpSetTensorDescriptorOnchipDataType(input_desc_, execution_dtype);
    mluOpSetTensorDescriptorEx_v2(output_desc_, MLUOP_LAYOUT_ARRAY,
                                  output_data_type, ndim, output_dim_size,
                                  output_dim_stride);
    size_t reservespaceSizeInBytes_ = 64;
    size_t workspaceSizeInBytes_ = 64;
    size_t *reservespace_size = &reservespaceSizeInBytes_;
    size_t *workspace_size = &workspaceSizeInBytes_;

    mluOpStatus_t status;
    if (handle_ != nullptr && fft_plan_ != nullptr) {
      status =
          mluOpMakeFFTPlanMany(handle_, fft_plan_, input_desc_, output_desc_,
                               rank, n, reservespace_size, workspace_size);

      if (status != MLUOP_STATUS_SUCCESS) {
        destroy();
        return status;
      }
    }

    status = mluOpExecFFT(handle_, fft_plan_, input_, scale_factor_, workspace_,
                          output_, direction_);
    destroy();

    return status;
  }

 protected:
  virtual void SetUp() {
    handle_ = nullptr;
    fft_plan_ = nullptr;
    input_ = nullptr;
    workspace_ = nullptr;
    output_ = nullptr;
  }

  void destroy() {
    try {
      if (handle_) {
        CNRT_CHECK(cnrtQueueSync(handle_->queue));
        VLOG(4) << "Destroy handle_";
        MLUOP_CHECK(mluOpDestroy(handle_));
        handle_ = nullptr;
      }
      if (input_desc_) {
        VLOG(4) << "Destroy input_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_desc_));
        input_desc_ = nullptr;
      }
      if (output_desc_) {
        VLOG(4) << "Destroy output_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
        output_desc_ = nullptr;
      }
      if (input_) {
        VLOG(4) << "Destroy input_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_));
        input_ = nullptr;
      }
      if (output_) {
        VLOG(4) << "Destroy output_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_));
        output_ = nullptr;
      }
      if (fft_plan_) {
        VLOG(4) << "Destroy fft_plan_";
        MLUOP_CHECK(mluOpDestroyFFTPlan(fft_plan_));
        fft_plan_ = nullptr;
      }
      if (workspace_) {
        VLOG(4) << "Destroy workspace_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
        workspace_ = nullptr;
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in fft_ExecFFT";
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpFFTPlan_t fft_plan_ = nullptr;
  mluOpTensorDescriptor_t input_desc_ = nullptr;
  mluOpTensorDescriptor_t output_desc_ = nullptr;
  void *input_ = nullptr;
  float scale_factor_ = 0.1;
  void *workspace_ = nullptr;
  void *output_ = nullptr;
  int direction_ = 0;
  size_t workspace_size_ = 64;
};

TEST_F(fft_ExecFFT, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in fft_ExecFFT";
  }
}

TEST_F(fft_ExecFFT, BAD_PARAM_fft_plan_null) {
  try {
    setParam(true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in fft_ExecFFT";
  }
}

TEST_F(fft_ExecFFT, BAD_PARAM_input_null) {
  try {
    setParam(true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in fft_ExecFFT";
  }
}

TEST_F(fft_ExecFFT, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in fft_ExecFFT";
  }
}

TEST_F(fft_ExecFFT, BAD_PARAM_output_null) {
  try {
    setParam(true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in fft_ExecFFT";
  }
}

}  // namespace mluopapitest
