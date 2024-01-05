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
class active_rotated_filter_forward : public testing::Test {
 public:
  void setParam(bool handle, bool input_desc, bool input, bool indices_desc,
                bool indices, bool workspace, bool output_desc, bool output) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (input_desc) {
      input_desc_ = nullptr;
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
      std::vector<int> input_dims{256, 200};
      MLUOP_CHECK(mluOpSetTensorDescriptor(input_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, input_dims.size(),
                                           input_dims.data()));
    }

    if (input) {
      uint64_t i_bytes;
      if (input_desc) {
        i_bytes = mluOpGetTensorElementNum(input_desc_) *
                  mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      } else {
        i_bytes = 12 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      }
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&input_, i_bytes))
    }

    if (indices_desc) {
      indices_desc_ = nullptr;
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&indices_desc_));
      std::vector<int> indices_dims{200};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          indices_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
          indices_dims.size(), indices_dims.data()));
    }

    if (indices) {
      uint64_t id_bytes;
      if (indices_desc) {
        id_bytes = mluOpGetTensorElementNum(indices_desc_) *
                   mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      } else {
        id_bytes = 12 * mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      }
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&indices_, id_bytes))
    }

    if (workspace) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_size_))
    }

    if (output_desc) {
      output_desc_ = nullptr;
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));
      std::vector<int> output_dims{1, 256, 20, 20};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          output_desc_, MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT,
          output_dims.size(), output_dims.data()));
    }

    if (output) {
      uint64_t o_bytes;
      if (output_desc) {
        o_bytes = mluOpGetTensorElementNum(output_desc_) *
                  mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      } else {
        o_bytes = 12 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      }
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&input_, o_bytes))
    }
  }
  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpActiveRotatedFilterForward(
        handle_, input_desc_, input_, indices_desc_, indices_, workspace_,
        workspace_size_, output_desc_, output_);
    destroy();
    return status;
  }

 private:
  virtual void SetUp() {
    handle_ = nullptr;
    input_desc_ = nullptr;
    input_ = nullptr;
    indices_desc_ = nullptr;
    indices_ = nullptr;
    workspace_ = nullptr;
    output_desc_ = nullptr;
    output_ = nullptr;
  }

  void destroy() {
    try {
      uint32_t i = 0;
      VLOG(4) << "Destroy parameters";
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
      if (input_) {
        VLOG(4) << "Destroy input_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_));
        input_ = nullptr;
      }
      if (indices_desc_) {
        VLOG(4) << "Destroy indices_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(indices_desc_));
        indices_desc_ = nullptr;
      }
      if (indices_) {
        VLOG(4) << "Destroy indices_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(indices_));
        indices_ = nullptr;
      }
      if (workspace_) {
        VLOG(4) << "Destroy workspace_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
        workspace_ = nullptr;
      }
      if (output_desc_) {
        VLOG(4) << "Destroy output_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
        output_desc_ = nullptr;
      }
      if (output_) {
        VLOG(4) << "Destroy output_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_));
        output_ = nullptr;
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in active_rotated_filter_forward";
    }
  }

  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t input_desc_ = nullptr;
  void *input_ = nullptr;
  mluOpTensorDescriptor_t indices_desc_ = nullptr;
  void *indices_ = nullptr;
  void *workspace_ = nullptr;
  size_t workspace_size_ = 64;
  mluOpTensorDescriptor_t output_desc_ = nullptr;
  void *output_ = nullptr;
};

TEST_F(active_rotated_filter_forward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in active_rotated_filter_forward";
  }
}

TEST_F(active_rotated_filter_forward, BAD_PARAM_input_descs_null) {
  try {
    setParam(true, false, true, true, true, true, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in active_rotated_filter_forward";
  }
}

TEST_F(active_rotated_filter_forward, BAD_PARAM_input_null) {
  try {
    setParam(true, true, false, true, true, true, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in active_rotated_filter_forward";
  }
}

TEST_F(active_rotated_filter_forward, BAD_PARAM_indices_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in active_rotated_filter_forward";
  }
}

TEST_F(active_rotated_filter_forward, BAD_PARAM_indices_null) {
  try {
    setParam(true, true, true, true, false, true, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in active_rotated_filter_forward";
  }
}

TEST_F(active_rotated_filter_forward, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, true, true, true, false, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in active_rotated_filter_forward";
  }
}

TEST_F(active_rotated_filter_forward, BAD_PARAM_output_desc_null) {
  try {
    setParam(true, true, true, true, true, true, false, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in active_rotated_filter_forward";
  }
}

TEST_F(active_rotated_filter_forward, BAD_PARAM_output_null) {
  try {
    setParam(true, true, true, true, true, true, true, false);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in active_rotated_filter_forward";
  }
}

}  // namespace mluopapitest
