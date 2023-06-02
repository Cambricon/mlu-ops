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
class mutual_information_forward_workspace : public testing::Test {
 public:
  void setParam(bool handle, bool px_desc, bool py_desc, bool opt_boundary_desc,
                bool p_desc, bool ans_desc, bool workspace_size) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (px_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&px_desc_));
      std::vector<int> px_desc_dims{4, 15, 105};
      MLUOP_CHECK(mluOpSetTensorDescriptor(px_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           px_desc_dims.data()));
    }

    if (py_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&py_desc_));
      std::vector<int> py_desc_dims{4, 16, 106};
      MLUOP_CHECK(mluOpSetTensorDescriptor(py_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           py_desc_dims.data()));
    }

    if (opt_boundary_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&opt_boundary_desc_));
      std::vector<int> opt_boundary_desc_dims{4, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          opt_boundary_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
          opt_boundary_desc_dims.data()));
    }

    if (p_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&p_desc_));
      std::vector<int> p_desc_dims{4, 16, 105};
      MLUOP_CHECK(mluOpSetTensorDescriptor(p_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           p_desc_dims.data()));
    }

    if (ans_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&ans_desc_));
      std::vector<int> ans_desc_dims{4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(ans_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 1,
                                           ans_desc_dims.data()));
    }

    if (workspace_size) {
      workspace_size_ = &workspace_size__;
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpGetMutualInformationForwardWorkspaceSize(
        handle_, px_desc_, py_desc_, opt_boundary_desc_, p_desc_,
        ans_desc_, workspace_size_);
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

    if (px_desc_) {
      VLOG(4) << "Destroy px_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(px_desc_));
      px_desc_ = nullptr;
    }

    if (py_desc_) {
      VLOG(4) << "Destroy py_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(py_desc_));
      py_desc_ = nullptr;
    }

    if (opt_boundary_desc_) {
      VLOG(4) << "Destroy opt_boundary_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(opt_boundary_desc_));
      opt_boundary_desc_ = nullptr;
    }

    if (p_desc_) {
      VLOG(4) << "Destroy p_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(p_desc_));
      p_desc_ = nullptr;
    }

    if (ans_desc_) {
      VLOG(4) << "Destroy ans_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(ans_desc_));
      ans_desc_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t px_desc_ = nullptr;
  mluOpTensorDescriptor_t py_desc_ = nullptr;
  mluOpTensorDescriptor_t opt_boundary_desc_ = nullptr;
  mluOpTensorDescriptor_t p_desc_ = nullptr;
  mluOpTensorDescriptor_t ans_desc_ = nullptr;
  size_t workspace_size__ = 10;
  size_t *workspace_size_ = nullptr;
};

TEST_F(mutual_information_forward_workspace, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_forward_workspace";
  }
}

TEST_F(mutual_information_forward_workspace, BAD_PARAM_px_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_forward_workspace";
  }
}

TEST_F(mutual_information_forward_workspace, BAD_PARAM_py_desc_null) {
  try {
    setParam(true, true, false, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_forward_workspace";
  }
}

TEST_F(mutual_information_forward_workspace, BAD_PARAM_p_desc_null) {
  try {
    setParam(true, true, true, true, false, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_forward_workspace";
  }
}

TEST_F(mutual_information_forward_workspace, BAD_PARAM_ans_grad_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_forward_workspace";
  }
}

TEST_F(mutual_information_forward_workspace, BAD_PARAM_workspace_size_null) {
  try {
    setParam(true, true, true, true, true, true, false);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_forward_workspace";
  }
}
}  // namespace mluopapitest
