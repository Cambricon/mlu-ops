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
class mutual_information_backward : public testing::Test {
 public:
  void setParam(bool handle, bool px_desc, bool px, bool py_desc, bool py,
                bool opt_boundary_desc, bool opt_boundary, bool p_desc, bool p,
                bool ans_grad_desc, bool ans_grad, bool px_grad_desc,
                bool px_grad, bool py_grad_desc, bool py_grad, bool workspace) {
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

    if (px) {
      if (px_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&px_, MLUOP_DTYPE_FLOAT *
                                         mluOpGetTensorElementNum(px_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&px_, MLUOP_DTYPE_FLOAT * 2));
      }
    }

    if (py_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&py_desc_));
      std::vector<int> py_desc_dims{4, 16, 104};
      MLUOP_CHECK(mluOpSetTensorDescriptor(py_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           py_desc_dims.data()));
    }

    if (py) {
      if (py_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&py_, MLUOP_DTYPE_FLOAT *
                                         mluOpGetTensorElementNum(py_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&py_, MLUOP_DTYPE_FLOAT * 2));
      }
    }

    if (opt_boundary_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&opt_boundary_desc_));
      std::vector<int> opt_boundary_desc_dims{4, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          opt_boundary_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT64, 2,
          opt_boundary_desc_dims.data()));
    }

    if (opt_boundary) {
      if (opt_boundary_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&opt_boundary_,
                               MLUOP_DTYPE_INT64 * mluOpGetTensorElementNum(
                                                       opt_boundary_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&opt_boundary_, MLUOP_DTYPE_INT64 * 2));
      }
    }

    if (p_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&p_desc_));
      std::vector<int> p_desc_dims{4, 16, 105};
      MLUOP_CHECK(mluOpSetTensorDescriptor(p_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           p_desc_dims.data()));
    }

    if (p) {
      if (p_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&p_, MLUOP_DTYPE_FLOAT *
                                        mluOpGetTensorElementNum(p_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&p_, MLUOP_DTYPE_FLOAT * 2));
      }
    }

    if (ans_grad_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&ans_grad_desc_));
      std::vector<int> ans_grad_desc_dims{4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(ans_grad_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 1,
                                           ans_grad_desc_dims.data()));
    }

    if (ans_grad) {
      if (ans_grad_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&ans_grad_,
                               MLUOP_DTYPE_FLOAT *
                                   mluOpGetTensorElementNum(ans_grad_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&ans_grad_, MLUOP_DTYPE_FLOAT * 2));
      }
    }

    if (px_grad_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&px_grad_desc_));
      std::vector<int> px_grad_desc_dims{4, 15, 105};
      MLUOP_CHECK(mluOpSetTensorDescriptor(px_grad_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           px_grad_desc_dims.data()));
    }

    if (px_grad) {
      if (px_grad_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&px_grad_, MLUOP_DTYPE_FLOAT *
                                      mluOpGetTensorElementNum(px_grad_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&px_grad_, MLUOP_DTYPE_FLOAT * 2));
      }
    }

    if (py_grad_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&py_grad_desc_));
      std::vector<int> py_grad_desc_dims{4, 16, 104};
      MLUOP_CHECK(mluOpSetTensorDescriptor(py_grad_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           py_grad_desc_dims.data()));
    }

    if (py_grad) {
      if (py_grad_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&py_grad_, MLUOP_DTYPE_FLOAT *
                                      mluOpGetTensorElementNum(py_grad_desc_)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&py_grad_, MLUOP_DTYPE_FLOAT * 2));
      }
    }

    if (workspace) {
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&workspace_, MLUOP_DTYPE_FLOAT * workspace_size_));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpMutualInformationBackward(
        handle_, px_desc_, px_, py_desc_, py_, opt_boundary_desc_,
        opt_boundary_, p_desc_, p_, ans_grad_desc_, ans_grad_,
        overwrite_ans_grad_, workspace_, workspace_size_, px_grad_desc_,
        px_grad_, py_grad_desc_, py_grad_);
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

    if (px_) {
      VLOG(4) << "Destroy px_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(px_));
      px_ = nullptr;
    }

    if (py_desc_) {
      VLOG(4) << "Destroy py_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(py_desc_));
      py_desc_ = nullptr;
    }

    if (py_) {
      VLOG(4) << "Destroy py_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(py_));
      py_ = nullptr;
    }

    if (opt_boundary_desc_) {
      VLOG(4) << "Destroy opt_boundary_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(opt_boundary_desc_));
      opt_boundary_desc_ = nullptr;
    }

    if (opt_boundary_) {
      VLOG(4) << "Destroy opt_boundary_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(opt_boundary_));
      opt_boundary_ = nullptr;
    }

    if (p_desc_) {
      VLOG(4) << "Destroy p_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(p_desc_));
      p_desc_ = nullptr;
    }

    if (p_) {
      VLOG(4) << "Destroy p_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(p_));
      p_ = nullptr;
    }

    if (ans_grad_desc_) {
      VLOG(4) << "Destroy ans_grad_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(ans_grad_desc_));
      ans_grad_desc_ = nullptr;
    }

    if (ans_grad_) {
      VLOG(4) << "Destroy ans_grad_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(ans_grad_));
      ans_grad_ = nullptr;
    }

    if (px_grad_desc_) {
      VLOG(4) << "Destroy px_grad_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(px_grad_desc_));
      px_grad_desc_ = nullptr;
    }

    if (px_grad_) {
      VLOG(4) << "Destroy px_grad_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(px_grad_));
      px_grad_ = nullptr;
    }

    if (py_grad_desc_) {
      VLOG(4) << "Destroy py_grad_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(py_grad_desc_));
      py_grad_desc_ = nullptr;
    }

    if (py_grad_) {
      VLOG(4) << "Destroy py_grad_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(py_grad_));
      py_grad_ = nullptr;
    }

    if (workspace_) {
      VLOG(4) << "Destroy workspace_";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t px_desc_ = nullptr;
  void *px_ = nullptr;
  mluOpTensorDescriptor_t py_desc_ = nullptr;
  void *py_ = nullptr;
  mluOpTensorDescriptor_t opt_boundary_desc_ = nullptr;
  void *opt_boundary_ = nullptr;
  mluOpTensorDescriptor_t p_desc_ = nullptr;
  void *p_ = nullptr;
  mluOpTensorDescriptor_t ans_grad_desc_ = nullptr;
  void *ans_grad_ = nullptr;
  mluOpTensorDescriptor_t px_grad_desc_ = nullptr;
  void *px_grad_ = nullptr;
  mluOpTensorDescriptor_t py_grad_desc_ = nullptr;
  void *py_grad_ = nullptr;
  size_t workspace_size_ = 10;
  void *workspace_ = nullptr;
  bool overwrite_ans_grad_ = true;
};

TEST_F(mutual_information_backward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_backward";
  }
}

TEST_F(mutual_information_backward, BAD_PARAM_px_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_backward";
  }
}

TEST_F(mutual_information_backward, BAD_PARAM_px_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_backward";
  }
}

TEST_F(mutual_information_backward, BAD_PARAM_py_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_backward";
  }
}

TEST_F(mutual_information_backward, BAD_PARAM_py_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_backward";
  }
}

TEST_F(mutual_information_backward, BAD_PARAM_opt_boundary_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_backward";
  }
}

TEST_F(mutual_information_backward, BAD_PARAM_voxel_opt_boundary_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_backward";
  }
}

TEST_F(mutual_information_backward, BAD_PARAM_p_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_backward";
  }
}

TEST_F(mutual_information_backward, BAD_PARAM_p_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_backward";
  }
}

TEST_F(mutual_information_backward, BAD_PARAM_ans_grad_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false, true,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_backward";
  }
}

TEST_F(mutual_information_backward, BAD_PARAM_ans_grad_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, false,
             true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_backward";
  }
}

TEST_F(mutual_information_backward, BAD_PARAM_px_grad_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             false, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_backward";
  }
}

TEST_F(mutual_information_backward, BAD_PARAM_px_grad_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, false, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_backward";
  }
}

TEST_F(mutual_information_backward, BAD_PARAM_py_grad_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, false, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_backward";
  }
}

TEST_F(mutual_information_backward, BAD_PARAM_py_grad_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, true, false, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_backward";
  }
}

TEST_F(mutual_information_backward, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, true, true, false);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in mutual_information_backward";
  }
}
}  // namespace mluopapitest
