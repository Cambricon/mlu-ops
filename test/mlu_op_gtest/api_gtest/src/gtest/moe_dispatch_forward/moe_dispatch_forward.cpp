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
class moe_dispatch_forward : public testing::Test {
 public:
  void setParam(bool handle, bool gates_desc, bool gates, bool indices_desc,
                bool indices, bool locations_desc, bool locations,
                bool input_desc, bool input, bool dispatch_desc,
                bool dispatch) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (gates_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&gates_desc_));
      std::vector<int> gates_shape = {2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(gates_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 1,
                                           gates_shape.data()));
    }
    if (gates) {
      if (gates_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&gates_, mluOpGetTensorElementNum(gates_desc_) *
                                    mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&gates_, 2 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }
    if (indices_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&indices_desc_));
      std::vector<int> indices_shape = {2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(indices_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 1,
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
            cnrtMalloc(&indices_, 2 * mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      }
    }
    if (locations_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&locations_desc_));
      std::vector<int> locations_shape = {2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(locations_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 1,
                                           locations_shape.data()));
    }
    if (locations) {
      if (locations_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&locations_, mluOpGetTensorElementNum(locations_desc_) *
                                        mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&locations_, 2 * mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      }
    }
    if (input_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
      std::vector<int> input_shape = {2, 2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(input_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           input_shape.data()));
    }
    if (input) {
      if (input_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&input_, mluOpGetTensorElementNum(input_desc_) *
                                    mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&input_, 4 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }
    if (dispatch_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&dispatch_desc_));
      std::vector<int> dispatch_shape = {4, 2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(dispatch_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           dispatch_shape.data()));
    }
    if (dispatch) {
      if (dispatch_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&dispatch_, mluOpGetTensorElementNum(dispatch_desc_) *
                                       mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&dispatch_, 8 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpMoeDispatchForward(
        handle_, gates_desc_, gates_, indices_desc_, indices_, locations_desc_,
        locations_, input_desc_, input_, 2, 2, 2, 2, dispatch_desc_, dispatch_);
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
    if (gates_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(gates_desc_));
      gates_desc_ = NULL;
    }
    if (gates_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(gates_));
      gates_ = NULL;
    }
    if (indices_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indices_desc_));
      indices_desc_ = NULL;
    }
    if (indices_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(indices_));
      indices_ = NULL;
    }
    if (locations_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(locations_desc_));
      locations_desc_ = NULL;
    }
    if (locations_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(locations_));
      locations_ = NULL;
    }
    if (input_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_desc_));
      input_desc_ = NULL;
    }
    if (input_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_));
      input_ = NULL;
    }
    if (dispatch_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(dispatch_desc_));
      dispatch_desc_ = NULL;
    }
    if (dispatch_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(dispatch_));
      dispatch_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t gates_desc_ = NULL;
  void* gates_ = NULL;
  mluOpTensorDescriptor_t indices_desc_ = NULL;
  void* indices_ = NULL;
  mluOpTensorDescriptor_t locations_desc_ = NULL;
  void* locations_ = NULL;
  mluOpTensorDescriptor_t input_desc_ = NULL;
  void* input_ = NULL;
  mluOpTensorDescriptor_t dispatch_desc_ = NULL;
  void* dispatch_ = NULL;
};

TEST_F(moe_dispatch_forward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in moe_dispatch_forward";
  }
}

TEST_F(moe_dispatch_forward, BAD_PARAM_gates_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in moe_dispatch_forward";
  }
}

TEST_F(moe_dispatch_forward, BAD_PARAM_gates_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in moe_dispatch_forward";
  }
}

TEST_F(moe_dispatch_forward, BAD_PARAM_indices_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in moe_dispatch_forward";
  }
}

TEST_F(moe_dispatch_forward, BAD_PARAM_indices_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in moe_dispatch_forward";
  }
}

TEST_F(moe_dispatch_forward, BAD_PARAM_locations_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in moe_dispatch_forward";
  }
}

TEST_F(moe_dispatch_forward, BAD_PARAM_locations_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in moe_dispatch_forward";
  }
}

TEST_F(moe_dispatch_forward, BAD_PARAM_input_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in moe_dispatch_forward";
  }
}

TEST_F(moe_dispatch_forward, BAD_PARAM_input_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in moe_dispatch_forward";
  }
}

TEST_F(moe_dispatch_forward, BAD_PARAM_dispatch_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in moe_dispatch_forward";
  }
}

TEST_F(moe_dispatch_forward, BAD_PARAM_dispatch_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in moe_dispatch_forward";
  }
}

}  // namespace mluopapitest
