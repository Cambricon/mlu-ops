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
class moe_dispatch_backward_data : public testing::Test {
 public:
  void setParam(bool handle, bool gates_desc, bool gates, bool indices_desc,
                bool indices, bool locations_desc, bool locations,
                bool dispatch_desc, bool dispatch, bool grad_input_desc,
                bool grad_input) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (gates_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&gates_desc_));
      std::vector<int> gates_dims{1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(gates_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 1,
                                           gates_dims.data()));
    }

    if (gates) {
      if (gates_desc) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&gates_, mluOpGetTensorElementNum(gates_desc_) *
                                    mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&gates_, 64 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    if (indices_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&indices_desc_));
      std::vector<int> indices_dims{1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(indices_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 1,
                                           indices_dims.data()));
    }

    if (indices) {
      if (indices_desc) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&indices_, mluOpGetTensorElementNum(indices_desc_) *
                                      mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      } else {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&indices_, 64 * mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      }
    }

    if (locations_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&locations_desc_));
      std::vector<int> locations_dims{1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(locations_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 1,
                                           locations_dims.data()));
    }

    if (locations) {
      if (locations_desc) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&locations_, mluOpGetTensorElementNum(locations_desc_) *
                                        mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&locations_,
                               64 * mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      }
    }

    if (dispatch_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&dispatch_desc_));
      std::vector<int> dispatch_dims{4, 1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(dispatch_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           dispatch_dims.data()));
    }

    if (dispatch) {
      if (dispatch_desc) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&dispatch_, mluOpGetTensorElementNum(dispatch_desc_) *
                                       mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&dispatch_, 64 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    if (grad_input_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_input_desc_));
      std::vector<int> grad_input_dims{1, 1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(grad_input_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           grad_input_dims.data()));
    }

    if (grad_input) {
      if (grad_input_desc) {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&grad_input_,
                               mluOpGetTensorElementNum(grad_input_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&grad_input_,
                               64 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }
  }
  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpMoeDispatchBackwardData(
        handle_, gates_desc_, gates_, indices_desc_, indices_, locations_desc_,
        locations_, dispatch_desc_, dispatch_, samples_, capacity_, hidden_,
        num_experts_, grad_input_desc_, grad_input_);
    destroy();
    return status;
  }

 protected:
  void destroy() {
    if (handle_) {
      CNRT_CHECK(cnrtQueueSync(handle_->queue));
      VLOG(4) << "Destroy handle_";
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = nullptr;
    }

    if (gates_desc_) {
      VLOG(4) << "Destroy gates_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(gates_desc_));
      gates_desc_ = nullptr;
    }

    if (gates_) {
      VLOG(4) << "Destroy gates_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(gates_));
      gates_ = nullptr;
    }

    if (indices_desc_) {
      VLOG(4) << "Destroy indices_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(indices_desc_));
      indices_desc_ = nullptr;
    }

    if (indices_) {
      VLOG(4) << "Destroy indices_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(indices_));
      indices_ = nullptr;
    }

    if (locations_desc_) {
      VLOG(4) << "Destroy locations_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(locations_desc_));
      locations_desc_ = nullptr;
    }

    if (locations_) {
      VLOG(4) << "Destroy locations_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(locations_));
      locations_ = nullptr;
    }

    if (dispatch_desc_) {
      VLOG(4) << "Destroy dispatch_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(dispatch_desc_));
      dispatch_desc_ = nullptr;
    }

    if (dispatch_) {
      VLOG(4) << "Destroy dispatch_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(dispatch_));
      dispatch_ = nullptr;
    }

    if (grad_input_desc_) {
      VLOG(4) << "Destroy grad_input_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_input_desc_));
      grad_input_desc_ = nullptr;
    }

    if (grad_input_) {
      VLOG(4) << "Destroy grad_input_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(grad_input_));
      grad_input_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t gates_desc_ = nullptr;
  void *gates_ = nullptr;
  mluOpTensorDescriptor_t indices_desc_ = nullptr;
  void *indices_ = nullptr;
  mluOpTensorDescriptor_t locations_desc_ = nullptr;
  void *locations_ = nullptr;
  mluOpTensorDescriptor_t dispatch_desc_ = nullptr;
  void *dispatch_ = nullptr;
  int samples_ = 1;
  int capacity_ = 2;
  int hidden_ = 1;
  int num_experts_ = 2;
  mluOpTensorDescriptor_t grad_input_desc_ = nullptr;
  void *grad_input_ = nullptr;
};

TEST_F(moe_dispatch_backward_data, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in moe_dispatch_backward_data";
  }
}

TEST_F(moe_dispatch_backward_data, BAD_PARAM_gates_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in moe_dispatch_backward_data";
  }
}

TEST_F(moe_dispatch_backward_data, BAD_PARAM_gates_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in moe_dispatch_backward_data";
  }
}

TEST_F(moe_dispatch_backward_data, BAD_PARAM_indices_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in moe_dispatch_backward_data";
  }
}

TEST_F(moe_dispatch_backward_data, BAD_PARAM_indices_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in moe_dispatch_backward_data";
  }
}

TEST_F(moe_dispatch_backward_data, BAD_PARAM_locations_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in moe_dispatch_backward_data";
  }
}

TEST_F(moe_dispatch_backward_data, BAD_PARAM_locations_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in moe_dispatch_backward_data";
  }
}

TEST_F(moe_dispatch_backward_data, BAD_PARAM_dispatch_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in moe_dispatch_backward_data";
  }
}

TEST_F(moe_dispatch_backward_data, BAD_PARAM_dispatch_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in moe_dispatch_backward_data";
  }
}

TEST_F(moe_dispatch_backward_data, BAD_PARAM_grad_input_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in moe_dispatch_backward_data";
  }
}

TEST_F(moe_dispatch_backward_data, BAD_PARAM_grad_input_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in moe_dispatch_backward_data";
  }
}

}  // namespace mluopapitest
