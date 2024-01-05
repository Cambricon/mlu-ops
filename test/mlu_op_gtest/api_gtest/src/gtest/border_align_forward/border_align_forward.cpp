/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
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
class border_align_forward_test : public testing::Test {
 public:
  void set_params(bool handle, bool input_desc, bool input, bool boxes_desc,
                  bool boxes, bool output_desc, bool output,
                  bool argmax_idx_desc, bool argmax_idx) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (input_desc) {
      std::vector<int> input_dims{2, 10, 10, 20};
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(input_desc_, MLUOP_LAYOUT_NHWC,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           input_dims.data()));
    }
    if (input) {
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(&input_, 4000 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
    }
    if (boxes_desc) {
      std::vector<int> boxes_dims{2, 100, 4};
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&boxes_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(boxes_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 3,
                                           boxes_dims.data()));
    }
    if (boxes) {
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(&boxes_, 800 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
    }
    if (output_desc) {
      std::vector<int> output_dims{2, 100, 4, 5};
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(output_desc_, MLUOP_LAYOUT_NHWC,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           output_dims.data()));
    }
    if (output) {
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(&output_, 4000 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
    }
    if (argmax_idx_desc) {
      std::vector<int> argmax_idx_dims{2, 100, 4, 5};
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&argmax_idx_desc_));
      MLUOP_CHECK(mluOpSetTensorDescriptor(argmax_idx_desc_, MLUOP_LAYOUT_NHWC,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           argmax_idx_dims.data()));
    }
    if (argmax_idx) {
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&argmax_idx_,
                             4000 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
    }
  }
  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpBorderAlignForward(
        handle_, input_desc_, input_, boxes_desc_, boxes_, pool_size_,
        output_desc_, output_, argmax_idx_desc_, argmax_idx_);
    destroy();
    return status;
  }

 protected:
  virtual void SetUp() {
    handle_ = nullptr;
    input_desc_ = nullptr;
    input_ = nullptr;
    boxes_desc_ = nullptr;
    boxes_ = nullptr;
    pool_size_ = 9;
    output_desc_ = nullptr;
    output_ = nullptr;
    argmax_idx_desc_ = nullptr;
    argmax_idx_ = nullptr;
  }
  void destroy() {
    try {
      if (handle_) {
        CNRT_CHECK(cnrtQueueSync(handle_->queue));
        VLOG(4) << "Destroy handle_";
        MLUOP_CHECK(mluOpDestroy(handle_));
      }
      if (input_desc_) {
        VLOG(4) << "Destroy input_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_desc_));
      }
      if (input_) {
        VLOG(4) << "Destroy input_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(input_));
      }
      if (boxes_desc_) {
        VLOG(4) << "Destroy boxes_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(boxes_desc_));
      }
      if (boxes_) {
        VLOG(4) << "Destroy boxes_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(boxes_));
      }
      if (output_desc_) {
        VLOG(4) << "Destroy output_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
      }
      if (output_) {
        VLOG(4) << "Destroy output_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(output_));
      }
      if (argmax_idx_desc_) {
        VLOG(4) << "Destroy argmax_idx_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(argmax_idx_desc_));
      }
      if (argmax_idx_) {
        VLOG(4) << "Destroy argmax_idx_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(argmax_idx_));
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in border_align_forward";
    }
  }

 private:
  mluOpHandle_t handle_;
  mluOpTensorDescriptor_t input_desc_;
  void *input_;
  mluOpTensorDescriptor_t boxes_desc_;
  void *boxes_;
  int32_t pool_size_;
  mluOpTensorDescriptor_t output_desc_;
  void *output_;
  mluOpTensorDescriptor_t argmax_idx_desc_;
  void *argmax_idx_;
};

TEST_F(border_align_forward_test, BAD_PARAM_handle_null) {
  try {
    set_params(false, true, true, true, true, true, true, true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_forward";
  }
}

TEST_F(border_align_forward_test, BAD_PARAM_input_desc_null) {
  try {
    set_params(true, false, true, true, true, true, true, true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_forward";
  }
}

TEST_F(border_align_forward_test, BAD_PARAM_input_null) {
  try {
    set_params(true, true, false, true, true, true, true, true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_forward";
  }
}

TEST_F(border_align_forward_test, BAD_PARAM_boxes_desc_null) {
  try {
    set_params(true, true, true, false, true, true, true, true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_forward";
  }
}

TEST_F(border_align_forward_test, BAD_PARAM_boxes_null) {
  try {
    set_params(true, true, true, true, false, true, true, true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_forward";
  }
}

TEST_F(border_align_forward_test, BAD_PARAM_output_desc_null) {
  try {
    set_params(true, true, true, true, true, false, true, true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_forward";
  }
}

TEST_F(border_align_forward_test, BAD_PARAM_output_null) {
  try {
    set_params(true, true, true, true, true, true, false, true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_forward";
  }
}

TEST_F(border_align_forward_test, BAD_PARAM_argmax_idx_desc_null) {
  try {
    set_params(true, true, true, true, true, true, true, false, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_forward";
  }
}

TEST_F(border_align_forward_test, BAD_PARAM_argmax_idx_null) {
  try {
    set_params(true, true, true, true, true, true, true, true, false);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_forward";
  }
}

}  // namespace mluopapitest
