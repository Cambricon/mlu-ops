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

class border_align_backward : public testing::Test {
 public:
  void set_params(bool handle, bool grad_output_desc, bool grad_output,
                  bool boxes_desc, bool boxes, bool argmax_idx_desc,
                  bool argmax_idx, bool grad_input_desc, bool grad_input) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (grad_output_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_output_desc_));
      std::vector<int> grad_output_dims{1, 4, 4, 1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_output_desc_, MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
          grad_output_dims.size(), grad_output_dims.data()));
    }
    if (grad_output) {
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&grad_output_,
                             16 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
    }
    if (boxes_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&boxes_desc_));
      std::vector<int> boxes_dims{1, 4, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(boxes_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, boxes_dims.size(),
                                           boxes_dims.data()));
    }
    if (boxes) {
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(&boxes_, 16 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
    }
    if (argmax_idx_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&argmax_idx_desc_));
      std::vector<int> argmax_idx_dims{1, 4, 4, 1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          argmax_idx_desc_, MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_INT32,
          argmax_idx_dims.size(), argmax_idx_dims.data()));
    }
    if (argmax_idx) {
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(&argmax_idx_, 16 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
    }
    if (grad_input_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_input_desc_));
      std::vector<int> grad_input_dims{1, 2, 2, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_input_desc_, MLUOP_LAYOUT_NHWC, MLUOP_DTYPE_FLOAT,
          grad_input_dims.size(), grad_input_dims.data()));
    }
    if (grad_input) {
      GTEST_CHECK(
          CNRT_RET_SUCCESS ==
          cnrtMalloc(&grad_input_, 16 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
    }
  }
  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpBorderAlignBackward(
        handle_, grad_output_desc_, grad_output_, boxes_desc_, boxes_,
        argmax_idx_desc_, argmax_idx_, pool_size_, grad_input_desc_,
        grad_input_);
    destroy();
    return status;
  }

 protected:
  virtual void SetUp() {
    handle_ = nullptr;
    grad_output_desc_ = nullptr;
    grad_output_ = nullptr;
    boxes_desc_ = nullptr;
    boxes_ = nullptr;
    argmax_idx_desc_ = nullptr;
    argmax_idx_ = nullptr;
    pool_size_ = 10;
    grad_input_desc_ = nullptr;
    grad_input_ = nullptr;
  }

  void destroy() {
    try {
      if (handle_) {
        CNRT_CHECK(cnrtQueueSync(handle_->queue));
        VLOG(4) << "Destroy handle_";
        MLUOP_CHECK(mluOpDestroy(handle_));
        handle_ = nullptr;
      }
      if (grad_output_desc_) {
        VLOG(4) << "Destroy grad_output_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_output_desc_));
        grad_output_desc_ = nullptr;
      }
      if (grad_output_) {
        VLOG(4) << "Destroy grad_output_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_output_));
        grad_output_ = nullptr;
      }
      if (boxes_desc_) {
        VLOG(4) << "Destroy boxes_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(boxes_desc_));
        boxes_desc_ = nullptr;
      }
      if (boxes_) {
        VLOG(4) << "Destroy boxes_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(boxes_));
        boxes_ = nullptr;
      }
      if (argmax_idx_desc_) {
        VLOG(4) << "Destroy argmax_idx_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(argmax_idx_desc_));
        argmax_idx_desc_ = nullptr;
      }
      if (argmax_idx_) {
        VLOG(4) << "Destroy argmax_idx_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(argmax_idx_));
        argmax_idx_ = nullptr;
      }
      if (grad_input_desc_) {
        VLOG(4) << "Destroy grad_input_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_input_desc_));
        grad_input_desc_ = nullptr;
      }
      if (grad_input_) {
        VLOG(4) << "Destroy grad_input_";
        GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_input_));
        grad_input_ = nullptr;
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what()
             << " in border_align_backward";
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t grad_output_desc_ = nullptr;
  void *grad_output_ = nullptr;
  mluOpTensorDescriptor_t boxes_desc_ = nullptr;
  void *boxes_ = nullptr;
  mluOpTensorDescriptor_t argmax_idx_desc_ = nullptr;
  void *argmax_idx_ = nullptr;
  int pool_size_ = 10;
  mluOpTensorDescriptor_t grad_input_desc_ = nullptr;
  void *grad_input_ = nullptr;
};

TEST_F(border_align_backward, BAD_PARAM_handle_null) {
  try {
    set_params(false, true, true, true, true, true, true, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_backward";
  }
}

TEST_F(border_align_backward, BAD_PARAM_grad_output_desc_null) {
  try {
    set_params(true, false, true, true, true, true, true, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_backward";
  }
}

TEST_F(border_align_backward, BAD_PARAM_grad_output_null) {
  try {
    set_params(true, true, false, true, true, true, true, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_backward";
  }
}

TEST_F(border_align_backward, BAD_PARAM_boxes_desc_null) {
  try {
    set_params(true, true, true, false, true, true, true, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_backward";
  }
}

TEST_F(border_align_backward, BAD_PARAM_boxes_null) {
  try {
    set_params(true, true, true, true, false, true, true, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_backward";
  }
}

TEST_F(border_align_backward, BAD_PARAM_argmax_idx_desc_null) {
  try {
    set_params(true, true, true, true, true, false, true, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_backward";
  }
}

TEST_F(border_align_backward, BAD_PARAM_argmax_idx_null) {
  try {
    set_params(true, true, true, true, true, true, false, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_backward";
  }
}

TEST_F(border_align_backward, BAD_PARAM_grad_input_desc_null) {
  try {
    set_params(true, true, true, true, true, true, true, false, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_backward";
  }
}

TEST_F(border_align_backward, BAD_PARAM_grad_input_null) {
  try {
    set_params(true, true, true, true, true, true, true, true, false);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in border_align_backward";
  }
}

}  // namespace mluopapitest
