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
#include <iostream>
#include <vector>
#include <string>
#include <tuple>

#include "gtest/gtest.h"
#include "mlu_op.h"
#include "api_test_tools.h"
#include "core/context.h"
#include "core/logging.h"

namespace mluopapitest {
class fft_MakeFFTPlanMany : public testing::Test {
 public:
  void setParam(bool handle,
                bool fft_plan,
                bool input_desc,
                bool output_desc,
                bool reservespace_size,
                bool workspace_size) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (fft_plan) {
      MLUOP_CHECK(mluOpCreateFFTPlan(&fft_plan_));
    }

    if (input_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
      std::vector<int> input_dims{1, 400};
      const int input_dim_stride[2] = {400, 1};
      MLUOP_CHECK(mluOpSetTensorDescriptorEx(input_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
                                           input_dims.size(), input_dims.data(), input_dim_stride));
      MLUOP_CHECK(mluOpSetTensorDescriptorOnchipDataType(input_desc_, MLUOP_DTYPE_FLOAT));
    }

    if (output_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));
      std::vector<int> output_dims{1, 201};
      const int output_dim_stride[2] = {201, 1};
      MLUOP_CHECK(mluOpSetTensorDescriptorEx(output_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_COMPLEX_FLOAT, output_dims.size(),
                                           output_dims.data(), output_dim_stride));
    }

    if (reservespace_size) {
      reservespace_size_ = &reservespaceSizeInBytes_;
    }

    if (workspace_size) {
      workspace_size_ = &workspaceSizeInBytes_;
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpMakeFFTPlanMany(handle_, fft_plan_, input_desc_, output_desc_, rank,
                                              n, reservespace_size_, workspace_size_);
    destroy();
    return status;
  }

 protected:
  virtual void SetUp() {
    handle_ = nullptr;
    fft_plan_ = nullptr;
    input_desc_ = nullptr;
    output_desc_ = nullptr;
    reservespace_size_ = nullptr;
    workspace_size_ = nullptr;
  }

  void destroy() {
    try {
      if (handle_) {
        CNRT_CHECK(cnrtQueueSync(handle_->queue));
        VLOG(4) << "Destroy handle_";
        MLUOP_CHECK(mluOpDestroy(handle_));
        handle_ = nullptr;
      }
      if (fft_plan_) {
        VLOG(4) << "Destroy fft_plan_";
        MLUOP_CHECK(mluOpDestroyFFTPlan(fft_plan_));
        fft_plan_ = nullptr;
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
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in fft_MakeFFTPlanMany";
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpFFTPlan_t fft_plan_ = nullptr;
  mluOpTensorDescriptor_t input_desc_ = nullptr;
  mluOpTensorDescriptor_t output_desc_ = nullptr;
  int rank = 1;
  int n[1] = {400};
  size_t *reservespace_size_ = nullptr;
  size_t *workspace_size_ = nullptr;
  size_t reservespaceSizeInBytes_ = 64;
  size_t workspaceSizeInBytes_ = 64;
};

TEST_F(fft_MakeFFTPlanMany, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in fft_MakeFFTPlanMany";
  }
}

TEST_F(fft_MakeFFTPlanMany, BAD_PARAM_fft_plan_null) {
  try {
    setParam(true, false, true, true, true, true);
    EXPECT_EQ(MLUOP_STATUS_NOT_INITIALIZED, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in fft_MakeFFTPlanMany";
  }
}

TEST_F(fft_MakeFFTPlanMany, BAD_PARAM_input_desc_null) {
  try {
    setParam(true, true, false, true, true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in fft_MakeFFTPlanMany";
  }
}

TEST_F(fft_MakeFFTPlanMany, BAD_PARAM_output_desc_null) {
  try {
    setParam(true, true, true, false, true, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in fft_MakeFFTPlanMany";
  }
}

TEST_F(fft_MakeFFTPlanMany, BAD_PARAM_reservespace_size_null) {
  try {
    setParam(true, true, true, true, false, true);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in fft_MakeFFTPlanMany";
  }
}

TEST_F(fft_MakeFFTPlanMany, BAD_PARAM_workspace_size_null) {
  try {
    setParam(true, true, true, true, true, false);
    EXPECT_EQ(MLUOP_STATUS_BAD_PARAM, compute());
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in fft_MakeFFTPlanMany";
  }
}

}  // namespace mluopapitest
