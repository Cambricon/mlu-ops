/*************************************************************************
 * Copyright (C) 2021 by Cambricon, Inc. All rights reserved.
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
#include "core/logging.h"
#include "core/tensor.h"
#include "gtest/gtest.h"
#include "mlu_op.h"

namespace mluopapitest {
class psroipool_forward_workspace : public testing::Test {
 public:
  void setParam(bool handle, bool workspace_size, int output_dim = 9) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (workspace_size) {
      size_t size_temp = 0;
      workspace_size_ = &size_temp;
    }
    output_dim_ = output_dim;
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status =
        mluOpGetPsRoiPoolWorkspaceSize(handle_, output_dim_, workspace_size_);
    destroy();
    return status;
  }

 protected:
  void destroy() {
    if (handle_) {
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = NULL;
    }
    if (workspace_size_) {
      workspace_size_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  int output_dim_ = 1;
  size_t* workspace_size_ = NULL;
};

TEST_F(psroipool_forward_workspace, BAD_PARAM_handle_null) {
  try {
    setParam(false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_forward";
  }
}

TEST_F(psroipool_forward_workspace, BAD_PARAM_size_null) {
  try {
    setParam(true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in psroipool_forward";
  }
}
}  // mluopapitest