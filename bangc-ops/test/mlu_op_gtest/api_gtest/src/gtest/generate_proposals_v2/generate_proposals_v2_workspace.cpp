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
class generate_proposals_v2_workspace : public testing::Test {
 public:
  void setParam(bool handle, bool scores_desc, bool size) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (scores_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&scores_desc_));
      std::vector<int> dim_size = {2, 8, 16, 16};
      MLUOP_CHECK(mluOpSetTensorDescriptor(scores_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           dim_size.data()));
    }
    if (size) {
      size_t size_temp = 0;
      size_ = &size_temp;
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status =
        mluOpGetGenerateProposalsV2WorkspaceSize(handle_, scores_desc_, size_);
    destroy();
    return status;
  }

 protected:
  void destroy() {
    if (handle_) {
      CNRT_CHECK(cnrtQueueSync(handle_->queue));
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = NULL;
    }
    if (scores_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(scores_desc_));
      scores_desc_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t scores_desc_ = NULL;
  size_t* size_ = NULL;
};

TEST_F(generate_proposals_v2_workspace, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2_workspace, BAD_PARAM_scores_desc_null) {
  try {
    setParam(true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2_workspace, BAD_PARAM_size_null) {
  try {
    setParam(true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}
}  // namespace mluopapitest
