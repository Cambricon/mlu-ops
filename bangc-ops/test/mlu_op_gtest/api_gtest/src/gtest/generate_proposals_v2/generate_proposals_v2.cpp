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
#include "core/logging.h"
#include "core/tensor.h"
#include "gtest/gtest.h"
#include "mlu_op.h"
#include "core/context.h"

namespace mluopapitest {
class generate_proposals_v2 : public testing::Test {
 public:
  void setParam(bool handle, bool scores_desc, bool scores,
                bool bbox_deltas_desc, bool bbox_deltas, bool im_shape_desc,
                bool im_shape, bool anchors_desc, bool anchors,
                bool variances_desc, bool variances, bool workspace,
                bool rpn_rois_desc, bool rpn_rois, bool rpn_roi_probs_desc,
                bool rpn_roi_probs, bool rpn_rois_num_desc, bool rpn_rois_num,
                bool rpn_rois_batch_size) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (scores_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&scores_desc_));
      std::vector<int> scores_dims = {2, 8, 16, 16};
      MLUOP_CHECK(mluOpSetTensorDescriptor(scores_desc_, MLUOP_LAYOUT_NHWC,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           scores_dims.data()));
    }
    if (scores) {
      size_t scores_ele_num = 1 * 5 * 5 * 9;
      size_t scores_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t scores_bytes = scores_ele_num * scores_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&scores_, scores_bytes));
    }
    if (bbox_deltas_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&bbox_deltas_desc_));
      std::vector<int> bbox_deltas_dims = {2, 32, 16, 16};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          bbox_deltas_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 4,
          bbox_deltas_dims.data()));
    }
    if (bbox_deltas) {
      size_t bbox_deltas_ele_num = 2 * 32 * 16 * 16;
      size_t bbox_deltas_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t bbox_deltas_bytes = bbox_deltas_ele_num * bbox_deltas_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&bbox_deltas_, bbox_deltas_bytes));
    }
    if (im_shape_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&im_shape_desc_));
      std::vector<int> im_shape_dims = {2, 2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(im_shape_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           im_shape_dims.data()));
    }
    if (im_shape) {
      size_t im_shape_ele_num = 2 * 2;
      size_t im_shape_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t im_shape_bytes = im_shape_ele_num * im_shape_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&im_shape_, im_shape_bytes));
    }
    if (anchors_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&anchors_desc_));
      std::vector<int> anchors_dims = {8, 16, 16, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(anchors_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           anchors_dims.data()));
    }
    if (anchors) {
      size_t anchors_ele_num = 8 * 16 * 16 * 4;
      size_t anchors_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t anchors_bytes = anchors_ele_num * anchors_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&anchors_, anchors_bytes));
    }
    if (variances_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&variances_desc_));
      std::vector<int> variances_dims = {8, 16, 16, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(variances_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           variances_dims.data()));
    }
    if (variances) {
      size_t variances_ele_num = 8 * 16 * 16 * 4;
      size_t variances_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t variances_bytes = variances_ele_num * variances_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&variances_, variances_bytes));
    }
    if (workspace) {
      size_t workspace_ele_num = workspace_size_;
      size_t workspace_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t workspace_bytes = workspace_ele_num * workspace_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&workspace_, workspace_bytes));
    }
    if (rpn_rois_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&rpn_rois_desc_));
      std::vector<int> rpn_rois_dims = {5, 4};
      MLUOP_CHECK(mluOpSetTensorDescriptor(rpn_rois_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           rpn_rois_dims.data()));
    }
    if (rpn_rois) {
      size_t rpn_rois_ele_num = 5 * 4;
      size_t rpn_rois_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t rpn_rois_bytes = rpn_rois_ele_num * rpn_rois_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&rpn_rois_, rpn_rois_bytes));
    }
    if (rpn_roi_probs_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&rpn_roi_probs_desc_));
      std::vector<int> rpn_roi_probs_dims = {5, 1};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          rpn_roi_probs_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 2,
          rpn_roi_probs_dims.data()));
    }
    if (rpn_roi_probs) {
      size_t rpn_roi_probs_ele_num = 5 * 1;
      size_t rpn_roi_probs_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t rpn_roi_probs_bytes =
          rpn_roi_probs_ele_num * rpn_roi_probs_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&rpn_roi_probs_, rpn_roi_probs_bytes));
    }
    if (rpn_rois_num_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&rpn_rois_num_desc_));
      std::vector<int> rpn_rois_num_dims = {2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          rpn_rois_num_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 1,
          rpn_rois_num_dims.data()));
    }
    if (rpn_rois_num) {
      size_t rpn_rois_num_ele_num = 2;
      size_t rpn_rois_num_dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      size_t rpn_rois_num_bytes =
          rpn_rois_num_ele_num * rpn_rois_num_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&rpn_rois_num_, rpn_rois_num_bytes));
    }
    if (rpn_rois_batch_size) {
      size_t rpn_rois_batch_size_ele_num = 1;
      size_t rpn_rois_batch_size_dtype_bytes =
          mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      size_t rpn_rois_batch_size_bytes =
          rpn_rois_batch_size_ele_num * rpn_rois_batch_size_dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS ==
                  cnrtMalloc(&rpn_rois_batch_size_, rpn_rois_batch_size_bytes));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpGenerateProposalsV2(
        handle_, pre_nms_top_n_, post_nms_top_n_, nms_thresh_, min_size_, eta_,
        pixel_offset_, scores_desc_, scores_, bbox_deltas_desc_, bbox_deltas_,
        im_shape_desc_, im_shape_, anchors_desc_, anchors_, variances_desc_,
        variances_, workspace_, workspace_size_, rpn_rois_desc_, rpn_rois_,
        rpn_roi_probs_desc_, rpn_roi_probs_, rpn_rois_num_desc_, rpn_rois_num_,
        rpn_rois_batch_size_);
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
    if (scores_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(scores_));
      scores_ = NULL;
    }
    if (bbox_deltas_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(bbox_deltas_desc_));
      bbox_deltas_desc_ = NULL;
    }
    if (bbox_deltas_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(bbox_deltas_));
      bbox_deltas_ = NULL;
    }
    if (im_shape_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(im_shape_desc_));
      im_shape_desc_ = NULL;
    }
    if (im_shape_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(im_shape_));
      im_shape_ = NULL;
    }
    if (anchors_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(anchors_desc_));
      anchors_desc_ = NULL;
    }
    if (anchors_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(anchors_));
      anchors_ = NULL;
    }
    if (variances_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(variances_desc_));
      variances_desc_ = NULL;
    }
    if (variances_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(variances_));
      variances_ = NULL;
    }
    if (workspace_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(workspace_));
      workspace_ = NULL;
    }
    if (rpn_rois_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(rpn_rois_desc_));
      rpn_rois_desc_ = NULL;
    }
    if (rpn_rois_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(rpn_rois_));
      rpn_rois_ = NULL;
    }
    if (rpn_roi_probs_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(rpn_roi_probs_desc_));
      rpn_roi_probs_desc_ = NULL;
    }
    if (rpn_roi_probs_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(rpn_roi_probs_));
      rpn_roi_probs_ = NULL;
    }
    if (rpn_rois_num_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(rpn_rois_num_desc_));
      rpn_rois_num_desc_ = NULL;
    }
    if (rpn_rois_num_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(rpn_rois_num_));
      rpn_rois_num_ = NULL;
    }
    if (rpn_rois_batch_size_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(rpn_rois_batch_size_));
      rpn_rois_batch_size_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  int pre_nms_top_n_ = 15;
  int post_nms_top_n_ = 5;
  float nms_thresh_ = 0.8;
  float min_size_ = 4;
  float eta_ = 3;
  bool pixel_offset_ = false;
  mluOpTensorDescriptor_t scores_desc_ = NULL;
  void* scores_ = NULL;
  mluOpTensorDescriptor_t bbox_deltas_desc_ = NULL;
  void* bbox_deltas_ = NULL;
  mluOpTensorDescriptor_t im_shape_desc_ = NULL;
  void* im_shape_ = NULL;
  mluOpTensorDescriptor_t anchors_desc_ = NULL;
  void* anchors_ = NULL;
  mluOpTensorDescriptor_t variances_desc_ = NULL;
  void* variances_ = NULL;
  void* workspace_ = NULL;
  size_t workspace_size_ = 64;
  mluOpTensorDescriptor_t rpn_rois_desc_ = NULL;
  void* rpn_rois_ = NULL;
  mluOpTensorDescriptor_t rpn_roi_probs_desc_ = NULL;
  void* rpn_roi_probs_ = NULL;
  mluOpTensorDescriptor_t rpn_rois_num_desc_ = NULL;
  void* rpn_rois_num_ = NULL;
  void* rpn_rois_batch_size_ = NULL;
};

TEST_F(generate_proposals_v2, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_scores_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_scores_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_bbox_deltas_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_bbox_deltas_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_im_shape_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_im_shape_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_anchors_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_anchors_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_variances_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false, true,
             true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_variances_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, false,
             true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_workspace_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             false, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_rpn_rois_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_rpn_rois_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_rpn_roi_probs_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_rpn_roi_probs_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_rpn_rois_num_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_rpn_rois_num_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}

TEST_F(generate_proposals_v2, BAD_PARAM_rpn_rois_batch_size_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what()
           << " in generate_proposals_v2";
  }
}
}  // namespace mluopapitest
