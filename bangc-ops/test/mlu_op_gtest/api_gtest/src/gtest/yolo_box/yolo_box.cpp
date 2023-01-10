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
class yolo_box : public testing::Test {
 public:
  void setParam(bool handle, bool x_desc, bool x, bool img_size_desc,
                bool img_size, bool anchors_desc, bool anchors, bool boxes_desc,
                bool boxes, bool scores_desc, bool scores) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (x_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&x_desc_));
      std::vector<int> dim_size = {2, 160, 3, 3};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          x_desc_, MLUOP_LAYOUT_NCHW, MLUOP_DTYPE_FLOAT, 4, dim_size.data()));
    }
    if (img_size_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&img_size_desc_));
      std::vector<int> dim_size = {2, 2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(img_size_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 2,
                                           dim_size.data()));
    }
    if (anchors_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&anchors_desc_));
      std::vector<int> dim_size = {20};
      MLUOP_CHECK(mluOpSetTensorDescriptor(anchors_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 1,
                                           dim_size.data()));
    }
    if (boxes_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&boxes_desc_));
      std::vector<int> dim_size = {2, 10, 4, 9};
      MLUOP_CHECK(mluOpSetTensorDescriptor(boxes_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           dim_size.data()));
    }
    if (scores_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&scores_desc_));
      std::vector<int> dim_size = {2, 10, 10, 9};
      MLUOP_CHECK(mluOpSetTensorDescriptor(scores_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           dim_size.data()));
    }
    if (x) {
      size_t ele_num = 8;
      size_t dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t bytes = ele_num * dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&x_, bytes));
    }
    if (img_size) {
      size_t ele_num = 8;
      size_t dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      size_t bytes = ele_num * dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&img_size_, bytes));
    }
    if (anchors) {
      size_t ele_num = 8;
      size_t dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_INT32);
      size_t bytes = ele_num * dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&anchors_, bytes));
    }
    if (boxes) {
      size_t ele_num = 8;
      size_t dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t bytes = ele_num * dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&boxes_, bytes));
    }
    if (scores) {
      size_t ele_num = 8;
      size_t dtype_bytes = mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT);
      size_t bytes = ele_num * dtype_bytes;
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMalloc(&scores_, bytes));
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpYoloBox(
        handle_, x_desc_, x_, img_size_desc_, img_size_, anchors_desc_,
        anchors_, class_num_, conf_thresh_, downsample_ratio_, clip_bbox_,
        scale_, iou_aware_, iou_aware_factor_, boxes_desc_, boxes_,
        scores_desc_, scores_);
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
    if (x_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(x_desc_));
      x_desc_ = NULL;
    }
    if (img_size_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(img_size_desc_));
      img_size_desc_ = NULL;
    }
    if (anchors_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(anchors_desc_));
      anchors_desc_ = NULL;
    }
    if (boxes_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(boxes_desc_));
      boxes_desc_ = NULL;
    }
    if (scores_desc_) {
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(scores_desc_));
      scores_desc_ = NULL;
    }
    if (x_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(x_));
      x_ = NULL;
    }
    if (img_size_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(img_size_));
      img_size_ = NULL;
    }
    if (anchors_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(anchors_));
      anchors_ = NULL;
    }
    if (boxes_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(boxes_));
      boxes_ = NULL;
    }
    if (scores_) {
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(scores_));
      scores_ = NULL;
    }
  }

 private:
  mluOpHandle_t handle_ = NULL;
  mluOpTensorDescriptor_t x_desc_ = NULL;
  mluOpTensorDescriptor_t img_size_desc_ = NULL;
  mluOpTensorDescriptor_t anchors_desc_ = NULL;
  mluOpTensorDescriptor_t boxes_desc_ = NULL;
  mluOpTensorDescriptor_t scores_desc_ = NULL;
  void* x_ = NULL;
  void* img_size_ = NULL;
  void* anchors_ = NULL;
  void* boxes_ = NULL;
  void* scores_ = NULL;
  int class_num_ = 10;
  float conf_thresh_ = 0.1;
  int downsample_ratio_ = 16;
  bool clip_bbox_ = true;
  float scale_ = 0.5;
  bool iou_aware_ = true;
  float iou_aware_factor_ = 0.5;
};

TEST_F(yolo_box, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in yolo_box";
  }
}

TEST_F(yolo_box, BAD_PARAM_x_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in yolo_box";
  }
}

TEST_F(yolo_box, BAD_PARAM_x_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in yolo_box";
  }
}

TEST_F(yolo_box, BAD_PARAM_img_size_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in yolo_box";
  }
}

TEST_F(yolo_box, BAD_PARAM_img_size_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in yolo_box";
  }
}

TEST_F(yolo_box, BAD_PARAM_anchors_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in yolo_box";
  }
}

TEST_F(yolo_box, BAD_PARAM_anchors_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in yolo_box";
  }
}

TEST_F(yolo_box, BAD_PARAM_boxes_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in yolo_box";
  }
}

TEST_F(yolo_box, BAD_PARAM_boxes_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in yolo_box";
  }
}

TEST_F(yolo_box, BAD_PARAM_scores_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false, true);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in yolo_box";
  }
}

TEST_F(yolo_box, BAD_PARAM_scores_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, false);
    EXPECT_TRUE(MLUOP_STATUS_BAD_PARAM == compute());
  } catch (const std::exception& e) {
    FAIL() << "MLUOPAPITEST: catched " << e.what() << " in yolo_box";
  }
}
}  // namespace mluopapitest
