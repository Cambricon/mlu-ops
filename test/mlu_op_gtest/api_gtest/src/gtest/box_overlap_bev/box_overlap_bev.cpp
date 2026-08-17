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
#include <tuple>

#include "gtest/gtest.h"
#include "mlu_op.h"
#include "core/context.h"
#include "core/logging.h"
#include "core/tensor.h"
#include "api_test_tools.h"

namespace mluopapitest {

class box_overlap_bev : public testing::Test {
 public:
  void set_params(bool handle, bool boxes1_desc, bool boxes1, bool boxes2_desc,
                  bool boxes2, bool overlaps_desc, bool overlaps) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }
    if (boxes1_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&boxes1_desc_));
      std::vector<int> boxes1_dims{2, 7};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          boxes1_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
          boxes1_dims.size(), boxes1_dims.data()));
    }
    if (boxes1) {
      GTEST_CHECK(
          cnrtSuccess ==
          cnrtMalloc(&boxes1_, 14 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
    }
    if (boxes2_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&boxes2_desc_));
      std::vector<int> boxes2_dims{2, 7};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          boxes2_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
          boxes2_dims.size(), boxes2_dims.data()));
    }
    if (boxes2) {
      GTEST_CHECK(
          cnrtSuccess ==
          cnrtMalloc(&boxes2_, 14 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
    }
    if (overlaps_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&overlaps_desc_));
      std::vector<int> overlaps_dims{2, 2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          overlaps_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT,
          overlaps_dims.size(), overlaps_dims.data()));
    }
    if (overlaps) {
      GTEST_CHECK(
          cnrtSuccess ==
          cnrtMalloc(&overlaps_, 4 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
    }
  }
  mluOpStatus_t compute() {
    mluOpStatus_t status =
        mluOpBoxOverlapBev(handle_, boxes1_desc_, boxes1_, boxes2_desc_,
                           boxes2_, overlaps_desc_, overlaps_);
    destroy();
    return status;
  }

 protected:
  virtual void SetUp() {
    handle_ = nullptr;
    boxes1_desc_ = nullptr;
    boxes1_ = nullptr;
    boxes2_desc_ = nullptr;
    boxes2_ = nullptr;
    overlaps_desc_ = nullptr;
    overlaps_ = nullptr;
  }

  void destroy() {
    try {
      if (handle_) {
        CNRT_CHECK(cnrtQueueSync(handle_->queue));
        VLOG(4) << "Destroy handle_";
        MLUOP_CHECK(mluOpDestroy(handle_));
      }
      if (boxes1_desc_) {
        VLOG(4) << "Destroy boxes1_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(boxes1_desc_));
      }
      if (boxes1_) {
        VLOG(4) << "Destroy boxes1_";
        GTEST_CHECK(cnrtSuccess == cnrtFree(boxes1_));
        boxes1_ = nullptr;
      }
      if (boxes2_desc_) {
        VLOG(4) << "Destroy boxes2_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(boxes2_desc_));
      }
      if (boxes2_) {
        VLOG(4) << "Destroy boxes2_";
        GTEST_CHECK(cnrtSuccess == cnrtFree(boxes2_));
        boxes2_ = nullptr;
      }
      if (overlaps_desc_) {
        VLOG(4) << "Destroy overlaps_desc_";
        MLUOP_CHECK(mluOpDestroyTensorDescriptor(overlaps_desc_));
      }
      if (overlaps_) {
        VLOG(4) << "Destroy overlaps_";
        GTEST_CHECK(cnrtSuccess == cnrtFree(overlaps_));
        overlaps_ = nullptr;
      }
    } catch (const std::exception &e) {
      FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in box_overlap_bev";
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t boxes1_desc_ = nullptr;
  void *boxes1_ = nullptr;
  mluOpTensorDescriptor_t boxes2_desc_ = nullptr;
  void *boxes2_ = nullptr;
  mluOpTensorDescriptor_t overlaps_desc_ = nullptr;
  void *overlaps_ = nullptr;
};

TEST_F(box_overlap_bev, BAD_PARAM_handle_null) {
  try {
    set_params(false, true, true, true, true, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in box_overlap_bev";
  }
}

TEST_F(box_overlap_bev, BAD_PARAM_boxes1_desc_null) {
  try {
    set_params(true, false, true, true, true, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in box_overlap_bev";
  }
}

TEST_F(box_overlap_bev, BAD_PARAM_boxes1_null) {
  try {
    set_params(true, true, false, true, true, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in box_overlap_bev";
  }
}

TEST_F(box_overlap_bev, BAD_PARAM_boxes2_desc_null) {
  try {
    set_params(true, true, true, false, true, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in box_overlap_bev";
  }
}

TEST_F(box_overlap_bev, BAD_PARAM_boxes2_null) {
  try {
    set_params(true, true, true, true, false, true, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in box_overlap_bev";
  }
}

TEST_F(box_overlap_bev, BAD_PARAM_overlaps_desc_null) {
  try {
    set_params(true, true, true, true, true, false, true);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in box_overlap_bev";
  }
}

TEST_F(box_overlap_bev, BAD_PARAM_overlaps_null) {
  try {
    set_params(true, true, true, true, true, true, false);
    mluOpStatus_t status = compute();
    EXPECT_TRUE(status == MLUOP_STATUS_BAD_PARAM);
  } catch (const std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what() << " in box_overlap_bev";
  }
}

}  // namespace mluopapitest
