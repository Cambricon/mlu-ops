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
class ms_deform_attn_forward : public testing::Test {
 public:
  void setParam(bool handle, bool data_value_desc, bool data_value,
                bool data_spatial_shapes_desc, bool data_spatial_shapes,
                bool data_level_start_index_desc, bool data_level_start_index,
                bool data_sampling_loc_desc, bool data_sampling_loc,
                bool data_attn_weight_desc, bool data_attn_weight,
                bool data_col_desc, bool data_col) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (data_value_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_value_desc_));
      std::vector<int> data_value_desc_dims{2, 3, 4, 5};
      MLUOP_CHECK(mluOpSetTensorDescriptor(data_value_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           data_value_desc_dims.data()));
    }

    if (data_value) {
      if (data_value_desc) {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&data_value_,
                               MLUOP_DTYPE_INT32 *
                                   mluOpGetTensorElementNum(data_value_desc_)));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&data_value_, MLUOP_DTYPE_INT32 * 2));
      }
    }

    if (data_spatial_shapes_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_spatial_shapes_desc_));
      std::vector<int> data_spatial_shapes_desc_dims{6, 2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          data_spatial_shapes_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 2,
          data_spatial_shapes_desc_dims.data()));
    }

    if (data_spatial_shapes) {
      if (data_spatial_shapes_desc) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&data_spatial_shapes_,
                       MLUOP_DTYPE_INT32 * mluOpGetTensorElementNum(
                                               data_spatial_shapes_desc_)));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&data_spatial_shapes_, MLUOP_DTYPE_INT32 * 2));
      }
    }

    if (data_level_start_index_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_level_start_index_desc_));
      std::vector<int> data_level_start_index_desc_dims{6};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          data_level_start_index_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32,
          1, data_level_start_index_desc_dims.data()));
    }

    if (data_level_start_index) {
      if (data_level_start_index_desc) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&data_level_start_index_,
                       MLUOP_DTYPE_INT32 * mluOpGetTensorElementNum(
                                               data_level_start_index_desc_)));
      } else {
        GTEST_CHECK(cnrtSuccess == cnrtMalloc(&data_level_start_index_,
                                              MLUOP_DTYPE_INT32 * 2));
      }
    }

    if (data_sampling_loc_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_sampling_loc_desc_));
      std::vector<int> data_sampling_loc_desc_dims{2, 7, 4, 6, 8, 2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          data_sampling_loc_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
          data_sampling_loc_desc_dims.data()));
    }

    if (data_sampling_loc) {
      if (data_sampling_loc_desc) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&data_sampling_loc_,
                       MLUOP_DTYPE_INT32 *
                           mluOpGetTensorElementNum(data_sampling_loc_desc_)));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&data_sampling_loc_, MLUOP_DTYPE_INT32 * 2));
      }
    }

    if (data_attn_weight_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_attn_weight_desc_));
      std::vector<int> data_attn_weight_desc_dims{2, 7, 4, 6, 8};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          data_attn_weight_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 5,
          data_attn_weight_desc_dims.data()));
    }

    if (data_attn_weight) {
      if (data_attn_weight_desc) {
        GTEST_CHECK(cnrtSuccess == cnrtMalloc(&data_attn_weight_,
                                              MLUOP_DTYPE_INT32 *
                                                  mluOpGetTensorElementNum(
                                                      data_attn_weight_desc_)));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&data_attn_weight_, MLUOP_DTYPE_INT32 * 2));
      }
    }

    if (data_col_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&data_col_desc_));
      std::vector<int> data_col_desc_dims{2, 7, 4, 5};
      MLUOP_CHECK(mluOpSetTensorDescriptor(data_col_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           data_col_desc_dims.data()));
    }

    if (data_col) {
      if (data_col_desc) {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&data_col_,
                               MLUOP_DTYPE_INT32 *
                                   mluOpGetTensorElementNum(data_col_desc_)));
      } else {
        GTEST_CHECK(cnrtSuccess ==
                    cnrtMalloc(&data_col_, MLUOP_DTYPE_INT32 * 2));
      }
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpMsDeformAttnForward(
        handle_, data_value_desc_, data_value_, data_spatial_shapes_desc_,
        data_spatial_shapes_, data_level_start_index_desc_,
        data_level_start_index_, data_sampling_loc_desc_, data_sampling_loc_,
        data_attn_weight_desc_, data_attn_weight_, im2col_step_, data_col_desc_,
        data_col_);
    destroy();
    return status;
  }

 protected:
  void destroy() {
    if (handle_) {
      CNRT_CHECK(cnrtQueueSync(handle_->queue));
      VLOG(4) << "Destroy handle";
      MLUOP_CHECK(mluOpDestroy(handle_));
      handle_ = nullptr;
    }

    if (data_value_desc_) {
      VLOG(4) << "Destroy data_value_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(data_value_desc_));
      data_value_desc_ = nullptr;
    }

    if (data_value_) {
      VLOG(4) << "Destroy data_value";
      GTEST_CHECK(cnrtSuccess == cnrtFree(data_value_));
      data_value_ = nullptr;
    }

    if (data_spatial_shapes_desc_) {
      VLOG(4) << "Destroy data_spatial_shapes_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(data_spatial_shapes_desc_));
      data_spatial_shapes_desc_ = nullptr;
    }

    if (data_spatial_shapes_) {
      VLOG(4) << "Destroy data_spatial_shapes";
      GTEST_CHECK(cnrtSuccess == cnrtFree(data_spatial_shapes_));
      data_spatial_shapes_ = nullptr;
    }

    if (data_level_start_index_desc_) {
      VLOG(4) << "Destroy data_level_start_index_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(data_level_start_index_desc_));
      data_level_start_index_desc_ = nullptr;
    }

    if (data_level_start_index_) {
      VLOG(4) << "Destroy data_level_start_index";
      GTEST_CHECK(cnrtSuccess == cnrtFree(data_level_start_index_));
      data_level_start_index_ = nullptr;
    }

    if (data_sampling_loc_desc_) {
      VLOG(4) << "Destroy data_sampling_loc_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(data_sampling_loc_desc_));
      data_sampling_loc_desc_ = nullptr;
    }

    if (data_sampling_loc_) {
      VLOG(4) << "Destroy data_sampling_loc";
      GTEST_CHECK(cnrtSuccess == cnrtFree(data_sampling_loc_));
      data_sampling_loc_ = nullptr;
    }

    if (data_attn_weight_desc_) {
      VLOG(4) << "Destroy data_attn_weight_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(data_attn_weight_desc_));
      data_attn_weight_desc_ = nullptr;
    }

    if (data_attn_weight_) {
      VLOG(4) << "Destroy data_attn_weight";
      GTEST_CHECK(cnrtSuccess == cnrtFree(data_attn_weight_));
      data_attn_weight_ = nullptr;
    }

    if (data_col_desc_) {
      VLOG(4) << "Destroy data_col_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(data_col_desc_));
      data_col_desc_ = nullptr;
    }

    if (data_col_) {
      VLOG(4) << "Destroy data_col";
      GTEST_CHECK(cnrtSuccess == cnrtFree(data_col_));
      data_col_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t data_value_desc_ = nullptr;
  void *data_value_ = nullptr;
  mluOpTensorDescriptor_t data_spatial_shapes_desc_ = nullptr;
  void *data_spatial_shapes_ = nullptr;
  mluOpTensorDescriptor_t data_level_start_index_desc_ = nullptr;
  void *data_level_start_index_ = nullptr;
  mluOpTensorDescriptor_t data_sampling_loc_desc_ = nullptr;
  void *data_sampling_loc_ = nullptr;
  mluOpTensorDescriptor_t data_attn_weight_desc_ = nullptr;
  void *data_attn_weight_ = nullptr;
  mluOpTensorDescriptor_t data_col_desc_ = nullptr;
  void *data_col_ = nullptr;
  int32_t im2col_step_ = 1;
};

TEST_F(ms_deform_attn_forward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true, true,
             true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_forward";
  }
}

TEST_F(ms_deform_attn_forward, BAD_PARAM_data_value_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true, true,
             true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_forward";
  }
}

TEST_F(ms_deform_attn_forward, BAD_PARAM_data_value_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true, true,
             true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_forward";
  }
}

TEST_F(ms_deform_attn_forward, BAD_PARAM_data_spatial_shapes_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true, true,
             true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_forward";
  }
}

TEST_F(ms_deform_attn_forward, BAD_PARAM_spatial_shapes_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true, true,
             true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_forward";
  }
}

TEST_F(ms_deform_attn_forward, BAD_PARAM_data_level_start_index_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true, true,
             true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_forward";
  }
}

TEST_F(ms_deform_attn_forward, BAD_PARAM_level_start_index_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true, true,
             true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_forward";
  }
}

TEST_F(ms_deform_attn_forward, BAD_PARAM_data_sampling_loc_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true, true,
             true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_forward";
  }
}

TEST_F(ms_deform_attn_forward, BAD_PARAM_sampling_loc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true, true,
             true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_forward";
  }
}

TEST_F(ms_deform_attn_forward, BAD_PARAM_data_attn_weight_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false, true,
             true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_forward";
  }
}

TEST_F(ms_deform_attn_forward, BAD_PARAM_attn_weight_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, false,
             true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_forward";
  }
}

TEST_F(ms_deform_attn_forward, BAD_PARAM_data_col_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             false, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_forward";
  }
}

TEST_F(ms_deform_attn_forward, BAD_PARAM_data_col_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, false);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_forward";
  }
}

}  // namespace mluopapitest
