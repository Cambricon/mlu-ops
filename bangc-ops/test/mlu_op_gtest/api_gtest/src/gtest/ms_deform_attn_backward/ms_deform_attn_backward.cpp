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
class ms_deform_attn_backward : public testing::Test {
 public:
  void setParam(bool handle, bool value_desc, bool value,
                bool spatial_shapes_desc, bool spatial_shapes,
                bool level_start_index_desc, bool level_start_index,
                bool sampling_loc_desc, bool sampling_loc,
                bool attn_weight_desc, bool attn_weight, bool grad_output_desc,
                bool grad_output, bool grad_value_desc, bool grad_value,
                bool grad_sampling_loc_desc, bool grad_sampling_loc,
                bool grad_attn_weight_desc, bool grad_attn_weight) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (value_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&value_desc_));
      std::vector<int> value_desc_dims{2, 3, 4, 5};
      MLUOP_CHECK(mluOpSetTensorDescriptor(value_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           value_desc_dims.data()));
    }

    if (value) {
      if (value_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&value_, mluOpGetTensorElementNum(value_desc_) *
                                    mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&value_, 2 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    if (spatial_shapes_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&spatial_shapes_desc_));
      std::vector<int> spatial_shapes_desc_dims{6, 2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          spatial_shapes_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 2,
          spatial_shapes_desc_dims.data()));
    }

    if (spatial_shapes) {
      if (spatial_shapes_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&spatial_shapes_,
                               mluOpGetTensorElementNum(spatial_shapes_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&spatial_shapes_,
                               2 * mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      }
    }

    if (level_start_index_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&level_start_index_desc_));
      std::vector<int> level_start_index_desc_dims{6};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          level_start_index_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_INT32, 1,
          level_start_index_desc_dims.data()));
    }

    if (level_start_index) {
      if (level_start_index_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&level_start_index_,
                       mluOpGetTensorElementNum(level_start_index_desc_) *
                           mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&level_start_index_,
                               2 * mluOpDataTypeBytes(MLUOP_DTYPE_INT32)));
      }
    }

    if (sampling_loc_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&sampling_loc_desc_));
      std::vector<int> sampling_loc_desc_dims{2, 7, 4, 6, 8, 2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          sampling_loc_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
          sampling_loc_desc_dims.data()));
    }

    if (sampling_loc) {
      if (sampling_loc_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&sampling_loc_,
                               mluOpGetTensorElementNum(sampling_loc_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&sampling_loc_,
                               2 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    if (attn_weight_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&attn_weight_desc_));
      std::vector<int> attn_weight_desc_dims{2, 7, 4, 6, 8};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          attn_weight_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 5,
          attn_weight_desc_dims.data()));
    }

    if (attn_weight) {
      if (attn_weight_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&attn_weight_,
                               mluOpGetTensorElementNum(attn_weight_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&attn_weight_,
                               2 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    if (grad_output_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_output_desc_));
      std::vector<int> grad_output_desc_dims{2, 7, 4, 5};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_output_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 4,
          grad_output_desc_dims.data()));
    }

    if (grad_output) {
      if (grad_output_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&grad_output_,
                               mluOpGetTensorElementNum(grad_output_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&grad_output_,
                               2 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    if (grad_value_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_value_desc_));
      std::vector<int> grad_value_desc_dims{2, 3, 4, 5};
      MLUOP_CHECK(mluOpSetTensorDescriptor(grad_value_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 4,
                                           grad_value_desc_dims.data()));
    }

    if (grad_value) {
      if (grad_value_desc) {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&grad_value_,
                               mluOpGetTensorElementNum(grad_value_desc_) *
                                   mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&grad_value_,
                               2 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    if (grad_sampling_loc_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_sampling_loc_desc_));
      std::vector<int> grad_sampling_loc_desc_dims{2, 7, 4, 6, 8, 2};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_sampling_loc_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 6,
          grad_sampling_loc_desc_dims.data()));
    }

    if (grad_sampling_loc) {
      if (grad_sampling_loc_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&grad_sampling_loc_,
                       mluOpGetTensorElementNum(grad_sampling_loc_desc_) *
                           mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&grad_sampling_loc_,
                               2 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }

    if (grad_attn_weight_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&grad_attn_weight_desc_));
      std::vector<int> grad_attn_weight_desc_dims{2, 7, 4, 6, 8};
      MLUOP_CHECK(mluOpSetTensorDescriptor(
          grad_attn_weight_desc_, MLUOP_LAYOUT_ARRAY, MLUOP_DTYPE_FLOAT, 5,
          grad_attn_weight_desc_dims.data()));
    }

    if (grad_attn_weight) {
      if (grad_attn_weight_desc) {
        GTEST_CHECK(
            CNRT_RET_SUCCESS ==
            cnrtMalloc(&grad_attn_weight_,
                       mluOpGetTensorElementNum(grad_attn_weight_desc_) *
                           mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      } else {
        GTEST_CHECK(CNRT_RET_SUCCESS ==
                    cnrtMalloc(&grad_attn_weight_,
                               2 * mluOpDataTypeBytes(MLUOP_DTYPE_FLOAT)));
      }
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpMsDeformAttnBackward(
        handle_, value_desc_, value_, spatial_shapes_desc_, spatial_shapes_,
        level_start_index_desc_, level_start_index_, sampling_loc_desc_,
        sampling_loc_, attn_weight_desc_, attn_weight_, grad_output_desc_,
        grad_output_, im2col_step_, grad_value_desc_, grad_value_,
        grad_sampling_loc_desc_, grad_sampling_loc_, grad_attn_weight_desc_,
        grad_attn_weight_);
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

    if (value_desc_) {
      VLOG(4) << "Destroy value_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(value_desc_));
      value_desc_ = nullptr;
    }

    if (value_) {
      VLOG(4) << "Destroy value";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(value_));
      value_ = nullptr;
    }

    if (spatial_shapes_desc_) {
      VLOG(4) << "Destroy spatial_shapes_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(spatial_shapes_desc_));
      spatial_shapes_desc_ = nullptr;
    }

    if (spatial_shapes_) {
      VLOG(4) << "Destroy spatial_shapes";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(spatial_shapes_));
      spatial_shapes_ = nullptr;
    }

    if (level_start_index_desc_) {
      VLOG(4) << "Destroy level_start_index_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(level_start_index_desc_));
      level_start_index_desc_ = nullptr;
    }

    if (level_start_index_) {
      VLOG(4) << "Destroy level_start_index";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(level_start_index_));
      level_start_index_ = nullptr;
    }

    if (sampling_loc_desc_) {
      VLOG(4) << "Destroy sampling_loc_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(sampling_loc_desc_));
      sampling_loc_desc_ = nullptr;
    }

    if (sampling_loc_) {
      VLOG(4) << "Destroy sampling_loc";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(sampling_loc_));
      sampling_loc_ = nullptr;
    }

    if (attn_weight_desc_) {
      VLOG(4) << "Destroy attn_weight_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(attn_weight_desc_));
      attn_weight_desc_ = nullptr;
    }

    if (attn_weight_) {
      VLOG(4) << "Destroy attn_weight";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(attn_weight_));
      attn_weight_ = nullptr;
    }

    if (grad_output_desc_) {
      VLOG(4) << "Destroy grad_output_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_output_desc_));
      grad_output_desc_ = nullptr;
    }

    if (grad_output_) {
      VLOG(4) << "Destroy grad_output";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_output_));
      grad_output_ = nullptr;
    }

    if (grad_value_desc_) {
      VLOG(4) << "Destroy grad_value_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_value_desc_));
      grad_value_desc_ = nullptr;
    }

    if (grad_value_) {
      VLOG(4) << "Destroy grad_value";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_value_));
      grad_value_ = nullptr;
    }

    if (grad_sampling_loc_desc_) {
      VLOG(4) << "Destroy grad_sampling_loc_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_sampling_loc_desc_));
      grad_sampling_loc_desc_ = nullptr;
    }

    if (grad_sampling_loc_) {
      VLOG(4) << "Destroy grad_sampling_loc";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_sampling_loc_));
      grad_sampling_loc_ = nullptr;
    }

    if (grad_attn_weight_desc_) {
      VLOG(4) << "Destroy grad_attn_weight_desc";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(grad_attn_weight_desc_));
      grad_attn_weight_desc_ = nullptr;
    }

    if (grad_attn_weight_) {
      VLOG(4) << "Destroy grad_attn_weight";
      GTEST_CHECK(CNRT_RET_SUCCESS == cnrtFree(grad_attn_weight_));
      grad_attn_weight_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t value_desc_ = nullptr;
  void *value_ = nullptr;
  mluOpTensorDescriptor_t spatial_shapes_desc_ = nullptr;
  void *spatial_shapes_ = nullptr;
  mluOpTensorDescriptor_t level_start_index_desc_ = nullptr;
  void *level_start_index_ = nullptr;
  mluOpTensorDescriptor_t sampling_loc_desc_ = nullptr;
  void *sampling_loc_ = nullptr;
  mluOpTensorDescriptor_t attn_weight_desc_ = nullptr;
  void *attn_weight_ = nullptr;
  mluOpTensorDescriptor_t grad_output_desc_ = nullptr;
  void *grad_output_ = nullptr;
  mluOpTensorDescriptor_t grad_value_desc_ = nullptr;
  void *grad_value_ = nullptr;
  mluOpTensorDescriptor_t grad_sampling_loc_desc_ = nullptr;
  void *grad_sampling_loc_ = nullptr;
  mluOpTensorDescriptor_t grad_attn_weight_desc_ = nullptr;
  void *grad_attn_weight_ = nullptr;
  int32_t im2col_step_ = 1;
};

TEST_F(ms_deform_attn_backward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true, true, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_value_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true, true, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_value_null) {
  try {
    setParam(true, true, false, true, true, true, true, true, true, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_spatial_shapes_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true, true, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_spatial_shapes_null) {
  try {
    setParam(true, true, true, true, false, true, true, true, true, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_level_start_index_desc_null) {
  try {
    setParam(true, true, true, true, true, false, true, true, true, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_level_start_index_null) {
  try {
    setParam(true, true, true, true, true, true, false, true, true, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_sampling_loc_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, false, true, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_sampling_loc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, false, true, true,
             true, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_attn_weight_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, false, true,
             true, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_attn_weight_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, false,
             true, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_grad_output_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             false, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_grad_output_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, false, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_grad_value_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, false, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_grad_value_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, true, false, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_grad_sampling_loc_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, true, true, false, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_grad_sampling_loc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, true, true, true, false, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_grad_attn_weight_desc_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, true, true, true, true, false, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}

TEST_F(ms_deform_attn_backward, BAD_PARAM_grad_attn_weight_null) {
  try {
    setParam(true, true, true, true, true, true, true, true, true, true, true,
             true, true, true, true, true, true, true, false);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in ms_deform_attn_backward";
  }
}
}  // namespace mluopapitest