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
class focal_loss_sigmoid_forward : public testing::Test {
 public:
  void setParam(bool handle, bool input_desc, bool input, bool target_desc,
                bool target, bool weight, bool output_desc, bool output) {
    if (handle) {
      MLUOP_CHECK(mluOpCreate(&handle_));
    }

    if (input_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&input_desc_));
      std::vector<int> input_desc_dims{100, 100};
      MLUOP_CHECK(mluOpSetTensorDescriptor(input_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           input_desc_dims.data()));
    }

    if (input) {
      if (input_desc) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&input_, MLUOP_DTYPE_FLOAT *
                                    mluOpGetTensorElementNum(input_desc_)));
      } else {
        GTEST_CHECK(cnrtSuccess == cnrtMalloc(&input_, MLUOP_DTYPE_FLOAT * 2));
      }
    }

    if (target_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&target_desc_));
      std::vector<int> target_desc_dims{100};
      MLUOP_CHECK(mluOpSetTensorDescriptor(target_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_INT32, 1,
                                           target_desc_dims.data()));
    }

    if (target) {
      if (target_desc) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&target_, MLUOP_DTYPE_INT32 *
                                     mluOpGetTensorElementNum(target_desc_)));
      } else {
        GTEST_CHECK(cnrtSuccess == cnrtMalloc(&target_, MLUOP_DTYPE_FLOAT * 2));
      }
    }

    MLUOP_CHECK(mluOpCreateTensorDescriptor(&weight_desc_));
    std::vector<int> weight_desc_dims{100};
    MLUOP_CHECK(mluOpSetTensorDescriptor(weight_desc_, MLUOP_LAYOUT_ARRAY,
                                         MLUOP_DTYPE_FLOAT, 1,
                                         weight_desc_dims.data()));
    if (weight) {
      GTEST_CHECK(
          cnrtSuccess ==
          cnrtMalloc(&weight_, MLUOP_DTYPE_FLOAT *
                                   mluOpGetTensorElementNum(weight_desc_)));
    }

    if (output_desc) {
      MLUOP_CHECK(mluOpCreateTensorDescriptor(&output_desc_));
      std::vector<int> output_desc_dims{100, 100};
      MLUOP_CHECK(mluOpSetTensorDescriptor(output_desc_, MLUOP_LAYOUT_ARRAY,
                                           MLUOP_DTYPE_FLOAT, 2,
                                           output_desc_dims.data()));
    }

    if (output) {
      if (output_desc) {
        GTEST_CHECK(
            cnrtSuccess ==
            cnrtMalloc(&output_, MLUOP_DTYPE_FLOAT *
                                     mluOpGetTensorElementNum(output_desc_)));
      } else {
        GTEST_CHECK(cnrtSuccess == cnrtMalloc(&output_, MLUOP_DTYPE_FLOAT * 2));
      }
    }
  }

  mluOpStatus_t compute() {
    mluOpStatus_t status = mluOpFocalLossSigmoidForward(
        handle_, prefer_, reduction_, input_desc_, input_, target_desc_,
        target_, weight_desc_, weight_, alpha_, gamma_, output_desc_, output_);
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

    if (input_desc_) {
      VLOG(4) << "Destroy input_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(input_desc_));
      input_desc_ = nullptr;
    }

    if (input_) {
      VLOG(4) << "Destroy input_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(input_));
      input_ = nullptr;
    }

    if (target_desc_) {
      VLOG(4) << "Destroy target_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(target_desc_));
      target_desc_ = nullptr;
    }

    if (target_) {
      VLOG(4) << "Destroy target_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(target_));
      target_ = nullptr;
    }

    if (weight_desc_) {
      VLOG(4) << "Destroy weight_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(weight_desc_));
      weight_desc_ = nullptr;
    }

    if (weight_) {
      VLOG(4) << "Destroy weight_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(weight_));
      weight_ = nullptr;
    }

    if (output_desc_) {
      VLOG(4) << "Destroy output_desc_";
      MLUOP_CHECK(mluOpDestroyTensorDescriptor(output_desc_));
      output_desc_ = nullptr;
    }

    if (output_) {
      VLOG(4) << "Destroy output_";
      GTEST_CHECK(cnrtSuccess == cnrtFree(output_));
      output_ = nullptr;
    }
  }

 private:
  mluOpHandle_t handle_ = nullptr;
  mluOpTensorDescriptor_t input_desc_ = nullptr;
  void *input_ = nullptr;
  mluOpTensorDescriptor_t target_desc_ = nullptr;
  void *target_ = nullptr;
  mluOpTensorDescriptor_t weight_desc_ = nullptr;
  void *weight_ = nullptr;
  mluOpTensorDescriptor_t output_desc_ = nullptr;
  void *output_ = nullptr;
  mluOpComputationPreference_t prefer_ = MLUOP_COMPUTATION_HIGH_PRECISION;
  mluOpLossReduction_t reduction_ = MLUOP_LOSS_REDUCTION_NONE;
  float alpha_ = 0.2;
  float gamma_ = 0.2;
};

TEST_F(focal_loss_sigmoid_forward, BAD_PARAM_handle_null) {
  try {
    setParam(false, true, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in focal_loss_sigmoid_forward";
  }
}

TEST_F(focal_loss_sigmoid_forward, BAD_PARAM_input_desc_null) {
  try {
    setParam(true, false, true, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in focal_loss_sigmoid_forward";
  }
}

TEST_F(focal_loss_sigmoid_forward, BAD_PARAM_input_null) {
  try {
    setParam(true, true, false, true, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in focal_loss_sigmoid_forward";
  }
}

TEST_F(focal_loss_sigmoid_forward, BAD_PARAM_targe_desc_null) {
  try {
    setParam(true, true, true, false, true, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in focal_loss_sigmoid_forward";
  }
}

TEST_F(focal_loss_sigmoid_forward, BAD_PARAM_targe_null) {
  try {
    setParam(true, true, true, true, false, true, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in focal_loss_sigmoid_forward";
  }
}

TEST_F(focal_loss_sigmoid_forward, BAD_PARAM_weight_null) {
  try {
    setParam(true, true, true, true, true, false, true, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in focal_loss_sigmoid_forward";
  }
}

TEST_F(focal_loss_sigmoid_forward, BAD_PARAM_output_desc_null) {
  try {
    setParam(true, true, true, true, true, true, false, true);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in focal_loss_sigmoid_forward";
  }
}

TEST_F(focal_loss_sigmoid_forward, BAD_PARAM_output_null) {
  try {
    setParam(true, true, true, true, true, true, true, false);
    EXPECT_EQ(compute(), MLUOP_STATUS_BAD_PARAM);
  } catch (std::exception &e) {
    FAIL() << "MLUOPAPIGTEST: catched " << e.what()
           << " in focal_loss_sigmoid_forward";
  }
}
}  // namespace mluopapitest
