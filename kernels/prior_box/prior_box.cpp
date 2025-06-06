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
#include "prior_box.h"

#include <string>

#include "core/gen_case.h"
#include "core/runtime/device.h"

#define api "mluOpPriorBox"
#define MLU500SERIERS_MAX_SUPPORT 2100
#define MLU300SERIERS_MAX_SUPPORT 2900

// policy function
static void policyFuncPriorBox(const mluOpHandle_t handle, cnrtDim3_t *k_dim,
                               cnrtFunctionType_t *k_type, const int count) {
  *k_type = cnrtFuncTypeBlock;
  uint32_t cluster_max = mluop::runtime::getClusterLimitCapability(handle);
  uint32_t core_num_per_cluster =
      mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  uint32_t core_max = cluster_max * core_num_per_cluster;
  uint32_t core_used = core_max > count ? count : core_max;
  k_dim->x = core_used;
  k_dim->y = 1;
  k_dim->z = 1;
}

static int getNumPriors(const mluOpTensorDescriptor_t min_sizes_desc,
                        const mluOpTensorDescriptor_t aspect_ratios_desc,
                        const mluOpTensorDescriptor_t max_sizes_desc) {
  int num_priors =
      min_sizes_desc->getDimIndex(0) * aspect_ratios_desc->getDimIndex(0);
  if (max_sizes_desc->getTotalElementNum() != 0) {
    num_priors += max_sizes_desc->getDimIndex(0);
  }
  return num_priors;
}

mluOpStatus_t mluOpPriorBoxParamCheck(
    const mluOpHandle_t handle, const mluOpTensorDescriptor_t min_sizes_desc,
    const void *min_sizes, const mluOpTensorDescriptor_t aspect_ratios_desc,
    const void *aspect_ratios, const mluOpTensorDescriptor_t variances_desc,
    const void *variances, const mluOpTensorDescriptor_t max_sizes_desc,
    const void *max_sizes, const int height, const int width,
    const int im_height, const int im_width, const float step_h,
    const float step_w, const float offset, const bool clip,
    const bool min_max_aspect_ratios_order,
    const mluOpTensorDescriptor_t output_desc, const void *output,
    const mluOpTensorDescriptor_t var_desc, const void *var) {
  // check input
  PARAM_CHECK(api, handle != nullptr);
  PARAM_CHECK(api, min_sizes_desc != nullptr);
  PARAM_CHECK(api, aspect_ratios_desc != nullptr);
  PARAM_CHECK(api, variances_desc != nullptr);
  PARAM_CHECK(api, max_sizes_desc != nullptr);
  PARAM_CHECK(api, output_desc != nullptr);
  PARAM_CHECK(api, var_desc != nullptr);
  // check dim
  PARAM_CHECK(api, min_sizes_desc->getDim() == 1);
  PARAM_CHECK(api, aspect_ratios_desc->getDim() == 1);
  PARAM_CHECK(api, variances_desc->getDim() == 1);
  PARAM_CHECK(api, max_sizes_desc->getDim() == 1);
  PARAM_CHECK(api, output_desc->getDim() == 4);
  PARAM_CHECK(api, var_desc->getDim() == 4);
  // check shape
  PARAM_CHECK(api, variances_desc->getDimIndex(0) == 4);
  PARAM_CHECK(api, output_desc->getDimIndex(0) == height);
  PARAM_CHECK(api, output_desc->getDimIndex(1) == width);
  PARAM_CHECK(api, output_desc->getDimIndex(3) == 4);
  PARAM_CHECK(api, var_desc->getDimIndex(3) == 4);
  PARAM_CHECK_GE(api, height, 0);
  PARAM_CHECK_GE(api, width, 0);
  // check data type
  PARAM_CHECK(api, min_sizes_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(api, aspect_ratios_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(api, variances_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(api, max_sizes_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(api, output_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  PARAM_CHECK(api, var_desc->getDtype() == MLUOP_DTYPE_FLOAT);
  // check scalar param
  PARAM_CHECK_GT(api, step_h, 0);
  PARAM_CHECK_GT(api, step_w, 0);
  // check stride
  STRIDE_TENSOR_CHECK("[mluOpPriorBox]:", min_sizes_desc,
                      "min_sizes_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpPriorBox]:", aspect_ratios_desc,
                      "aspect_ratios_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpPriorBox]:", variances_desc,
                      "variances_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpPriorBox]:", max_sizes_desc,
                      "max_sizes_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpPriorBox]:", output_desc,
                      "output_desc must be contiguous");
  STRIDE_TENSOR_CHECK("[mluOpPriorBox]:", var_desc,
                      "var_desc must be contiguous");

  // check param depand

  for (int i = 0; i < output_desc->getDim(); i++) {
    std::string i_str = "i: " + std::to_string(i);
    PARAM_CHECK(api, output_desc->getDimIndex(i) == var_desc->getDimIndex(i),
                i_str);
  }
  if (max_sizes_desc->getTotalElementNum() != 0) {
    PARAM_CHECK(
        api, max_sizes_desc->getDimIndex(0) == min_sizes_desc->getDimIndex(0));
    PARAM_CHECK(api, max_sizes_desc->getDimIndex(0) +
                             min_sizes_desc->getDimIndex(0) *
                                 aspect_ratios_desc->getDimIndex(0) ==
                         output_desc->getDimIndex(2));
  } else {
    PARAM_CHECK(api, min_sizes_desc->getDimIndex(0) *
                             aspect_ratios_desc->getDimIndex(0) ==
                         output_desc->getDimIndex(2));
  }
  const int num_priors =
      getNumPriors(min_sizes_desc, aspect_ratios_desc, max_sizes_desc);
  // check num_priors limit
  const int max_support_num_priors = (handle->arch > 500)
                                         ? MLU500SERIERS_MAX_SUPPORT
                                         : MLU300SERIERS_MAX_SUPPORT;
  if (num_priors > max_support_num_priors) {
    LOG(ERROR) << api << " Support max num_priors is " << max_support_num_priors
               << ",but now is " << num_priors;
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  // check large tensor
  if ((mluOpGetTensorElementNum(min_sizes_desc) >= LARGE_TENSOR_NUM) ||
      (mluOpGetTensorElementNum(aspect_ratios_desc) >= LARGE_TENSOR_NUM) ||
      (mluOpGetTensorElementNum(max_sizes_desc) >= LARGE_TENSOR_NUM) ||
      (mluOpGetTensorElementNum(output_desc) >= LARGE_TENSOR_NUM) ||
      (mluOpGetTensorElementNum(var_desc) >= LARGE_TENSOR_NUM)) {
    LOG(ERROR) << api << " Overflow max tensor num."
               << " Currently, MLU-OPS supports tensor num smaller than 2^31.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpPriorBox(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t min_sizes_desc,
    const void *min_sizes, const mluOpTensorDescriptor_t aspect_ratios_desc,
    const void *aspect_ratios, const mluOpTensorDescriptor_t variances_desc,
    const void *variances, const mluOpTensorDescriptor_t max_sizes_desc,
    const void *max_sizes, const int height, const int width,
    const int im_height, const int im_width, const float step_h,
    const float step_w, const float offset, const bool clip,
    const bool min_max_aspect_ratios_order,
    const mluOpTensorDescriptor_t output_desc, void *output,
    const mluOpTensorDescriptor_t var_desc, void *var) {
  // param check
  mluOpStatus_t pb_status = mluOpPriorBoxParamCheck(
      handle, min_sizes_desc, min_sizes, aspect_ratios_desc, aspect_ratios,
      variances_desc, variances, max_sizes_desc, max_sizes, height, width,
      im_height, im_width, step_h, step_w, offset, clip,
      min_max_aspect_ratios_order, output_desc, output, var_desc, var);
  if (pb_status != MLUOP_STATUS_SUCCESS) {
    return pb_status;
  }
  // check zero element
  if ((mluOpGetTensorElementNum(min_sizes_desc) == 0) ||
      (mluOpGetTensorElementNum(aspect_ratios_desc) == 0) ||
      (mluOpGetTensorElementNum(variances_desc) == 0)) {
    LOG(ERROR) << api << " Zero element tensor failure.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if ((mluOpGetTensorElementNum(output_desc)) == 0 ||
      (mluOpGetTensorElementNum(var_desc)) == 0) {
    VLOG(5) << api << " Input skip zero element tensor..";
    return MLUOP_STATUS_SUCCESS;
  }
  // check ptr
  PARAM_CHECK(api, min_sizes != nullptr);
  PARAM_CHECK(api, aspect_ratios != nullptr);
  PARAM_CHECK(api, variances != nullptr);
  PARAM_CHECK(api, output != nullptr);
  PARAM_CHECK(api, var != nullptr);
  if (max_sizes_desc->getTotalElementNum() > 0) {
    PARAM_CHECK(api, max_sizes != nullptr);
  }

  const int min_sizes_num = min_sizes_desc->getDimIndex(0);
  const int aspect_ratios_num = aspect_ratios_desc->getDimIndex(0);
  const int variances_num = variances_desc->getDimIndex(0);
  const int max_sizes_num = max_sizes_desc->getDimIndex(0);
  const int output_size = output_desc->getTotalElementNum();
  const int var_size = var_desc->getTotalElementNum();
  const int num_priors = max_sizes_num > 0
                             ? min_sizes_num * aspect_ratios_num + max_sizes_num
                             : min_sizes_num * aspect_ratios_num;

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("prior_box", "PRIOR_BOX");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA_REAL(true, "input1", min_sizes, min_sizes_desc);
    GEN_CASE_DATA_REAL(true, "input2", aspect_ratios, aspect_ratios_desc);
    GEN_CASE_DATA_REAL(true, "input3", variances, variances_desc);
    GEN_CASE_DATA_REAL(true, "input4", max_sizes, max_sizes_desc);
    GEN_CASE_OP_PARAM_SINGLE(0, "prior_box", "height", height);
    GEN_CASE_OP_PARAM_SINGLE(1, "prior_box", "width", width);
    GEN_CASE_OP_PARAM_SINGLE(2, "prior_box", "im_height", im_height);
    GEN_CASE_OP_PARAM_SINGLE(3, "prior_box", "im_width", im_width);
    GEN_CASE_OP_PARAM_SINGLE(4, "prior_box", "step_h", step_h);
    GEN_CASE_OP_PARAM_SINGLE(5, "prior_box", "step_w", step_w);
    GEN_CASE_OP_PARAM_SINGLE(6, "prior_box", "offset", offset);
    /*
    The community's prior_box reference link: paddle/phi/kernels/gpu/prior_box_kernel.cu

    The prior_box community logic includes a flip parameter that modifies the aspect_ratios.

    In mluOpPriorBox, the aspect_ratios are already the result after modification
    (presumably handled by the framework layer),
    and mluOpPriorBox does not directly handle the flip parameter.

    However, the generator uses the flip parameter to modify the aspect_ratios.

    Currently, the logic of "framework + mlu-ops" is equivalent to that of the "generator",
    but the operator's logic is coupled, making it difficult to make lightweight modifications.

    A temporary workaround is to uniformly set flip to false during case generation,
    meaning the generated results do not require further modification.

    For the true cases dumped by the framework, if the case is regenerated through the generator,
    the generator code can handle it correctly (aspect_ratios have been processed,
    and the generator will not reprocess them).
    */
    GEN_CASE_OP_PARAM_SINGLE(7, "prior_box", "flip", false);
    GEN_CASE_OP_PARAM_SINGLE(8, "prior_box", "clip", clip);
    GEN_CASE_OP_PARAM_SINGLE(9, "prior_box", "min_max_aspect_ratios_order",
                             min_max_aspect_ratios_order);
    GEN_CASE_DATA(false, "output", output, output_desc, 0, 0);
    GEN_CASE_DATA(false, "var", var, var_desc, 0, 0);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }
  cnrtDim3_t k_dim_box;
  cnrtFunctionType_t k_type;
  policyFuncPriorBox(handle, &k_dim_box, &k_type, height);
  VLOG(5) << api << "Begin Launch KernelPriorBox [" << k_type << ", "
          << k_dim_box.x << ", " << k_dim_box.y << ", " << k_dim_box.z << "].";
  CHECK_RETURN(
      "[mluOpPriorBox]",
      KernelPriorBox(k_dim_box, k_type, handle->queue, min_sizes, min_sizes_num,
                     aspect_ratios, aspect_ratios_num, variances, variances_num,
                     max_sizes, max_sizes_num, height, width, im_height,
                     im_width, step_h, step_w, offset, num_priors, clip,
                     min_max_aspect_ratios_order, output, output_size, var,
                     var_size));
  VLOG(5) << "End KernelPriorBox kernel";
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
