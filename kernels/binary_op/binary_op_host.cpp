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
#include "binary_op_host.h"

#include <algorithm>
#include <vector>

#include "mlu_op.h"
#include "kernels/kernel.h"
#include "kernels/debug.h"
#include "core/tensor.h"
#include "core/type.h"
#include "core/context.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "kernels/tensor_stride_process/tensor_stride_process_host.h"

void binaryOpPolicyFunc(mluOpHandle_t handle, const int pad_up_size,
                        cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type,
                        const mluOpTensorDescriptor_t desc) {
  const uint64_t union_number =
      mluop::runtime::getClusterLimitCapability(handle);
  const uint64_t core_dim =
      mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  const uint64_t core_number = union_number * core_dim;

  const uint64_t dim = mluOpGetTensorElementNum(desc);
  uint64_t size = dim * mluop::getSizeOfDataType(desc->getDtype());
  size = PAD_UP(size, pad_up_size);

  // Union1 policyFunc
  *k_type = cnrtFuncTypeUnion1;
  k_dim->x = core_dim;
  const uint64_t maximum_partitions = PAD_UP(size / pad_up_size, core_dim);
  if (maximum_partitions < core_number) {
    k_dim->y = maximum_partitions / core_dim;
  } else {
    k_dim->y = core_number / core_dim;
  }
  k_dim->z = 1;
}

void binaryOpBlockPolicyFunc(mluOpHandle_t handle,
                             const mluOpTensorDescriptor_t desc,
                             uint32_t pad_up_size, cnrtDim3_t &k_dim,
                             cnrtFunctionType_t &k_type,
                             size_t &normal_core_elem_num,
                             size_t &tail_core_elem_num) {
  k_type = cnrtFuncTypeBlock;
  const uint32_t core_number =
      mluop::runtime::getMaxParallelJobNum(handle, k_type);
  const size_t element_num = mluOpGetTensorElementNum(desc);
  const uint32_t dtype_size = mluop::getSizeOfDataType(desc->getDtype());
  if (MLUOP_MLU590 == handle->arch) {
    const uint32_t llc_pending_size = 512;
    pad_up_size = std::max(llc_pending_size, pad_up_size);
  }
  const uint32_t aligned_num = DIV_UP(pad_up_size, dtype_size);
  normal_core_elem_num =
      std::max(static_cast<size_t>(aligned_num),
               CEIL_ALIGN(element_num / core_number, aligned_num));
  const uint32_t rem_num = element_num % normal_core_elem_num;
  size_t task_num = element_num / normal_core_elem_num;
  if (0 < rem_num && task_num < core_number) {
    task_num = task_num + 1;
    tail_core_elem_num = rem_num;
  } else { /*rem_num == 0 or task_num == core_number, tail is normal or more*/
    tail_core_elem_num = normal_core_elem_num + rem_num;
  }

  k_dim.x = task_num;
  k_dim.y = 1;
  k_dim.z = 1;
}

static inline bool isSupportType(const mluOpDataType_t check_type,
                                 const mluOpDataType_t support_type[],
                                 const int len) {
  for (int i = 0; i < len; ++i) {
    if (check_type == support_type[i]) {
      return true;
    }
  }
  return false;
}

mluOpStatus_t binaryOpParamCheck(
    const std::string &op_name, const mluOpHandle_t handle,
    const mluOpTensorDescriptor_t input1_desc, const void *input1,
    const mluOpTensorDescriptor_t input2_desc, const void *input2,
    const mluOpTensorDescriptor_t output_desc, const void *output,
    const mluOpDataType_t support_type[], const int len, bool &zero_element,
    bool isSupportBroadcast) {
  // check descriptor
  PARAM_CHECK(op_name, handle != NULL);
  PARAM_CHECK(op_name, input1_desc != NULL);
  PARAM_CHECK(op_name, input2_desc != NULL);
  PARAM_CHECK(op_name, output_desc != NULL);

  // check dtype equal
  PARAM_CHECK_EQ(op_name, input1_desc->getDtype(), input2_desc->getDtype());
  PARAM_CHECK_EQ(op_name, input1_desc->getDtype(), output_desc->getDtype());

  // check dim less than MLUOP_DIM_MAX
  PARAM_CHECK_LE(op_name, input1_desc->getDim(), MLUOP_DIM_MAX);
  PARAM_CHECK_LE(op_name, input2_desc->getDim(), MLUOP_DIM_MAX);
  PARAM_CHECK_LE(op_name, output_desc->getDim(), MLUOP_DIM_MAX);
  PARAM_CHECK_GT(op_name, input1_desc->getDim(), 0);
  PARAM_CHECK_GT(op_name, input2_desc->getDim(), 0);
  PARAM_CHECK_GT(op_name, output_desc->getDim(), 0);

  // check data type support
  if (!isSupportType(input1_desc->getDtype(), support_type, len)) {
    LOG(ERROR) << op_name << ":input1_desc's data type is not supported.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if (isSupportBroadcast) {
    int32_t left_dim_num = input1_desc->getDim();
    int32_t right_dim_num = input2_desc->getDim();
    int32_t max_dim = std::max(left_dim_num, right_dim_num);
    std::vector<int> left_aligned_dims(input1_desc->getDims(),
                                       input1_desc->getDims() + input1_desc->getDim());
    std::vector<int> right_aligned_dims(input2_desc->getDims(),
                                        input2_desc->getDims() + input2_desc->getDim());

    // aligning dimensions to max_dim
    if (left_dim_num < max_dim) {
      left_aligned_dims.insert(left_aligned_dims.begin(),
                               max_dim - left_dim_num, 1);
    }

    if (right_dim_num < max_dim) {
      right_aligned_dims.insert(right_aligned_dims.begin(),
                                max_dim - right_dim_num, 1);
    }

    if (output_desc->getDim() != max_dim) {
      LOG(ERROR)
          << op_name
          << " The dimension size of the output tensors does not meet the "
             "requirements of broadcast.";
      return MLUOP_STATUS_BAD_PARAM;
    }

    for (int i = 0; i < max_dim; ++i) {
      if (left_aligned_dims[i] != right_aligned_dims[i] &&
          left_aligned_dims[i] != 1 && right_aligned_dims[i] != 1) {
        LOG(ERROR)
            << op_name << " The shape of the two inputs do not meet the"
            << " requirements of broadcast. In the broadcast dimension, the"
            << " dimension size of x is " << left_aligned_dims[i]
            << " the dimension size of y is " << right_aligned_dims[i] << ".";
        return MLUOP_STATUS_BAD_PARAM;
      }
      int max_dim_value = std::max(left_aligned_dims[i], right_aligned_dims[i]);
      int min_dim_value = std::min(left_aligned_dims[i], right_aligned_dims[i]);
      if ((min_dim_value > 0 && max_dim_value != output_desc->getDimIndex(i)) ||
          (min_dim_value == 0 && output_desc->getDimIndex(i) != 0)) {
        LOG(ERROR) << op_name
                   << " The shape of the inferred tensors is not equal"
                   << " to the output tensor. In the broadcast shape"
                   << ", the shape of x is "
                   << array2String(input1_desc->getDim(), input1_desc->getDims())
                   << ", the shape of y is "
                   << array2String(input2_desc->getDim(), input2_desc->getDims())
                   << ", the shape of z is "
                   << array2String(output_desc->getDim(), output_desc->getDims());
        return MLUOP_STATUS_BAD_PARAM;
      }
    }
  } else {
    mluOpStatus_t sameShapeStatus = binaryOpParamSameShapeCheck(
        op_name, input1_desc, input2_desc, output_desc);
    if (sameShapeStatus != MLUOP_STATUS_SUCCESS) {
      return sameShapeStatus;
    }
  }

  // check 0 element
  if ((mluOpGetTensorElementNum(input1_desc) == 0) ||
      (mluOpGetTensorElementNum(input2_desc) == 0) ||
      (mluOpGetTensorElementNum(output_desc) == 0)) {
    VLOG(5) << op_name << " skip zero element tensor.";
    zero_element = true;
    return MLUOP_STATUS_SUCCESS;
  }

  // check device pointer
  PARAM_CHECK(op_name, input1 != NULL);
  PARAM_CHECK(op_name, input2 != NULL);
  PARAM_CHECK(op_name, output != NULL);

  // large tensor check
  if (handle->arch < MLUOP_MLU590) {
    uint64_t num_output = mluOpGetTensorElementNum(output_desc);
    TENSOR_NUM_CHECK(op_name, num_output, LARGE_TENSOR_NUM,
                     "output tensor num is too large. ");

    if (mluop::strideCaseWithNotConsistentDense(3, input1_desc, input2_desc,
                                                output_desc)) {
      uint64_t num_input1_with_stride = shapeStrideCount(input1_desc);
      uint64_t num_input2_with_stride = shapeStrideCount(input2_desc);
      uint64_t num_output_with_stride = shapeStrideCount(output_desc);
      TENSOR_NUM_CHECK(op_name, num_input1_with_stride, LARGE_TENSOR_NUM,
                       "input1 tensor num with stride is too large. ");
      TENSOR_NUM_CHECK(op_name, num_input2_with_stride, LARGE_TENSOR_NUM,
                       "input2 tensor num with stride is too large. ");
      TENSOR_NUM_CHECK(op_name, num_output_with_stride, LARGE_TENSOR_NUM,
                       "output tensor num with stride is too large. ");
    }
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t binaryOpParamSameShapeCheck(
    const std::string &op_name, const mluOpTensorDescriptor_t input1_desc,
    const mluOpTensorDescriptor_t input2_desc,
    const mluOpTensorDescriptor_t output_desc) {
  // check dim
  PARAM_CHECK_EQ("[" + op_name + "]", input1_desc->getDim(), input2_desc->getDim());
  PARAM_CHECK_EQ("[" + op_name + "]", input1_desc->getDim(), output_desc->getDim());

  // check shape
  for (int i = 0; i < input1_desc->getDim(); i++) {
    if (input1_desc->getDimIndex(i) != input2_desc->getDimIndex(i)) {
      LOG(ERROR) << op_name << ":The shape of input1 should be equal to input2"
                 << ". But now input1's shape[" << i << "] is "
                 << input1_desc->getDimIndex(i) << ", input2's shape[" << i << "] is "
                 << input2_desc->getDimIndex(i) << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
    if (input1_desc->getDimIndex(i) != output_desc->getDimIndex(i)) {
      LOG(ERROR) << op_name << ":The shape of input1 should be equal to output"
                 << ". But now input1's shape[" << i << "] is "
                 << input1_desc->getDimIndex(i) << ", output's shape[" << i << "] is "
                 << output_desc->getDimIndex(i) << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  return MLUOP_STATUS_SUCCESS;
}

std::string array2String(int32_t dim_num,const int64_t *dims) {
  std::string res;
  res.push_back('[');
  for (int i = 0; i < dim_num; i++) {
    res.append(std::to_string(dims[i]));
    if (i + 1 != dim_num) {
      res.push_back(',');
    }
  }
  res.push_back(']');
  return res;
}
