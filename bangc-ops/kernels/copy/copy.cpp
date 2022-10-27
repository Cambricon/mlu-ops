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
#include <string>

#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "copy_mlu.h"
#include "kernels/tensor_stride_process/tensor_stride_process.h"

// According to test, the threshold is about 4KB.
const int POLICY_UNION1_UPLIMIT = 4 * 1024;  // 4KB

// Minimum stride of the last dim to enable SMC copy besides IPC
const int POLICY_MIN_LAST_DIM_STRIDE_FOR_SMC = 1 * 1024;  // 1 KB

// policy function
static mluOpStatus_t policyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                                cnrtFunctionType_t *k_type,
                                const size_t total_num) {
  // Choose best task dimension
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = mluop::runtime::getCoreNumOfEachUnionCapability(handle);
  k_dim->z = 1;

  // According to test, the threshold is about 4KB.
  int threshold = POLICY_UNION1_UPLIMIT;
  if (total_num > threshold) {
    unsigned int max_dimy = mluop::runtime::getClusterLimitCapability(handle);
    k_dim->y = (max_dimy > (total_num / threshold)) ? (total_num / threshold)
                                                    : max_dimy;
  } else {
    k_dim->y = 1;
  }
  return MLUOP_STATUS_SUCCESS;
}

static size_t shapeStrideCount(const mluOpTensorDescriptor_t desc) {
  size_t total = 1;
  for (int i = 0; i < desc->dim; ++i) {
    if (desc->dims[i] == 0) {
      total = 0;
      break;
    }
    total += (desc->dims[i] - 1) * desc->strides[i];
  }
  return total;
}

/* The API for mluOpCopy. This operator copy a tensor to another tensor.
 *  The operator supports data types of INT8, INT16, INT32, HALF and FLOAT32.
 *  And the kernel can be launched on UNION1 job types with multi jobs.
 *  parameters:
 *    handle - the context of the current kernel
 *    input_desc - the tensor descriptor of the input tensor
 *    input - pointer to the input tensor on DDR
 *    output_desc - the tensor descriptor of the output tensor
 *    output - pointer to the output tensor on DDR
 *  returns:
 *    MLUOP_STATUS_SUCCESS - The kernel is performed successfully
 *    MLUOP_STATUS_EXECUTION_FAILED - The job configuration is invalid
 */
mluOpStatus_t MLUOP_WIN_API mluOpCopy(mluOpHandle_t handle,
                                      const mluOpTensorDescriptor_t input_desc,
                                      const void *input,
                                      const mluOpTensorDescriptor_t output_desc,
                                      void *output) {
  PARAM_CHECK("[mluOpCopy]", handle != NULL);
  PARAM_CHECK("[mluOpCopy]", input_desc != NULL);
  PARAM_CHECK("[mluOpCopy]", output_desc != NULL);
  PARAM_CHECK("[mluOpCopy]", input_desc->dtype != MLUOP_DTYPE_INVALID);
  PARAM_CHECK("[mluOpCopy]", output_desc->dtype != MLUOP_DTYPE_INVALID);
  PARAM_CHECK("[mluOpCopy]", input_desc->dtype == output_desc->dtype);
  size_t num_input = mluOpGetTensorElementNum(input_desc);
  size_t num_output = mluOpGetTensorElementNum(output_desc);
  size_t size_input =
      shapeStrideCount(input_desc) * getSizeOfDataType(input_desc->dtype);
  size_t size_output =
      shapeStrideCount(output_desc) * getSizeOfDataType(output_desc->dtype);
  bool stride_kernel = false;
  if (strideCaseWithNotConsistentDense(2, input_desc, output_desc)) {
    stride_kernel = true;
    TENSOR_SIZE_CHECK("[mluOpCopy]", size_input, LARGE_TENSOR_SIZE,
                      "input tensor size is too large. ");
    TENSOR_SIZE_CHECK("[mluOpCopy]", size_output, LARGE_TENSOR_SIZE,
                      "output tensor size is too large. ");
  } else {
    TENSOR_NUM_CHECK("[mluOpCopy]", num_input, LARGE_TENSOR_NUM, "");
  }
  if (num_input != num_output) {
    LOG(ERROR) << "[mluOpCopy] the size of input should be the same as output"
               << ". But now the size of input is " << num_input
               << ", and the size of output is " << num_output << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (num_input == 0) {
    VLOG(5) << "mluOpCopy skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }
  PARAM_CHECK("[mluOpCopy]", input != NULL);
  PARAM_CHECK("[mluOpCopy]", output != NULL);

  // initialize parameters
  mluOpDataType_t k_data_type = input_desc->dtype;
  size_t total_num = num_input;

  // set kernel datatype to int8, set total_num to size
  const int kDTypeSize = getSizeOfDataType(k_data_type);
  total_num = num_input * kDTypeSize;

  // generate copy prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("copy");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", input, input_desc, 100, 0);
    GEN_CASE_DATA(false, "output", output, output_desc, 0, 0);
    if (output_desc->dtype == MLUOP_DTYPE_COMPLEX_HALF ||
        output_desc->dtype == MLUOP_DTYPE_COMPLEX_FLOAT) {
      GEN_CASE_TEST_PARAM_NEW(false, false, true, 0, 0, 0, 0, 0, 0);

    } else {
      GEN_CASE_TEST_PARAM_NEW(false, false, true, 0, 0, 0);
    }
  }
  // generate copy prototxt end!

  // dimxyz policy
  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_BLOCK;
  cnrtDim3_t k_dim;

  // Choose best task dimension
  policyFunc(handle, &k_dim, &k_type, total_num);

  if (stride_kernel) {
    PARAM_CHECK("[mluOpCopy]", input_desc->dim <= MLUOP_DIM_MAX);
    TensorShape input_shape;
    TensorShape output_shape;
    getTensorShape(input_desc, &input_shape);
    getTensorShape(output_desc, &output_shape);
    VLOG(5) << "Launch Kernel mluOpUnion1KernelCopyWithStride <<<Union1"
            << ", Dim3{" << k_dim.x << ", " << k_dim.y << ", " << k_dim.z
            << "} >>>";

    // FIXME(taokai): this whole section is a hotfix for now, better strategy is
    // needed.
    bool use_SMC = false;
    if (handle->arch > 290) {
      // case 1: large input dim_stride at the last dim
      int last_dim_stride =
          input_shape.tensor_strides[MLUOP_DIM_MAX - 1] * kDTypeSize;
      VLOG(5) << "last_dim_stride: " << last_dim_stride;

      if (last_dim_stride >= POLICY_MIN_LAST_DIM_STRIDE_FOR_SMC) {
        use_SMC = true;
      }
      // case 2: expand the last dim [N, 1] -> [N, M], N > 10000
      // FIXME(taokai): N > 10000 is pure empirical based on current dataset,
      //                better reasoning and strategy are needed.
      if (input_shape.tensor_strides[MLUOP_DIM_MAX - 1] == 0 &&
          input_shape.tensor_strides[MLUOP_DIM_MAX - 2] > 0 &&
          input_shape.tensor_dims[MLUOP_DIM_MAX - 2] > 10000 &&
          output_shape.tensor_strides[MLUOP_DIM_MAX - 1] == 1) {
        VLOG(5) << "Expand last dim. ";
        use_SMC = true;
      }
    }
    VLOG(5) << "use_SMC: " << use_SMC;
    KERNEL_CHECK((mluOpUnion1KernelCopyWithStride(
        k_dim, k_type, handle->queue, input, input_shape, output, output_shape,
        num_input, kDTypeSize, use_SMC)));
  } else {
    VLOG(5) << "Launch Kernel mluOpUnion1KernelCopy <<<Union1"
            << ", Dim3{" << k_dim.x << ", " << k_dim.y << ", " << k_dim.z
            << "} >>>";
    KERNEL_CHECK((mluOpUnion1KernelCopy(k_dim, k_type, handle->queue, input,
                                        output, num_input, kDTypeSize)));
  }

  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
