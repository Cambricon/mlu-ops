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
#ifndef KERNELS_TENSOR_STRIDE_PROCESS_TENSOR_STRIDE_PROCESS_HOST_H_
#define KERNELS_TENSOR_STRIDE_PROCESS_TENSOR_STRIDE_PROCESS_HOST_H_
#include "mlu_op.h"
#include "kernels/kernel.h"
#include "core/context.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "core/tool.h"

namespace mluop {

struct TensorShape {
  int64_t tensor_dims[MLUOP_DIM_MAX];
  int64_t tensor_strides[MLUOP_DIM_MAX];
  uint64_t total_num = 1;
  uint64_t total_stride = 1;
  bool is_contiguous = 1;
};

bool strideCaseWithNotConsistentDense(int tensor_num, ...);

bool isDenseStrideTensor(const mluOpTensorDescriptor_t tensor_desc);

bool ifNeedTensorStrideProcess(const mluOpTensorDescriptor_t tensor_desc);

bool isTransPadStride(TensorShape &tensor_shape, int64_t *dims,
                      int64_t *strides);

void getTensorShape(const mluOpTensorDescriptor_t tensor_desc,
                    TensorShape *tensor_shape);

void getExpandTensorShape(const mluOpTensorDescriptor_t tensor_desc,
                          int64_t *target_shape, int target_dim,
                          TensorShape *tensor_shape);

}  // namespace mluop

mluOpStatus_t MLUOP_WIN_API mluOpTensorStrideIn(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, void *output);

mluOpStatus_t MLUOP_WIN_API mluOpTensorStrideOut(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, void *output);

mluOpStatus_t MLUOP_WIN_API
mluOpContiguous(mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
                const void *input, void *output);

mluOpStatus_t MLUOP_WIN_API KernelTensorStrideIn(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, mluop::TensorShape input_shape, void *output,
    mluOpDataType_t dtype);

mluOpStatus_t MLUOP_WIN_API
KernelTensorStrideOut(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                      cnrtQueue_t queue, const void *input, void *output,
                      mluop::TensorShape output_shape, mluOpDataType_t dtype);

#endif  // KERNELS_TENSOR_STRIDE_PROCESS_TENSOR_STRIDE_PROCESS_HOST_H_
