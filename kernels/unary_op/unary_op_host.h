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
#ifndef KERNELS_UNARY_OP_UNARY_OP_HOST_H_
#define KERNELS_UNARY_OP_UNARY_OP_HOST_H_

#include <string>
#include "mlu_op.h"
#include "core/context.h"
#include "core/logging.h"
#include "core/tool.h"
#include "core/gen_case.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/tensor_stride_process/tensor_stride_process_host.h"

#define FLAG_UNARY(DType, Pref) \
  (x_desc->dtype == MLUOP_DTYPE_##DType && prefer == MLUOP_COMPUTATION_##Pref)

#define FLAG_UNARY_DTYPE(DType) (x_desc->dtype == MLUOP_DTYPE_##DType)

void unaryOpPolicyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type,
                       mluOpTensorDescriptor_t desc);

bool needStrideProcess(const mluOpTensorDescriptor_t x_desc,
                       const mluOpTensorDescriptor_t y_desc);
/* user param check
 * step1:check desc and data ptr is not nullptr_t
 * step2:check shape and data type
 * */

void unaryOpPolicyFuncBlock(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                            cnrtFunctionType_t *k_type,
                            mluOpTensorDescriptor_t desc);

// Divide tasks in host, use with UNARY_OP_KERNEL_3PIPELINE_V2
void unaryOpPolicyFuncBlock_v2(mluOpHandle_t handle,
                               const mluOpTensorDescriptor_t desc,
                               size_t single_core_min_load_size,
                               cnrtDim3_t &k_dim, cnrtFunctionType_t &k_type,
                               size_t &normal_core_elem_num,
                               size_t &tail_core_elem_num);

/* user param check
 * step1:check desc and data ptr is not nullptr_t
 * step2:check shape and data type
 * */

mluOpStatus_t unaryOpParamCheck(std::string op_name, const mluOpHandle_t handle,
                                const mluOpTensorDescriptor_t x_desc,
                                const void *x,
                                const mluOpTensorDescriptor_t y_desc,
                                const void *y,
                                const mluOpDataType_t support_type[],
                                const int type_len, bool &zero_element);
#endif  // KERNELS_UNARY_OP_UNARY_OP_HOST_H_
