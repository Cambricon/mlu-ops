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
#ifndef KERNELS_BINARY_OP_BINARY_OP_HOST_H_
#define KERNELS_BINARY_OP_BINARY_OP_HOST_H_

#include <string>

#include "mlu_op.h"

void binaryOpPolicyFunc(const mluOpHandle_t &handle,
                        const mluOpTensorDescriptor_t &desc,
                        const int &align_param, cnrtDim3_t *k_dim,
                        cnrtFunctionType_t *k_type);

/* user param check
 * step1:check desc and data ptr is not nullptr_t
 * step2:check shape and data type
 * */
mluOpStatus_t binaryOpParamCheck(
    const std::string &op_name, const mluOpHandle_t &handle,
    const mluOpTensorDescriptor_t &input1_desc, const void *input1,
    const mluOpTensorDescriptor_t &input2_desc, const void *input2,
    const mluOpTensorDescriptor_t &output_desc, const void *output,
    const mluOpDataType_t support_type[], const int &len, bool &zero_element);
#endif  //  KERNELS_BINARY_OP_BINARY_OP_HOST_H_
