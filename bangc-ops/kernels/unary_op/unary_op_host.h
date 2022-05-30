/*************************************************************************
 * Copyright (C) 2021 by Cambricon, Inc. All rights reserved.
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
#include "core/mlu_op_core.h"

void unaryOpPolicyFunc(const mluOpHandle_t &handle,
                       const mluOpTensorDescriptor_t &desc, cnrtDim3_t *k_dim,
                       cnrtFunctionType_t *k_type);

/* user param check
 * step1:check desc and data ptr is not nullptr_t
 * step2:check shape and data type
 * */
mluOpStatus_t unaryOpParamCheck(const std::string &op_name,
                                const mluOpHandle_t &handle,
                                const mluOpTensorDescriptor_t &x_desc,
                                const void *x,
                                const mluOpTensorDescriptor_t &y_desc,
                                const void *y,
                                const mluOpDataType_t support_type[],
                                const int &type_len, bool &zero_element);
#endif  // KERNELS_UNARY_OP_UNARY_OP_HOST_H_
