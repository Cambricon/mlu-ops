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

#include <new>
#include <string>

#include "core/logging.h"
#include "core/type.h"

#include "kernels/get_indice_pairs/get_indice_pairs_structs.h"
#include "mlu_op.h"

mluOpStatus_t MLUOP_WIN_API mluOpCreateSparseConvolutionDescriptor(
    mluOpSparseConvolutionDescriptor_t *desc) {
  if (desc == NULL) {
    LOG(ERROR) << "mluOpCreateSparseConvolutionDescriptor failed, "
               << "can't create desc when desc == NULL.";
    return MLUOP_STATUS_NOT_INITIALIZED;
  }
  mluOpSparseConvolutionStruct *ts =
      new (std::nothrow) mluOpSparseConvolutionStruct();
  *desc = ts;
  return MLUOP_STATUS_SUCCESS;
}

/* set sparse convolution descriptor.
 * pad_dim_num = input_dim_num - 2, and each dim need two pad value.
 */
mluOpStatus_t MLUOP_WIN_API mluOpSetSparseConvolutionDescriptor(
    mluOpSparseConvolutionDescriptor_t sparse_conv_desc, int dimNb,
    int batch, const int pad[], const int stride[], const int dilation[],
    const int input_space[], const int filter_space[], const int output_space[],
    const int sub_m, const int transpose, const int inverse) {
  std::string interface_name = "[mluOpSetSparseConvolutionDescriptor]";
  PARAM_CHECK(interface_name, sparse_conv_desc != NULL);
  PARAM_CHECK(interface_name, pad != NULL);
  PARAM_CHECK(interface_name, stride != NULL);
  PARAM_CHECK(interface_name, dilation != NULL);
  PARAM_CHECK(interface_name, input_space != NULL);
  PARAM_CHECK(interface_name, filter_space != NULL);
  PARAM_CHECK(interface_name, output_space != NULL);
  if (dimNb != 5) {
    LOG(ERROR) << interface_name << " only "
               << "support 3D_conv, dimnb should be 5. now dimNb is " << dimNb
               << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  sparse_conv_desc->dimNb = dimNb;

  if (batch <= 0) {
    LOG(ERROR) << interface_name << " only "
               << "support postive batch. now batch is " << batch
               << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  sparse_conv_desc->batch = batch;

  sparse_conv_desc->sub_m = sub_m;

  if (transpose != 0) {
    LOG(ERROR) << interface_name << " : not "
               << "support transpose . now transpose is " << transpose << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  sparse_conv_desc->transpose = transpose;

  if (inverse != 0) {
    LOG(ERROR) << interface_name << " : not "
               << "support inverse. now inverse is " << inverse << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  sparse_conv_desc->inverse = inverse;

  int kernel_dim = dimNb - 2;
  for (int idx = 0; idx < kernel_dim; idx++) {
    PARAM_CHECK_GE(interface_name, pad[idx], 0);
    sparse_conv_desc->pad[idx] = pad[idx];
    PARAM_CHECK_GE(interface_name, stride[idx], 1);
    sparse_conv_desc->stride[idx] = stride[idx];
    PARAM_CHECK_GE(interface_name, dilation[idx], 1);
    sparse_conv_desc->dilation[idx] = dilation[idx];
    PARAM_CHECK_GE(interface_name, input_space[idx], 1);
    sparse_conv_desc->input_space[idx] = input_space[idx];
    PARAM_CHECK_GE(interface_name, filter_space[idx], 1);
    sparse_conv_desc->filter_space[idx] = filter_space[idx];
    PARAM_CHECK_GE(interface_name, output_space[idx], 1);
    sparse_conv_desc->output_space[idx] = output_space[idx];
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetSparseConvolutionNumActOut(
    mluOpSparseConvolutionDescriptor_t desc,
    int *num_act_out) {
  if (desc == NULL || num_act_out == NULL) {
    LOG(ERROR) << "mluOpCreateSparseConvolutionDescriptor or "
               << "num_act_out failed "
               << " Passing NULL ptr to this API.";
    return MLUOP_STATUS_NOT_INITIALIZED;
  }
  int size = 0;
  size = desc->num_act_out;
  num_act_out[0] =  size;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpDestroySparseConvolutionDescriptor(
    mluOpSparseConvolutionDescriptor_t desc) {
  if (desc == NULL) {
    LOG(ERROR) << "mluOpDestroySparseConvolutionDescriptor fail. Passing NULL "
                  "ptr to this API.";
    return MLUOP_STATUS_EXECUTION_FAILED;
  }
  delete desc;
  return MLUOP_STATUS_SUCCESS;
}
