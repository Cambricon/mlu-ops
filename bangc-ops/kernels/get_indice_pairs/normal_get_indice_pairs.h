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

#ifndef KERNELS_GET_INDICE_PAIRS_NORMAL_GET_INDICE_PAIRS_H_
#define KERNELS_GET_INDICE_PAIRS_NORMAL_GET_INDICE_PAIRS_H_

#include <string>

#include "mlu_op.h"
#include "kernels/get_indice_pairs/get_indice_pairs_structs.h"

mluOpStatus_t getNormalGetIndicePairsWorkspaceSize(
    mluOpHandle_t handle, const std::string interface_name,
    const mluOpSparseConvolutionDescriptor_t sparse_conv_desc,
    const mluOpTensorDescriptor_t indices_desc,
    const mluOpTensorDescriptor_t indice_pairs_desc,
    const mluOpTensorDescriptor_t out_indices_desc,
    const mluOpTensorDescriptor_t indice_num_desc, size_t *return_ws);

mluOpStatus_t normalGetIndicePairs(
    mluOpHandle_t handle, const std::string interface_name,
    const mluOpSparseConvolutionDescriptor_t sparse_conv_desc,
    const mluOpTensorDescriptor_t indices_desc, const void *indices,
    void *workspace, size_t workspace_size,
    const mluOpTensorDescriptor_t indice_pairs_desc, void *indice_pairs,
    const mluOpTensorDescriptor_t out_indices_desc, void *out_indices,
    const mluOpTensorDescriptor_t indice_num_desc, void *indice_num,
    const bool is_get_workspace, size_t *return_ws);
#endif  // KERNELS_GET_INDICE_PAIRS_NORMAL_GET_INDICE_PAIRS_H_
