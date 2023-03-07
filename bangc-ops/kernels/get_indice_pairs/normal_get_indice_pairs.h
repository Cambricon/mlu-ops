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

void MLUOP_WIN_API mluOpBlockDefaultGetIndicePairKernel1(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    void *mask_all_ws, void *indice_index_in_ws, void *indice_out_expand_ws,
    void *indices, FilterSpace filter_space, InputSpace input_space,
    OutputSpace output_space, Stride stride, Dilation dilation,
    Padding padding, int32_t core_num_l, int32_t input_active_site,
    int32_t batch_size);

void MLUOP_WIN_API mluOpBlockDefaultGetIndicePairKernel2(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    void *step_index_ptr, int32_t num_act_out, int32_t core_num_l);

void MLUOP_WIN_API mluOpBlockDefaultGetIndicePairKernel3(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    void *indice_pairs, void *input_addr, void *mask_addr,
    int32_t input_active_site, int32_t kernel_volume, int32_t core_num_l);

void MLUOP_WIN_API mluOpBlockDefaultGetIndicePairKernel4(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    void *out_indices, void *input_addr, OutputSpace host_output_space,
    int32_t len_l, int32_t core_num_l);

void MLUOP_WIN_API mluOpBlockBalanceGetIndicePairKernel(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    void *balance_input, void *balance_mask, void *balance_output,
    int32_t len_l, int32_t kernel_volume, int32_t core_num_l,
    int32_t output_size);

void MLUOP_WIN_API mluOpBlockSubmGetIndicePairKernel1(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    void *mask_all_ptr, void *indice_index_in_ptr, void *indice_in_expand_ptr,
    void *out_indices_expand_ptr, void *indices, FilterSpace host_filter_space,
    InputSpace host_input_space, OutputSpace host_output_space,
    Stride host_stride, Dilation host_dilation, Padding host_padding,
    int32_t core_num_l, int32_t input_active_site, int32_t batch_size);

void MLUOP_WIN_API mluOpBlockSubmGetIndicePairKernel2(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    void *out_indices, void *mask_all_ptr, void *out_indices_expand_ptr,
    void *indices, int32_t len_1_one, int32_t len_l_two,
    int32_t core_num_l_one, int32_t core_num_l_two);

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
