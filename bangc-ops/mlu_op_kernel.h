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
#ifndef MLU_OP_KERNEL_H_
#define MLU_OP_KERNEL_H_

#include <stdint.h>
#include "cnrt.h"

#ifndef MLUOP_WIN_API
#ifdef _WIN32
#define MLUOP_WIN_API __stdcall
#else
#define MLUOP_WIN_API
#endif  // _WIN32
#endif  // MLUOP_WIN_API

#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus

/* Abs */
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineAbsHalfFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineAbsFloatFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);

void MLUOP_WIN_API mluOpBlockKernel5StagePipelineAbsHalfFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);
void MLUOP_WIN_API mluOpBlockKernel5StagePipelineAbsFloatFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);

/* Div */
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineDivHalfFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, const void *y, void *z, int num);
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineDivHalfHighAcc(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, const void *y, void *z, int num);
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineDivFloatFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, const void *y, void *z, int num);

/* FillZero */
void MLUOP_WIN_API mluOpBlockKernelFillZeroByte(cnrtDim3_t k_dim,
                                                cnrtFunctionType_t k_type,
                                                cnrtQueue_t queue,
                                                const int num_byte, void *x);

/* Log */
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineLogHalfFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num, float coef);
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineLogHalfHighAcc(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num, float coef);
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineLogFloatFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num, float coef);

void MLUOP_WIN_API mluOpBlockKernel5StagePipelineLogHalfFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num, float coef);
void MLUOP_WIN_API mluOpBlockKernel5StagePipelineLogHalfHighAcc(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num, float coef);
void MLUOP_WIN_API mluOpBlockKernel5StagePipelineLogFloatFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num, float coef);

/* poly_nms */
void MLUOP_WIN_API mluOpBlockKernelPolyNmsCalcAreaFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const float *boxes, const int box_num, const int real_width,
    float *dev_area);

void MLUOP_WIN_API mluOpBlockKernelPolyNmsGenMaskFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const float *boxes, const int box_num, const int real_width,
    const float iou_threshold, float *dev_area, uint32_t *dev_mask,
    int *dev_sort_info);

void MLUOP_WIN_API mluOpBlockKernelPolyNmsGenResultFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const int box_num, uint32_t *dev_mask, int *dev_sort_info, int *output,
    int *output_size);

/* PSRoIPool */
void MLUOP_WIN_API mluOpBlockKernelPsRoiPoolForwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *bottom_data, const void *bottom_rois, void *top_data,
    void *mapping_channel, const int batch_size, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const int output_dim, const int group_size,
    const int rois_num, const int rois_offset, const float spatial_scale);
    
/* RoICrop*/
void MLUOP_WIN_API mluOpBlockKernelRoiCropForwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, const void *grid, const int batch, const int height,
    const int width, const int channels, const int grid_n, const int output_h,
    const int output_w, void *output);
void MLUOP_WIN_API mluOpBlockKernelRoiCropBackwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *grad_output, const void *grid, const int batch,
    const int height, const int width, const int channels, const int grid_n,
    const int output_h, const int output_w, void *grad_input);

/* Sqrt */
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineSqrtHalfFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineSqrtHalfHighAcc(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineSqrtFloatFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);

void MLUOP_WIN_API mluOpBlockKernel5StagePipelineSqrtHalfFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);
void MLUOP_WIN_API mluOpBlockKernel5StagePipelineSqrtHalfHighAcc(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);
void MLUOP_WIN_API mluOpBlockKernel5StagePipelineSqrtFloatFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, void *y, int num);

/* SqrtBackward */
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineSqrtBackwardHalfHighAcc(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *y, const void *diff_y, void *x, int num);
void MLUOP_WIN_API mluOpBlockKernel3StagePipelineSqrtBackwardFloatFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *y, const void *diff_y, void *x, int num);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // MLU_OP_KERNEL_H_
