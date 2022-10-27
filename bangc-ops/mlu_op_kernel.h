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

/* generate_proposals_v2 */
void MLUOP_WIN_API mluOpUBestKernelGenerateProposalsV2Float(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const float *scores, const float *bbox_deltas, const float *im_shape,
    const float *anchors, const float *variances, float *workspace,
    float *rpn_rois, float *rpn_roi_probs, int *rpn_rois_num,
    int *rpn_rois_batch_size, const int pre_nms_top_n, const int post_nms_top_n,
    const float nms_thresh, const float min_size, const float eta,
    bool pixel_offset, const int batch_size, const int Anchors_num, const int H,
    const int W);

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

void MLUOP_WIN_API mluOpBlockKernelPsRoiPoolBackwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *top_grad, const void *mapping_channel, const void *rois,
    void *bottom_grad, const int batch_size, const int height, const int width,
    const int channels, const int pooled_height, const int pooled_width,
    const int output_dim, const int rois_num, const int rois_offset,
    const float spatial_scale);

/*PriorBox*/
void MLUOP_WIN_API mluOpBlockKernelPriorBoxFloat(
    cnrtDim3_t k_dim_box, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *min_sizes, const int min_sizes_num, const void *aspect_ratios,
    const int aspect_ratios_num, const void *variances, const int variances_num,
    const void *max_sizes, const int max_sizes_num, const int height,
    const int width, const int im_height, const int im_width,
    const float step_h, const float step_w, const float offset,
    const int num_priors, const bool clip,
    const bool min_max_aspect_ratios_order, void *output, const int output_size,
    void *var, const int var_size);

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

/* yolo_box */
void MLUOP_WIN_API mluOpBlockKernelYoloBoxFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *x, const void *img_size, const void *anchors,
    const int class_num, const float conf_thresh, const int downsample_ratio,
    const bool clip_bbox, const float scale, const bool iou_aware,
    const float iou_aware_factor, const int n_in, const int anchor_s,
    const int c_in, const int h_in, const int w_in, void *boxes, void *scores);

/* ThreeInterpolateForward*/
void MLUOP_WIN_API mluOpUnionKernelThreeInterpolateForwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *features, const void *indices, const void *weights, const int b,
    const int c, const int m, const int n, const int c_limit_size,
    const int m_limit_size, const int n_limit_size, void *output);
void MLUOP_WIN_API mluOpUnionKernelThreeInterpolateForwardHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *features, const void *indices, const void *weights, const int b,
    const int c, const int m, const int n, const int c_limit_size,
    const int m_limit_size, const int n_limit_size, void *output);

/* Expand */
void MLUOP_WIN_API mluOpUnion1KernelExpandTensor(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, void *output, const int input_1, const int input_2,
    const int input_3, const int input_4, const int input_5, const int input_6,
    const int input_7, const int input_8, const int output_1,
    const int output_2, const int output_3, const int output_4,
    const int output_5, const int output_6, const int output_7,
    const int output_8, const int dtype_size);

void MLUOP_WIN_API mluOpUnion1KernelExpandOneDim(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, void *output, const int high_num, const int expand_num,
    const int low_num, const int dtype_size);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // MLU_OP_KERNEL_H_
