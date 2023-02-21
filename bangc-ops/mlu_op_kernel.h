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

/* RoiAlignRotated */
struct mluOpRoiAlignRotatedParams {
  int pooled_height;
  int pooled_width;
  int sample_ratio;
  float spatial_scale;
  bool aligned;
  bool clockwise;
};

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

/* Carafe */
void MLUOP_WIN_API mluOpBlockKernelCarafeForwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, const void *mask, void *output,
    int input_dimN,    // number of input samples
    int input_dimH,    // input feature map height
    int input_dimW,    // input feature map width
    int input_dimC,    // number of input channels
    int kernel_size,   // upsampling kernel side size
    int group_size,    // number of channel groups
    int scale_factor,  // upscaling factor
    int block_dimH,    // block height
    int block_dimW,    // block width
    int block_dimG,    // block of channel groups
    int block_dimC,    // block of channels in each group
    int grid_dimH,     // nb. of blocks along height
    int grid_dimW,     // nb. of blocks along width
    int grid_dimG,     // nb. of group blocks
    int grid_dimC);     // nb. of channel blocks

void MLUOP_WIN_API mluOpBlockKernelCarafeForwardHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, const void *mask, void *output,
    int input_dimN,    // number of input samples
    int input_dimH,    // input feature map height
    int input_dimW,    // input feature map width
    int input_dimC,    // number of input channels
    int kernel_size,   // upsampling kernel side size
    int group_size,    // number of channel groups
    int scale_factor,  // upscaling factor
    int block_dimH,    // block height
    int block_dimW,    // block width
    int block_dimG,    // block of channel groups
    int block_dimC,    // block of channels in each group
    int grid_dimH,     // nb. of blocks along height
    int grid_dimW,     // nb. of blocks along width
    int grid_dimG,     // nb. of group blocks
    int grid_dimC);     // nb. of channel blocks

void MLUOP_WIN_API mluOpBlockKernelCarafeBackwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    void *input, void *mask,
    void *grad_output, void *grad_input,
    void *grad_mask, int n, int hi, int wi,
    int c, int k_up, int group, int scale);

void MLUOP_WIN_API mluOpBlockKernelCarafeBackwardHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    void *input, void *mask,
    void *grad_output, void *grad_input,
    void *grad_mask, int n, int hi, int wi,
    int c, int k_up, int group, int scale);

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
    const bool pixel_offset, const int batch_size, const int Anchors_num,
    const int H, const int W);

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

/* RoiawarePool3d */
void MLUOP_WIN_API mluOpUnionKernelPtsIdxOfVoxelsHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const int pool_method, const int boxes_num, const int pts_num,
    const int max_pts_each_voxel, const int out_x,
    const int out_y, const int out_z, const void *rois, const void *pts,
    void *pts_idx_of_voxels);

void MLUOP_WIN_API mluOpUnionKernelPtsIdxOfVoxelsFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const int pool_method, const int boxes_num, const int pts_num,
    const int max_pts_each_voxel, const int out_x,
    const int out_y, const int out_z, const void *rois, const void *pts,
    void *pts_idx_of_voxels);

void MLUOP_WIN_API mluOpUnionKernelRoiawarePool3dForwardHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const int pool_method, const int boxes_num, const int pts_num,
    const int channels, const int max_pts_each_voxel, const int out_x,
    const int out_y, const int out_z, const void *pts_feature,
    const void *pts_idx_of_voxels, void *pooled_features, void *argmax);

void MLUOP_WIN_API mluOpUnionKernelRoiawarePool3dForwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const int pool_method, const int boxes_num, const int pts_num,
    const int channels, const int max_pts_each_voxel, const int out_x,
    const int out_y, const int out_z, const void *pts_feature,
    const void *pts_idx_of_voxels, void *pooled_features, void *argmax);

void MLUOP_WIN_API mluOpUnionKernelRoiawarePool3dBackwardHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const int pool_method, const int boxes_num, const int out_x,
    const int out_y, const int out_z, const int channels,
    const int max_pts_each_voxel, const void *pts_idx_of_voxels,
    const void *argmax, const void *grad_out, void *grad_in);

void MLUOP_WIN_API mluOpUnionKernelRoiawarePool3dBackwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const int pool_method, const int boxes_num, const int out_x,
    const int out_y, const int out_z, const int channels,
    const int max_pts_each_voxel, const void *pts_idx_of_voxels,
    const void *argmax, const void *grad_out, void *grad_in);

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

/* VoxelPooling */
void MLUOP_WIN_API mluOpUnionKernelVoxelPoolingForwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const int batch_size, const int num_points, const int num_channels,
    const int num_voxel_x, const int num_voxel_y, const int num_voxel_z,
    const void *geom_xyz, const void *input_features, void *output_features,
    void *pos_memo);

/* BoxIouRotated */
void MLUOP_WIN_API mluOpUnionKernelBoxIouRotatedFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *box1, const void *box2, void *ious, const int num_box1,
    const int num_box2, const int mode, const bool aligned);

/* BboxOverlaps */
void MLUOP_WIN_API mluOpUnion1BboxOverlapsKernelFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *bbox1, const void *bbox2, void *ious, const int32_t num_bboxl,
    const int32_t num_bbox2, const int32_t mode, const bool aligned,
    const int32_t offset);
void MLUOP_WIN_API mluOpUnion1BboxOverlapsKernelHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *bbox1, const void *bbox2, void *ious, const int32_t num_bboxl,
    const int32_t num_bbox2, const int32_t mode, const bool aligned,
    const int32_t offset);

/* RoiAlignRotated */
void MLUOP_WIN_API mluOpBlockKernelRoiAlignRotatedForwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *features, const void *rois, const int batch, const int height,
    const int width, const int channel, const int rois_num,
    const mluOpRoiAlignRotatedParams rroiAlignParams, void *output);
void MLUOP_WIN_API mluOpBlockKernelRoiAlignRotatedForwardHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *features, const void *rois, const int batch, const int height,
    const int width, const int channel, const int rois_num,
    const mluOpRoiAlignRotatedParams rroiAlignParams, void *output);

void MLUOP_WIN_API mluOpBlockKernelRoiAlignRotatedBackwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *top_grad, const void *rois, const int batch, const int height,
    const int width, const int channel, const int rois_num,
    const mluOpRoiAlignRotatedParams rroiAlignParams, void *bottom_grad);
void MLUOP_WIN_API mluOpBlockKernelRoiAlignRotatedBackwardHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *top_grad, const void *rois, const int batch, const int height,
    const int width, const int channel, const int rois_num,
    const mluOpRoiAlignRotatedParams rroiAlignParams, void *bottom_grad);

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

/* RotatedFeatureAlign */
void MLUOP_WIN_API mluOpBlockKernelRotatedFeatureAlignForwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, const void *bboxes, const int batches, const int height,
    const int width, const int channels, const int offset_rois,
    const float spatial_scale, const int points, void *output);
void MLUOP_WIN_API mluOpBlockKernelRotatedFeatureAlignForwardHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, const void *bboxes, const int batches, const int height,
    const int width, const int channels, const int offset_rois,
    const float spatial_scale, const int points, void *output);

void MLUOP_WIN_API mluOpBlockKernelRotatedFeatureAlignBackwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *top_output, const void *bboxes, const int batches,
    const int height, const int width, const int channels,
    const int offset_rois, const float spatial_scale, const int points,
    void *bottom_input);
void MLUOP_WIN_API mluOpBlockKernelRotatedFeatureAlignBackwardHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *top_output, const void *bboxes, const int batches,
    const int height, const int width, const int channels,
    const int offset_rois, const float spatial_scale, const int points,
    void *bottom_input);

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

/* ThreeInterpolate*/
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

void MLUOP_WIN_API mluOpUnionKernelThreeInterpolateBackwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *grad_output, const void *indices, const void *weights,
    const int b, const int c, const int m, const int n,
    const int c_limit_size, const int m_limit_size, const int n_limit_size,
    void *grad_features);

void MLUOP_WIN_API mluOpUnionKernelThreeInterpolateBackwardHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *grad_output, const void *indices, const void *weights,
    const int b, const int c, const int m, const int n,
    const int c_limit_size, const int m_limit_size, const int n_limit_size,
    void *grad_features);

/* Expand */
void MLUOP_WIN_API mluOpUnion1KernelExpandTensor(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, void *output, const uint32_t input_1,
    const uint32_t input_2, const uint32_t input_3, const uint32_t input_4,
    const uint32_t input_5, const uint32_t input_6, const uint32_t input_7,
    const uint32_t input_8, const uint32_t output_1, const uint32_t output_2,
    const uint32_t output_3, const uint32_t output_4, const uint32_t output_5,
    const uint32_t output_6, const uint32_t output_7, const uint32_t output_8,
    const int dtype_size);

void MLUOP_WIN_API mluOpUnion1KernelExpandOneDim(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, void *output, const uint32_t high_num,
    const uint32_t expand_num, const uint32_t low_num, const int dtype_size);

/* Psamask */
typedef enum {
  COLLECT = 0,
  DISTRIBUTE = 1,
} psamaskType_t;

typedef enum {
  PARTITION_N = 0,
  PARTITION_H = 1,
} dimPartitionType_t;

struct PartitionSeg {
  int h_per_cluster;
  int n_per_cluster;
  int h_per_core;
  int n_per_core;
  dimPartitionType_t cluster_partition;
  dimPartitionType_t core_partition;
};

struct Shape {
  int n;
  int h;
  int w;
  int c;
};

struct LimitParam {
  int n;
  int h;
  int w;
};

struct PositionInCore {
  int n_start;
  int n_end;
  int h_start;
  int h_end;
  int w_start;
  int w_end;
};

void MLUOP_WIN_API mluOpUnion1KernelPsamaskForwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const float *x, float *y, const psamaskType_t psa_type,
    const dimPartitionType_t core_partition,
    const dimPartitionType_t cluster_partition, const int batch,
    const int h_feature, const int w_feature, const int h_mask,
    const int w_mask, const int x_c, const int y_c, const int half_h_mask,
    const int half_w_mask, const int n_per_core, const int h_per_core,
    const int n_per_cluster, const int h_per_cluster, const int limit_n_seg,
    const int limit_h_seg, const int limit_w_seg);

void MLUOP_WIN_API mluOpUnion1KernelPsamaskBackwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const float *y, float *x, const psamaskType_t psa_type,
    const dimPartitionType_t core_partition,
    const dimPartitionType_t cluster_partition, const int batch,
    const int h_feature, const int w_feature, const int h_mask,
    const int w_mask, const int x_c, const int y_c, const int half_h_mask,
    const int half_w_mask, const int n_per_core, const int h_per_core,
    const int n_per_cluster, const int h_per_cluster, const int limit_n_seg,
    const int limit_h_seg, const int limit_w_seg);

/* voxelization */
void MLUOP_WIN_API mluOpUnionKernelDynamicVoxelize(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *points, const void *voxel_size, const void *coors_range,
    void *coors, const int32_t num_points, const int32_t num_features);

void MLUOP_WIN_API mluOpUnionKernelPoint2Voxel(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue, void *coors,
    void *point_to_pointidx, void *point_to_voxelidx, const int32_t num_points,
    const int32_t max_points);

void MLUOP_WIN_API mluOpUnionKernelCalcPointsPerVoxel(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    void *point_to_pointidx, void *point_to_voxelidx, void *coor_to_voxelidx,
    void *num_points_per_voxel, void *voxel_num, const int32_t max_voxels,
    const int32_t num_points);

void MLUOP_WIN_API mluOpUnionKernelAssignVoxelsCoors(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *points, void *temp_coors, void *point_to_voxelidx,
    void *coor_to_voxelidx, void *voxels, void *coors, const int32_t max_points,
    const int32_t num_points, const int32_t num_features);

/*ActiveRotatedFilter*/
void MLUOP_WIN_API mluOpUnionXKernelActiveRotatedFilterForwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const int output_planes, const int input_planes, const int orientations,
    const int kH, const int kW, const int rotations, const void *input_gdram,
    const void *indices_gdram, const void *workspace_gdram, void *output_gdram);

void MLUOP_WIN_API mluOpUnionXKernelActiveRotatedFilterForwardHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const int output_planes, const int input_planes, const int orientations,
    const int kH, const int kW, const int rotations, const void *input_gdram,
    const void *indices_gdram, const void *workspace_gdram, void *output_gdram);

void MLUOP_WIN_API MLUUnion1DeformRoiPoolForwardHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, const void *rois, const void *offset, void *output,
    const int batches, const int channels, const int height, const int width,
    const int num_rois, const int pooled_height, const int pooled_width,
    const float spatial_scale, const int sampling_ratio, const float gamma);

void MLUOP_WIN_API MLUUnion1DeformRoiPoolForwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *input, const void *rois, const void *offset, void *output,
    const int batches, const int channels, const int height, const int width,
    const int num_rois, const int pooled_height, const int pooled_width,
    const float spatial_scale, const int sampling_ratio, const float gamma);

void MLUOP_WIN_API MLUUnion1DeformRoiPoolBackwardHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *grad_output, const void *input, const void *rois,
    const void *offset, void *grad_input, void *grad_offset, const int batches,
    const int channels, const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width, const float spatial_scale,
    const int sampling_ratio, const float gamma);

void MLUOP_WIN_API MLUUnion1DeformRoiPoolBackwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const void *grad_output, const void *input, const void *rois,
    const void *offset, void *grad_input, void *grad_offset, const int batches,
    const int channels, const int height, const int width, const int num_rois,
    const int pooled_height, const int pooled_width, const float spatial_scale,
    const int sampling_ratio, const float gamma);

/* three_nn */
void MLUOP_WIN_API mluOpUnion1KernelThreeNNForwardFloat(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue, const int b,
    const int n, const int m, const void *unknown, const void *known,
    void *dist2, void *idx);

void MLUOP_WIN_API mluOpUnion1KernelThreeNNForwardHalf(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue, const int b,
    const int n, const int m, const void *unknown, const void *known,
    void *dist2, void *idx);

#if defined(__cplusplus)
}
#endif  // __cplusplus

#endif  // MLU_OP_KERNEL_H_
