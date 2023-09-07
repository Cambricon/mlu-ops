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
#ifndef KERNELS_PSAMASK_PSAMASK_H
#define KERNELS_PSAMASK_PSAMASK_H

#include "mlu_op.h"

constexpr int TRANSPOSE_ALIGN_BASE = 64;
constexpr int NRAMSET_ELEM_COUNT_ALIGN_BASE = 128;
constexpr int NRAMSET_DST_ALIGN_BASE = 64;

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

mluOpStatus_t MLUOP_WIN_API KernelPsamaskForward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const float *x, float *y, const psamaskType_t psa_type,
    const dimPartitionType_t core_partition,
    const dimPartitionType_t cluster_partition, const int batch,
    const int h_feature, const int w_feature, const int h_mask,
    const int w_mask, const int x_c, const int y_c, const int half_h_mask,
    const int half_w_mask, const int n_per_core, const int h_per_core,
    const int n_per_cluster, const int h_per_cluster, const int limit_n_seg,
    const int limit_h_seg, const int limit_w_seg);

mluOpStatus_t MLUOP_WIN_API KernelPsamaskBackward(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const float *y, float *x, const psamaskType_t psa_type,
    const dimPartitionType_t core_partition,
    const dimPartitionType_t cluster_partition, const int batch,
    const int h_feature, const int w_feature, const int h_mask,
    const int w_mask, const int x_c, const int y_c, const int half_h_mask,
    const int half_w_mask, const int n_per_core, const int h_per_core,
    const int n_per_cluster, const int h_per_cluster, const int limit_n_seg,
    const int limit_h_seg, const int limit_w_seg);

#endif  // KERNELS_PSAMASK_PSAMASK_H
