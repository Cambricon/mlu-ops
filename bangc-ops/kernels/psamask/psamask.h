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
#ifndef KERNELS_PSAMASK_PSAMASK_H_
#define KERNELS_PSAMASK_PSAMASK_H_

#include "core/type.h"
#include "kernels/kernel.h"

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
#endif  // KERNELS_PSAMASK_PSAMASK_H_
