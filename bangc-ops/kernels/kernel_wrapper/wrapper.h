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
#ifndef KERNELS_KERNEL_WRAPPER_WRAPPER_H
#define KERNELS_KERNEL_WRAPPER_WRAPPER_H

#include <algorithm>
#include <string>

#include "mlu_op.h"
#include "export_statement.h"

#define KERNEL_REGISTER(OP_NAME, PARAMS, ...)                                 \
    class OP_NAME##Wrapper {                                                  \
     public:                                                                  \
      OP_NAME##Wrapper() {}                                                   \
      ~OP_NAME##Wrapper() {}                                                  \
      mluOpStatus_t invoke(PARAMS);                                           \
      std::string op_name = #OP_NAME;                                         \
    };

/* Kernel param types macro defination */

#define ADDN_PARAM_TYPE                                                       \
    mluOpHandle_t, const mluOpTensorDescriptor_t[], const void *const *,      \
    uint32_t, const mluOpTensorDescriptor_t, void *

#define ADDNV2_PARAM_TYPE                                                     \
    mluOpHandle_t, const mluOpTensorDescriptor_t[], const void *const *,      \
    uint32_t, const mluOpTensorDescriptor_t, void *, void *, size_t

#define COPY_PARAM_TYPE                                                       \
    mluOpHandle_t, const mluOpTensorDescriptor_t, const void *,               \
    const mluOpTensorDescriptor_t, void *

#define EXPAND_PARAM_TYPE                                                     \
    mluOpHandle_t, const mluOpTensorDescriptor_t, const void *,               \
    const mluOpTensorDescriptor_t, void *

#define FILL_PARAM_TYPE                                                       \
    mluOpHandle_t, float, const mluOpTensorDescriptor_t, void *

#define FILL_V2_PARAM_TYPE                                                    \
    mluOpHandle_t, const mluOpTensorDescriptor_t, const void *,               \
    const mluOpTensorDescriptor_t, void *

#define FILL_V3_PARAM_TYPE                                                    \
    mluOpHandle_t, const mluOpPointerMode_t, const void *,                    \
    const mluOpTensorDescriptor_t, void *

#define MATMUL_PARAM_TYPE                                                     \
    mluOpHandle_t, const bool, const bool, const void *,                      \
    const mluOpTensorDescriptor_t, const void *,                              \
    const mluOpTensorDescriptor_t, const void *, const void *,                \
    const mluOpTensorDescriptor_t, void *

#define MATMUL_V2_PARAM_TYPE                                                  \
    mluOpHandle_t, mluOpMatMulDescriptor_t, mluOpMatMulAlgo_t, const void *,  \
    const mluOpTensorDescriptor_t, const void *,                              \
    const mluOpTensorDescriptor_t, const void *,                              \
    const void *, const mluOpTensorDescriptor_t, void *, void *, size_t,      \
    const mluOpTensorDescriptor_t, void *

#define UNIQUE_PARAM_TYPE                                                     \
    mluOpHandle_t, const mluOpUniqueDescriptor_t,                             \
    const mluOpTensorDescriptor_t, const void *, const int, void *,           \
    void *, int *, int *

#define UNIQUE_V2_PARAM_TYPE                                                  \
    mluOpHandle_t, const mluOpUniqueDescriptor_t,                             \
    const mluOpTensorDescriptor_t, const void *, void *, const size_t, int *, \
    const mluOpTensorDescriptor_t, void *, const mluOpTensorDescriptor_t,     \
    void *, const mluOpTensorDescriptor_t, void *

#define NMS_PARAM_TYPE                                                        \
    mluOpHandle_t, const mluOpNmsDescriptor_t,                                \
    const mluOpTensorDescriptor_t, const void *,                              \
    const mluOpTensorDescriptor_t, const void *,                              \
    void *, size_t,                                                           \
    const mluOpTensorDescriptor_t, void *,                                    \
    void *

#define ROIALIGN_FORWARD_PARAM_TYPE                                           \
    mluOpHandle_t, const mluOpRoiAlignForwardDescriptor_t,                    \
    const mluOpTensorDescriptor_t, const void *,                              \
    const mluOpTensorDescriptor_t, const void *,                              \
    const mluOpTensorDescriptor_t, void *

#define ROIALIGN_FORWARD_V2_PARAM_TYPE                                        \
    mluOpHandle_t, const mluOpRoiAlignForwardDescriptor_t,                    \
    const mluOpTensorDescriptor_t, const void *,                              \
    const mluOpTensorDescriptor_t, const void *,                              \
    const mluOpTensorDescriptor_t, void *,                                    \
    const mluOpTensorDescriptor_t, void *,                                    \
    const mluOpTensorDescriptor_t, void *

#define SCATTER_ND_PARAM_TYPE                                                 \
    mluOpHandle_t, const mluOpTensorDescriptor_t, const void *,               \
    const mluOpTensorDescriptor_t, const void *,                              \
    const mluOpTensorDescriptor_t, void *

#define SCATTER_ND_V2_PARAM_TYPE                                              \
    mluOpHandle_t, mluOpScatterNdMode_t, const mluOpTensorDescriptor_t,       \
    const void *, const mluOpTensorDescriptor_t, const void *,                \
    const mluOpTensorDescriptor_t, const void *,                              \
    const mluOpTensorDescriptor_t, void *

#define GATHER_ND_PARAM_TYPE                                                  \
    mluOpHandle_t, const mluOpTensorDescriptor_t, const void *,               \
    const mluOpTensorDescriptor_t, const void *,                              \
    const mluOpTensorDescriptor_t, void *

#define TRANSPOSE_PARAM_TYPE                                                  \
    mluOpHandle_t, const mluOpTransposeDescriptor_t,                          \
    const mluOpTensorDescriptor_t, const void *,                              \
    const mluOpTensorDescriptor_t, void *y

#define TRANSPOSE_V2_PARAM_TYPE                                               \
    mluOpHandle_t, const mluOpTransposeDescriptor_t,                          \
    const mluOpTensorDescriptor_t, const void *,                              \
    const mluOpTensorDescriptor_t, void *, void *, size_t

#define REDUCE_PARAM_TYPE                                                     \
    mluOpHandle_t, const mluOpReduceDescriptor_t, void *, size_t,             \
    const void *, const mluOpTensorDescriptor_t, const void *,                \
    const size_t, void *, const void *, const mluOpTensorDescriptor_t,        \
    void *

#define ROIALIGNBACKWARD_PARAM_TYPE                                           \
    mluOpHandle_t,                                                            \
    const float,                                                              \
    const int,                                                                \
    const bool,                                                               \
    const mluOpTensorDescriptor_t,                                            \
    const void *,                                                             \
    const mluOpTensorDescriptor_t,                                            \
    const void *,                                                             \
    const mluOpTensorDescriptor_t,                                            \
    void *

#define ROIALIGNBACKWARD_V2_PARAM_TYPE                                        \
    mluOpHandle_t,                                                            \
    const mluOpTensorDescriptor_t,                                            \
    const void *,                                                             \
    const mluOpTensorDescriptor_t,                                            \
    const void *,                                                             \
    const mluOpTensorDescriptor_t,                                            \
    const void *,                                                             \
    const mluOpTensorDescriptor_t,                                            \
    const void *,                                                             \
    const float,                                                              \
    const int,                                                                \
    const bool,                                                               \
    const int,                                                                \
    const mluOpTensorDescriptor_t,                                            \
    void *

#define ROIPOOLINGFORWARD_PARAM_TYPE                                          \
    mluOpHandle_t,                                                            \
    mluOpPoolingMode_t,                                                       \
    const mluOpTensorDescriptor_t,                                            \
    const void *,                                                             \
    const mluOpTensorDescriptor_t,                                            \
    const void *,                                                             \
    float,                                                                    \
    const mluOpTensorDescriptor_t,                                            \
    void *,                                                                   \
    int *

#define ROIPOOLINGBACKWARD_PARAM_TYPE                                         \
    mluOpHandle_t,                                                            \
    mluOpPoolingMode_t,                                                       \
    const mluOpTensorDescriptor_t,                                            \
    const void *,                                                             \
    const mluOpTensorDescriptor_t,                                            \
    const void *,                                                             \
    const mluOpTensorDescriptor_t,                                            \
    const int *,                                                              \
    const float,                                                              \
    const mluOpTensorDescriptor_t,                                            \
    void *

#define TRANSFORM_PARAM_TYPE                                                  \
    mluOpHandle_t,                                                            \
    const mluOpPointerMode_t,                                                 \
    const void *,                                                             \
    const mluOpTensorDescriptor_t,                                            \
    const void *,                                                             \
    const void *,                                                             \
    const mluOpTensorDescriptor_t,                                            \
    void *

#define SYNCBATCHNORMSTATS_PARAM_TYPE                                       \
  mluOpHandle_t, const mluOpTensorDescriptor_t, const void *, const float,  \
      const mluOpTensorDescriptor_t, void *, const mluOpTensorDescriptor_t, \
      void *

#define SYNCBATCHNORMSTATS_V2_PARAM_TYPE                                      \
  mluOpHandle_t, const mluOpTensorDescriptor_t, const void *, void *, size_t, \
      const float, const mluOpTensorDescriptor_t, void *,                     \
      const mluOpTensorDescriptor_t, void *

#define SYNCBATCHNORMGATHERSTATSWITHCOUNTS_PARAM_TYPE                       \
  mluOpHandle_t, const mluOpTensorDescriptor_t, const void *,               \
      const mluOpTensorDescriptor_t, const void *,                          \
      const mluOpTensorDescriptor_t, void *, const mluOpTensorDescriptor_t, \
      void *, float, float, const mluOpTensorDescriptor_t, const void *,    \
      const mluOpTensorDescriptor_t, void *, const mluOpTensorDescriptor_t, \
      void *

#define SYNCBATCHNORMELEMT_PARAM_TYPE                         \
  mluOpHandle_t, const mluOpTensorDescriptor_t, const void *, \
      const mluOpTensorDescriptor_t, const void *,            \
      const mluOpTensorDescriptor_t, const void *,            \
      const mluOpTensorDescriptor_t, const void *,            \
      const mluOpTensorDescriptor_t, const void *,            \
      const mluOpTensorDescriptor_t, void *

#define SYNCBATCHNORMBACKWADREDUCE_PARAM_TYPE                               \
  mluOpHandle_t, const mluOpTensorDescriptor_t, const void *,               \
      const mluOpTensorDescriptor_t, const void *,                          \
      const mluOpTensorDescriptor_t, const void *,                          \
      const mluOpTensorDescriptor_t, const void *,                          \
      const mluOpTensorDescriptor_t, void *, const mluOpTensorDescriptor_t, \
      void *, const mluOpTensorDescriptor_t, void *,                        \
      const mluOpTensorDescriptor_t, void *, const bool, const bool,        \
      const bool

#define SYNCBATCHNORMBACKWADREDUCE_V2_PARAM_TYPE                            \
  mluOpHandle_t, const mluOpTensorDescriptor_t, const void *,               \
      const mluOpTensorDescriptor_t, const void *,                          \
      const mluOpTensorDescriptor_t, const void *,                          \
      const mluOpTensorDescriptor_t, const void *, void *, size_t,          \
      const mluOpTensorDescriptor_t, void *, const mluOpTensorDescriptor_t, \
      void *, const mluOpTensorDescriptor_t, void *,                        \
      const mluOpTensorDescriptor_t, void *, const bool, const bool,        \
      const bool

#define SYNCBATCHNORMBACKWARDELEMT_PARAM_TYPE                 \
  mluOpHandle_t, const mluOpTensorDescriptor_t, const void *, \
      const mluOpTensorDescriptor_t, const void *,            \
      const mluOpTensorDescriptor_t, const void *,            \
      const mluOpTensorDescriptor_t, const void *,            \
      const mluOpTensorDescriptor_t, const void *,            \
      const mluOpTensorDescriptor_t, const void *,            \
      const mluOpTensorDescriptor_t, const void *,            \
      const mluOpTensorDescriptor_t, void *

#define SYNCBATCHNORMBACKWARDELEMT_V2_PARAM_TYPE                 \
  mluOpHandle_t, const mluOpTensorDescriptor_t, const void *,    \
    const mluOpTensorDescriptor_t,                               \
      const void *, const mluOpTensorDescriptor_t, const void *, \
      const mluOpTensorDescriptor_t, const void *,               \
      const mluOpTensorDescriptor_t, const void *,               \
      const mluOpTensorDescriptor_t, const void *,               \
      const mluOpTensorDescriptor_t, const void *,               \
      const mluOpTensorDescriptor_t, const void *,               \
      const mluOpTensorDescriptor_t, void *diff_x

#define TRANSFORM_PARAM_TYPE                                                  \
    mluOpHandle_t,                                                            \
    const mluOpPointerMode_t,                                                 \
    const void *,                                                             \
    const mluOpTensorDescriptor_t,                                            \
    const void *,                                                             \
    const void *,                                                             \
    const mluOpTensorDescriptor_t,                                            \
    void *

#define STRIDEDSLICE_PARAM_TYPE                               \
  mluOpHandle_t, const mluOpTensorDescriptor_t, const void *, \
      const int *, const int *, const int *,      \
      const mluOpTensorDescriptor_t, void *

/* Kernel register */
KERNEL_REGISTER(addN, ADDN_PARAM_TYPE);
KERNEL_REGISTER(addNV2, ADDNV2_PARAM_TYPE);
KERNEL_REGISTER(copy, COPY_PARAM_TYPE);
KERNEL_REGISTER(expand, EXPAND_PARAM_TYPE);
KERNEL_REGISTER(fill, FILL_PARAM_TYPE);
KERNEL_REGISTER(fillV2, FILL_V2_PARAM_TYPE);
KERNEL_REGISTER(fillV3, FILL_V3_PARAM_TYPE);
KERNEL_REGISTER(matmul, MATMUL_PARAM_TYPE);
KERNEL_REGISTER(matmulV2, MATMUL_V2_PARAM_TYPE);
KERNEL_REGISTER(unique, UNIQUE_PARAM_TYPE);
KERNEL_REGISTER(uniqueV2, UNIQUE_V2_PARAM_TYPE);
KERNEL_REGISTER(roiAlignForward, ROIALIGN_FORWARD_PARAM_TYPE);
KERNEL_REGISTER(roiAlignForwardV2, ROIALIGN_FORWARD_V2_PARAM_TYPE);
KERNEL_REGISTER(scatterNd, SCATTER_ND_PARAM_TYPE);
KERNEL_REGISTER(scatterNdV2, SCATTER_ND_V2_PARAM_TYPE);
KERNEL_REGISTER(gatherNd, GATHER_ND_PARAM_TYPE);
KERNEL_REGISTER(transpose, TRANSPOSE_PARAM_TYPE);
KERNEL_REGISTER(transposeV2, TRANSPOSE_V2_PARAM_TYPE);
KERNEL_REGISTER(nms, NMS_PARAM_TYPE);
KERNEL_REGISTER(reduce, REDUCE_PARAM_TYPE);
KERNEL_REGISTER(RoiAlignBackward, ROIALIGNBACKWARD_PARAM_TYPE);
KERNEL_REGISTER(RoiAlignBackwardV2, ROIALIGNBACKWARD_V2_PARAM_TYPE);
KERNEL_REGISTER(RoiPoolingForward, ROIPOOLINGFORWARD_PARAM_TYPE);
KERNEL_REGISTER(RoiPoolingBackward, ROIPOOLINGBACKWARD_PARAM_TYPE);
KERNEL_REGISTER(transform, TRANSFORM_PARAM_TYPE);
KERNEL_REGISTER(SyncBatchNormStats, SYNCBATCHNORMSTATS_PARAM_TYPE);
KERNEL_REGISTER(SyncBatchNormStatsV2, SYNCBATCHNORMSTATS_V2_PARAM_TYPE);
KERNEL_REGISTER(SyncBatchNormGatherStatsWithCounts,
                SYNCBATCHNORMGATHERSTATSWITHCOUNTS_PARAM_TYPE);
KERNEL_REGISTER(SyncBatchNormElemt, SYNCBATCHNORMELEMT_PARAM_TYPE);
KERNEL_REGISTER(SyncBatchnormBackwardReduce,
        SYNCBATCHNORMBACKWADREDUCE_PARAM_TYPE);
KERNEL_REGISTER(SyncBatchnormBackwardReduceV2,
        SYNCBATCHNORMBACKWADREDUCE_V2_PARAM_TYPE);
KERNEL_REGISTER(SyncBatchNormBackwardElemt,
        SYNCBATCHNORMBACKWARDELEMT_PARAM_TYPE);
KERNEL_REGISTER(SyncBatchNormBackwardElemtV2,
        SYNCBATCHNORMBACKWARDELEMT_V2_PARAM_TYPE);
KERNEL_REGISTER(transform, TRANSFORM_PARAM_TYPE);
KERNEL_REGISTER(StridedSlice, STRIDEDSLICE_PARAM_TYPE);
#endif  // KERNELS_KERNEL_WRAPPER_WRAPPER_H
