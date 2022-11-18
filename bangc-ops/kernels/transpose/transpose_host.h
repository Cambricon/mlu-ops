/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
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
#ifndef KERNELS_TRANSPOSE_TRANSPOSE_HOST_H_
#define KERNELS_TRANSPOSE_TRANSPOSE_HOST_H_

#include "kernels/transpose/transpose.h"

#include <stdio.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include "core/context.h"
#include "core/logging.h"
#include "core/gen_case.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/tool.h"

// TRANS_MIN_BYTE indicates the min bytes of transpose
#define TRANS_MIN_BYTE (64 * 64)
#define TRANS_MAX (uint32_t(-1))
#define TRANS_NRAM_LIMIT (handle_->nram_size + REM_FOR_STACK - RESERVED_NRAM)
#define NRAM_BYTE_2D_CPU (TRANS_NRAM_LIMIT / 16 * 8)
#define TRANS_MEMCPY_ALIGN 128
#define TRANS_SPLIT_RATIO (6)
#define TRANS_HN_RATIO (25)
#define TRANS_N_LIMIT (2)
#define TRANS_NW_RATIO (3)
#define TRANS_H_LIMIT (3136)
#define TRANS_ONCE_NUM (2048)
#define TRANS_X4_CLUSTER_NUM (8)
#define TRANS_TOTAL_NUM_RATIO (10)
#define TRANS_LOOP_RATIO (5)
#define TRANS_LOOP_LIMIT (9)
#define TRANS_N (96)
#define TRANS_PAD_UP(x, y) ((x % y) ? (x / y + 1) : (x / y))
#define MLUOP_TRANSPOSE_PROFILING (0)

// enum transDim shows how to deal with layout in internal transpose operation.
// for 2D, layout is [N, H]
// for 3D, layout is [N, H, W]
// for 4D, layout is [N, H, W, C]
// for nD, dimension will be treated as 3D with permute=[0,2,1] for N loops.
enum transDim {
  TR_N,
  TR_H,
  TR_W,
  TR_C,
};

// in some 2D or 3D cases, when one dimension is much bigger than another,
// we treat some cases as permute=[0,2,3,1] for better performance.
// for TRANSPOSE_3D_021, layout is [N, H, W], this macro indicates of W / H
// for TRANSPOSE_2D, layout is [N, H], this macro indicates H / N
#define TRANS_THRESHOLD_RATIO (150)
#define TRANS_HN_THRESHOLD_RATIO (32)
#define TRANS_HW_THRESHOLD_RATIO (8)

struct transRawInfo {
  std::vector<int> input_raw{0, 0, 0, 0};
  std::vector<int> permute_raw{0, 0, 0, 0};
  mluOpDataType_t dtype;
  size_t size_origin;
};

struct transFoldInfo {
  std::vector<int> input_fold{0, 0, 0, 0};
  std::vector<int> permute_fold{0, 0, 0, 0};
};

struct policyInfo {
  cnrtFunctionType_t kType;
  cnrtDim3_t kDim;
  mluOpTransposeStrategy_t strategy;
  mluOpTranspose3DStrategy_t st3D;
  mluOpTranspose2DStrategy_t st2D;
};

// This structure save the split information for TRANSPOSE_3D_021.
// num_split: the number of ipu core used
// num_processed: the total processed number for ipu core, only interger
// segment. num_processed_ceil: the total processed number for ipu core,
// including interger segment and remainder segment num_processed_limit: the
// once processed number for one loop in ipu core, used with num_processed.
// num_processed_ceil_limit: the once processed number for one loop in ipu core,
// used with num_processed_limit. for example:  x_fold = (n, h, w), permute_fold
// = (0, 2, 1), num_split = (n1, n2, n3), then: num_processed = (n / n1,  h /
// n2,  w / n3), num_processed_ceil = ( n / n1 + n % n1,  h / n2 + h % n2, w /
// n3 + w % n3).
struct splitInfo {
  std::vector<int> num_split{0, 0, 0, 0};
  std::vector<int> num_processed{0, 0, 0, 0};
  std::vector<int> num_processed_limit{0, 0, 0, 0};
  std::vector<int> num_processed_ceil{0, 0, 0, 0};
  std::vector<int> num_processed_ceil_limit{0, 0, 0, 0};
  bool is_mul_overflow;
  bool is_split_lowest_dim;
  bool split_h;
  bool split_w;
};

struct inferNdFor021TransInfo {
  // When the dimension after folding is bigger than 4, and strategy is
  // TRANSPOSE_COMMON, we treat TRANSPOSE_COMMON as TRANSPOSE_3D_021 by looping
  // N times. for example: x = [A, B, C, D, E], permute = [2,1,0,4,3]
  //        x = [A, B, C, D, E]
  // loop 1           |
  //                  v
  //        x = [C, D, E, A, B]
  // loop 2           |
  //                  v
  //        x = [C, B, D, E, A]
  // loop 3           |
  //                  v
  //        x = [C, B, A, D, E]
  // loop 4           |
  //                  v
  //        x = [C, B, A, E, D]
  // the information for every loop will be infered by inferNextParam.
  std::queue<std::vector<int>> infer_nd_for_021trans;
  // When the dimension after folding is bigger than 4, and strategy is
  // TRANSPOSE_COMMON, in the past, we treated TRANSPOSE_COMMON as
  // TRANSPOSE_3D_021 by looping N times, now, in order to implement best path
  // for TRANSPOSE_COMMON, TRANSPOSE_COMMON implements by using TRANSPOSE_2D,
  // TRANSPOSE_3D_021, TRANSPOSE_3D_102, or TRANSPOSE_4D_0213. For example: x =
  // [A, B, C, D, E], permute = [2,1,0,4,3]
  //        x = [A, B, C, D, E]
  // loop 1           |
  //                  v
  //        x = [C, D, A, B, E]
  // loop 2           |
  //                  v
  //        x = [C, B, E, D, A]
  // loop 3           |
  //                  v
  //        x = [C, B, A, E, D]
  // The method implement TRANSPOSE_COMMON which is only three steps is smaller
  // than before four steps.
  std::vector<std::vector<int>> infer_nd_path;
};

class Transpose {
 protected:
  mluOpHandle_t handle_;
  int split_dim_nd_for_small;
  int cluster_num;
  int core_num_per_cluster;
  uint64_t ele_num;
  int trans_aligned;
  // nram_limit indicates the max processed number of IPU CORE at once time.
  int nram_limit;
  // fold dims according to permute. The dimensions which in order in permute
  // will be folded to one dimension, such as x = [a,b,c,d,e,f,g], while permute
  // = [2,1,3,4,0,5,6] will be folded to x = [a,b,c,(d*e),(f*g)], and new
  // permute will be [2,1,3,0,4]
  void dimensionFolder();
  // get strategy according to the size of fold input and fold permute.
  void getTransposeStrategy();
  // reduce dimension, when the dimension is 1 in input,
  // such as x = [1,a,b], permute = [0,2,1], new x will be [a,b], and new
  // permute will be [1,0].
  void dimensionReduction();
  // judge whether the multiplier of input dimensions is bigger than Unit32.
  bool isMulOverflow() const;
  // judge whether the lowest dim need to be splited, only for 2D and 3D_021.
  bool isSplitLowestDim() const;
  void policyFunc2D(mluOpHandle_t handle_, const int &h, const int &w,
                    const int &size_dt, const int &number,
                    mluOpTranspose2DStrategy_t &st2D, const int &trans_aligned,
                    bool &split_h);
  void policyFunc();
  // split lowest dimension for TRANSPOSE_2D and TRANSPOSE_3D_021 when
  // split_info.is_split_lowest_dim = true
  void splitLowestDim();
  void getCoreSplitInfo();
  void getNumProcessedLimit(const std::vector<int> &num_processed,
                            const std::vector<bool> &align_for3d4d,
                            std::vector<int> &num_processed_limit);

  void spliter();
  // infer params for Nd transpose cases.
  void inferNextParam();
  void inferTransPath();

 public:
  transRawInfo trans_raw_info;
  transFoldInfo trans_fold_info;
  policyInfo policy_info;
  splitInfo split_info;
  inferNdFor021TransInfo infer_info;
  std::vector<bool> align_for3d4d{0, 0, 0, 0};
  Transpose(mluOpHandle_t handle, const int *input,
            const std::vector<int> &permute, const mluOpDataType_t dtype);
  // preprocess for the original input data, including folding dimension,
  // reducting dimension and getting strategy.
  void preProcess();
  bool planner();
  inline int64_t getEleNum() const { return ele_num; }
  mluOpStatus_t launchKernel(const std::string op_name,
                             const mluOpTransposeDescriptor_t &desc,
                             const mluOpTensorDescriptor_t x_desc,
                             const void *x,
                             const mluOpTensorDescriptor_t y_desc, void *y,
                             void *workspace);
};

inline mluOpStatus_t genPrototxt(const mluOpHandle_t handle,
                                 const mluOpTransposeDescriptor_t &desc,
                                 const mluOpTensorDescriptor_t &x_desc,
                                 const void *x,
                                 const mluOpTensorDescriptor_t &y_desc,
                                 const void *y) {
  // generate  transpose prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    int p[TRANSPOSE_MAX_DIM] = {0};
    for (int i = 0; i < desc->dim; i++) {
      p[i] = desc->permute[i];
    }
    GEN_CASE_START("transpose");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "x", x, x_desc, 10, -10);
    GEN_CASE_DATA(false, "y", y, y_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "transpose", "dim", desc->dim);
    GEN_CASE_OP_PARAM_ARRAY(2, "transpose", "permute", p, desc->dim);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0, 0, 0);
  }
  // generate transpose prototxt end!
  return MLUOP_STATUS_SUCCESS;
}

int paramCheck(const std::string op_name, const mluOpHandle_t &handle,
               const mluOpTransposeDescriptor_t &desc,
               const mluOpTensorDescriptor_t &x_desc, const void *x,
               const mluOpTensorDescriptor_t &y_desc, const void *y,
               const void *workspace, const size_t workspace_size);
#endif  // KERNELS_TRANSPOSE_TRANSPOSE_HOST_H_
