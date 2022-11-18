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
#include "kernels/transpose/transpose.h"

#include <stdio.h>

#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

#include "core/context.h"
#include "core/logging.h"
#include "core/gen_case.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/tool.h"
#include "kernels/transpose/transpose_host.h"

mluOpStatus_t MLUOP_WIN_API
mluOpCreateTransposeDescriptor(mluOpTransposeDescriptor_t *desc) {
  PARAM_CHECK("[mluOpCreateTransposeDescriptor]", desc != NULL);
  mluOpTransposeStruct *s = new (std::nothrow) mluOpTransposeStruct();
  if (s == NULL) {
    LOG(ERROR) << "[mluOpCreateTransposeDescriptor] TransposeDescriptor malloc "
                  "failed.";
    return MLUOP_STATUS_ALLOC_FAILED;
  }
  *desc = s;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpDestroyTransposeDescriptor(mluOpTransposeDescriptor_t desc) {
  PARAM_CHECK("[mluOpDestroyTransposeDescriptor]", desc != NULL);
  delete desc;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetTransposeDescriptor(
    mluOpTransposeDescriptor_t desc, const int dims, const int *permute) {
  PARAM_CHECK("[mluOpSetTransposeDescriptor]", desc != NULL);
  if (dims > TRANSPOSE_MAX_DIM) {
    LOG(ERROR) << "[mluOpSetTransposeDescriptor] The dim size of permute "
                  "should be less than or "
               << "equal to " << TRANSPOSE_MAX_DIM << ". But now is " << dims
               << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  desc->dim = dims;
  { std::vector<int>().swap(desc->permute); }
  for (int i = 0; i < dims; i++) {
    desc->permute.push_back(permute[i]);
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetTransposeWorkspaceSize(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t x_desc,
    const mluOpTransposeDescriptor_t desc, size_t *size) {
  PARAM_CHECK("[mluOpGetTransposeWorkspaceSize]", handle != NULL);
  PARAM_CHECK("[mluOpGetTransposeWorkspaceSize]", x_desc != NULL);
  PARAM_CHECK("[mluOpGetTransposeWorkspaceSize]", desc != NULL);
  PARAM_CHECK("[mluOpGetTransposeWorkspaceSize]", size != NULL);
  if (x_desc->dim < 4) {
    *size = 0;
    return MLUOP_STATUS_SUCCESS;
  }

  Transpose trans(handle, x_desc->dims, desc->permute, x_desc->dtype);
  trans.preProcess();
  int64_t ele_num = 1;
  const int dim_fold = trans.trans_fold_info.input_fold.size();
  if (dim_fold < 4) {
    ele_num = 0;
  } else {
    ele_num = trans.getEleNum();
  }
  *size = ele_num * trans.trans_raw_info.size_origin;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpTranspose(mluOpHandle_t handle, const mluOpTransposeDescriptor_t desc,
               const mluOpTensorDescriptor_t x_desc, const void *x,
               const mluOpTensorDescriptor_t y_desc, void *y) {
  LOG_FIRST_N(WARNING, 1)
      << "[mluOpTranspose]: This api will be deprecated in the further release,"
      << " please use [mluOpTranspose_v2] instead.";
  int res = paramCheck("[mluOpTranspose]", handle, desc, x_desc, x, y_desc, y,
                       NULL, 0);
  // res:-1: zero ele, 1: bad param, 0: pass
  if (-1 == res) {
    return MLUOP_STATUS_SUCCESS;
  } else if (1 == res) {
    return MLUOP_STATUS_BAD_PARAM;
  } else {
    PARAM_CHECK("[mluOpTranspose]", genPrototxt(handle, desc, x_desc, x, y_desc,
                                                y) == MLUOP_STATUS_SUCCESS);

    Transpose trans(handle, x_desc->dims, desc->permute, x_desc->dtype);
    trans.preProcess();
    trans.planner();
    PARAM_CHECK("[mluOpTranspose]",
                trans.launchKernel("[mluOpTranspose]", desc, x_desc, x, y_desc,
                                   y, NULL) == MLUOP_STATUS_SUCCESS);
    GEN_CASE_END();
    return MLUOP_STATUS_SUCCESS;
  }
}

mluOpStatus_t MLUOP_WIN_API
mluOpTranspose_v2(mluOpHandle_t handle, const mluOpTransposeDescriptor_t desc,
                  const mluOpTensorDescriptor_t x_desc, const void *x,
                  const mluOpTensorDescriptor_t y_desc, void *y,
                  void *workspace, size_t workspace_size) {
  int res = paramCheck("[mluOpTranspose_v2]", handle, desc, x_desc, x, y_desc,
                       y, workspace, workspace_size);
  // res:-1: zero ele, 1: bad param, 0: pass
  if (-1 == res) {
    return MLUOP_STATUS_SUCCESS;
  } else if (1 == res) {
    return MLUOP_STATUS_BAD_PARAM;
  } else {
    PARAM_CHECK("[mluOpTranspose_v2]",
                genPrototxt(handle, desc, x_desc, x, y_desc, y) ==
                    MLUOP_STATUS_SUCCESS);
    Transpose trans(handle, x_desc->dims, desc->permute, x_desc->dtype);
    trans.preProcess();
    trans.planner();
    PARAM_CHECK(
        "[mluOpTranspose_v2]",
        trans.launchKernel("[mluOpTranspose_v2]", desc, x_desc, x, y_desc, y,
                           workspace) == MLUOP_STATUS_SUCCESS);
    GEN_CASE_END();
    return MLUOP_STATUS_SUCCESS;
  }
}
