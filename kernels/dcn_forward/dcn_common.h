/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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
#ifndef KERNELS_DCN_COMMON_DCN_COMMON_H
#define KERNELS_DCN_COMMON_DCN_COMMON_H
#include <limits.h>
#include <math.h>
#include <vector>

#include "kernels/utils/cnnl_helper.h"

#define DCN_API "mluOpDCN"

mluOpStatus_t MLUOP_WIN_API
mluOpCreateDCNDescriptor(mluOpDCNDescriptor_t *dcn_desc) {
  PARAM_CHECK(DCN_API, dcn_desc != NULL);
  CHECK_FUNC_RETURN(cnnlCreateDCNDescriptor(dcn_desc), CNNL_STATUS_SUCCESS,
                    "[mluOpDcn] Internal error accured in "
                    "mluOpCreateDCNDescriptor.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpDestroyDCNDescriptor(mluOpDCNDescriptor_t dcn_desc) {
  PARAM_CHECK(DCN_API, dcn_desc != NULL);
  CHECK_FUNC_RETURN(cnnlDestroyDCNDescriptor(dcn_desc), CNNL_STATUS_SUCCESS,
                    "[mluOpDcn] Internal error accured in "
                    "mluOpDestroyDCNDescriptor.",
                    MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetDCNDescriptor(
    mluOpDCNDescriptor_t dcn_desc, int dimNb, const int pad[],
    const int stride[], const int dilation[], int deformable_group,
    int conv_group, int im2col_step, const mluOpDataType_t compute_type) {
  PARAM_CHECK(DCN_API, dcn_desc != NULL);
  CHECK_FUNC_RETURN(
      cnnlSetDCNDescriptor(dcn_desc, dimNb, pad, stride, dilation,
                           deformable_group, conv_group, im2col_step,
                           cnnlDataType_t(compute_type)),
      CNNL_STATUS_SUCCESS,
      "[mluOpDcn] Internal error accured in "
      "mluOpSetDCNDescriptor.",
      MLUOP_STATUS_INTERNAL_ERROR);
  return MLUOP_STATUS_SUCCESS;
}

#endif  // KERNELS_DCN_COMMON_DCN_COMMON_H
