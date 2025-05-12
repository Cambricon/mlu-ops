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
#include "complex_matrix_dot_vector.h"
#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "kernels/kernel.h"

static void policyFunc(const mluOpHandle_t &handle,
                       cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type, int col_num, int row_num, bool row_major, bool *large_col) {
  int num_deal = 0;
  if(row_major) {
    num_deal = handle->nram_size  / (8 * sizeof(float));
    VLOG(5) << "nram_size: " << handle->nram_size;
    // if (col_num > 10) {
    if (col_num > num_deal) {
        *large_col = true;
    } else {
        *large_col = false;
    }
  } else {
    num_deal = handle->nram_size  / (6 * sizeof(float));
    if (col_num <= num_deal) {
      *large_col = false;
    } else {
      *large_col = true;
    }
  }
  VLOG(5) << "if large col: " << *large_col << " col_num: " << col_num
          << "num_deal: " << num_deal;
  // *k_type = cnrtFuncTypeUnion1;
  *k_type = cnrtFuncTypeBlock;
  // k_dim->x = 4;
  k_dim->x = 1;
  k_dim->y = 1;
  k_dim->z = 1;
  // *k_type = cnrtFuncTypeUnion1;
  // k_dim->x = handle->core_num_per_cluster;
  // k_dim->y = mluop::runtime::getClusterLimitCapability(handle);
  // k_dim->z = 1;
}

mluOpStatus_t MLUOP_WIN_API
mluOpComplexMatrixDotVector(mluOpHandle_t handle, const mluOpTensorDescriptor_t vector_desc, void *vector_input, const mluOpTensorDescriptor_t matrix_desc, void *matrix_input,
              const int pad_num, bool row_major, const int output_type, const mluOpTensorDescriptor_t output_desc, void *output) {

  // policy select


  const int64_t dims = matrix_desc->getDim();
  int batch = 0;
  int col_num = 1;
  int row_num = 1;
  if(output_type == 0) {
    if(row_major) {
      if ( dims == 1) {
          batch = 1;
          col_num = matrix_desc->getDimIndex(0);
      } else if(dims == 2) {
          col_num = matrix_desc->getDimIndex(1);
          batch = matrix_desc->getDimIndex(0);
      } else if(dims == 3) {
          col_num = matrix_desc->getDimIndex(2);
          row_num = matrix_desc->getDimIndex(1);
          batch = matrix_desc->getDimIndex(0);
      }
    } else {
      if(dims == 2) {
          col_num = matrix_desc->getDimIndex(1);
          row_num = matrix_desc->getDimIndex(0);
          batch = 1;
      } else if (dims == 3) {
          col_num = matrix_desc->getDimIndex(2);
          row_num = matrix_desc->getDimIndex(1);
          batch = matrix_desc->getDimIndex(0);
      }
    }
  } else if(output_type == 1) {
    if(row_major) {
      if ( dims == 1) {
          batch = 1;
          col_num = output_desc->getDimIndex(0);
      } else if(dims == 2) {
          col_num = output_desc->getDimIndex(1);
          batch = output_desc->getDimIndex(0);
      } else if(dims == 3) {
          col_num = output_desc->getDimIndex(2);
          row_num = output_desc->getDimIndex(1);
          batch = output_desc->getDimIndex(0);
      }
    } else {
      if(dims == 2) {
          col_num = output_desc->getDimIndex(1);
          row_num = output_desc->getDimIndex(0);
          batch = 1;
      } else if (dims == 3) {
          col_num = output_desc->getDimIndex(2);
          row_num = output_desc->getDimIndex(1);
          batch = output_desc->getDimIndex(0);
      }
    }
  }

  bool real_input = false;
  if (matrix_desc->getDtype() == MLUOP_DTYPE_FLOAT) {
      real_input = true;
  }

  VLOG(5) << "batch: " << batch;
  VLOG(5) << "col_num: " << col_num;
  VLOG(5) << "row_num: " << row_num;
  VLOG(5) << "pad_num: " << pad_num;
  VLOG(5) << "row_major: " << row_major;
  VLOG(5) << "output_type: " << output_type;

  VLOG(5) << "kernel KernelComplexMatrixDotVector.";

  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  bool large_col = false;
  policyFunc(handle, &k_dim, &k_type, col_num, row_num, row_major, &large_col);
  VLOG(5) << "[policyFunc] launch kernel policyFUnc[" << k_dim.x << ", "
          << k_dim.y << ", " << k_dim.z << "]";
  CHECK_RETURN("[KernelComplexMatrixDotVector] ", KernelComplexMatrixDotVector(k_dim, k_type, handle->queue,
    vector_input, matrix_input, output, batch, row_num, col_num, pad_num, row_major, real_input, large_col, output_type));
  return MLUOP_STATUS_SUCCESS;
}
