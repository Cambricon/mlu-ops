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

#include "transpose_cpu.h"
#include <vector>
#include "core/tensor.h"


template <typename T>
static void transposeCpuNd(const int loop_d, T *x, T *y, const uint64_t sum,
                           uint64_t *dim, uint64_t *DIM, uint64_t *permute) {
  for (int loop_t = 0; loop_t < loop_d; loop_t++) {
    T *output = (T *)(y + sum * loop_t);
    T *input = (T *)(x + sum * loop_t);
    uint64_t in_index = 0, out_index = 0;

    for (dim[0] = 0; dim[0] < DIM[0]; dim[0]++) {
      for (dim[1] = 0; dim[1] < DIM[1]; dim[1]++) {
        for (dim[2] = 0; dim[2] < DIM[2]; dim[2]++) {
          for (dim[3] = 0; dim[3] < DIM[3]; dim[3]++) {
            for (dim[4] = 0; dim[4] < DIM[4]; dim[4]++) {
              for (dim[5] = 0; dim[5] < DIM[5]; dim[5]++) {
                for (dim[6] = 0; dim[6] < DIM[6]; dim[6]++) {
                  for (dim[7] = 0; dim[7] < DIM[7]; dim[7]++) {
                    in_index =
                        dim[0] * DIM[1] * DIM[2] * DIM[3] * DIM[4] * DIM[5] *
                            DIM[6] * DIM[7] +
                        dim[1] * DIM[2] * DIM[3] * DIM[4] * DIM[5] * DIM[6] *
                            DIM[7] +
                        dim[2] * DIM[3] * DIM[4] * DIM[5] * DIM[6] * DIM[7] +
                        dim[3] * DIM[4] * DIM[5] * DIM[6] * DIM[7] +
                        dim[4] * DIM[5] * DIM[6] * DIM[7] +
                        dim[5] * DIM[6] * DIM[7] + dim[6] * DIM[7] + dim[7];
                    out_index =
                        dim[permute[0]] * DIM[permute[1]] * DIM[permute[2]] *
                            DIM[permute[3]] * DIM[permute[4]] *
                            DIM[permute[5]] * DIM[permute[6]] *
                            DIM[permute[7]] +
                        dim[permute[1]] * DIM[permute[2]] * DIM[permute[3]] *
                            DIM[permute[4]] * DIM[permute[5]] *
                            DIM[permute[6]] * DIM[permute[7]] +
                        dim[permute[2]] * DIM[permute[3]] * DIM[permute[4]] *
                            DIM[permute[5]] * DIM[permute[6]] *
                            DIM[permute[7]] +
                        dim[permute[3]] * DIM[permute[4]] * DIM[permute[5]] *
                            DIM[permute[6]] * DIM[permute[7]] +
                        dim[permute[4]] * DIM[permute[5]] * DIM[permute[6]] *
                            DIM[permute[7]] +
                        dim[permute[5]] * DIM[permute[6]] * DIM[permute[7]] +
                        dim[permute[6]] * DIM[permute[7]] + dim[permute[7]];
                    output[out_index] = input[in_index];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

mluOpStatus_t mluOpTransposeCpu(const int64_t dim_desc,
                                const std::vector<int> permute_desc,
                                const mluOpTensorDescriptor_t x_desc,
                                const void *x,
                                const mluOpTensorDescriptor_t y_desc, void *y) {
  PARAM_CHECK("[cnnlTransposeCpu]", x_desc != NULL);
  PARAM_CHECK("[cnnlTransposeCpu]", y_desc != NULL);
  uint64_t sum = mluOpGetTensorElementNum(x_desc);
  // zero elements, return success
  if (sum == 0 || x_desc->getDim() == 0 || y_desc->getDim() == 0) {
    VLOG(5) << "cnnlTransposeCpu:: zero elements, return success.";
    return MLUOP_STATUS_SUCCESS;
  }
  PARAM_CHECK("[cnnlTransposeCpu]", x != NULL);
  PARAM_CHECK("[cnnlTransposeCpu]", y != NULL);

  const uint64_t dim_all = dim_desc;
  auto data_type = x_desc->getDtype();
  int loop_d = 1;
  if (data_type == MLUOP_DTYPE_INT31) {
    loop_d = 2;
  }
  // do not change the inited value(8) in permute
  // 8 is used to match TRANSPOSE_MAX_DIM, which can make the loop below
  // applies to all-dims transpose, from 2D transpose to 8D transpose
  // if you change macro TRANSPOSE_MAX_DIM, the inited value(8) should alse be
  // changed to TRANSPOSE_MAX_DIM. And the loop level should be equal to
  // TRANSPOSE_MAX_DIM
  uint64_t permute[TRANSPOSE_MAX_DIM] = {8, 8, 8, 8, 8, 8, 8, 8};
  uint64_t DIM[TRANSPOSE_MAX_DIM + 1] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  uint64_t dim[TRANSPOSE_MAX_DIM + 1] = {0};

  if (x_desc->getDim() != dim_all || y_desc->getDim() != dim_all) {
    LOG(ERROR)
        << "cnnlTransposeCpu: dimension information mismatch, dim of x: "
        << x_desc->getDim() << ", dim of y: " << y_desc->getDim()
        << ", dim of descriptor: " << dim_all;
    return MLUOP_STATUS_BAD_PARAM;
  }

  for (int i = 0; i < dim_all; i++) {
    permute[i] = permute_desc[i];
    DIM[i] = x_desc->getDimIndex(i);
  }
  if (MLUOP_DTYPE_INT31 == data_type) {
    transposeCpuNd(loop_d, (int16_t *)x, (int16_t *)y, sum, dim, DIM, permute);
  } else if (MLUOP_DTYPE_COMPLEX_HALF == data_type ||
             MLUOP_DTYPE_COMPLEX_FLOAT == data_type) {
    transposeCpuNd(loop_d, (double *)x, (double *)y, sum, dim, DIM, permute);
  } else {
    transposeCpuNd(loop_d, (float *)x, (float *)y, sum, dim, DIM, permute);
  }
  return MLUOP_STATUS_SUCCESS;
}
