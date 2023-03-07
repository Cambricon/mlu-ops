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
#include <vector>
#include "transpose.h"
#include "kernels/kernel_wrapper/export_statement.h"
#define TRANSPOSE_MAX_DIM (8)

struct mluOpTransposeStruct {
  int dim;
  std::vector<int> permute;
};

namespace mluoptest {
template <typename T>
static void transposeCpuNd(const int loop_d,
                           T *x,
                           T *y,
                           const uint64_t sum,
                           uint64_t *dim,
                           uint64_t *DIM,
                           uint64_t *permute) {
  for (int loop_t = 0; loop_t < loop_d; loop_t++) {
    T *output = (T *)(y + sum * loop_t);
    T *input = (T *)(x + sum * loop_t);
    uint64_t in_index = 0, out_index = 0;    for (dim[0] = 0; dim[0] < DIM[0]; dim[0]++) {  // NOLINT
      for (dim[1] = 0; dim[1] < DIM[1]; dim[1]++) {
        for (dim[2] = 0; dim[2] < DIM[2]; dim[2]++) {
          for (dim[3] = 0; dim[3] < DIM[3]; dim[3]++) {
            for (dim[4] = 0; dim[4] < DIM[4]; dim[4]++) {
              for (dim[5] = 0; dim[5] < DIM[5]; dim[5]++) {
                for (dim[6] = 0; dim[6] < DIM[6]; dim[6]++) {
                  for (dim[7] = 0; dim[7] < DIM[7]; dim[7]++) {
                    in_index =
                        dim[0] * DIM[1] * DIM[2] * DIM[3] * DIM[4] * DIM[5] * DIM[6] * DIM[7] +  // NOLINT
                        dim[1] * DIM[2] * DIM[3] * DIM[4] * DIM[5] * DIM[6] * DIM[7] +  // NOLINT
                        dim[2] * DIM[3] * DIM[4] * DIM[5] * DIM[6] * DIM[7] +
                        dim[3] * DIM[4] * DIM[5] * DIM[6] * DIM[7] +
                        dim[4] * DIM[5] * DIM[6] * DIM[7] + dim[5] * DIM[6] * DIM[7] +  // NOLINT
                        dim[6] * DIM[7] + dim[7];
                    out_index =
                        dim[permute[0]] * DIM[permute[1]] * DIM[permute[2]] * DIM[permute[3]] *  // NOLINT
                            DIM[permute[4]] * DIM[permute[5]] * DIM[permute[6]] * DIM[permute[7]] +  // NOLINT
                        dim[permute[1]] * DIM[permute[2]] * DIM[permute[3]] * DIM[permute[4]] *  // NOLINT
                            DIM[permute[5]] * DIM[permute[6]] * DIM[permute[7]] +  // NOLINT
                        dim[permute[2]] * DIM[permute[3]] * DIM[permute[4]] * DIM[permute[5]] *  // NOLINT
                            DIM[permute[6]] * DIM[permute[7]] +
                        dim[permute[3]] * DIM[permute[4]] * DIM[permute[5]] * DIM[permute[6]] *  // NOLINT
                            DIM[permute[7]] +
                        dim[permute[4]] * DIM[permute[5]] * DIM[permute[6]] * DIM[permute[7]] +  // NOLINT
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

mluOpStatus_t mluOpTransposeCpu(const mluOpTransposeDescriptor_t desc,
                                const mluOpTensorDescriptor_t x_desc,
                                const void *x,
                                const mluOpTensorDescriptor_t y_desc,
                                void *y) {
  uint64_t sum = mluOpGetTensorElementNum(x_desc);
  // zero elements, return success
  if (sum == 0 || x_desc->dim == 0 || y_desc->dim == 0) {
    VLOG(5) << "mluOpTransposeCpu:: zero elements, return success.";
    return MLUOP_STATUS_SUCCESS;
  }
  const uint64_t dim_all = desc->dim;
  auto data_type = x_desc->dtype;
  int loop_d = 1;
  // do not change the inited value(8) in permute
  // 8 is used to match TRANSPOSE_MAX_DIM, which can make the loop below
  // applies to all-dims transpose, from 2D transpose to 8D transpose
  // if you change macro TRANSPOSE_MAX_DIM, the inited value(8) should alse be
  // changed to TRANSPOSE_MAX_DIM. And the loop level should be equal to
  // TRANSPOSE_MAX_DIM
  uint64_t permute[TRANSPOSE_MAX_DIM] = {8, 8, 8, 8, 8, 8, 8, 8};
  uint64_t DIM[TRANSPOSE_MAX_DIM + 1] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  uint64_t dim[TRANSPOSE_MAX_DIM + 1] = {0};
  if (x_desc->dim != dim_all || y_desc->dim != dim_all) {
    LOG(ERROR)
        << "mluOpTransposeCpu: dimension information mismatch, dim of x: "
        << x_desc->dim
        << ", dim of y: " << y_desc->dim << ", dim of descriptor: " << dim_all;
    return MLUOP_STATUS_BAD_PARAM;
  }
  for (int i = 0; i < dim_all; i++) {
    permute[i] = desc->permute[i];
    DIM[i] = x_desc->dims[i];
  }
  if (MLUOP_DTYPE_COMPLEX_HALF == data_type ||
      MLUOP_DTYPE_COMPLEX_FLOAT == data_type) {
    transposeCpuNd(loop_d, (double *)x, (double *)y, sum, dim, DIM, permute);
  } else {
    transposeCpuNd(loop_d, (float *)x, (float *)y, sum, dim, DIM, permute);
  }
  return MLUOP_STATUS_SUCCESS;
}

void TransposeExecutor::paramCheck() {
  if (parser_->getInputNum() != 1) {
    LOG(ERROR) << "transpose input number is wrong. ";
  }
  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "transpose output number is wrong. ";
  }
  flag_quant_mode_ = NO_QUANT;
}

void TransposeExecutor::compute() {
  VLOG(4) << "TransposeExecutor compute ";
  auto d = parser_->getProtoNode()->transpose_param().dim();
  auto x = tensor_desc_[0].tensor;
  auto y = tensor_desc_[1].tensor;
  auto x_ptr = data_vector_[0].device_ptr;
  // data_vector_[1].device_ptr = data_vector_[0].device_ptr;
  auto y_ptr = data_vector_[1].device_ptr;
  VLOG(4) << "call mluOpTranspose()";
  int permute[8];
  for (int i = 0; i < d; i++) {
    permute[i] = parser_->getProtoNode()->transpose_param().permute(i);
  }  mluOpTransposeDescriptor_t trans_desc = nullptr;
  trans_desc = cpu_runtime_.allocate(mluOpCreateTransposeDescriptor,
                                     mluOpDestroyTransposeDescriptor);
  MLUOP_CHECK(mluOpSetTransposeDescriptor(trans_desc, d, permute));
  auto workspace = workspace_.at(0);
  interface_timer_.start();
  MLUOP_CHECK(mluOpTranspose_v2(handle_, trans_desc, x, x_ptr, y, y_ptr,
                                workspace, size_workspace_));
  interface_timer_.stop();
  cpu_runtime_.deallocate(trans_desc);
}

void TransposeExecutor::cpuCompute() {
  assert(parser_->getInputNum() == 1);
  assert(parser_->getOutputNum() == 1);
  auto d = parser_->getProtoNode()->transpose_param().dim();
  auto x = tensor_desc_[0].tensor;
  auto y = tensor_desc_[1].tensor;
  auto count1 = parser_->getInputDataCount(0);
  auto count2 = parser_->getOutputDataCount(0);
  assert(count1 == count2);
  int permute[8];
  for (int i = 0; i < d; i++) {
    permute[i] = parser_->getProtoNode()->transpose_param().permute(i);
  }
  mluOpTransposeDescriptor_t trans_desc;
  trans_desc = cpu_runtime_.allocate(mluOpCreateTransposeDescriptor,
                                     mluOpDestroyTransposeDescriptor);
  MLUOP_CHECK(mluOpSetTransposeDescriptor(trans_desc, d, permute));
  VLOG(4) << "call mluOpTransposeHost()";
  MLUOP_CHECK(mluOpTransposeCpu(trans_desc, x, cpu_fp32_input_[0], y,
                                cpu_fp32_output_[0]));
  cpu_runtime_.deallocate(trans_desc);
}

void TransposeExecutor::workspaceMalloc() {
  auto x_desc = tensor_desc_[0].tensor;
  auto d = parser_->getProtoNode()->transpose_param().dim();
  int permute[8];
  for (int i = 0; i < d; i++) {
    permute[i] = parser_->getProtoNode()->transpose_param().permute(i);
  }
  mluOpTransposeDescriptor_t trans_desc;
  trans_desc = cpu_runtime_.allocate(mluOpCreateTransposeDescriptor,
                                     mluOpDestroyTransposeDescriptor);
  MLUOP_CHECK(mluOpSetTransposeDescriptor(trans_desc, d, permute));
  MLUOP_CHECK(mluOpGetTransposeWorkspaceSize(handle_, x_desc, trans_desc,
                                             &size_workspace_));
  VLOG(4) << "Malloc workspace space.";
  void* temp = nullptr;
  if (size_workspace_ != 0) {
    temp = mlu_runtime_.allocate(size_workspace_);
  }
  workspace_.push_back(temp);
  VLOG(4) << "Malloc addr: " << temp << " , size: " << size_workspace_;
  eva_->setMluWorkspaceSize(size_workspace_);
}

void TransposeExecutor::workspaceFree() {
  auto temp = workspace_.at(0);
  mlu_runtime_.deallocate(temp);
}

int64_t TransposeExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->getOutputDataCount(0);
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}
}  // namespace mluoptest
