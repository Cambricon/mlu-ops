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

namespace mluoptest {

void TransposeExecutor::paramCheck() {
  if (parser_->getInputNum() != 1) {
    LOG(ERROR) << "transpose input number is wrong. ";
  }
  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "transpose output number is wrong. ";
  }
  flag_quant_mode_ = NO_QUANT;
}

void TransposeExecutor::prepareComputeParam() {
    dims_ = parser_->getProtoNode()->transpose_param().dim();
    x_desc_ = tensor_desc_[0].tensor;
    y_desc_ = tensor_desc_[1].tensor;
    for (int i = 0; i < dims_; i++) {
      permute_[i] = parser_->getProtoNode()->transpose_param().permute(i);
    }
    trans_desc_ = cpu_runtime_.allocate(mluOpCreateTransposeDescriptor,
                                        mluOpDestroyTransposeDescriptor);
    MLUOP_CHECK(mluOpSetTransposeDescriptor(trans_desc_, dims_, permute_));
}

void TransposeExecutor::workspaceMalloc() {
    prepareComputeParam();
    MLUOP_CHECK(mluOpGetTransposeWorkspaceSize(handle_, x_desc_, trans_desc_,
                                               &size_workspace_));
    VLOG(4) << "Malloc workspace space.";
    void *temp = nullptr;
    if (size_workspace_ != 0) {
        temp = mlu_runtime_.allocate(size_workspace_);
  }
  workspace_.push_back(temp);
  VLOG(4) << "Malloc addr: " << temp << " , size: " << size_workspace_;
  eva_->setMluWorkspaceSize(size_workspace_);
}

void TransposeExecutor::compute() {
  VLOG(4) << "TransposeExecutor compute ";
  auto x_ptr = data_vector_[0].device_ptr;
  auto y_ptr = data_vector_[1].device_ptr;
  VLOG(4) << "call mluOpTranspose()";
  auto workspace = workspace_.at(0);
  interface_timer_.start();
  MLUOP_CHECK(mluOpTranspose_v2(handle_, trans_desc_, x_desc_, x_ptr, y_desc_,
                                y_ptr, workspace, size_workspace_));
  interface_timer_.stop();
}

void TransposeExecutor::cpuCompute() {
  auto x_cpu = cpu_fp32_input_[0];
  auto y_cpu = cpu_fp32_output_[0];
  uint64_t element_num = mluOpGetTensorElementNum(x_desc_);

  // do not change the inited value(8) in permute
  // 8 is used to match TRANSPOSE_MAX_DIM, which can make the loop below
  // applies to all-dims transpose, from 2D transpose to 8D transpose
  // if you change macro TRANSPOSE_MAX_DIM, the inited value(8) should alse be
  // changed to TRANSPOSE_MAX_DIM. And the loop level should be equal to
  // TRANSPOSE_MAX_DIM
  uint64_t permute_8d[TRANSPOSE_MAX_DIM] = {8, 8, 8, 8, 8, 8, 8, 8};
  uint64_t DIM[TRANSPOSE_MAX_DIM + 1] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  uint64_t dim[TRANSPOSE_MAX_DIM + 1] = {0};

  for (int i = 0; i < dims_; i++) {
      permute_8d[i] = permute_[i];
      DIM[i] = x_desc_->dims[i];
  }
  auto data_type = x_desc_->dtype;
  if (MLUOP_DTYPE_COMPLEX_HALF == data_type ||
      MLUOP_DTYPE_COMPLEX_FLOAT == data_type) {
    transposeCpuNd((double *)y_cpu, (double *)x_cpu, dim, DIM, permute_8d,
                   element_num, 1);
  } else {
    transposeCpuNd((float *)y_cpu, (float *)x_cpu, dim, DIM, permute_8d,
                   element_num, 1);
  }
}

template <typename T>
void TransposeExecutor::transposeCpuNd(T *y,
                                       const T *x,
                                       uint64_t *dim,
                                       const uint64_t *DIM,
                                       const uint64_t *permute,
                                       const uint64_t element_num,
                                       const int loop_d) {
  for (int loop_t = 0; loop_t < loop_d; loop_t++) {
    T *output = (T *)(y + element_num * loop_t);
    T *input = (T *)(x + element_num * loop_t);
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
