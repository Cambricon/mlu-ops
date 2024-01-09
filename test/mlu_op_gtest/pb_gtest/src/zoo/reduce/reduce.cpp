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
#include <set>
#include <algorithm>
#include <functional>
#include "reduce.h"
namespace mluoptest {
void ReduceExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_reduce_param()) {
    LOG(ERROR) << "Lose reduce param. ";
  }
  if (parser_->getInputNum() != 1) {
    LOG(ERROR) << "reduce tensor input number is wrong. ";
  }
  flag_quant_mode_ = NO_QUANT;
}

bool ReduceExecutor::isinf(float input) {
  if (*(int *)(&input) == 0x7f800000 || *(int *)(&input) == 0xff800000) {
    return true;
  } else {
    return false;
  }
}

bool ReduceExecutor::isnan(float input) {
  if (*(int *)(&input) >= 0x7f800001) {
    return true;
  } else {
    return false;
  }
}

void ReduceExecutor::kahanAdd(double input, double &sum, double &delta) {
  if (isinf((float)input) || isinf((float)sum) ||
      isnan((float)input) || isnan((float)sum)) {
    sum += input;
  } else {
    double y = input - delta;
    double t = sum + y;
    delta = t - sum - y;
    sum = t;
  }
}

void ReduceExecutor::compute() {
  VLOG(4) << "ReduceExecutor compute ";
  if (!parser_->getProtoNode()->has_reduce_param()) {
    LOG(ERROR) << "Lose reduce param. ";
  }
  std::vector<int> axis;
  int axis_num = 0;
  if (parser_->getProtoNode()->reduce_param().axises_size() > 0) {
    axis_num = parser_->getProtoNode()->reduce_param().axises_size();
    for (int i = 0; i < axis_num; i++) {
      int axis_temp = parser_->getProtoNode()->reduce_param().axises(i);
      axis.push_back(axis_temp);
    }  } else if (parser_->getProtoNode()->reduce_param().has_axis() == true) {
    int axis_temp = parser_->getProtoNode()->reduce_param().axis();
    axis_num = 1;
    axis.push_back(axis_temp);
  } else {
    LOG(ERROR) << "axis or axises must choose one at least";
  }
  mluOpReduceOp_t mode =
      (mluOpReduceOp_t)parser_->getProtoNode()->reduce_param().mode();
  mluOpReduceIndices_t is_indices =
      (mluOpReduceIndices_t)parser_->getProtoNode()->reduce_param().is_indices();  // NOLINT
  mluOpIndicesType_t indices_type =
      (mluOpIndicesType_t)parser_->getProtoNode()->reduce_param().indices_type();  // NOLINT
  float p = parser_->getProtoNode()->reduce_param().p();
  auto tensor_a = tensor_desc_[0].tensor;
  auto tensor_c = tensor_desc_[1].tensor;
  auto dev_a = data_vector_[0].device_ptr;
  auto dev_c = data_vector_[1].device_ptr;
  VLOG(4) << "call Reduce  Tensor()";
  mluOpReduceDescriptor_t reduce_desc;
  mluOpDataType_t dt = tensor_a->dtype;
  if (mode == MLUOP_REDUCE_ADD || mode == MLUOP_REDUCE_AVG) {
    dt = (mluOpDataType_t)parser_->getProtoNode()->
                                       reduce_param().compute_dtype();
  }
  reduce_desc = cpu_runtime_.allocate(mluOpCreateReduceDescriptor,
                                      mluOpDestroyReduceDescriptor);
  uint32_t indices_size_inbytes = 0;
  void *dev_indices = NULL;
  if (is_indices == MLUOP_REDUCE_NO_INDICES) {
    MLUOP_CHECK(mluOpSetReduceDescriptor(reduce_desc, axis.data(), axis_num,
                                         mode, dt, MLUOP_NOT_PROPAGATE_NAN,
                                         is_indices, indices_type));
    if (mode == MLUOP_REDUCE_NORMP) {
      MLUOP_CHECK(mluOpSetReduceDescriptor_v2(reduce_desc, axis.data(),
                                              axis_num, mode, dt,
                                              MLUOP_NOT_PROPAGATE_NAN,
                                              is_indices, indices_type, p));
    }
  } else if (is_indices == MLUOP_REDUCE_FLATTENED_INDICES) {
    if (tensor_desc_.size() != 3 || data_vector_.size() != 3) {
      LOG(ERROR)
          << "The data vector size must be 3, when is_indices = MLUOP_REDUCE_FLATTENED_INDICES";  // NOLINT
      return;
    }
    auto tensor_indices = tensor_desc_[2].tensor;
    dev_indices = data_vector_[2].device_ptr;
    uint32_t indices_dim = 1;
    for (int i = 0; i < tensor_indices->dim; i++) {
      indices_dim = indices_dim * tensor_indices->dims[i];
    }
    if (indices_type == MLUOP_32BIT_INDICES) {
      indices_size_inbytes = indices_dim * sizeof(int32_t);
    } else {
      indices_size_inbytes = indices_dim * sizeof(int16_t);
    }
    MLUOP_CHECK(mluOpSetReduceDescriptor(reduce_desc, axis.data(), axis_num,
                                         mode, dt, MLUOP_NOT_PROPAGATE_NAN,
                                         is_indices, indices_type));
  } else {
    dev_indices = data_vector_[1].device_ptr;
    dev_c = NULL;
    MLUOP_CHECK(mluOpSetReduceDescriptor(reduce_desc, axis.data(), axis_num,
                                         mode, dt, MLUOP_NOT_PROPAGATE_NAN,
                                         is_indices, indices_type));
  }
  interface_timer_.start();
  float alpha = parser_->getProtoNode()->reduce_param().alpha();
  float beta = parser_->getProtoNode()->reduce_param().beta();
  if (tensor_a->dtype == MLUOP_DTYPE_INT32) {
    int alpha_int = (int)alpha;
    int beta_int = (int)beta;
    MLUOP_CHECK(mluOpReduce(handle_, reduce_desc, workspace_ptr,
                            workspace_size_, &alpha_int, tensor_a,
                            dev_a, indices_size_inbytes, dev_indices,
                            &beta_int, tensor_c, dev_c));
  } else {
    MLUOP_CHECK(mluOpReduce(handle_, reduce_desc, workspace_ptr,
                            workspace_size_, &alpha, tensor_a,
                            dev_a, indices_size_inbytes, dev_indices,
                            &beta, tensor_c, dev_c));
  }
  interface_timer_.stop();
}

void ReduceExecutor::workspaceMalloc() {
  std::vector<int> axis;
  int axis_num = 0;
  if (parser_->getProtoNode()->reduce_param().axises_size() > 0) {
    axis_num = parser_->getProtoNode()->reduce_param().axises_size();
    for (int i = 0; i < axis_num; i++) {
      int axis_temp = parser_->getProtoNode()->reduce_param().axises(i);
      axis.push_back(axis_temp);
    }
  } else if (parser_->getProtoNode()->reduce_param().has_axis() == true) {
    int axis_temp = parser_->getProtoNode()->reduce_param().axis();
    axis_num = 1;
    axis.push_back(axis_temp);
  } else {
    LOG(ERROR) << "axis or axises must choose one at least";
  }
  mluOpReduceOp_t mode = (mluOpReduceOp_t)parser_->getProtoNode()->reduce_param().mode();  // NOLINT
  mluOpReduceIndices_t is_indices =
      (mluOpReduceIndices_t)parser_->getProtoNode()->reduce_param().is_indices();  // NOLINT
  mluOpIndicesType_t indices_type =
      (mluOpIndicesType_t)parser_->getProtoNode()->reduce_param().indices_type();  // NOLINT
  float p = parser_->getProtoNode()->reduce_param().p();
  auto tensor_a = tensor_desc_[0].tensor;
  auto tensor_c = tensor_desc_[1].tensor;
  auto dev_a = data_vector_[0].device_ptr;
  auto dev_c = data_vector_[1].device_ptr;
  VLOG(4) << "call Reduce  Tensor()";
  mluOpReduceDescriptor_t reduce_desc;
  mluOpDataType_t dt = tensor_a->dtype;
  if (mode == MLUOP_REDUCE_ADD || mode == MLUOP_REDUCE_AVG) {
    dt = (mluOpDataType_t)parser_->getProtoNode()->reduce_param().compute_dtype();  // NOLINT
  }
  reduce_desc = cpu_runtime_.allocate(mluOpCreateReduceDescriptor,
                                      mluOpDestroyReduceDescriptor);
  uint32_t indices_size_inbytes = 0;
  void *dev_indices = NULL;
  if (is_indices == MLUOP_REDUCE_NO_INDICES) {
    MLUOP_CHECK(mluOpSetReduceDescriptor(reduce_desc, axis.data(), axis_num,
                                         mode, dt, MLUOP_NOT_PROPAGATE_NAN,
                                         is_indices, indices_type));
    if (mode == MLUOP_REDUCE_NORMP) {
      MLUOP_CHECK(mluOpSetReduceDescriptor_v2(reduce_desc, axis.data(),
                                              axis_num, mode, dt,
                                              MLUOP_NOT_PROPAGATE_NAN,
                                              is_indices, indices_type, p));
    }  } else if (is_indices == MLUOP_REDUCE_FLATTENED_INDICES) {
    if (tensor_desc_.size() != 3 || data_vector_.size() != 3) {
      LOG(ERROR)
          << "The data vector size must be 3, when is_indices = MLUOP_REDUCE_FLATTENED_INDICES";  // NOLINT
      return;
    }
    auto tensor_indices = tensor_desc_[2].tensor;
    dev_indices = data_vector_[2].device_ptr;
    uint32_t indices_dim = 1;
    for (int i = 0; i < tensor_indices->dim; i++) {
      indices_dim = indices_dim * tensor_indices->dims[i];
    }
    if (indices_type == MLUOP_32BIT_INDICES) {
      indices_size_inbytes = indices_dim * sizeof(int32_t);
    } else {
      indices_size_inbytes = indices_dim * sizeof(int16_t);
    }
    MLUOP_CHECK(mluOpSetReduceDescriptor(reduce_desc, axis.data(), axis_num,
                                         mode, dt, MLUOP_NOT_PROPAGATE_NAN,
                                         is_indices, indices_type));
  } else {
    dev_indices = data_vector_[1].device_ptr;
    dev_c = NULL;
    MLUOP_CHECK(mluOpSetReduceDescriptor(reduce_desc, axis.data(), axis_num,
                                         mode, dt, MLUOP_NOT_PROPAGATE_NAN,
                                         is_indices, indices_type));
  }
  VLOG(4) << "execute mluOpGetReduceOpWorkspaceSize";
  MLUOP_CHECK(
      mluOpGetReduceOpWorkspaceSize(handle_, tensor_a, tensor_c, reduce_desc,
                                    &workspace_size_));
  VLOG(4) << "Malloc workspace space.";
  if (workspace_size_ != 0) {
    workspace_ptr = mlu_runtime_.allocate(workspace_size_);
    workspace_.push_back(workspace_ptr);
    eva_->setMluWorkspaceSize(workspace_size_);
  }
  VLOG(4) << "Malloc addr: " << workspace_ptr << " , size: " << workspace_size_;
}

void ReduceExecutor::workspaceFree() {
  if (workspace_ptr != nullptr) {
    mlu_runtime_.deallocate(workspace_ptr);
    workspace_ptr = nullptr;
  }
}

// reset indice to 0, if declared in pb but empty.
void ReduceExecutor::diffPreprocess() {
  mluOpReduceIndices_t is_indices =
      (mluOpReduceIndices_t)parser_->getProtoNode()->reduce_param().is_indices();  // NOLINT
  if (parser_->getOutputNum() == 1 &&
          is_indices != MLUOP_REDUCE_ONLY_INDICES) {
    return;
  }
  if (is_indices == MLUOP_REDUCE_NO_INDICES && parser_->getOutputNum() == 2) {
    // dev_indices is not output
    int indices_size = parser_->getOutputDataCount(1);
    for (int i = 0; i < indices_size; ++i) {
      mlu_fp32_output_[1][i] = 0.0f;
    }
    return;
  }
  if (parser_->getOutputNum() == 2 &&
      is_indices == MLUOP_REDUCE_ONLY_INDICES) {
    int indices_size = parser_->getOutputDataCount(1);
    for (int i = 0; i < indices_size; ++i) {
      mlu_fp32_output_[1][i] = 0.0f;
    }
    return;
  }
  mluOpReduceOp_t mode =
     (mluOpReduceOp_t)parser_->getProtoNode()->reduce_param().mode();
  if ((mode != MLUOP_REDUCE_MAX && mode != MLUOP_REDUCE_MIN))
    return;
}

void ReduceExecutor::cpuCompute() {
  assert(parser_->getInputNum() == 1);
  std::vector<int> axis;
  int axis_num = 0;
  if (parser_->getProtoNode()->reduce_param().axises_size() > 0) {
    axis_num = parser_->getProtoNode()->reduce_param().axises_size();
    for (int i = 0; i < axis_num; i++) {
      int axis_temp = parser_->getProtoNode()->reduce_param().axises(i);
      axis.push_back(axis_temp);
    }
  } else if (parser_->getProtoNode()->reduce_param().has_axis() == true) {
    int axis_temp = parser_->getProtoNode()->reduce_param().axis();
    axis_num = 1;
    axis.push_back(axis_temp);
  } else {
    LOG(ERROR) << "axis or axises must choose one at least";
  }
  mluOpReduceOp_t mode =
      (mluOpReduceOp_t)parser_->getProtoNode()->reduce_param().mode();
  mluOpReduceIndices_t is_indices =
      (mluOpReduceIndices_t)parser_->getProtoNode()->reduce_param().is_indices();  // NOLINT
  float p = parser_->getProtoNode()->reduce_param().p();
  auto tensor_a = tensor_desc_[0].tensor;  int a_dims = tensor_a->dim;
  std::vector<int> input_dims(tensor_a->dims, tensor_a->dims + tensor_a->dim);
  if (is_indices == MLUOP_REDUCE_NO_INDICES) {
    std::vector<int> output_dims(tensor_desc_[1].tensor->dims,
                                 tensor_desc_[1].tensor->dims + tensor_desc_[1].tensor->dim);  // NOLINT
    reduceComputeValue(input_dims, output_dims, axis, mode);
  } else if (is_indices == MLUOP_REDUCE_FLATTENED_INDICES ||
             is_indices == MLUOP_REDUCE_ONLY_INDICES) {
    reduceComputeIndex(input_dims, axis, mode, is_indices);
  } else {
    LOG(ERROR) << "reduce indices do not support this indices mode";
  }
}

void ReduceExecutor::reduceComputeIndex(std::vector<int> input_dims,
                                        std::vector<int> axis,
                                        mluOpReduceOp_t mode,
                                        mluOpReduceIndices_t is_indices) {
  assert(axis.size() == 1);
  int input_sum = std::accumulate(input_dims.begin(),
                                  input_dims.end(), 1, std::multiplies<int>());
  int left = 1;
  int mid = 1;
  int right = 1;
  if (axis[0] == -1) {
    left = 1;
    mid = input_sum;
    right = 1;
  } else {
    for (size_t dim = 0; dim < input_dims.size(); ++dim) {
      if (axis[0] == dim) {
        mid *= input_dims[dim];
      } else if (axis[0] > dim) {
        left *= input_dims[dim];
      } else {
        right *= input_dims[dim];
      }
    }
  }
  for (int i = 0; i < left; i++) {
    for (int j = 0; j < right; j++) {
      float temp = 0.0;
      uint32_t index = 0;
      switch (mode) {
        case MLUOP_REDUCE_MAX: {
          temp = -1.0 / 0.0;
          for (int m = 0; m < mid; m++) {
            if (temp < cpu_fp32_input_[0][i * mid * right + m * right + j]) {
              temp = cpu_fp32_input_[0][i * mid * right + m * right + j];
              index = m;
            }
          }
          break;
        }
        case MLUOP_REDUCE_MIN: {
          temp = 1.0 / 0.0;
          for (int m = 0; m < mid; m++) {
            if (temp > cpu_fp32_input_[0][i * mid * right + m * right + j]) {
              temp = cpu_fp32_input_[0][i * mid * right + m * right + j];
              index = m;
            }
          }
          break;
        }
        case MLUOP_REDUCE_MAX_LAST_INDEX: {
          temp = -1.0 / 0.0;
          for (int m = mid - 1; m >= 0; m--) {
            if (temp < cpu_fp32_input_[0][i * mid * right + m * right + j]) {
              temp = cpu_fp32_input_[0][i * mid * right + m * right + j];
              index = m;
            }
          }
          break;
        }
        case MLUOP_REDUCE_MIN_LAST_INDEX: {
          temp = 1.0 / 0.0;
          for (int m = mid - 1; m >= 0; m--) {
            if (temp > cpu_fp32_input_[0][i * mid * right + m * right + j]) {
              temp = cpu_fp32_input_[0][i * mid * right + m * right + j];
              index = m;
            }
          }
          break;
        }
        default: {
          LOG(ERROR) << "[mluOpReduce] Do not support this op mode now";
        }
      }
      if (is_indices == MLUOP_REDUCE_FLATTENED_INDICES) {
        cpu_fp32_output_[0][i * right + j] = temp;
        cpu_fp32_output_[1][i * right + j] = index;
      } else if (is_indices == MLUOP_REDUCE_ONLY_INDICES) {
        cpu_fp32_output_[0][i * right + j] = index;
      }
    }
  }
}

void ReduceExecutor::reduceComputeValue(std::vector<int> input_dims,
                                        std::vector<int> output_dims,
                                        std::vector<int> axis,
                                        mluOpReduceOp_t mode) {
  std::sort(axis.begin(), axis.end());
  int input_sum = std::accumulate(input_dims.begin(),
                                  input_dims.end(), 1, std::multiplies<int>());
  int output_sum =
      std::accumulate(output_dims.begin(), output_dims.end(), 1,
                      std::multiplies<int>());
  std::vector<float> input_tmp(input_sum, 0.);
  for (int iter = 0; iter < input_sum; ++iter) {
    input_tmp[iter] = cpu_fp32_input_[0][iter];
  }
  int last_time_mid = 1;
  for (auto axis_iter : axis) {
    int left = 1;
    int mid = 1;
    int right = 1;
    if (axis_iter == -1) {
      left = 1;
      mid = input_sum;
      right = 1;
    } else {
      for (size_t dim = 0; dim < input_dims.size(); ++dim) {
        if (axis_iter == dim) {
          mid *= input_dims[dim];
        } else if (axis_iter > dim) {
          left *= input_dims[dim];
        } else {
          right *= input_dims[dim];
        }
      }
    }
    left /= last_time_mid;
    std::vector<float> output_tmp(right * left, 0.);
    for (int i = 0; i < left; i++) {
      for (int j = 0; j < right; j++) {
        switch (mode) {
          case MLUOP_REDUCE_ADD: {
            double sum = 0.0;
            double c = 0.0;
            for (int m = 0; m < mid; m++) {
              kahanAdd((double)(input_tmp[i * mid * right + m * right + j]), sum, c);  // NOLINT
            }
            output_tmp[i * right + j] = (float)sum;
            if (tensor_desc_[0].tensor->dtype == MLUOP_DTYPE_INT32) {
              output_tmp[i * right + j] = (int)(sum);
            } else {
              output_tmp[i * right + j] = (float)sum;
            }
            break;
          }
          case MLUOP_REDUCE_AVG: {
            double sum = 0.0;
            double c = 0.0;
            for (int m = 0; m < mid; m++) {
              kahanAdd((double)(input_tmp[i * mid * right + m * right + j]), sum, c);  // NOLINT
            }
            if (tensor_desc_[0].tensor->dtype == MLUOP_DTYPE_INT32) {
              output_tmp[i * right + j] = (int)(sum / mid);
            } else {
              output_tmp[i * right + j] = (float)(sum / mid);
            }
            break;
          }
          case MLUOP_REDUCE_MUL: {
            float result = 1.0;
            for (int m = 0; m < mid; m++) {
              result = result * input_tmp[i * mid * right + m * right + j];
            }
            if (tensor_desc_[0].tensor->dtype == MLUOP_DTYPE_INT32) {
              output_tmp[i * right + j] = (int)(result);
            } else {
              output_tmp[i * right + j] = result;
            }
            break;
          }
          case MLUOP_REDUCE_MAX: {
            float result = -1.0 / 0.0;
            for (int m = 0; m < mid; m++) {
              if (result < input_tmp[i * mid * right + m * right + j]) {
                result = input_tmp[i * mid * right + m * right + j];
              }
            }
            output_tmp[i * right + j] = result;
            break;
          }
          case MLUOP_REDUCE_MIN: {
            float result = 1.0 / 0.0;
            for (int m = 0; m < mid; m++) {
              if (result > input_tmp[i * mid * right + m * right + j]) {
                result = input_tmp[i * mid * right + m * right + j];
              }
            }
            output_tmp[i * right + j] = result;
            break;
          }
          case MLUOP_REDUCE_AND: {
            float result = 1.0;
            for (int m = 0; m < mid; m++) {
              result = result && input_tmp[(i * mid * right + m * right + j)];
            }
            output_tmp[i * right + j] = result;
            break;
          }
          case MLUOP_REDUCE_OR: {
            float result = 0.0;
            for (int m = 0; m < mid; m++) {
              result = result || input_tmp[(i * mid * right + m * right + j)];
            }
            output_tmp[i * right + j] = result;
            break;
          }
          case MLUOP_REDUCE_NORM1: {
            float result = 0.0;
            for (int m = 0; m < mid; m++) {
              result += fabs(input_tmp[i * mid * right + m * right + j]);
            }
            output_tmp[i * right + j] = result;
            break;
          }
          case MLUOP_REDUCE_NORM2: {
            float result = 0.0;
            for (int m = 0; m < mid; m++) {
              result +=
                  powf(input_tmp[i * mid * right + m * right + j], (float)2.0);
            }
            output_tmp[i * right + j] = sqrt(result);
            break;
          }
          case MLUOP_REDUCE_NORMP: {
            float result = 0.0;
            float p = parser_->getProtoNode()->reduce_param().p();
            for (int m = 0; m < mid; m++) {
              float input_data =
                  fabs(input_tmp[i * mid * right + m * right + j]);
              if (p == 0.0) {
                result += input_data > 0 ? 1 : 0;
              } else {
                result += powf(input_data, (float)p);
              }
            }
            if (p != 0.0) {
              output_tmp[i * right + j] = powf(result, (float)(1.0 / p));
            } else {
              output_tmp[i * right + j] = result;
            }
            break;
          }
          case MLUOP_REDUCE_ASUM: {
            float result = 0.0;
            for (int m = 0; m < mid; m++) {
              result += fabs(input_tmp[i * mid * right + m * right + j]);
            }
            output_tmp[i * right + j] = result;
            break;
          }
          case MLUOP_REDUCE_SUMSQ: {
            float result = 0.0;
            for (int m = 0; m < mid; m++) {
              if (axis_iter == axis[0]) {
                input_tmp[i * mid * right + m * right + j] =
                  powf(input_tmp[i * mid * right + m * right + j], (float)2.0);
              }
              result += input_tmp[i * mid * right + m * right + j];
            }
            output_tmp[i * right + j] = result;
            break;
          }
          default: {
            LOG(ERROR) << "[mluOpReduce] Do not support this op mode now";
          }
        }
      }
    }
    input_tmp = output_tmp;
    last_time_mid *= mid;
    if (axis_iter == axis[axis.size() - 1]) {
      for (int iter = 0; iter < output_sum; ++iter) {
        float alpha = parser_->getProtoNode()->reduce_param().alpha();
        float beta = parser_->getProtoNode()->reduce_param().beta();
        cpu_fp32_output_[0][iter] = output_tmp[iter]  * alpha + beta;
      }
    }
  }
}

int64_t ReduceExecutor::getTheoryOps() {
  mluOpReduceOp_t mode =
      (mluOpReduceOp_t)parser_->getProtoNode()->reduce_param().mode();
  int64_t cp_count = 0;
  switch (mode) {
    case MLUOP_REDUCE_MAX:
    case MLUOP_REDUCE_MIN:
    case MLUOP_REDUCE_ADD:
    case MLUOP_REDUCE_ASUM:
    case MLUOP_REDUCE_SUMSQ:
    case MLUOP_REDUCE_MUL:
    case MLUOP_REDUCE_OR:
    case MLUOP_REDUCE_AND:
    case MLUOP_REDUCE_AVG:
    case MLUOP_REDUCE_MAX_LAST_INDEX:
    case MLUOP_REDUCE_MIN_LAST_INDEX: {
      cp_count = 1;
      break;
    }
    case MLUOP_REDUCE_NORM1:
    case MLUOP_REDUCE_NORM2:
    case MLUOP_REDUCE_NORMP: {
      cp_count = 2;
      break;
    }
    default: { VLOG(4) << "Reduce param wrong!"; }
  }
  int64_t theory_ops = parser_->getInputDataCount(0) * cp_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

std::set<Evaluator::Formula> ReduceExecutor::getCriterionsUse() const {
  auto platform = exe_context_->handle->arch;
  mluOpReduceOp_t mode =
      (mluOpReduceOp_t)parser_->getProtoNode()->reduce_param().mode();
  if (platform == MLUOP_MLU220 ||
      platform == MLUOP_MLU270 ||
      platform == MLUOP_MLU290) {
    if (mode == MLUOP_REDUCE_ADD ||
        mode == MLUOP_REDUCE_AVG ||
        mode == MLUOP_REDUCE_MUL) {
      return { Evaluator::DIFF1, Evaluator::DIFF2 };
    }
  }
  return Executor::getCriterionsUse();
}
}  // namespace mluoptest
