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
#include <stdarg.h>
#include <algorithm>
#include <vector>

#include "core/context.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"
#include "tensor_stride_process_mlu.h"
#include "tensor_stride_process.h"

using std::vector;

// Check if tensor need stride process.
bool ifNeedTensorStrideProcess(const mluOpTensorDescriptor_t tensor_desc) {
  bool needStrideProcess = false;
  int tensor_dim = tensor_desc->dim;
  int stride_base = 1;
  for (int i = tensor_dim - 1; i >= 0; i--) {
    if (tensor_desc->dims[i] != 1) {
      if (tensor_desc->strides[i] == stride_base) {
        stride_base *= tensor_desc->dims[i];
      } else {
        needStrideProcess = true;
        break;
      }
    }
  }
  return needStrideProcess;
}

bool isDenseStrideTensor(const mluOpTensorDescriptor_t tensor_desc) {
  int tensor_dim = tensor_desc->dim;
  std::vector<int> dims;
  std::vector<int> strides;
  std::vector<int> perm;
  for (int i = 0; i < tensor_dim; i++) {
    dims.emplace_back(tensor_desc->dims[i]);
    strides.emplace_back(tensor_desc->strides[i]);
    perm.emplace_back(i);
  }

  if (tensor_dim == 1) {
    return dims[0] < 2 || strides[0] == 1;
  }
  std::sort(perm.begin(), perm.end(), [&](int a, int b) {
    if (dims[a] < 2) {
      return false;
    } else if (dims[b] < 2) {
      return true;
    }
    return strides[a] < strides[b];
  });

  auto require_stride = 1;
  for (auto i = 0; i < tensor_dim; i++) {
    const auto size_perm_i = dims[perm[i]];
    if (size_perm_i < 2) {
      return true;
    }
    if (strides[perm[i]] != require_stride) {
      return false;
    }
    require_stride *= size_perm_i;
  }
  return true;
}

// Tensor may be a stride case, but if the stride of a tensor is dense,
// and all tensors are consistent dense,
// they can be processd with common default stride method with a high
// performance in element-wise kernel.
bool strideCaseWithNotConsistentDense(int tensor_num, ...) {
  bool is_stride_case = false;
  va_list ap;
  va_start(ap, tensor_num);
  // if one of a tensor is strided, set is_fake_stride_case true.
  for (auto i = 0; i < tensor_num; i++) {
    mluOpTensorDescriptor_t this_tensor = va_arg(ap, mluOpTensorDescriptor_t);
    if (ifNeedTensorStrideProcess(this_tensor)) {
      is_stride_case = true;
      break;
    }
  }
  if (!is_stride_case) {
    va_end(ap);
    return false;
  } else {
    va_start(ap, tensor_num);
    mluOpTensorDescriptor_t first_tensor = va_arg(ap, mluOpTensorDescriptor_t);
    if (!isDenseStrideTensor(first_tensor)) {
      va_end(ap);
      return true;
    }
    int first_dims[MLUOP_DIM_MAX] = {0};
    int first_stride[MLUOP_DIM_MAX] = {0};
    auto first_dim = first_tensor->dim;
    // get first tensor's shape and stride info
    for (auto i = 0; i < first_dim; i++) {
      first_dims[i] = first_tensor->dims[i];
      first_stride[i] = first_tensor->strides[i];
    }
    // judge whether shapes and strides of tensors are same,
    // if not, need stride process
    // note: we should ignore the dim if the shape at this dim is 1.
    for (auto i = 0; i < tensor_num - 1; i++) {
      mluOpTensorDescriptor_t this_tensor = va_arg(ap, mluOpTensorDescriptor_t);
      if (this_tensor->dim != first_dim) {
        va_end(ap);
        return true;
      }
      for (auto j = 0; j < first_dim; j++) {
        if (first_dims[j] != this_tensor->dims[j] ||
            ((first_dims[j] != 1) &&
             (first_stride[j] != this_tensor->strides[j]))) {
          va_end(ap);
          return true;
        }
      }
    }
  }
  va_end(ap);
  return false;
}

// From tensor_desc get tensor's dims and strides.
void getTensorShape(const mluOpTensorDescriptor_t tensor_desc,
                    TensorShape *tensor_shape) {
  if (!ifNeedTensorStrideProcess(tensor_desc)) {
    tensor_shape->is_contiguous = true;
  } else {
    tensor_shape->is_contiguous = false;
  }
  int tensor_dim = tensor_desc->dim;
  int tensor_dims[MLUOP_DIM_MAX];
  int tensor_strides[MLUOP_DIM_MAX];
  int tensor_temp_dims[MLUOP_DIM_MAX];
  int tensor_temp_strides[MLUOP_DIM_MAX];
  int total_num = 1;
  // dims:    (2, 3) -> (1, 1, 1, 1, 1, 1, 2, 3)
  // strides: (2, 3) -> (0, 0, 0, 0, 0, 0, 2, 3)
  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    if (i < MLUOP_DIM_MAX - tensor_dim) {
      tensor_dims[i] = 1;
      tensor_strides[i] = 1;
    } else {
      total_num *= tensor_desc->dims[i + tensor_dim - MLUOP_DIM_MAX];
      tensor_dims[i] = tensor_desc->dims[i + tensor_dim - MLUOP_DIM_MAX];
      tensor_strides[i] = tensor_desc->strides[i + tensor_dim - MLUOP_DIM_MAX];
    }
  }
  tensor_shape->total_num = total_num;
  // dims:    (1, 1, 1, 1, 2, 1, 1, 3) -> (1, 1, 1, 1, 1, 1, 2, 3)
  // strides: (0, 0, 0, 0, 2, 4, 5, 3) -> (0, 0, 0, 0, 0, 0, 2, 3)
  for (int i = MLUOP_DIM_MAX - 1, j = MLUOP_DIM_MAX - 1; j >= 0; --i) {
    if (i < 0) {
      tensor_temp_dims[j] = 1;
      tensor_temp_strides[j] = 0;
      --j;
    } else if (tensor_dims[i] != 1) {
      tensor_temp_dims[j] = tensor_dims[i];
      tensor_temp_strides[j] = tensor_strides[i];
      --j;
    }
  }
  // dims:    (1, 1, 1, 1,   2,  3, 4, 5) -> (1, 1, 1, 1, 1, 1,   2, 60)
  // strides: (0, 0, 0, 0, 500, 20, 5, 1) -> (0, 0, 0, 0, 0, 0, 500,  1)
  int offset = 0;
  for (int i = MLUOP_DIM_MAX - 1; i > 0; i--) {
    if (tensor_temp_strides[i] == 1 &&
        tensor_temp_strides[i - 1] == tensor_temp_dims[i]) {
      tensor_temp_dims[i - 1] *= tensor_temp_dims[i];
      tensor_temp_strides[i - 1] = 1;
      offset++;
    } else {
      break;
    }
  }
  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    if (i < offset) {
      tensor_shape->tensor_dims[i] = 1;
      tensor_shape->tensor_strides[i] = 0;
    } else {
      tensor_shape->tensor_dims[i] = tensor_temp_dims[i - offset];
      tensor_shape->tensor_strides[i] = tensor_temp_strides[i - offset];
    }
  }
}

// From tensor_desc and target_shape get the soft expand tensor's dims and
// strides. attention: this function will not check the legal of target expand
// shape from original shape. input: tensor_desc: the original tensor
// descriptor; target_shape: the traget expand shape; target_dim: the target
// expand dimension; output: tensor_shape: used for stride input and output
// kernel.
void getExpandTensorShape(const mluOpTensorDescriptor_t tensor_desc,
                          int *target_shape, int target_dim,
                          TensorShape *tensor_shape) {
  tensor_shape->is_contiguous = false;
  int tensor_dim = target_dim;
  int tensor_dims[MLUOP_DIM_MAX];
  int tensor_strides[MLUOP_DIM_MAX];
  int tensor_temp_dims[MLUOP_DIM_MAX];
  int tensor_temp_strides[MLUOP_DIM_MAX];
  int total_num = 1;
  // target_shape:      (7, 3, 4, 5)
  // tensor_desc_shape:    (3, 1, 5)
  // tensor_desc_stride:   (s1, s2, s3)
  // dims:    (1, 1, 1, 1, 7, 3, 4, 5)
  // strides: (0, 0, 0, 0, 0, s1, 0, s3)
  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    tensor_strides[i] = 0;
    if (i < MLUOP_DIM_MAX - tensor_dim) {  // add 1 at high dim.
      tensor_dims[i] = 1;
    } else {
      total_num *= target_shape[i + tensor_dim - MLUOP_DIM_MAX];
      tensor_dims[i] =
          target_shape[i + tensor_dim - MLUOP_DIM_MAX];  // set shape
      if (i >=
          MLUOP_DIM_MAX -
              tensor_desc->dim) {  // set stride if tensor_desc has this stride
        if (tensor_desc->dims[i + tensor_desc->dim - MLUOP_DIM_MAX] != 1) {
          tensor_strides[i] =
              tensor_desc->strides[i + tensor_desc->dim - MLUOP_DIM_MAX];
        }
      }
    }
  }
  tensor_shape->total_num = total_num;
  // dims:    (1, 1, 1, 1, 2, 1, 1, 3) -> (1, 1, 1, 1, 1, 1, 2, 3)
  // strides: (0, 0, 0, 0, 2, 4, 5, 3) -> (0, 0, 0, 0, 0, 0, 2, 3)
  for (int i = MLUOP_DIM_MAX - 1, j = MLUOP_DIM_MAX - 1; j >= 0; --i) {
    if (i < 0) {
      tensor_temp_dims[j] = 1;
      tensor_temp_strides[j] = 0;
      --j;
    } else if (tensor_dims[i] != 1) {
      tensor_temp_dims[j] = tensor_dims[i];
      tensor_temp_strides[j] = tensor_strides[i];
      --j;
    }
  }
  // dims:    (1, 1, 1, 1,   2,  3, 4, 5) -> (1, 1, 1, 1, 1, 1,   2, 60)
  // strides: (0, 0, 0, 0, 500, 20, 5, 1) -> (0, 0, 0, 0, 0, 0, 500,  1)
  int offset = 0;
  for (int i = MLUOP_DIM_MAX - 1; i > 0; i--) {
    if (tensor_temp_strides[i] == 1 &&
        tensor_temp_strides[i - 1] == tensor_temp_dims[i]) {
      tensor_temp_dims[i - 1] *= tensor_temp_dims[i];
      tensor_temp_strides[i - 1] = 1;
      offset++;
    } else {
      break;
    }
  }
  for (int i = 0; i < MLUOP_DIM_MAX; i++) {
    if (i < offset) {
      tensor_shape->tensor_dims[i] = 1;
      tensor_shape->tensor_strides[i] = 0;
    } else {
      tensor_shape->tensor_dims[i] = tensor_temp_dims[i - offset];
      tensor_shape->tensor_strides[i] = tensor_temp_strides[i - offset];
    }
  }
}

// Policy function
static mluOpStatus_t policyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                                cnrtFunctionType_t *k_type, int total_num) {
  *k_type = CNRT_FUNC_TYPE_UNION1;
  uint32_t union_number = mluop::runtime::getClusterLimitCapability(handle);

  // Split to different cores according to total_num.
  int need_cluster = total_num / CORE_DIM;
  need_cluster = (need_cluster == 0) ? 1 : need_cluster;
  k_dim->x = CORE_DIM;
  k_dim->y = need_cluster > union_number ? union_number : need_cluster;
  k_dim->z = 1;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpTensorStrideIn(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, void *output) {
  TensorShape input_shape;
  getTensorShape(input_desc, &input_shape);
  mluOpDataType_t data_type = input_desc->dtype;
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  policyFunc(handle, &k_dim, &k_type, input_shape.total_num);

  VLOG(5) << "Launch Kernel mluOpUnion1KernelTensorStrideIn<<<Union"
          << k_type / CORE_DIM << ", " << k_dim.x << ", " << k_dim.y << ", "
          << k_dim.z << ">>>";
  KERNEL_CHECK((mluOpUnion1KernelTensorStrideIn(
      k_dim, k_type, handle->queue, input, input_shape, output, data_type)));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpTensorStrideOut(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, void *output) {
  TensorShape output_shape;
  getTensorShape(input_desc, &output_shape);

  mluOpDataType_t data_type = input_desc->dtype;
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  policyFunc(handle, &k_dim, &k_type, output_shape.total_num);

  VLOG(5) << "Launch Kernel mluOpUnion1KernelTensorStrideOut<<<Union"
          << k_type / CORE_DIM << ", " << k_dim.x << ", " << k_dim.y << ", "
          << k_dim.z << ">>>";
  KERNEL_CHECK((mluOpUnion1KernelTensorStrideOut(
      k_dim, k_type, handle->queue, input, output, output_shape, data_type)));
  return MLUOP_STATUS_SUCCESS;
}

static vector<int> getDefaultStride(int *dims, int dim) {
  vector<int> default_stride(dim, 1);
  int temp = 1;
  for (int i = 0; i < dim; i++) {
    int offset = dim - 1 - i;
    default_stride[offset] = temp;
    temp *= dims[offset];
  }
  return default_stride;
}

mluOpStatus_t MLUOP_WIN_API
mluOpContiguous(mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
                const void *input, void *output) {
  auto default_stride = getDefaultStride(input_desc->dims, input_desc->dim);
  mluOpTensorDescriptor_t temp_desc = nullptr;
  mluOpCreateTensorDescriptor(&temp_desc);
  mluOpSetTensorDescriptorEx(temp_desc, input_desc->layout, input_desc->dtype,
                             input_desc->dim, input_desc->dims,
                             default_stride.data());
  auto status_copy = mluOpCopy(handle, input_desc, input, temp_desc, output);
  if (status_copy != MLUOP_STATUS_SUCCESS) {
    KERNEL_CALL_CHECK("mluOpContiguous", "mluOpCopy", status_copy, "");
  }
  mluOpDestroyTensorDescriptor(temp_desc);
  return MLUOP_STATUS_SUCCESS;
}
