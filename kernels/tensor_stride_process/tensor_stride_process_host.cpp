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
#include "kernels/tensor_stride_process/tensor_stride_process_host.h"

#include <stdarg.h>

#include <algorithm>
#include <vector>

#include "kernels/tensor_stride_process/tensor_stride_process_mlu.h"
#include "kernels/utils/cnnl_helper.h"

using std::vector;

namespace mluop {
// Tensor may be a stride case, but if the stride of a tensor is dense,
// and all tensors are consistent dense,
// they can be processed with common default stride method with a high
// performance in element-wise kernel. Also, if one of the input is scalar, it
// will be regarded as a broadcasting tensor with the same shape and strides as
// other tensors.
//
// For example, the following 2 patterns will return false in this function:
//
// 1. input.shape = [2, 3], input.stride = [1, 3]   (dense & consistent)
//    output.shape = [2, 3], output.stride = [1, 3] (dense & consistent)
//
// 2. input1.shape = [2, 3], input1.stride = [1, 3] (dense & consistent)
//    input2.shape = [1]                            (scalar)
//    output.shape = [2, 3], output.stride = [1, 3] (dense & consistent)

bool strideCaseWithNotConsistentDense(int tensor_num, ...) {
  bool is_stride_case = false;
  mluOpTensorDescriptor_t first_stride_tensor = NULL;
  va_list ap;
  va_start(ap, tensor_num);
  // if one of a tensor is strided, set is_fake_stride_case t101.rue.
  // scalar is excluded.
  for (auto i = 0; i < tensor_num; i++) {
    const mluOpTensorDescriptor_t this_tensor =
        va_arg(ap, mluOpTensorDescriptor_t);
    if (ifNeedTensorStrideProcess(this_tensor)) {
      is_stride_case = true;
      first_stride_tensor = this_tensor;
      break;
    }
  }

  if (!is_stride_case) {
    va_end(ap);
    return false;
  } else {
    va_start(ap, tensor_num);
    if (!isDenseStrideTensor(first_stride_tensor)) {
      va_end(ap);
      return true;
    }
    int64_t first_dims[MLUOP_DIM_MAX] = {0};
    int64_t first_stride[MLUOP_DIM_MAX] = {0};
    auto first_dim = first_stride_tensor->dim;
    // get first stride tensor's shape and stride info
    for (auto i = 0; i < first_dim; i++) {
      first_dims[i] = first_stride_tensor->dims[i];
      first_stride[i] = first_stride_tensor->strides[i];
    }
    // judge whether shapes and strides of tensors are same,
    // if not, need stride process
    // note: we should ignore the dim if the shape at this dim is 1.
    for (auto i = 0; i < tensor_num; i++) {
      const mluOpTensorDescriptor_t this_tensor =
          va_arg(ap, mluOpTensorDescriptor_t);
      if (mluOpGetTensorElementNum(this_tensor) == 1) {
        // ignore scalar
        continue;
      }
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

bool isDenseStrideTensor(const mluOpTensorDescriptor_t tensor_desc) {
  int tensor_dim = tensor_desc->dim;
  std::vector<int64_t> dims;
  std::vector<int64_t> strides;
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

// Check if tensor need stride process.
bool ifNeedTensorStrideProcess(const mluOpTensorDescriptor_t tensor_desc) {
  bool needStrideProcess = false;
  int tensor_dim = tensor_desc->dim;
  int64_t stride_base = 1;
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

// Check if stride out is 021 trans and dimension 1 or 2 pad
// for stride in, the operation is crop actually
// dims_ptr != nullptr will fill tensor_shape with merged stride and dim
bool isTransPadStride(TensorShape &tensor_shape, int64_t *dims_ptr,
                      int64_t *strides_ptr) {
  // get valid dims and merging dims
  vector<int64_t> dims;
  vector<int64_t> strides;
  int begin = 0;
  while (begin < MLUOP_DIM_MAX) {
    // skip the leading 1
    if (tensor_shape.tensor_dims[begin] != 1) {
      // start scanning on following dims and stride
      int dim = tensor_shape.tensor_dims[begin];
      int stride = tensor_shape.tensor_strides[begin];
      while (begin + 1 < MLUOP_DIM_MAX) {
        // means can be merged
        if (tensor_shape.tensor_strides[begin] ==
            (tensor_shape.tensor_strides[begin + 1] *
             tensor_shape.tensor_dims[begin + 1])) {
          dim *= tensor_shape.tensor_dims[begin + 1];
          stride = tensor_shape.tensor_strides[begin + 1];
          begin++;
        } else {
          break;
        }
      }
      // fillin the merged dim and stride
      dims.push_back(dim);
      strides.push_back(stride);
    }
    begin++;
  }
  // only handle three dimension of which notation is nhw
  if (dims.size() != 3) {
    return false;
  }
  // the stride of h == 1 means transpose
  bool is_trans = false;
  if (strides[1] == 1) {
    is_trans = true;
  }
  // there are two kinds of pad: pad ho or pad wo
  bool is_pad = false;
  // if pad ho, wo should be equal to hi
  if (strides[2] == dims[1] && (strides[0] % strides[2]) == 0 &&
      (strides[0] / strides[2]) > dims[2]) {
    is_pad = true;
  }
  // if pad wo, ho should be equal to wi
  if (strides[2] > dims[1] && strides[0] == (dims[2] * strides[2])) {
    is_pad = true;
  }
  // return merges dims and strides
  if (is_trans && is_pad) {
    if (dims_ptr != nullptr) {
      for (int i = 0; i < dims.size(); i++) {
        dims_ptr[i] = dims[i];
      }
    }
    if (strides_ptr != nullptr) {
      for (int i = 0; i < strides.size(); i++) {
        strides_ptr[i] = strides[i];
      }
    }
  }
  return is_trans && is_pad;
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
  int64_t tensor_dims[MLUOP_DIM_MAX];
  int64_t tensor_strides[MLUOP_DIM_MAX];
  int64_t tensor_temp_dims[MLUOP_DIM_MAX];
  int64_t tensor_temp_strides[MLUOP_DIM_MAX];
  uint64_t total_num = 1;
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
  tensor_shape->total_stride = shapeStrideCount(tensor_desc);
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
                          int64_t *target_shape, int target_dim,
                          TensorShape *tensor_shape) {
  tensor_shape->is_contiguous = false;
  int tensor_dim = target_dim;
  int64_t tensor_dims[MLUOP_DIM_MAX];
  int64_t tensor_strides[MLUOP_DIM_MAX];
  int64_t tensor_temp_dims[MLUOP_DIM_MAX];
  int64_t tensor_temp_strides[MLUOP_DIM_MAX];
  uint64_t total_num = 1;
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
  // shape expand, but stride won't grow up, can use tensor_desc as usual.
  tensor_shape->total_stride = shapeStrideCount(tensor_desc);
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

}  // namespace mluop

// Policy function
static mluOpStatus_t policyFunc(mluOpHandle_t handle, cnrtDim3_t *k_dim,
                                cnrtFunctionType_t *k_type,
                                uint64_t total_num) {
  if (handle->sram_size <= 0) {
    *k_type = CNRT_FUNC_TYPE_BLOCK;
  } else {
    *k_type = CNRT_FUNC_TYPE_UNION1;
  }
  uint32_t union_number = mluop::runtime::getClusterLimitCapability(handle);

  // Split to different cores according to total_num.
  uint64_t need_cluster = total_num / CORE_DIM;
  need_cluster = (need_cluster == 0) ? 1 : need_cluster;
  k_dim->x = CORE_DIM;
  k_dim->y = need_cluster > union_number ? union_number : need_cluster;
  k_dim->z = 1;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpTensorStrideIn(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, void *output) {
  LOG_FIRST_N(WARNING, 1) << "[mluOpTensorStrideIn] is deprecated and will be "
                             "removed in the future release, "
                          << "please use [mluOpContiguous] instead.";

  if (handle->arch < MLUOP_MLU590) {
    uint64_t num_with_stride = shapeStrideCount(input_desc);
    TENSOR_NUM_CHECK("[mluOpTensorStrideIn]", num_with_stride, LARGE_TENSOR_NUM,
                     "input tensor num with stride is too large. ");
  }

  mluop::TensorShape input_shape;
  mluop::getTensorShape(input_desc, &input_shape);
  mluOpDataType_t data_type = input_desc->dtype;
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  policyFunc(handle, &k_dim, &k_type, input_shape.total_num);

  VLOG(5) << "Launch Kernel MLUUnionKernelTensorStrideIn<<<Union"
          << k_type / CORE_DIM << ", " << k_dim.x << ", " << k_dim.y << ", "
          << k_dim.z << ">>>";
  CHECK_RETURN("[mluOpTensorStrideIn]",
               KernelTensorStrideIn(k_dim, k_type, handle->queue, input,
                                    input_shape, output, data_type));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpTensorStrideOut(
    mluOpHandle_t handle, const mluOpTensorDescriptor_t input_desc,
    const void *input, void *output) {
  mluop::TensorShape output_shape;
  mluop::getTensorShape(input_desc, &output_shape);

  if (handle->arch < MLUOP_MLU590) {
    uint64_t num_with_stride = shapeStrideCount(input_desc);
    TENSOR_NUM_CHECK("[mluOpTensorStrideOut]", num_with_stride,
                     LARGE_TENSOR_NUM,
                     "input tensor num with stride is too large. ");
  }

  mluOpDataType_t data_type = input_desc->dtype;
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;

  policyFunc(handle, &k_dim, &k_type, output_shape.total_num);

  VLOG(5) << "Launch Kernel MLUUnionKernelTensorStrideOut<<<Union"
          << k_type / CORE_DIM << ", " << k_dim.x << ", " << k_dim.y << ", "
          << k_dim.z << ">>>";
  CHECK_RETURN("[mluOpTensorStrideOut]",
               KernelTensorStrideOut(k_dim, k_type, handle->queue, input,
                                     output, output_shape, data_type));
  return MLUOP_STATUS_SUCCESS;
}

static vector<int64_t> getDefaultStride(int64_t *dims, int dim) {
  vector<int64_t> default_stride(dim, 1);
  int64_t temp = 1;
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
  mluOpSetTensorDescriptorEx_v2(temp_desc, input_desc->layout,
                                input_desc->dtype, input_desc->dim,
                                input_desc->dims, default_stride.data());
  DEFINE_CREATE_AND_SET_CNNL_HANDLE(handle, cnnl_handle);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(input_desc, cnnl_input_desc);
  DEFINE_CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(temp_desc, cnnl_temp_desc);
  CALL_CNNL(
      cnnlCopy(cnnl_handle, cnnl_input_desc, input, cnnl_temp_desc, output));
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_input_desc);
  DESTROY_CNNL_TENSOR_DESCRIPTOR(cnnl_temp_desc);
  DESTROY_CNNL_HANDLE(cnnl_handle);
  mluOpDestroyTensorDescriptor(temp_desc);
  return MLUOP_STATUS_SUCCESS;
}
