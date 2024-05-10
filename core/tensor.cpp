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
#include <iomanip>
#include <algorithm>

#include "core/tensor.h"
#include "core/logging.h"
#include "core/type.h"

#define SET_PARAM_FOR_POINTER(ptr, val) \
  if (ptr != nullptr) {                 \
    *ptr = val;                         \
  }

#define SET_ARRAY_PARAM_FOR_POINTER(ptr, arr, num) \
  if (ptr != nullptr) {                            \
    for (int i = 0; i < num; ++i) {                \
      ptr[i] = arr[i];                             \
    }                                              \
  }

/* mluOpTensorStruct */

mluOpStatus_t mluOpTensorStruct::tensorDimN(size_t &dim_out) {
  size_t index;
  switch (layout) {
    case MLUOP_LAYOUT_NCHW:
    case MLUOP_LAYOUT_NHWC:
    case MLUOP_LAYOUT_NDHWC:
      index = 0;
      break;
    case MLUOP_LAYOUT_HWCN:
      index = 3;
      break;
    default:
      LOG(ERROR) << "mluOpTensorStruct::tensorDimN, "
                 << "illegal layout in descriptor: " << layout;
      return MLUOP_STATUS_BAD_PARAM;
  }
  if (index > dim) {
    LOG(ERROR) << "mluOpTensorStruct::tensorDimN, "
               << "mismatch layout and dimension. layout: " << layout;
    return MLUOP_STATUS_NOT_INITIALIZED;
  }
  dim_out = dims[index];
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpTensorStruct::tensorDimC(size_t &dim_out) {
  size_t index;
  switch (layout) {
    case MLUOP_LAYOUT_NCHW:
      index = 1;
      break;
    case MLUOP_LAYOUT_NHWC:
      index = 3;
      break;
    case MLUOP_LAYOUT_NDHWC:
      index = 4;
      break;
    case MLUOP_LAYOUT_HWCN:
      index = 2;
      break;
    default:
      LOG(ERROR) << "mluOpTensorStruct::tensorDimC, "
                 << "illegal layout in descriptor: " << layout;
      return MLUOP_STATUS_BAD_PARAM;
  }
  if (index > dim) {
    LOG(ERROR) << "mluOpTensorStruct::tensorDimC, "
               << "mismatch layout and dimension. layout: " << layout;
    return MLUOP_STATUS_NOT_INITIALIZED;
  }
  dim_out = dims[index];
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpTensorStruct::tensorDimH(size_t &dim_out) {
  size_t index;
  switch (layout) {
    case MLUOP_LAYOUT_NCHW:
      index = 2;
      break;
    case MLUOP_LAYOUT_NHWC:
      index = 1;
      break;
    case MLUOP_LAYOUT_NDHWC:
      index = 2;
      break;
    case MLUOP_LAYOUT_HWCN:
      index = 0;
      break;
    default:
      LOG(ERROR) << "mluOpTensorStruct::tensorDimH, "
                 << "illegal layout in descriptor: " << layout;
      return MLUOP_STATUS_BAD_PARAM;
  }
  if (index > dim) {
    LOG(ERROR) << "mluOpTensorStruct::tensorDimH, "
               << "mismatch layout and dimension. layout: " << layout;
    return MLUOP_STATUS_NOT_INITIALIZED;
  }
  dim_out = dims[index];
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpTensorStruct::tensorDimW(size_t &dim_out) {
  size_t index;
  switch (layout) {
    case MLUOP_LAYOUT_NCHW:
      index = 3;
      break;
    case MLUOP_LAYOUT_NHWC:
      index = 2;
      break;
    case MLUOP_LAYOUT_NDHWC:
      index = 3;
      break;
    case MLUOP_LAYOUT_HWCN:
      index = 1;
      break;
    default:
      LOG(ERROR) << "mluOpTensorStruct::tensorDimW, "
                 << "illegal layout in descriptor: " << layout;
      return MLUOP_STATUS_BAD_PARAM;
  }
  if (index > dim) {
    LOG(ERROR) << "mluOpTensorStruct::tensorDimW, "
               << "mismatch layout and dimension. layout: " << layout;
    return MLUOP_STATUS_NOT_INITIALIZED;
  }
  dim_out = dims[index];
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetSizeOfDataType(mluOpDataType_t data_type,
                                                   size_t *size) {
  PARAM_CHECK("[mluOpGetSizeOfDataType]", size != NULL);

  if (MLUOP_DTYPE_INVALID != data_type) {
    *size = mluop::getSizeOfDataType(data_type);
    return MLUOP_STATUS_SUCCESS;
  } else {
    LOG(ERROR) << "[mluOpGetSizeOfDataType]:data_type should not be "
                  "MLUOP_DTYPE_INVALID. ";
    return MLUOP_STATUS_BAD_PARAM;
  }
}

mluOpStatus_t MLUOP_WIN_API
mluOpCreateSeqDataDescriptor(mluOpSeqDataDescriptor_t *seq_data_desc) {
  PARAM_CHECK("[mluOpCreateSeqDataDescriptor]", seq_data_desc != NULL);
  mluOpSeqDataStruct *ts = new (std::nothrow) mluOpSeqDataStruct();
  *seq_data_desc = ts;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetSeqDataDescriptorBase(
    mluOpSeqDataDescriptor_t seq_data_desc, mluOpSeqDataLayout_t layout,
    mluOpDataType_t dtype, int dimNb, const void *dimSize,
    int seqLengthArraySize, const void *seqLengthArray, void *paddingFill) {
  PARAM_CHECK("[mluOpSetSeqDataDescriptor]", seq_data_desc != NULL);
  PARAM_CHECK("[mluOpSetSeqDataDescriptor]", dimSize != NULL);
  PARAM_CHECK_GE("[mluOpSetSeqDataDescriptor]", layout, 0);
  PARAM_CHECK_GE("[mluOpSetSeqDataDescriptor]", dtype, 0);
  PARAM_CHECK_GT("[mluOpSetSeqDataDescriptor]", dimNb, 0);
  PARAM_CHECK_GE("[mluOpSetSeqDataDescriptor]", seqLengthArraySize, 0);

  seq_data_desc->dim = dimNb;
  seq_data_desc->layout = layout;
  seq_data_desc->dtype = dtype;
  seq_data_desc->seq_length_size = seqLengthArraySize;
  seq_data_desc->padding_fill = paddingFill;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetSeqDataDescriptor(
    mluOpSeqDataDescriptor_t seq_data_desc, mluOpSeqDataLayout_t layout,
    mluOpDataType_t dtype, int dimNb, const int *dimSize,
    int seqLengthArraySize, const int *seqLengthArray, void *paddingFill) {
  CHECK_RETURN("[mluOpSetSeqDataDescriptor]",
               mluOpSetSeqDataDescriptorBase(
                   seq_data_desc, layout, dtype, dimNb, (void *)dimSize,
                   seqLengthArraySize, (void *)seqLengthArray, paddingFill));

  seq_data_desc->dims.clear();
  for (int i = 0; i < dimNb; ++i) {
    PARAM_CHECK_GE("[mluOpSetSeqDataDescriptor]", dimSize[i], 0);
    seq_data_desc->dims.push_back(static_cast<int64_t>(dimSize[i]));
  }
  seq_data_desc->seq_length.clear();
  if (seqLengthArray != nullptr) {
    for (int i = 0; i < seqLengthArraySize; ++i) {
      PARAM_CHECK_GT("[mluOpSetSeqDataDescriptor]", seqLengthArray[i], 0);
      seq_data_desc->seq_length.push_back(
          static_cast<int64_t>(seqLengthArray[i]));
    }
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetSeqDataDescriptor_v2(
    mluOpSeqDataDescriptor_t seq_data_desc, mluOpSeqDataLayout_t layout,
    mluOpDataType_t dtype, int dimNb, const int64_t *dimSize,
    int seqLengthArraySize, const int *seqLengthArray, void *paddingFill) {
  CHECK_RETURN("[mluOpSetSeqDataDescriptor_v2]",
               mluOpSetSeqDataDescriptorBase(
                   seq_data_desc, layout, dtype, dimNb, (void *)dimSize,
                   seqLengthArraySize, (void *)seqLengthArray, paddingFill));

  seq_data_desc->dims.clear();
  for (int i = 0; i < dimNb; ++i) {
    PARAM_CHECK_GE("[mluOpSetSeqDataDescriptor_v2]", dimSize[i], 0);
    seq_data_desc->dims.push_back(dimSize[i]);
  }
  seq_data_desc->seq_length.clear();
  if (seqLengthArray != nullptr) {
    for (int i = 0; i < seqLengthArraySize; ++i) {
      PARAM_CHECK_GT("[mluOpSetSeqDataDescriptor_v2]", seqLengthArray[i], 0);
      seq_data_desc->seq_length.push_back(seqLengthArray[i]);
    }
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetSeqDataDescriptor_v2(
    const mluOpSeqDataDescriptor_t seq_data_desc, mluOpSeqDataLayout_t *layout,
    mluOpDataType_t *dtype, int *dimNb, int64_t *dimSize,
    int64_t *seqLengthArraySize, int64_t *seqLengthArray, void *paddingFill) {
  PARAM_CHECK_NE("[mluOpGetSeqDataDescriptor]", seq_data_desc, NULL);

  SET_PARAM_FOR_POINTER(layout, seq_data_desc->layout);
  SET_PARAM_FOR_POINTER(dtype, seq_data_desc->dtype);
  SET_PARAM_FOR_POINTER(dimNb, seq_data_desc->dim);
  SET_ARRAY_PARAM_FOR_POINTER(dimSize, seq_data_desc->dims, seq_data_desc->dim);
  SET_PARAM_FOR_POINTER(seqLengthArraySize, seq_data_desc->seq_length_size);
  SET_ARRAY_PARAM_FOR_POINTER(seqLengthArray, seq_data_desc->seq_length,
                              seq_data_desc->seq_length_size);

  if (paddingFill != nullptr && seq_data_desc->padding_fill != nullptr) {
    int size_pd = mluop::getSizeOfDataType(*dtype);
    std::memcpy(paddingFill, seq_data_desc->padding_fill, size_pd);
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetSeqDataDescriptorPositionAndScale(
    mluOpSeqDataDescriptor_t desc, int position, float scale) {
  PARAM_CHECK("[mluOpSetSeqDataDescriptorPositionAndScale]", desc != NULL);

  desc->position = position;
  desc->scale = scale;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetSeqDataDescriptorPositionAndScale(
    const mluOpSeqDataDescriptor_t desc, int *position, float *scale) {
  PARAM_CHECK_NE("[mluOpGetSeqDataDescriptorPositionAndScale]", desc, NULL);
  PARAM_CHECK_NE("[mluOpGetSeqDataDescriptorPositionAndScale]", position, NULL);
  PARAM_CHECK_NE("[mluOpGetSeqDataDescriptorPositionAndScale]", scale, NULL);

  *position = desc->position;
  *scale = desc->scale;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpDestroySeqDataDescriptor(mluOpSeqDataDescriptor_t seq_data_desc) {
  PARAM_CHECK_NE("[mluOpDestroySeqDataDescriptor]", seq_data_desc, NULL);

  delete seq_data_desc;

  return MLUOP_STATUS_SUCCESS;
}

#if MLUOP_TENSOR_QUEUE_ENABLE
static mluOpTensorDescriptorQueueStruct *queue_array = NULL;
static std::hash<std::thread::id> hasher;

MLUOP_ATTRIBUTE_CONSTRUCTOR MLUOP_ATTRIBUTE_VISIBILITY_HIDDEN void mluOpInit() {
  if (!queue_array) {
    queue_array =
        new (std::nothrow) mluOpTensorDescriptorQueueStruct[QUEUE_ARRAY_LENGTH];
  }
}

MLUOP_ATTRIBUTE_DESTRUCTOR MLUOP_ATTRIBUTE_VISIBILITY_HIDDEN void mluOpExit() {
  if (queue_array) {
    delete[] queue_array;
    queue_array = NULL;
  }
}
#endif
/* MLUOP interface */
mluOpStatus_t MLUOP_WIN_API
mluOpCreateTensorDescriptor(mluOpTensorDescriptor_t *desc) {
  PARAM_CHECK("[mluOpCreateTensorDescriptor]", desc != NULL);

#if MLUOP_TENSOR_QUEUE_ENABLE
  size_t id = hasher(std::this_thread::get_id()) % QUEUE_ARRAY_LENGTH;
  queue_array[id].lock();
  if (MLUOP_PREDICT_FALSE(queue_array[id].queue.empty())) {
    queue_array[id].extend(queue_array[id].extend_num);
    queue_array[id].extend_num *= 2;
  }
  *desc = queue_array[id].queue.front();
  queue_array[id].queue.pop();
  queue_array[id].unlock();
#else
  mluOpTensorStruct *ts = new (std::nothrow) mluOpTensorStruct();
  *desc = ts;
#endif

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpCreateGroupTensorDescriptors(
    mluOpTensorDescriptor_t **group_desc, const int desc_num) {
  PARAM_CHECK("[mluOpCreateGroupTensorDescriptors]", group_desc != NULL);
  PARAM_CHECK("[mluOpCreateGroupTensorDescriptors]", desc_num > 0);

#if MLUOP_TENSOR_QUEUE_ENABLE
  size_t id = hasher(std::this_thread::get_id()) % QUEUE_ARRAY_LENGTH;
  queue_array[id].lock();
  if (MLUOP_PREDICT_FALSE(queue_array[id].queue.empty() ||
                          (size_t)desc_num >
                              (size_t)queue_array[id].queue.size())) {
    queue_array[id].extend(
        std::max((size_t)queue_array[id].extend_num, (size_t)desc_num));
    queue_array[id].extend_num =
        2 * std::max((size_t)queue_array[id].extend_num, (size_t)desc_num);
  }
  for (int i = 0; i < desc_num; ++i) {
    *(group_desc[i]) = queue_array[id].queue.front();
    queue_array[id].queue.pop();
  }
  queue_array[id].unlock();
#else
  for (int i = 0; i < desc_num; ++i) {
    mluOpTensorStruct *ts = new (std::nothrow) mluOpTensorStruct();
    *(group_desc[i]) = ts;
  }
#endif

  return MLUOP_STATUS_SUCCESS;
}

static inline mluOpStatus_t mluOpSetTensorDescriptorZeroDim(mluOpTensorDescriptor_t desc) {
  if (desc->pointer_mode == MLUOP_POINTER_MODE_HOST) {
    desc->dim = 0;
    desc->total_element_num = 1;
    desc->total_tensor_size = mluop::getSizeOfDataType(desc->dtype);
    return MLUOP_STATUS_SUCCESS;
  } else {
    LOG(ERROR)
        << "[mluOpSetTensorDescriptorDim]: Currently, the dim can be set to 0"
        << " only when the pointer mode of desc is MLUOP_POINTER_MODE_HOST. "
        << "Please use [mluOpSetTensorDescriptorPointerMode] to set the "
           "pointer "
        << "mode of desc.";
    return MLUOP_STATUS_BAD_PARAM;
  }
}

mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptor(
    mluOpTensorDescriptor_t desc, mluOpTensorLayout_t layout,
    mluOpDataType_t dtype, int dimNb, const int *dimSize) {
  PARAM_CHECK("[mluOpSetTensorDescriptor]", desc != NULL);
  PARAM_CHECK("[mluOpSetTensorDescriptor]", layout >= 0);
  PARAM_CHECK("[mluOpSetTensorDescriptor]", dtype >= 0);

  desc->dtype = dtype;
  desc->layout = layout;

  if (dimNb == 0) {
    return mluOpSetTensorDescriptorZeroDim(desc);
  } else {
    PARAM_CHECK("[mluOpSetTensorDescriptor]", dimNb > 0);
    PARAM_CHECK("[mluOpSetTensorDescriptor]", dimSize != NULL);
    return mluOpSetTensorDescriptorDim(desc, dimNb, dimSize);
  }
}

mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptor_v2(
    mluOpTensorDescriptor_t desc, mluOpTensorLayout_t layout,
    mluOpDataType_t dtype, int dimNb, const int64_t *dimSize) {
  PARAM_CHECK("[mluOpSetTensorDescriptor]", desc != NULL);
  PARAM_CHECK("[mluOpSetTensorDescriptor]", layout >= 0);
  PARAM_CHECK("[mluOpSetTensorDescriptor]", dtype >= 0);

  desc->dtype = dtype;
  desc->layout = layout;

  if (dimNb == 0) {
    return mluOpSetTensorDescriptorZeroDim(desc);
  } else {
    PARAM_CHECK("[mluOpSetTensorDescriptor]", dimNb > 0);
    PARAM_CHECK("[mluOpSetTensorDescriptor]", dimSize != NULL);

    return mluOpSetTensorDescriptorDim_v2(desc, dimNb, dimSize);
  }
}

// Internal interface. Caller should guarantee parameter validity.
static inline void mluOpSetTensorDescriptorDimBase(
  mluOpTensorDescriptor_t desc, int dimNb) {
  desc->dim = dimNb;

  if (MLUOP_PREDICT_FALSE(desc->larger_dims != NULL)) {
    delete[] desc->larger_dims;
    desc->larger_dims = NULL;
  }
  if (MLUOP_PREDICT_FALSE(desc->larger_strides != NULL)) {
    delete[] desc->larger_strides;
    desc->larger_strides = NULL;
  }
  if (MLUOP_PREDICT_FALSE(dimNb > MLUOP_DIM_MAX)) {
    desc->larger_dims = new (std::nothrow) int64_t[dimNb];
    desc->larger_strides = new (std::nothrow) int64_t[dimNb];
    desc->dims = desc->larger_dims;
    desc->strides = desc->larger_strides;
  } else {
    desc->dims = desc->normal_dims;
    desc->strides = desc->normal_strides;
  }

  return;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorDim(
    mluOpTensorDescriptor_t desc, int dimNb, const int *dimSize) {
  if (dimNb == 0) {
    CHECK_RETURN("[mluOpSetTensorDescriptorDim]",
                 mluOpSetTensorDescriptorZeroDim(desc));
  } else {
    mluOpSetTensorDescriptorDimBase(desc, dimNb);
    std::copy(dimSize, dimSize + dimNb, desc->dims);
  }

  // infer strides of dimNb dimensions and compute total_num & total_size
  uint64_t stride_base = 1;
  bool is_overflow = false;
  int tmp_num = 0;
  for (int i = dimNb - 1; i >= 0; --i) {
    desc->strides[i] = stride_base;
    is_overflow |= __builtin_smul_overflow(stride_base, dimSize[i], &tmp_num);
    stride_base *= dimSize[i];
  }
  desc->total_element_num = stride_base;
  desc->total_tensor_size =
      desc->total_element_num * mluop::getSizeOfDataType(desc->dtype);
  // judge int overflow situation
  if (MLUOP_PREDICT_FALSE(is_overflow)) {
    std::stringstream tensor_info;
    tensor_info << "dims:(";
    for (int i = 0; i < dimNb - 1; ++i) {
      tensor_info << dimSize[i] << ", ";
    }

    tensor_info << dimSize[dimNb - 1]
                << "), data width:" << mluop::getSizeOfDataType(desc->dtype)
                << ".";
    LOG(WARNING) << "[mluOpSetTensorDescriptor]: overflow max tensor num. "
                 << "Currently, mluop supports tensor num smaller than 2^31, "
                 << "now tensor " << tensor_info.str();
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorDim_v2(
    mluOpTensorDescriptor_t desc, int dimNb, const int64_t *dimSize) {
  mluOpSetTensorDescriptorDimBase(desc, dimNb);

  memcpy(desc->dims, dimSize, dimNb * sizeof(int64_t));

  // infer strides of dimNb dimensions and compute total_num & total_size
  uint64_t stride_base = 1;
  bool is_overflow = false;
  int64_t tmp_num = 0;
  for (int i = dimNb - 1; i >= 0; --i) {
    desc->strides[i] = stride_base;
    is_overflow |= __builtin_smull_overflow(stride_base, dimSize[i], &tmp_num);
    stride_base *= dimSize[i];
  }
  desc->total_element_num = stride_base;
  desc->total_tensor_size =
      desc->total_element_num * mluop::getSizeOfDataType(desc->dtype);
  // judge int overflow situation
  if (MLUOP_PREDICT_FALSE(is_overflow)) {
    std::stringstream tensor_info;
    tensor_info << "dims:(";
    for (int i = 0; i < dimNb - 1; ++i) {
      tensor_info << dimSize[i] << ", ";
    }

    tensor_info << dimSize[dimNb - 1]
                << "), data width:" << mluop::getSizeOfDataType(desc->dtype)
                << ".";
    LOG(WARNING) << "[mluOpSetTensorDescriptor_v2]: overflow max tensor num. "
                 << "Currently, mluop supports tensor num smaller than 2^63, "
                 << "now tensor " << tensor_info.str();
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetGroupTensorDescriptors(
    mluOpTensorDescriptor_t **group_desc,
    const mluOpTensorLayout_t *group_layout, const mluOpDataType_t *group_dtype,
    const int *group_dimNb, const int *group_dimSize, const int desc_num) {
  PARAM_CHECK("[mluOpSetGroupTensorDescriptors]", group_desc != NULL);
  PARAM_CHECK("[mluOpSetGroupTensorDescriptors]", group_layout != NULL);
  PARAM_CHECK("[mluOpSetGroupTensorDescriptors]", group_dtype != NULL);
  PARAM_CHECK("[mluOpSetGroupTensorDescriptors]", group_dimNb != NULL);
  PARAM_CHECK("[mluOpSetGroupTensorDescriptors]", group_dimSize != NULL);
  PARAM_CHECK("[mluOpSetGroupTensorDescriptors]", desc_num > 0);

  int group_dimSize_iterator = 0;
  for (int i = 0; i < desc_num; ++i) {
    (*(group_desc[i]))->dim = group_dimNb[i];
    (*(group_desc[i]))->dtype = group_dtype[i];
    (*(group_desc[i]))->layout = group_layout[i];

    if (MLUOP_PREDICT_FALSE(group_dimNb[i] > MLUOP_DIM_MAX)) {
      (*(group_desc[i]))->larger_dims =
          new (std::nothrow) int64_t[group_dimNb[i]];
      (*(group_desc[i]))->larger_strides =
          new (std::nothrow) int64_t[group_dimNb[i]];
      (*(group_desc[i]))->dims = (*(group_desc[i]))->larger_dims;
      (*(group_desc[i]))->strides = (*(group_desc[i]))->larger_strides;
    } else {
      (*(group_desc[i]))->dims = (*(group_desc[i]))->normal_dims;
      (*(group_desc[i]))->strides = (*(group_desc[i]))->normal_strides;
    }
    std::copy(group_dimSize + group_dimSize_iterator,
              group_dimSize + group_dimSize_iterator + group_dimNb[i],
              (*(group_desc[i]))->dims);

    // infer strides of dimNb dimensions and compute total_num and total_size
    int strideBase = 1;
    for (int j = group_dimNb[i] - 1; j >= 0; --j) {
      (*(group_desc[i]))->strides[j] = strideBase;
      strideBase *= (*(group_desc[i]))->dims[j];
    }
    (*(group_desc[i]))->total_element_num = strideBase;
    (*(group_desc[i]))->total_tensor_size =
        (*(group_desc[i]))->total_element_num *
        mluop::getSizeOfDataType(group_dtype[i]);

    // compute new iterator for next loop.
    group_dimSize_iterator += group_dimNb[i];
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetGroupTensorDescriptors_v2(
    mluOpTensorDescriptor_t **group_desc,
    const mluOpTensorLayout_t *group_layout, const mluOpDataType_t *group_dtype,
    const int *group_dimNb, const int64_t *group_dimSize, const int desc_num) {
  PARAM_CHECK("[mluOpSetGroupTensorDescriptors]", group_desc != NULL);
  PARAM_CHECK("[mluOpSetGroupTensorDescriptors]", group_layout != NULL);
  PARAM_CHECK("[mluOpSetGroupTensorDescriptors]", group_dtype != NULL);
  PARAM_CHECK("[mluOpSetGroupTensorDescriptors]", group_dimNb != NULL);
  PARAM_CHECK("[mluOpSetGroupTensorDescriptors]", group_dimSize != NULL);
  PARAM_CHECK("[mluOpSetGroupTensorDescriptors]", desc_num > 0);

  int group_dimSize_iterator = 0;
  for (int i = 0; i < desc_num; ++i) {
    (*(group_desc[i]))->dim = group_dimNb[i];
    (*(group_desc[i]))->dtype = group_dtype[i];
    (*(group_desc[i]))->layout = group_layout[i];

    if (MLUOP_PREDICT_FALSE(group_dimNb[i] > MLUOP_DIM_MAX)) {
      (*(group_desc[i]))->larger_dims =
          new (std::nothrow) int64_t[group_dimNb[i]];
      (*(group_desc[i]))->larger_strides =
          new (std::nothrow) int64_t[group_dimNb[i]];
      (*(group_desc[i]))->dims = (*(group_desc[i]))->larger_dims;
      (*(group_desc[i]))->strides = (*(group_desc[i]))->larger_strides;
    } else {
      (*(group_desc[i]))->dims = (*(group_desc[i]))->normal_dims;
      (*(group_desc[i]))->strides = (*(group_desc[i]))->normal_strides;
    }
    memcpy((*(group_desc[i]))->dims, group_dimSize + group_dimSize_iterator,
           group_dimNb[i] * sizeof(int64_t));

    // infer strides of dimNb dimensions and compute total_num and total_size
    int strideBase = 1;
    for (int j = group_dimNb[i] - 1; j >= 0; --j) {
      (*(group_desc[i]))->strides[j] = strideBase;
      strideBase *= (*(group_desc[i]))->dims[j];
    }
    (*(group_desc[i]))->total_element_num = strideBase;
    (*(group_desc[i]))->total_tensor_size =
        (*(group_desc[i]))->total_element_num *
        mluop::getSizeOfDataType(group_dtype[i]);

    // compute new iterator for next loop.
    group_dimSize_iterator += group_dimNb[i];
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpResetTensorDescriptor(mluOpTensorDescriptor_t desc) {
  PARAM_CHECK("[mluOpResetTensorDescriptor]", desc != NULL);
  desc->reset();

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorEx(
    mluOpTensorDescriptor_t desc, mluOpTensorLayout_t layout,
    mluOpDataType_t dtype, int dimNb, const int *dimSize,
    const int *dimStride) {
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", desc != NULL);
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", layout >= 0);
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", dtype >= 0);

  desc->dtype = dtype;
  desc->layout = layout;

  if (dimNb == 0) {
    return mluOpSetTensorDescriptorZeroDim(desc);
  } else {
    PARAM_CHECK("[mluOpSetTensorDescriptorEx]", dimSize != NULL);
    PARAM_CHECK("[mluOpSetTensorDescriptorEx]", dimStride != NULL);
    PARAM_CHECK("[mluOpSetTensorDescriptorEx]", dimNb > 0);

    mluOpSetTensorDescriptorDimBase(desc, dimNb);
    std::copy(dimSize, dimSize + dimNb, desc->dims);
    std::copy(dimStride, dimStride + dimNb, desc->strides);

    // assign total_element_num and total_tensor_size
    desc->total_element_num = 1;
    for (int i = 0; i < dimNb; ++i) {
      desc->total_element_num *= dimSize[i];
    }
    desc->total_tensor_size =
        desc->total_element_num * mluop::getSizeOfDataType(dtype);

    return MLUOP_STATUS_SUCCESS;
  }
}

mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorEx_v2(
    mluOpTensorDescriptor_t desc, mluOpTensorLayout_t layout,
    mluOpDataType_t dtype, int dimNb, const int64_t *dimSize,
    const int64_t *dimStride) {
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", desc != NULL);
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", layout >= 0);
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", dtype >= 0);

  desc->dtype = dtype;
  desc->layout = layout;

  if MLUOP_PREDICT_FALSE(dimNb == 0) {
    return mluOpSetTensorDescriptorZeroDim(desc);
  } else {
    PARAM_CHECK("[mluOpSetTensorDescriptorEx]", dimSize != NULL);
    PARAM_CHECK("[mluOpSetTensorDescriptorEx]", dimStride != NULL);
    PARAM_CHECK("[mluOpSetTensorDescriptorEx]", dimNb > 0);

    mluOpSetTensorDescriptorDimBase(desc, dimNb);
    memcpy(desc->dims, dimSize, dimNb * sizeof(int64_t));
    memcpy(desc->strides, dimStride, dimNb * sizeof(int64_t));

    // assign total_element_num and total_tensor_size
    desc->total_element_num = 1;
    for (int i = 0; i < dimNb; ++i) {
      desc->total_element_num *= dimSize[i];
    }

    desc->total_tensor_size =
        desc->total_element_num * mluop::getSizeOfDataType(dtype);

    return MLUOP_STATUS_SUCCESS;
  }
}

mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorOnchipDataType(
    mluOpTensorDescriptor_t desc, mluOpDataType_t onchip_dtype) {
  PARAM_CHECK("[mluOpSetTensorDescriptorOnchipDataType]", desc != NULL);

  desc->onchip_dtype = onchip_dtype;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpSetTensorDescriptorPosition(mluOpTensorDescriptor_t desc, int position) {
  PARAM_CHECK("[mluOpSetTensorDescriptorPosition]", desc != NULL);

  desc->position = position;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorPositionAndScale(
    mluOpTensorDescriptor_t desc, int position, float scale) {
  PARAM_CHECK("[mluOpSetTensorDescriptorPositionAndScale]", desc != NULL);

  desc->position = position;
  desc->scale = scale;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorPositionScaleAndOffset(
    mluOpTensorDescriptor_t desc, int position, float scale, int offset) {
  PARAM_CHECK("[mluOpSetTensorDescriptorPositionScaleAndOffset]", desc != NULL);

  desc->position = position;
  desc->scale = scale;
  desc->offset = offset;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorPointerMode(
    mluOpTensorDescriptor_t desc, mluOpPointerMode_t pointer_mode) {
  PARAM_CHECK("[mluOpSetTensorDescriptorPointerMode]", desc != NULL);
  PARAM_CHECK("[mluOpSetTensorDescriptorPointerMode]", pointer_mode >= 0);

  desc->pointer_mode = pointer_mode;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptorEx(
    const mluOpTensorDescriptor_t desc, mluOpTensorLayout_t *layout,
    mluOpDataType_t *dtype, int *dimNb, int *dimSize, int *dimStride) {
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", desc != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", layout != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", dtype != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", dimNb != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", dimSize != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", dimStride != NULL);

  *layout = desc->layout;
  *dtype = desc->dtype;
  *dimNb = desc->dim;
  for (int i = 0; i < *dimNb; ++i) {
    dimSize[i] = static_cast<int>(desc->dims[i]);
    dimStride[i] = static_cast<int>(desc->strides[i]);
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptorEx_v2(
    const mluOpTensorDescriptor_t desc, mluOpTensorLayout_t *layout,
    mluOpDataType_t *dtype, int *dimNb, int64_t *dimSize, int64_t *dimStride) {
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", desc != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", layout != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", dtype != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", dimNb != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", dimSize != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", dimStride != NULL);

  *layout = desc->layout;
  *dtype = desc->dtype;
  *dimNb = desc->dim;
  for (int i = 0; i < *dimNb; ++i) {
    dimSize[i] = desc->dims[i];
    dimStride[i] = desc->strides[i];
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptor(
    const mluOpTensorDescriptor_t desc, mluOpTensorLayout_t *layout,
    mluOpDataType_t *dtype, int *dimNb, int *dimSize) {
  PARAM_CHECK("[mluOpGetTensorDescriptor]", desc != NULL);

  SET_PARAM_FOR_POINTER(layout, desc->layout);
  SET_PARAM_FOR_POINTER(dtype, desc->dtype);
  SET_PARAM_FOR_POINTER(dimNb, desc->dim);
  SET_ARRAY_PARAM_FOR_POINTER(dimSize, desc->dims, desc->dim);

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptor_v2(
    const mluOpTensorDescriptor_t desc, mluOpTensorLayout_t *layout,
    mluOpDataType_t *dtype, int *dimNb, int64_t *dimSize) {
  PARAM_CHECK("[mluOpGetTensorDescriptor]", desc != NULL);

  SET_PARAM_FOR_POINTER(layout, desc->layout);
  SET_PARAM_FOR_POINTER(dtype, desc->dtype);
  SET_PARAM_FOR_POINTER(dimNb, desc->dim);
  SET_ARRAY_PARAM_FOR_POINTER(dimSize, desc->dims, desc->dim);

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptorOnchipDataType(
    const mluOpTensorDescriptor_t desc, mluOpDataType_t *onchip_dtype) {
  PARAM_CHECK("[mluOpGetTensorDescriptorOnchipDataType]", desc != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorOnchipDataType]", onchip_dtype != NULL);

  *onchip_dtype = desc->onchip_dtype;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpGetTensorDescriptorPosition(mluOpTensorDescriptor_t desc, int *position) {
  PARAM_CHECK("[mluOpGetTensorDescriptorPosition]", desc != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorPosition]", position != NULL);

  *position = desc->position;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptorPositionAndScale(
    mluOpTensorDescriptor_t desc, int *position, float *scale) {
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionAndScale]", desc != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionAndScale]", position != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionAndScale]", scale != NULL);

  *position = desc->position;
  *scale = desc->scale;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptorPositionScaleAndOffset(
    mluOpTensorDescriptor_t desc, int *position, float *scale, int *offset) {
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionScaleAndOffset]", desc != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionScaleAndOffset]",
              position != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionScaleAndOffset]",
              scale != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionScaleAndOffset]",
              offset != NULL);

  *position = desc->position;
  *scale = desc->scale;
  *offset = desc->offset;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptorPointerMode(
    mluOpTensorDescriptor_t desc, mluOpPointerMode_t *pointer_mode) {
  PARAM_CHECK("[mluOpGetTensorDescriptorPointerMode]", desc != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorPointerMode]", pointer_mode != NULL);

  SET_PARAM_FOR_POINTER(pointer_mode, desc->pointer_mode);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpDestroyTensorDescriptor(mluOpTensorDescriptor_t desc) {
  PARAM_CHECK("[mluOpDestroyTensorDescriptor]", desc != NULL);
  desc->reset();

#if MLUOP_TENSOR_QUEUE_ENABLE
  size_t id = hasher(std::this_thread::get_id()) % QUEUE_ARRAY_LENGTH;
  queue_array[id].lock();
  queue_array[id].queue.emplace(desc);
  queue_array[id].unlock();
#else
  delete desc;
#endif

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpDestroyGroupTensorDescriptors(
    mluOpTensorDescriptor_t **group_desc, const int desc_num) {
  PARAM_CHECK("[mluOpDestroyGroupTensorDescriptors]", group_desc != NULL);
  PARAM_CHECK("[mluOpDestroyGroupTensorDescriptors]", desc_num > 0);

#if MLUOP_TENSOR_QUEUE_ENABLE
  size_t id = hasher(std::this_thread::get_id()) % QUEUE_ARRAY_LENGTH;
  queue_array[id].lock();
  for (int i = 0; i < desc_num; ++i) {
    (*(group_desc[i]))->reset();
    queue_array[id].queue.emplace(*(group_desc[i]));
  }
  queue_array[id].unlock();
#else
  for (int i = 0; i < desc_num; ++i) {
    (*(group_desc[i]))->reset();
    delete (*(group_desc[i]));
  }
#endif

  return MLUOP_STATUS_SUCCESS;
}

// usr interface.
uint64_t MLUOP_WIN_API
mluOpGetTensorElementNum(const mluOpTensorDescriptor_t desc) {
  CHECK(desc != NULL);
  return desc->total_element_num;
}

uint64_t mluOpGetSeqDataElementNum(mluOpSeqDataDescriptor_t desc) {
  CHECK(desc != NULL);
  uint64_t tensor_num = 1;
  auto return_status = desc->seqDataElementsNumber(tensor_num);
  CHECK(return_status == MLUOP_STATUS_SUCCESS);
  return tensor_num;
}

mluOpStatus_t MLUOP_WIN_API mluOpCreateTensorSetDescriptor(
    mluOpTensorSetDescriptor_t *tensorSet, const int tensorSetDimNb,
    const int *tensorSetDimSize) {
  mluOpTensorSetStruct *tss = new (std::nothrow) mluOpTensorSetStruct();
  tss->dim_num = tensorSetDimNb;
  int set_size = 1;
  for (int i = 0; i < tensorSetDimNb; i++) {
    set_size *= tensorSetDimSize[i];
    tss->dim_set.push_back(tensorSetDimSize[i]);
    int j = i + 1;
    int offset_base = 1;
    while (j < tensorSetDimNb) {
      offset_base *= tensorSetDimSize[j];
      j++;
    }
    tss->dim_offset_base.push_back(offset_base);
  }
  for (int i = 0; i < set_size; i++) {
    auto ts = std::make_shared<mluOpTensorStruct>();
    tss->tensor_set.push_back(ts);
  }
  tss->tensor_num = set_size;
  tss->dataOffsetInit(set_size);
  tss->user_indices.resize(set_size);
  *tensorSet = tss;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetTensorSetDescriptor(
    mluOpTensorSetDescriptor_t tensorSet, int *tensorSetDimNb, int *dimSize) {
  *tensorSetDimNb = tensorSet->dim_num;
  for (int i = 0; i < tensorSet->dim_num; i++) {
    dimSize[i] = tensorSet->dim_set[i];
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpDestroyTensorSetDescriptor(mluOpTensorSetDescriptor_t tensorSet) {
  PARAM_CHECK("[mluOpDestroyTensorSetDescriptor]", tensorSet != NULL);
  tensorSet->tensor_set.clear();
  delete tensorSet;
  tensorSet = NULL;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpInitTensorSetMemberDescriptor(
    mluOpTensorSetDescriptor_t tensorSet, const int tensorSetDimNb,
    const int *tensorIndex, mluOpTensorLayout_t layout, mluOpDataType_t dtype,
    const int dimNb, const int *dimSize) {
  PARAM_CHECK("[mluOpInitTensorSetMemberDescriptor]",
              tensorSet->dim_num == tensorSetDimNb);
  auto ts = tensorSet->getTensor(tensorIndex);
  PARAM_CHECK("[mluOpInitTensorSetMemberDescriptor]",
              MLUOP_STATUS_SUCCESS ==
                  mluOpSetTensorDescriptor(ts, layout, dtype, dimNb, dimSize));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpInitTensorSetMemberDescriptor_v2(
    mluOpTensorSetDescriptor_t tensorSet, const int tensorSetDimNb,
    const int *tensorIndex, mluOpTensorLayout_t layout, mluOpDataType_t dtype,
    const int dimNb, const int64_t *dimSize) {
  PARAM_CHECK("[mluOpInitTensorSetMemberDescriptor_v2]",
              tensorSet->dim_num == tensorSetDimNb);
  auto ts = tensorSet->getTensor(tensorIndex);
  PARAM_CHECK("[mluOpInitTensorSetMemberDescriptor_v2]",
              MLUOP_STATUS_SUCCESS == mluOpSetTensorDescriptor_v2(
                                          ts, layout, dtype, dimNb, dimSize));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpInitTensorSetMemberDescriptorPositionAndScale(
    mluOpTensorSetDescriptor_t tensorSet, const int tensorSetDimNb,
    const int *tensorIndex, const int position, const float scale) {
  PARAM_CHECK("[mluOpInitTensorSetMemberDescriptorPositionAndScale]",
              tensorSet->dim_num == tensorSetDimNb);
  auto ts = tensorSet->getTensor(tensorIndex);
  PARAM_CHECK("[mluOpInitTensorSetMemberDescriptorPositionAndScale]",
              MLUOP_STATUS_SUCCESS == mluOpSetTensorDescriptorPositionAndScale(
                                          ts, position, scale));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetTensorSetDescriptorSize(
    mluOpTensorSetDescriptor_t tensorSet, int *sizeInBytes) {
  PARAM_CHECK("[mluOpGetTensorSetDescriptorSize]", tensorSet != NULL);
  int tensor_set_size = tensorSet->getSize();
  *sizeInBytes = tensor_set_size;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetTensorAndDataFromTensorSet(
    mluOpTensorSetDescriptor_t tensorSet, const int tensorSetDimNb,
    const int *tensorIndex, void *data, mluOpTensorDescriptor_t *tensorDesc,
    void **dataAddrInDevice) {
  PARAM_CHECK("[mluOpGetTensorAndDataFromTensorSet]", tensorSet != NULL);
  PARAM_CHECK("[mluOpGetTensorAndDataFromTensorSet]",
              tensorSet->dim_num == tensorSetDimNb);
  *tensorDesc = tensorSet->getTensor(tensorIndex);
  auto offset = tensorSet->getOffset(tensorIndex);
  *dataAddrInDevice = (void *)((char *)data + offset);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetSeqDataDescriptorOnchipDataType(
    mluOpSeqDataDescriptor_t desc, mluOpDataType_t onchip_dtype) {
  PARAM_CHECK("[mluOpSetSeqDataDescriptorOnchipDataType]", desc != NULL);

  desc->onchip_dtype = onchip_dtype;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpGetSeqDataDescriptorOnchipDataType(
    mluOpSeqDataDescriptor_t desc, mluOpDataType_t *onchip_dtype) {
  PARAM_CHECK("[mluOpGetSeqDataDescriptorOnchipDataType]", desc != NULL);
  PARAM_CHECK("[mluOpGetSeqDataDescriptorOnchipDataType]",
              onchip_dtype != NULL);

  *onchip_dtype = desc->onchip_dtype;
  return MLUOP_STATUS_SUCCESS;
}
