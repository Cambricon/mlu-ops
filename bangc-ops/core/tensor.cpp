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

mluOpStatus_t mluOpGetSizeOfDataType(mluOpDataType_t data_type, size_t *size) {
  PARAM_CHECK("[mluOpGetSizeOfDataType]", size != NULL);

  if (MLUOP_DTYPE_INVALID != data_type) {
    *size = getSizeOfDataType(data_type);
    return MLUOP_STATUS_SUCCESS;
  } else {
    LOG(ERROR) << "[mluOpGetSizeOfDataType]:data_type should not be "
                  "MLUOP_DTYPE_INVALID. ";
    return MLUOP_STATUS_BAD_PARAM;
  }
}

#if MLUOP_TENSOR_QUEUE_ENABLE
mluOpTensorDescriptorQueueStruct *queue_array = NULL;
std::hash<std::thread::id> hasher;

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
mluOpStatus_t mluOpCreateTensorDescriptor(mluOpTensorDescriptor_t *desc) {
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

mluOpStatus_t mluOpCreateGroupTensorDescriptors(
    mluOpTensorDescriptor_t *group_desc[], const int desc_num) {
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

mluOpStatus_t mluOpSetTensorDescriptor(mluOpTensorDescriptor_t desc,
                                       mluOpTensorLayout_t layout,
                                       mluOpDataType_t dtype, int dimNb,
                                       const int dimSize[]) {
  PARAM_CHECK("[mluOpSetTensorDescriptor]", desc != NULL);
  PARAM_CHECK("[mluOpSetTensorDescriptor]", dimNb > 0);
  PARAM_CHECK("[mluOpSetTensorDescriptor]", dimSize != NULL);
  PARAM_CHECK("[mluOpSetTensorDescriptor]", layout >= 0);
  PARAM_CHECK("[mluOpSetTensorDescriptor]", dtype >= 0);

  desc->dim = dimNb;
  desc->dtype = dtype;
  desc->layout = layout;

  if (MLUOP_PREDICT_FALSE(dimNb > MLUOP_DIM_MAX)) {
    desc->larger_dims = new int[dimNb];
    desc->larger_strides = new int[dimNb];
    desc->dims = desc->larger_dims;
    desc->strides = desc->larger_strides;
  } else {
    desc->dims = desc->normal_dims;
    desc->strides = desc->normal_strides;
  }
  memcpy(desc->dims, dimSize, dimNb * sizeof(int));

  // infer strides of dimNb dimensions and compute total_num & total_size
  int strideBase = 1;
  for (int i = dimNb - 1; i >= 0; --i) {
    desc->strides[i] = strideBase;
    strideBase *= desc->dims[i];
  }
  desc->total_element_num = strideBase;
  desc->total_tensor_size = desc->total_element_num * getSizeOfDataType(dtype);
  // judge int overflow situation
  int total_size = desc->total_tensor_size;
  if (total_size != 0) {
    int last_num = total_size / getSizeOfDataType(dtype);
    for (int i = 0; i < dimNb - 1; ++i) {
      last_num = last_num / dimSize[i];
    }
    if (last_num < dimSize[dimNb - 1]) {
      std::stringstream tensor_info;
      tensor_info << "dims:(";
      for (int i = 0; i < dimNb - 1; ++i) {
        tensor_info << dimSize[i] << ", ";
      }

      tensor_info << dimSize[dimNb - 1]
                  << "), data width:" << getSizeOfDataType(dtype) << ".";
      LOG(WARNING) << "[mluOpSetTensorDescriptor]: overflow tensor size with "
                      "type 'int'. Currently, mluOp supports tensor size "
                      "smaller than 2^31, now tensor "
                   << tensor_info.str();
    }
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpSetGroupTensorDescriptors(
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
      (*(group_desc[i]))->larger_dims = new (std::nothrow) int[group_dimNb[i]];
      (*(group_desc[i]))->larger_strides =
          new (std::nothrow) int[group_dimNb[i]];
      (*(group_desc[i]))->dims = (*(group_desc[i]))->larger_dims;
      (*(group_desc[i]))->strides = (*(group_desc[i]))->larger_strides;
    } else {
      (*(group_desc[i]))->dims = (*(group_desc[i]))->normal_dims;
      (*(group_desc[i]))->strides = (*(group_desc[i]))->normal_strides;
    }
    memcpy((*(group_desc[i]))->dims, group_dimSize + group_dimSize_iterator,
           group_dimNb[i] * sizeof(int));

    // infer strides of dimNb dimensions and compute total_num and total_size
    int strideBase = 1;
    for (int j = group_dimNb[i] - 1; j >= 0; --j) {
      (*(group_desc[i]))->strides[j] = strideBase;
      strideBase *= (*(group_desc[i]))->dims[j];
    }
    (*(group_desc[i]))->total_element_num = strideBase;
    (*(group_desc[i]))->total_tensor_size =
        (*(group_desc[i]))->total_element_num *
        getSizeOfDataType(group_dtype[i]);

    // compute new iterator for next loop.
    group_dimSize_iterator += group_dimNb[i];
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpResetTensorDescriptor(mluOpTensorDescriptor_t desc) {
  PARAM_CHECK("[mluOpResetTensorDescriptor]", desc != NULL);
  desc->reset();

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpSetTensorDescriptorEx(mluOpTensorDescriptor_t desc,
                                         mluOpTensorLayout_t layout,
                                         mluOpDataType_t dtype, int dimNb,
                                         const int dimSize[],
                                         const int dimStride[]) {
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", desc != NULL);
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", dimSize != NULL);
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", dimStride != NULL);
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", layout >= 0);
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", dtype >= 0);
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", dimNb > 0);

  desc->dim = dimNb;
  memcpy(desc->dims, dimSize, dimNb * sizeof(int));
  memcpy(desc->strides, dimStride, dimNb * sizeof(int));

  // assign total_element_num and total_tensor_size
  desc->total_element_num = 1;
  for (int i = 0; i < dimNb; ++i) {
    desc->total_element_num *= dimSize[i];
  }
  desc->total_tensor_size = desc->total_element_num * getSizeOfDataType(dtype);

  desc->dtype = dtype;
  desc->layout = layout;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpSetTensorDescriptorOnchipDataType(
    mluOpTensorDescriptor_t desc, mluOpDataType_t onchip_dtype) {
  PARAM_CHECK("[mluOpSetTensorDescriptorOnchipDataType]", desc != NULL);

  desc->onchip_dtype = onchip_dtype;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpSetTensorDescriptorPosition(mluOpTensorDescriptor_t desc,
                                               int position) {
  PARAM_CHECK("[mluOpSetTensorDescriptorPosition]", desc != NULL);

  desc->position = position;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpSetTensorDescriptorPositionAndScale(
    mluOpTensorDescriptor_t desc, int position, float scale) {
  PARAM_CHECK("[mluOpSetTensorDescriptorPositionAndScale]", desc != NULL);

  desc->position = position;
  desc->scale = scale;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpSetTensorDescriptorPositionScaleAndOffset(
    mluOpTensorDescriptor_t desc, int position, float scale, int offset) {
  PARAM_CHECK("[mluOpSetTensorDescriptorPositionScaleAndOffset]", desc != NULL);

  desc->position = position;
  desc->scale = scale;
  desc->offset = offset;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetTensorDescriptorEx(const mluOpTensorDescriptor_t desc,
                                         mluOpTensorLayout_t *layout,
                                         mluOpDataType_t *dtype, int *dimNb,
                                         int dimSize[], int dimStride[]) {
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

mluOpStatus_t mluOpGetTensorDescriptor(const mluOpTensorDescriptor_t desc,
                                       mluOpTensorLayout_t *layout,
                                       mluOpDataType_t *dtype, int *dimNb,
                                       int dimSize[]) {
  PARAM_CHECK("[mluOpGetTensorDescriptor]", desc != NULL);

  if (layout != nullptr) {
    *layout = desc->layout;
  }
  if (dtype != nullptr) {
    *dtype = desc->dtype;
  }
  if (dimNb != nullptr) {
    *dimNb = desc->dim;
  }
  if (dimSize != nullptr) {
    for (int i = 0; i < *dimNb; ++i) {
      dimSize[i] = desc->dims[i];
    }
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetTensorDescriptorOnchipDataType(
    const mluOpTensorDescriptor_t desc, mluOpDataType_t *onchip_dtype) {
  PARAM_CHECK("[mluOpGetTensorDescriptorOnchipDataType]", desc != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorOnchipDataType]", onchip_dtype != NULL);

  *onchip_dtype = desc->onchip_dtype;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetTensorDescriptorPosition(
    const mluOpTensorDescriptor_t desc, int *position) {
  PARAM_CHECK("[mluOpGetTensorDescriptorPosition]", desc != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorPosition]", position != NULL);

  *position = desc->position;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetTensorDescriptorPositionAndScale(
    const mluOpTensorDescriptor_t desc, int *position, float *scale) {
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionAndScale]", desc != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionAndScale]", position != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionAndScale]", scale != NULL);

  *position = desc->position;
  *scale = desc->scale;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetTensorDescriptorPositionScaleAndOffset(
    const mluOpTensorDescriptor_t desc, int *position, float *scale,
    int *offset) {
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

mluOpStatus_t mluOpDestroyTensorDescriptor(mluOpTensorDescriptor_t desc) {
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

mluOpStatus_t mluOpDestroyGroupTensorDescriptors(
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
uint64_t mluOpGetTensorElementNum(const mluOpTensorDescriptor_t desc) {
  CHECK(desc != NULL);
  uint64_t tensor_num = 1;
  auto return_status = desc->tensorElementsNumber(tensor_num);
  return tensor_num;
}

mluOpStatus_t mluOpCreateTensorSetDescriptor(
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

mluOpStatus_t mluOpGetTensorSetDescriptor(mluOpTensorSetDescriptor_t tensorSet,
                                          int *tensorSetDimNb, int *dimSize) {
  *tensorSetDimNb = tensorSet->dim_num;
  for (int i = 0; i < tensorSet->dim_num; i++) {
    dimSize[i] = tensorSet->dim_set[i];
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpDestroyTensorSetDescriptor(
    mluOpTensorSetDescriptor_t tensorSet) {
  PARAM_CHECK("[mluOpDestroyTensorSetDescriptor]", tensorSet != NULL);
  tensorSet->tensor_set.clear();
  delete tensorSet;
  tensorSet = NULL;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpInitTensorSetMemberDescriptor(
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

mluOpStatus_t mluOpInitTensorSetMemberDescriptorPositionAndScale(
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

mluOpStatus_t mluOpGetTensorSetDescriptorSize(
    mluOpTensorSetDescriptor_t tensorSet, int *sizeInBytes) {
  PARAM_CHECK("[mluOpGetTensorSetDescriptorSize]", tensorSet != NULL);

  int tensor_set_size = tensorSet->getSize();
  *sizeInBytes = tensor_set_size;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetTensorAndDataFromTensorSet(
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
