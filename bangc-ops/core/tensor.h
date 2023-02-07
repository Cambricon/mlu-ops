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
#ifndef CORE_TENSOR_H_
#define CORE_TENSOR_H_

#include <vector>
#include <list>
#include <memory>
#include <queue>
#include <thread>  // NOLINT
#include <atomic>
#include <cstring>

#include "core/macros.h"
#include "core/logging.h"
#include "core/type.h"
#include "mlu_op.h"

struct mluOpTensorStruct {
  mluOpTensorStruct()
      : dim(0),
        dtype(MLUOP_DTYPE_FLOAT),
        onchip_dtype(MLUOP_DTYPE_INVALID),
        layout(MLUOP_LAYOUT_ARRAY),
        position(0),
        scale(1.0),
        offset(0) {
    /* explicit set initial values for document use.
     */
  }
  ~mluOpTensorStruct() {
    /* please do NOT implement any codes here.
     * a state-less struct should not hold any resources.
     */
  }
  /* methods */
  mluOpStatus_t tensorDimN(size_t &dim);
  mluOpStatus_t tensorDimC(size_t &dim);
  mluOpStatus_t tensorDimH(size_t &dim);
  mluOpStatus_t tensorDimW(size_t &dim);
  inline mluOpStatus_t tensorElementsNumber(size_t &elements) const {
    elements = total_element_num;
    return MLUOP_STATUS_SUCCESS;
  }
  inline mluOpStatus_t tensorSize(size_t &tensor_size) const {
    tensor_size = total_tensor_size;
    return MLUOP_STATUS_SUCCESS;
  }

  /* struct */
  int dim = 0;
  uint64_t total_element_num = 0;
  uint64_t total_tensor_size = 0;
  // if dimNb > MLUOP_DIM_MAX (8), using larger_dims, malloc it and dims point
  // it. else, using normal_dims, dont need malloc and free.
  int normal_dims[MLUOP_DIM_MAX] = {-1};
  int *larger_dims = NULL;
  int *dims = normal_dims;  // point the normal dims as default

  int normal_strides[MLUOP_DIM_MAX] = {-1};
  int *larger_strides = NULL;
  int *strides = normal_strides;  // point the normal strides as default

  mluOpDataType_t dtype;
  mluOpDataType_t onchip_dtype;
  mluOpTensorLayout_t layout;
  int position;
  float scale;
  int offset;
  int channelNb;
  std::vector<int> positions;
  std::vector<float> scales;
  std::vector<int> offsets;
  inline void init() {  // reset random value after malloc.
    // init these pointer.
    // if not, when call reset() will free invalid pointer.
    larger_dims = NULL;
    larger_strides = NULL;

    dim = 0;
    total_element_num = 0;
    total_tensor_size = 0;
    dims = normal_dims;
    strides = normal_strides;
  }
  inline void reset() {  // reset variable as default.
    if (MLUOP_PREDICT_FALSE(larger_dims != NULL)) {
      delete[] larger_dims;
      larger_dims = NULL;
    }
    if (MLUOP_PREDICT_FALSE(larger_strides != NULL)) {
      delete[] larger_strides;
      larger_strides = NULL;
    }
    dims = normal_dims;
    strides = normal_strides;
    dtype = MLUOP_DTYPE_FLOAT;
    onchip_dtype = MLUOP_DTYPE_INVALID;
    layout = MLUOP_LAYOUT_ARRAY;

    position = 0;
    scale = 1.0f;
    offset = 0;

    dim = 0;
    total_element_num = 0;
    total_tensor_size = 0;
  }
};

// dim_set(rnn)     [layer_num, direction, cap_of_cell]
// dim_offset_base  [direction * cap_of_cell, cap_of_cell, 1]
// tensor_set       [l1.forward.filter1, ..., l1.forward.filter9,
//                   l1.backward.filter1, ..., l1.backward.filter9,
//                   l2.forward.filter1, ..., l2.forward.filter9
//                   ...                                       ]
struct mluOpTensorSetStruct {
  mluOpTensorSetStruct() : tensor_num(0), dim_num(0) {
    /* explicit set initial values for document use.
     */
  }
  ~mluOpTensorSetStruct() {
    /* please do NOT implement any codes here.
     * a state-less struct should not hold any resources.
     */
  }
  /* methods */
  inline size_t getSize() {
    CHECK(!this->tensor_set.empty());
    size_t tensor_set_size = 0;
    for (int i = 0; i < tensor_set.size(); i++) {
      size_t size = 0;
      tensor_set[i]->tensorSize(size);
      tensor_set_size += size;
    }
    return tensor_set_size;
  }
  // tensor set (eg: rnn)
  inline int getIndex(const int tensorIndex[]) const {
    int index = 0;
    for (int i = 0; i < this->dim_set.size(); i++) {
      index += tensorIndex[i] * this->dim_offset_base[i];
    }
    return index;
  }

  inline size_t getOffset(const int tensorIndex[]) {
    int64_t offset = 0;
    int index = this->getIndex(tensorIndex);
    for (int i = 0; i < index; i++) {
      size_t ts_size = 0;
      this->tensor_set[i]->tensorSize(ts_size);
      offset += ts_size;
    }
    data_offset[index] = offset;
    return offset;
  }

  inline mluOpTensorDescriptor_t getTensor(const int tensorIndex[]) const {
    auto index = this->getIndex(tensorIndex);
    auto ts = this->tensor_set[index].get();
    return ts;
  }

  inline mluOpDataType_t getDatatype() const {
    CHECK(!this->tensor_set.empty());
    return this->tensor_set[0]->dtype;
  }

  inline mluOpTensorLayout_t getLayout() const {
    CHECK(!this->tensor_set.empty());
    return this->tensor_set[0]->layout;
  }

  inline void checkDataOffset() const {
    auto data_offset_array = data_offset.size();
    for (int i = 0; i < data_offset_array; i++) {
      if (i != 0 && data_offset[i] == 0) {
        LOG(ERROR) << "tensorSet's data not set, index:" << i << " of "
                   << tensor_num;
      }
    }
  }

  inline void dataOffsetInit(int set_size) {
    this->data_offset.resize(set_size);
  }

  inline std::vector<size_t> getDataOffsets() {
    if (data_offset.size() == 0) {
      return data_offset;
    }
    int offset = 0;
    data_offset[0] = offset;
    for (int i = 0; i < tensor_num - 1; i++) {
      size_t ts_size = 0;
      this->tensor_set[i]->tensorSize(ts_size);
      offset += ts_size;
      data_offset[i + 1] = offset;
    }
    return data_offset;
  }
  /* struct */
  int tensor_num = 0;
  int dim_num = 0;                   // dimension number
  std::vector<int> dim_set;          // the number for each dimension
  std::vector<int> dim_offset_base;  // offset for each dimension
  std::vector<std::shared_ptr<mluOpTensorStruct>>
      tensor_set;  // vector of tensorDesc

  std::vector<std::vector<int>> user_indices;  // releated tensor's index
  std::vector<size_t> data_offset;             // data's offset
};

#ifndef MLUOP_TENSOR_QUEUE_ENABLE
#define MLUOP_TENSOR_QUEUE_ENABLE 1
#endif

#if MLUOP_TENSOR_QUEUE_ENABLE
struct mluOpTensorDescriptorQueueStruct {
  mluOpTensorDescriptorQueueStruct() {
    extend(extend_num);
    extend_num *= 2;
  }
  explicit mluOpTensorDescriptorQueueStruct(size_t n) {
    extend_num = n;
    extend(extend_num);
    extend_num *= 2;
  }
  ~mluOpTensorDescriptorQueueStruct() {
    for (auto it : this->headers) {
      delete[] it;
    }
  }
  std::queue<mluOpTensorDescriptor_t> queue;
  std::list<mluOpTensorStruct *> headers;
  std::atomic_flag flag = ATOMIC_FLAG_INIT;
  inline void lock() {
    while (flag.test_and_set(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }
  inline void unlock() { flag.clear(std::memory_order_release); }
  inline void extend(size_t n) {
    mluOpTensorStruct *header = new (std::nothrow) mluOpTensorStruct[n];
    headers.emplace_back(header);
    for (size_t i = 0; i < n; ++i) {
      mluOpTensorStruct *desc = header + i;
      desc->init();  // reset random value.
      queue.emplace(desc);
    }
  }
  size_t extend_num = 100;
};
#endif

inline int mluOpDataTypeBytes(const mluOpDataType_t dt) {
  return mluop::getSizeOfDataType(dt);
}

inline int mluOpGetTensordimN(const mluOpTensorDescriptor_t desc) {
  switch (desc->layout) {
    case MLUOP_LAYOUT_NCHW:
    case MLUOP_LAYOUT_NHWC:
    case MLUOP_LAYOUT_NDHWC:
      return desc->dims[0];
    case MLUOP_LAYOUT_NCDHW:
      return desc->dims[0];
    case MLUOP_LAYOUT_HWCN:
      return desc->dims[3];
    default:
      LOG(ERROR) << "Failed to call dimN, illegal layout in "
                    "TensorDescriptor.\n";
  }
  return 0;
}

inline int mluOpGetTensordimD(const mluOpTensorDescriptor_t desc) {
  switch (desc->layout) {
    case MLUOP_LAYOUT_NDHWC:
      return desc->dims[1];
    case MLUOP_LAYOUT_NCDHW:
      return desc->dims[2];
    default:
      LOG(ERROR) << "Failed to call dimD, illegal layout in "
                    "TensorDescriptor.\n";
  }
  return 0;
}

inline int mluOpGetTensordimC(const mluOpTensorDescriptor_t desc) {
  switch (desc->layout) {
    case MLUOP_LAYOUT_NCHW:
      return desc->dims[1];
    case MLUOP_LAYOUT_NHWC:
      return desc->dims[3];
    case MLUOP_LAYOUT_NDHWC:
      return desc->dims[4];
    case MLUOP_LAYOUT_NCDHW:
      return desc->dims[1];
    case MLUOP_LAYOUT_HWCN:
      return desc->dims[2];
    default:
      LOG(ERROR) << "Failed to call dimC, illegal layout in "
                    "TensorDescriptor.\n";
  }
  return 0;
}

inline int mluOpGetTensordimH(const mluOpTensorDescriptor_t desc) {
  switch (desc->layout) {
    case MLUOP_LAYOUT_NCHW:
      return desc->dims[2];
    case MLUOP_LAYOUT_NHWC:
      return desc->dims[1];
    case MLUOP_LAYOUT_NDHWC:
      return desc->dims[2];
    case MLUOP_LAYOUT_NCDHW:
      return desc->dims[3];
    case MLUOP_LAYOUT_HWCN:
      return desc->dims[0];
    default:
      LOG(ERROR) << "Failed to call dimH, illegal layout in "
                    "TensorDescriptor.\n";
  }
  return 0;
}

inline int mluOpGetTensordimW(const mluOpTensorDescriptor_t desc) {
  switch (desc->layout) {
    case MLUOP_LAYOUT_NCHW:
      return desc->dims[3];
    case MLUOP_LAYOUT_NHWC:
      return desc->dims[2];
    case MLUOP_LAYOUT_NDHWC:
      return desc->dims[3];
    case MLUOP_LAYOUT_NCDHW:
      return desc->dims[4];
    case MLUOP_LAYOUT_HWCN:
      return desc->dims[1];
    default:
      LOG(ERROR) << "Failed to call dimW, illegal layout in "
                    "TensorDescriptor.\n";
  }
  return 0;
}

#endif  // CORE_TENSOR_H_
