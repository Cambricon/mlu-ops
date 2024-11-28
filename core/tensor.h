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
#include <string>
#include "mlu_op.h"
#include "core/macros.h"
#include "core/logging.h"
#include "core/type.h"

#define QUEUE_ARRAY_LENGTH 4

struct alignas(64) mluOpTensorStruct {
  /** default constructor */
  mluOpTensorStruct() = default;

  /** copy constructor */
  mluOpTensorStruct(mluOpTensorStruct const &other) { *this = other; }

  /** move constructor */
  mluOpTensorStruct(mluOpTensorStruct const &&) = delete;

  /** destructor */
  ~mluOpTensorStruct() {
    if MLUOP_PREDICT_FALSE (dims != normal_dims) {
      delete[] dims;
    }
    if MLUOP_PREDICT_FALSE (strides != normal_strides) {
      delete[] strides;
    }
  }

  /** copy assignment operator */
  mluOpTensorStruct &operator=(mluOpTensorStruct const &other) {
    if (dim > MLUOP_DIM_MAX && (dim < other.dim || other.dim < MLUOP_DIM_MAX)) {
      delete[] dims;
      delete[] strides;
      if (other.dim < MLUOP_DIM_MAX) {
        dims = normal_dims;
        strides = normal_strides;
      } else {
        dims = new (std::nothrow) int64_t[dim];
        strides = new (std::nothrow) int64_t[dim];
      }
    }

    dim = other.dim;
    dtype = other.dtype;
    layout = other.layout;
    onchip_dtype = other.onchip_dtype;
    pointer_mode = other.pointer_mode;

    total_element_num = other.total_element_num;
    total_tensor_size = other.total_tensor_size;

    memcpy(dims, other.dims, sizeof(int64_t) * dim);
    memcpy(strides, other.strides, sizeof(int64_t) * dim);

    position = other.position;
    scale = other.scale;
    offset = other.offset;

    positions = other.positions;
    scales = other.scales;
    offsets = other.offsets;

    return *this;
  }

  mluOpTensorStruct &operator=(mluOpTensorStruct const &&other) = delete;

  /* methods */
  mluOpStatus_t tensorDimN(size_t &dim);
  mluOpStatus_t tensorDimC(size_t &dim);
  mluOpStatus_t tensorDimH(size_t &dim);
  mluOpStatus_t tensorDimW(size_t &dim);

  inline bool isSameDims(const mluOpTensorStruct &other) const;
  inline bool isSameDims(const mluOpTensorStruct *other) const;
  inline bool isCpuScalar() const;

 public:
  inline float getOffset() const { return this->offset; }
  inline void setOffset(float newOffset) { this->offset = newOffset; }

  inline float getScale() const { return this->scale; }
  inline void setScale(float newScale) { this->scale = newScale; }

  inline mluOpTensorLayout_t getLayout() const { return this->layout; }
  inline void setLayout(mluOpTensorLayout_t newLayout) {
    this->layout = newLayout;
  }

  inline uint64_t getTotalTensorSize() const { return this->total_tensor_size; }
  inline void setTotalTensorSize(uint64_t newSize) {
    this->total_tensor_size = newSize;
  }
  inline uint64_t getTotalElementNum() const { return this->total_element_num; }
  inline void setTotalElementNum(uint64_t newNum) {
    this->total_element_num = newNum;
  }

  inline int getPosition() const { return this->position; }
  inline void setPosition(int newPosition) { this->position = newPosition; }

  inline mluOpDataType_t getDtype() const { return this->dtype; }
  inline void setDtype(mluOpDataType_t newDtype) { this->dtype = newDtype; }
  inline mluOpDataType_t getOnchipDtype() const { return this->onchip_dtype; }
  inline void setOnchipDtype(mluOpDataType_t newDtype) {
    this->onchip_dtype = newDtype;
  }

  inline int getDim() const { return this->dim; }
  inline void setDim(int newDim) { this->dim = newDim; }

  inline void releaseDims() { delete[] this->dims; }
  inline int64_t *getDims() const { return this->dims; }
  inline int64_t getDimIndex(size_t index) {
    if (index >= this->dim) {
      throw std::out_of_range("Index out of range");
    }
    return (this->dims)[index];
  }
  inline void setDims(int64_t *newDims) { this->dims = newDims; }

  inline void releaseStrides() { delete[] this->strides; }
  inline int64_t *getStrides() const { return this->strides; }
  inline int64_t getStrideIndex(size_t index) const {
    if (index >= this->dim) {
      throw std::out_of_range("Index out of range");
    }
    return (this->strides)[index];
  }
  inline void setStrideIndex(size_t index, int64_t newStride) {
    this->strides[index] = newStride;
  }

  inline void setStrides(int64_t *newStrides) { this->strides = newStrides; }

  inline mluOpPointerMode_t getPointerMode() const {
    return this->pointer_mode;
  }
  inline void setPointerMode(mluOpPointerMode_t new_pointer_mode) {
    this->pointer_mode = new_pointer_mode;
  }

  inline int64_t *getNormalDims() { return this->normal_dims; }
  inline int64_t *getNormalStrides() { return this->normal_strides; }
  // Definition of function in tensor.cpp
  static inline mluOpStatus_t mluOpSetTensorDescriptorZeroDim(mluOpTensorDescriptor_t desc);
  mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptor(
      mluOpTensorDescriptor_t desc, mluOpTensorLayout_t layout,
      mluOpDataType_t dtype, int dimNb, const int *dimSize);

  mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptor_v2(
      mluOpTensorDescriptor_t desc, mluOpTensorLayout_t layout,
      mluOpDataType_t dtype, int dimNb, const int64_t *dimSize);

  static inline void mluOpSetTensorDescriptorDimBase(
       mluOpTensorDescriptor_t desc, int dimNb);

  mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorDim(
       mluOpTensorDescriptor_t desc, int dimNb, const int *dimSize);

  mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorDim_v2(
       mluOpTensorDescriptor_t desc, int dimNb, const int64_t *dimSize);

  mluOpStatus_t MLUOP_WIN_API
  mluOpResetTensorDescriptor( mluOpTensorDescriptor_t desc);

  mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorEx(
      mluOpTensorDescriptor_t desc, mluOpTensorLayout_t layout,
      mluOpDataType_t dtype, int dimNb, const int *dimSize,
      const int *dimStride);

  mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorEx_v2(
      mluOpTensorDescriptor_t desc, mluOpTensorLayout_t layout,
      mluOpDataType_t dtype, int dimNb, const int64_t *dimSize,
      const int64_t *dimStride);

  mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorOnchipDataType(
      mluOpTensorDescriptor_t desc, mluOpDataType_t onchip_dtype);

  mluOpStatus_t MLUOP_WIN_API
  mluOpSetTensorDescriptorPosition(mluOpTensorDescriptor_t desc, int position);

  mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorPositionAndScale(
  mluOpTensorDescriptor_t desc, int position, float scale);


  mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorPositionScaleAndOffset(
      mluOpTensorDescriptor_t desc, int position, float scale, int offset);

  mluOpStatus_t MLUOP_WIN_API mluOpSetTensorDescriptorPointerMode(
      mluOpTensorDescriptor_t desc, mluOpPointerMode_t pointer_mode);

  mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptorEx(
      const mluOpTensorDescriptor_t desc, mluOpTensorLayout_t *layout,
      mluOpDataType_t *dtype, int *dimNb, int *dimSize, int *dimStride);

  mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptorEx_v2(
    const mluOpTensorDescriptor_t desc, mluOpTensorLayout_t *layout,
    mluOpDataType_t *dtype, int *dimNb, int64_t *dimSize, int64_t *dimStride);

  mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptor(
      const mluOpTensorDescriptor_t desc, mluOpTensorLayout_t *layout,
      mluOpDataType_t *dtype, int *dimNb, int *dimSize);

  mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptor_v2(
      const mluOpTensorDescriptor_t desc, mluOpTensorLayout_t *layout,
      mluOpDataType_t *dtype, int *dimNb, int64_t *dimSize);

  mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptorOnchipDataType(
      const mluOpTensorDescriptor_t desc, mluOpDataType_t *onchip_dtype);

  mluOpStatus_t MLUOP_WIN_API
  mluOpGetTensorDescriptorPosition(mluOpTensorDescriptor_t desc, int *position);

  mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptorPositionAndScale(
      mluOpTensorDescriptor_t desc, int *position, float *scale);

  mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptorPositionScaleAndOffset(
      mluOpTensorDescriptor_t desc, int *position, float *scale, int *offset);

  mluOpStatus_t MLUOP_WIN_API mluOpGetTensorDescriptorPointerMode(
      mluOpTensorDescriptor_t desc, mluOpPointerMode_t *pointer_mode);

  mluOpStatus_t MLUOP_WIN_API
  mluOpDestroyTensorDescriptor(mluOpTensorDescriptor_t desc);

  uint64_t MLUOP_WIN_API
  mluOpGetTensorElementNum(mluOpTensorDescriptor_t desc);
  
  mluOpStatus_t MLUOP_WIN_API mluOpSetGroupTensorDescriptors(
    mluOpTensorDescriptor_t **group_desc,
    const mluOpTensorLayout_t *group_layout, const mluOpDataType_t *group_dtype,
    const int *group_dimNb, const int *group_dimSize, const int desc_num);

  mluOpStatus_t MLUOP_WIN_API mluOpSetGroupTensorDescriptors_v2(
    mluOpTensorDescriptor_t **group_desc,
    const mluOpTensorLayout_t *group_layout, const mluOpDataType_t *group_dtype,
    const int *group_dimNb, const int64_t *group_dimSize, const int desc_num);

  // private:
  /* Try to pack and align the struct */
  /*  ------------------- 64 Bytes - 1 -------------------*/
  int64_t normal_dims[MLUOP_DIM_MAX];

  /*  ------------------- 64 Bytes - 2 -------------------*/
  int64_t normal_strides[MLUOP_DIM_MAX];

  /*  ------------------- 64 Bytes - 3 -------------------*/
  /* Offset - 0 */
  uint64_t total_element_num = 0;
  uint64_t total_tensor_size = 0;
  int64_t *dims = normal_dims;        // point the normal dims as default
  int64_t *strides = normal_strides;  // point the normal strides as default
  /* Offset - 32 */
  int dim = 0;
  mluOpDataType_t dtype = MLUOP_DTYPE_FLOAT;
  mluOpDataType_t onchip_dtype = MLUOP_DTYPE_INVALID;
  mluOpTensorLayout_t layout = MLUOP_LAYOUT_ARRAY;
  mluOpPointerMode_t pointer_mode = MLUOP_POINTER_MODE_DEVICE;

  /* Offset - 52 */
  /* To be removed*/
  int position = 0;
  float scale = 1;
  int offset = 0;
  std::vector<int> positions;
  std::vector<float> scales;
  std::vector<int> offsets;
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
      tensor_set_size += tensor_set[i]->getTotalTensorSize();
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
      offset += tensor_set[i]->getTotalTensorSize();
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
    return this->tensor_set[0]->getDtype();
  }

  inline mluOpTensorLayout_t getLayout() const {
    CHECK(!this->tensor_set.empty());
    return this->tensor_set[0]->getLayout();
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
      offset += tensor_set[i]->getTotalTensorSize();
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

struct mluOpSeqDataStruct {
  mluOpSeqDataStruct()
      : dim(0),
        dtype(MLUOP_DTYPE_FLOAT),
        layout(MLUOP_SEQDATA_NBTC),
        position(0),
        scale(1.0),
        offset(0),
        padding_fill(nullptr) {
    /* explicit set initial values for document use.
     */
  }
  ~mluOpSeqDataStruct() {
    /* please do NOT implement any codes here.
     * a state-less struct should not hold any resources.
     */
  }
  /* methods */
  inline mluOpStatus_t seqDataElementsNumber(size_t &elements) const {
    uint64_t elements_counter = 1;
    for (size_t i = 0; i < dim; ++i) {
      elements_counter *= dims[i];
    }
    elements = elements_counter;
    return MLUOP_STATUS_SUCCESS;
  }
  inline int getSeqenceArrayBytes() const {
    int seq_array_size = 0;
    if (!seq_length.empty()) {
      seq_array_size = seq_length.size() * sizeof(int);
    }
    return seq_array_size;
  }
  /* struct */
  int dim;
  std::vector<int64_t> dims;
  // int* dims;
  mluOpDataType_t dtype;
  mluOpDataType_t onchip_dtype;
  mluOpSeqDataLayout_t layout;
  int64_t seq_length_size;
  std::vector<int64_t> seq_length;
  int position;
  float scale;
  int offset;
  void *padding_fill;
};

inline int mluOpDataTypeBytes(const mluOpDataType_t dt) {
  return mluop::getSizeOfDataType(dt);
}

inline int64_t mluOpGetTensordimN(const mluOpTensorDescriptor_t desc) {
  switch (desc->getLayout()) {
    case MLUOP_LAYOUT_NCHW:
    case MLUOP_LAYOUT_NHWC:
    case MLUOP_LAYOUT_NDHWC:
    case MLUOP_LAYOUT_NLC:
      return desc->getDimIndex(0);
    case MLUOP_LAYOUT_NCDHW:
      return desc->getDimIndex(0);
    case MLUOP_LAYOUT_HWCN:
      return desc->getDimIndex(3);
    default:
      LOG(ERROR)
          << "Failed to call dimN, illegal layout in TensorDescriptor.\n";
  }
  return 0;
}

inline int64_t mluOpGetTensordimD(const mluOpTensorDescriptor_t desc) {
  switch (desc->getLayout()) {
    case MLUOP_LAYOUT_NDHWC:
      return desc->getDimIndex(1);
    case MLUOP_LAYOUT_NCDHW:
      return desc->getDimIndex(2);
    default:
      LOG(ERROR)
          << "Failed to call dimD, illegal layout in TensorDescriptor.\n";
  }
  return 0;
}

inline int64_t mluOpGetTensordimC(const mluOpTensorDescriptor_t desc) {
  switch (desc->getLayout()) {
    case MLUOP_LAYOUT_NCHW:
      return desc->getDimIndex(1);
    case MLUOP_LAYOUT_NHWC:
      return desc->getDimIndex(3);
    case MLUOP_LAYOUT_NDHWC:
      return desc->getDimIndex(4);
    case MLUOP_LAYOUT_NCDHW:
      return desc->getDimIndex(1);
    case MLUOP_LAYOUT_HWCN:
    case MLUOP_LAYOUT_NLC:
      return desc->getDimIndex(2);
    default:
      LOG(ERROR)
          << "Failed to call dimC, illegal layout in TensorDescriptor.\n";
  }
  return 0;
}

inline int64_t mluOpGetTensordimH(const mluOpTensorDescriptor_t desc) {
  switch (desc->getLayout()) {
    case MLUOP_LAYOUT_NCHW:
      return desc->getDimIndex(2);
    case MLUOP_LAYOUT_NHWC:
      return desc->getDimIndex(1);
    case MLUOP_LAYOUT_NDHWC:
      return desc->getDimIndex(2);
    case MLUOP_LAYOUT_NCDHW:
      return desc->getDimIndex(3);
    case MLUOP_LAYOUT_HWCN:
      return desc->getDimIndex(0);
    default:
      LOG(ERROR)
          << "Failed to call dimH, illegal layout in TensorDescriptor.\n";
  }
  return 0;
}

inline int64_t mluOpGetTensordimW(const mluOpTensorDescriptor_t desc) {
  switch (desc->getLayout()) {
    case MLUOP_LAYOUT_NCHW:
      return desc->getDimIndex(3);
    case MLUOP_LAYOUT_NHWC:
      return desc->getDimIndex(2);
    case MLUOP_LAYOUT_NDHWC:
      return desc->getDimIndex(3);
    case MLUOP_LAYOUT_NCDHW:
      return desc->getDimIndex(4);
    case MLUOP_LAYOUT_HWCN:
    case MLUOP_LAYOUT_NLC:
      return desc->getDimIndex(1);
    default:
      LOG(ERROR)
          << "Failed to call dimW, illegal layout in TensorDescriptor.\n";
  }
  return 0;
}

uint64_t mluOpGetSeqDataElementNum(mluOpSeqDataDescriptor_t desc);

inline int64_t mluOpGetSeqDataDimN(const mluOpSeqDataDescriptor_t desc) {
  switch (desc->layout) {
    case MLUOP_SEQDATA_NBTC:
    case MLUOP_SEQDATA_NTBC:
    case MLUOP_SEQDATA_NC:
    case MLUOP_SEQDATA_NTC:
      return desc->dims[0];
    case MLUOP_SEQDATA_BNTC:
    case MLUOP_SEQDATA_TNBC:
    case MLUOP_SEQDATA_TNC:
    case MLUOP_SEQDATA_TNC_PACKED:
    case MLUOP_SEQDATA_TN:
      return desc->dims[1];
    case MLUOP_SEQDATA_BTNC:
    case MLUOP_SEQDATA_TBNC:
      return desc->dims[2];
    default:
      LOG(ERROR)
          << "Failed to call dimN, illegal layout in SeqDataDescriptor.\n";
  }
  return 0;
}

inline int64_t mluOpGetSeqDataDimB(const mluOpSeqDataDescriptor_t desc) {
  switch (desc->layout) {
    case MLUOP_SEQDATA_BNTC:
    case MLUOP_SEQDATA_BTNC:
      return desc->dims[0];
    case MLUOP_SEQDATA_TBNC:
    case MLUOP_SEQDATA_NBTC:
      return desc->dims[1];
    case MLUOP_SEQDATA_NTBC:
    case MLUOP_SEQDATA_TNBC:
      return desc->dims[2];
    default:
      LOG(ERROR)
          << "Failed to call dimB, illegal layout in SeqDataDescriptor.\n";
  }
  return 0;
}

inline int64_t mluOpGetSeqDataDimT(const mluOpSeqDataDescriptor_t desc) {
  switch (desc->layout) {
    case MLUOP_SEQDATA_TNC:
    case MLUOP_SEQDATA_TNC_PACKED:
    case MLUOP_SEQDATA_TNBC:
    case MLUOP_SEQDATA_TBNC:
    case MLUOP_SEQDATA_TN:
      return desc->dims[0];
    case MLUOP_SEQDATA_NTC:
    case MLUOP_SEQDATA_NTBC:
    case MLUOP_SEQDATA_BTNC:
      return desc->dims[1];
    case MLUOP_SEQDATA_NBTC:
    case MLUOP_SEQDATA_BNTC:
      return desc->dims[2];
    default:
      LOG(ERROR)
          << "Failed to call dimT, illegal layout in SeqDataDescriptor.\n";
  }
  return 0;
}

inline int64_t mluOpGetSeqDataDimC(const mluOpSeqDataDescriptor_t desc) {
  switch (desc->layout) {
    case MLUOP_SEQDATA_TNC:
    case MLUOP_SEQDATA_TNC_PACKED:
    case MLUOP_SEQDATA_NTC:
      return desc->dims[2];
    case MLUOP_SEQDATA_NC:
      return desc->dims[1];
    case MLUOP_SEQDATA_TNBC:
    case MLUOP_SEQDATA_TBNC:
    case MLUOP_SEQDATA_NBTC:
    case MLUOP_SEQDATA_NTBC:
    case MLUOP_SEQDATA_BNTC:
    case MLUOP_SEQDATA_BTNC:
      return desc->dims[3];
    default:
      LOG(ERROR)
          << "Failed to call dimC, illegal layout in SeqDataDescriptor.\n";
  }
  return 0;
}

inline uint64_t shapeStrideCount(const mluOpTensorDescriptor_t desc) {
  uint64_t total = 1;
  for (int i = 0; i < desc->getDim(); ++i) {
    if (desc->getDimIndex(i) == 0) {
      total = 0;
      break;
    }
    total += (desc->getDimIndex(i) - 1) * desc->getStrideIndex(i);
  }
  return total;
}

inline bool mluOpTensorStruct::isSameDims(
    const mluOpTensorStruct &other) const {
  if (dim == other.dim) {
    if (0 == memcmp(dims, other.dims, dim * sizeof(*dims))) {
      return true;
    }
  }
  return false;
}

inline bool mluOpTensorStruct::isSameDims(
    const mluOpTensorStruct *other) const {
  return isSameDims(*other);
}

inline bool mluOpTensorStruct::isCpuScalar() const {
  if (dim == 0 && pointer_mode == MLUOP_POINTER_MODE_HOST &&
      total_element_num == 1) {
    return true;
  }
  return false;
}
// Attention: Do not put operator data structures in this header file.

#endif  // CORE_TENSOR_H_
