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
#include "core/mlu_op_core.h"
#include "core/macros.h"
#include "core/logging.h"
#include "core/type.h"

#define QUEUE_ARRAY_LENGTH 4

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
  int dim               = 0;
  int total_element_num = 0;
  int total_tensor_size = 0;
  // if dimNb > MLUOP_DIM_MAX (8), using larger_dims, malloc it and dims point
  // it. else, using normal_dims, dont need malloc and free.
  int normal_dims[MLUOP_DIM_MAX] = {-1};
  int *larger_dims               = NULL;
  int *dims = normal_dims;  // point the normal dims as default

  int normal_strides[MLUOP_DIM_MAX] = {-1};
  int *larger_strides               = NULL;
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
    larger_dims    = NULL;
    larger_strides = NULL;

    dim               = 0;
    total_element_num = 0;
    total_tensor_size = 0;
    dims              = normal_dims;
    strides           = normal_strides;
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
    dims         = normal_dims;
    strides      = normal_strides;
    dtype        = MLUOP_DTYPE_FLOAT;
    onchip_dtype = MLUOP_DTYPE_INVALID;
    layout       = MLUOP_LAYOUT_ARRAY;

    position = 0;
    scale    = 1.0f;
    offset   = 0;

    dim               = 0;
    total_element_num = 0;
    total_tensor_size = 0;
  }
};

inline int mluOpDataTypeBytes(const mluOpDataType_t dt) {
  switch (dt) {
    case MLUOP_DTYPE_HALF:
      return 2;
    case MLUOP_DTYPE_FLOAT:
      return 4;
    case MLUOP_DTYPE_INT8:
    case MLUOP_DTYPE_UINT8:
    case MLUOP_DTYPE_BOOL:
      return 1;
    case MLUOP_DTYPE_INT16:
      return 2;
    // case MLUOP_DTYPE_INT23:   return 3;
    case MLUOP_DTYPE_INT32:
      return 4;
    case MLUOP_DTYPE_INT64:
      return 8;
    default:
      return -1;
  }
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
