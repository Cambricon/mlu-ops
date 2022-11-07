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

#ifndef TOC_RUNTIME_RUNTIME_TENSOR_H_
#define TOC_RUNTIME_RUNTIME_TENSOR_H_

#include <vector>
#include <list>
#include <memory>
#include <queue>
#include <thread>  // NOLINT
#include <atomic>
#include <cstring>
#include "cnrt.h"
#include "cn_api.h"

#define CONTEXT_DEVICENAME_BUFFER_SIZE 64
#define CONTEXT_DEVICENAME_LEAST_SIZE 6
#define RUNTIME_DIM_MAX 8
#define RUNTIME_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define RUNTIME_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))


/******************************************************************************
 * RUNTIME Data Type
 ******************************************************************************/
/*! @brief Enumeration variables describing the data types in RUNTIME. */
typedef enum {
  RUNTIME_DTYPE_INVALID = 0, /*!< The data is an invalid data type. */
  RUNTIME_DTYPE_HALF = 1,    /*!< The data is a 16-bit floating-point data type. */
  RUNTIME_DTYPE_FLOAT = 2,   /*!< The data is a 32-bit floating-point data type. */
  RUNTIME_DTYPE_INT8 = 3,    /*!< The data is a 8-bit signed integer data type. */
  RUNTIME_DTYPE_INT16 = 4,   /*!< The data is a 16-bit signed integer data type. */
  RUNTIME_DTYPE_INT31 = 5,   /*!< The data is a 31-bit signed integer data type. */
  RUNTIME_DTYPE_INT32 = 6,   /*!< The data is a 32-bit signed integer data type. */
  RUNTIME_DTYPE_INT64 = 9,   /*!< The data is a 64-bit signed integer data type. */
  RUNTIME_DTYPE_UINT8 = 7,   /*!< The data is a 8-bit unsigned integer data type. */
  RUNTIME_DTYPE_UINT16 = 13, /*!< The data is a 16-bit unsigned integer data type. */
  RUNTIME_DTYPE_UINT32 = 11, /*!< The data is a 32-bit unsigned integer data type. */
  RUNTIME_DTYPE_UINT64 = 12, /*!< The data is a 64-bit unsigned integer data type. */
  RUNTIME_DTYPE_BOOL = 8,    /*!< The data is a boolean data type. */
} RuntimeDataType_t;

/******************************************************************************
 * RUNTIME Return Status
 ******************************************************************************/
/*! @brief Enumeration variables describing function return status.
 */
typedef enum {
  RUNTIME_STATUS_SUCCESS         = 0, /*!< The operation was successfully completed. */
  RUNTIME_STATUS_NOT_INITIALIZED = 1,
  /*!< RUNTIME library was not initialized properly, which is usually caused by the
       failure of calling ::runtimeCreate, ::RuntimeCreateTensorDescriptor or ::RuntimeSetTensorDescriptor.
       Such error is usually due to incompatible MLU device or invalid driver environment.
       Notice that ::runtimeCreate should be called prior to any other runtime functions.*/
  RUNTIME_STATUS_ALLOC_FAILED = 2,
  /*!< This error occurs when the resource allocation failed, usually caused by the failure
       of cnMallocHost, probably because of the exceeded memory usage. Please make sure that
       the memory allocated previously is deallocated as much as possible.*/
  RUNTIME_STATUS_BAD_PARAM = 3,
  /*!< Invalid value or parameters passed to the function, including data type, layout,
       dimensions, etc.*/
  RUNTIME_STATUS_INTERNAL_ERROR = 4,
  /*!< Error occurred inside of the function, which may indicate an internal error or bug in
       the library. This error is usually due to the failure of cnrtMemcpyAsync.
       Please check whether the memory passed to the function was deallocated before the completion
       of the routine.*/
  RUNTIME_STATUS_ARCH_MISMATCH = 5,
  /*!< Invalid MLU device which was not supported by current function.*/
  RUNTIME_STATUS_EXECUTION_FAILED = 6,
  /*!< Error occurred when the function failed to execute on MLU device due to multiple reasons.
       You can check whether the hardware environment, driver version and other prerequisite
       libraries are correctly installed. For more information about prerequisite libraries,
       see "Cambricon RUNTIME User Guide".*/
  RUNTIME_STATUS_NOT_SUPPORTED = 7,
  /*!< Error when the requested functionality was not supported in
       this version but would be supported in the future. */
  RUNTIME_STATUS_NUMERICAL_OVERFLOW = 8,
  /*!< Numerical overflow occurred when executing the function,
       which is usually due to large scale or inappropriate range of value of input tensor.*/
} RuntimeStatus_t;


struct RuntimeTensorStruct {
  RuntimeTensorStruct()
      : dim(0),
        dtype(RUNTIME_DTYPE_FLOAT),
        onchip_dtype(RUNTIME_DTYPE_INVALID),
        // layout(RUNTIME_LAYOUT_ARRAY),
        position(0),
        scale(1.0),
        offset(0) {
    /* explicit set initial values for document use.
     */
  }
  ~RuntimeTensorStruct() {
    /* please do NOT implement any codes here.
     * a state-less struct should not hold any resources.
     */
  }
  /* struct */
  int dim               = 0;
  uint64_t total_element_num = 0;
  uint64_t total_tensor_size = 0;
  // if dimNb > RUNTIME_DIM_MAX (8), using larger_dims, malloc it and dims point it.
  // else, using normal_dims, dont need malloc and free.
  int normal_dims[RUNTIME_DIM_MAX] = {-1};
  int *larger_dims              = NULL;
  int *dims                     = normal_dims;  // point the normal dims as default

  int normal_strides[RUNTIME_DIM_MAX] = {-1};
  int *larger_strides              = NULL;
  int *strides                     = normal_strides;  // point the normal strides as default

  RuntimeDataType_t dtype;
  RuntimeDataType_t onchip_dtype;
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
    if (RUNTIME_PREDICT_FALSE(larger_dims != NULL)) {
      delete[] larger_dims;
      larger_dims = NULL;
    }
    if (RUNTIME_PREDICT_FALSE(larger_strides != NULL)) {
      delete[] larger_strides;
      larger_strides = NULL;
    }
    dims         = normal_dims;
    strides      = normal_strides;
    dtype        = RUNTIME_DTYPE_FLOAT;
    onchip_dtype = RUNTIME_DTYPE_INVALID;

    position = 0;
    scale    = 1.0f;
    offset   = 0;

    dim               = 0;
    total_element_num = 0;
    total_tensor_size = 0;
  }
};

typedef struct RuntimeTensorStruct *RuntimeTensorDescriptor_t;

RuntimeStatus_t RuntimeCreateTensorDescriptor(RuntimeTensorDescriptor_t *desc);

RuntimeStatus_t RuntimeSetTensorDescriptor(RuntimeTensorDescriptor_t desc,
                                           RuntimeDataType_t dtype,
                                           int dimNb,
                                           const int *dimSize);

typedef enum {
  RUNTIME_UNKNOWN_DEVICE = 0,
  // RUNTIME_MLU100 = 100,
  RUNTIME_MLU220 = 220,
  RUNTIME_MLU270 = 270,
  RUNTIME_MLU290 = 290,
  RUNTIME_CE3226 = 322,
  RUNTIME_MLU370 = 372,
  RUNTIME_MLU590 = 592,
} RuntimeDevType_t;

struct RuntimeDeviceName {
  char name[CONTEXT_DEVICENAME_BUFFER_SIZE];
  RuntimeDevType_t type;
};

struct RuntimeContext {
  CNdev device;
  cnrtQueue_t queue;
  RuntimeDevType_t arch;  // return arch type. e.g. RUNTIME_MLU270
  int32_t cluster_num;
  int32_t core_num_per_cluster;
  int32_t nram_size;
  int32_t wram_size;
  int32_t sram_size;
  int32_t capability_cluster_num;
  int32_t capability_job_limit;
};

typedef struct RuntimeContext *RuntimeHandle_t;

size_t GetSizeOfDataType(RuntimeDataType_t dtype);

union BangArgUnion32 {
  int32_t v_int32;
  uint32_t v_uint32;
  float v_float32;
};
#endif  // TOC_RUNTIME_RUNTIME_TENSOR_H_
