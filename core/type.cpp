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
#include "core/preprocessor.h"
#include "core/type.h"
#define to_string(a) #a

#define ENUM_CASE_HANDLE(e) \
  case e: {                 \
    return to_string(e);    \
  }
#define ENUM_CASE_NO_PREFIX_HANDLE(e) \
  case MLUOP_##e: {                   \
    return to_string(e);              \
  }

#define MLUOP_STATUS_ENUM_LIST                                   \
  MLUOP_STATUS_SUCCESS, MLUOP_STATUS_NOT_INITIALIZED,            \
      MLUOP_STATUS_ALLOC_FAILED, MLUOP_STATUS_BAD_PARAM,         \
      MLUOP_STATUS_INTERNAL_ERROR, MLUOP_STATUS_ARCH_MISMATCH,   \
      MLUOP_STATUS_EXECUTION_FAILED, MLUOP_STATUS_NOT_SUPPORTED, \
      MLUOP_STATUS_NUMERICAL_OVERFLOW

#define MLUOP_DATA_TYPE_ENUM_NO_PREFIX_LIST                                    \
  DTYPE_BOOL, DTYPE_INT8, DTYPE_UINT8, DTYPE_INT16, DTYPE_UINT16, DTYPE_INT31, \
      DTYPE_INT32, DTYPE_UINT32, DTYPE_INT64, DTYPE_UINT64, DTYPE_HALF,        \
      DTYPE_BFLOAT16, DTYPE_FLOAT, DTYPE_DOUBLE, DTYPE_COMPLEX_HALF,           \
      DTYPE_COMPLEX_FLOAT

#define MLUOP_TENSOR_LAYOUT_ENUM_NO_PREFIX_LIST                      \
  LAYOUT_NCHW, LAYOUT_NHWC, LAYOUT_HWCN, LAYOUT_NDHWC, LAYOUT_ARRAY, \
      LAYOUT_NCDHW, LAYOUT_TNC, LAYOUT_NTC, LAYOUT_NC, LAYOUT_NLC, LAYOUT_NCL

const char* MLUOP_WIN_API mluOpGetErrorString(mluOpStatus_t status) {
  CHECK_GE(status, 0);

  switch (status) { MLUOP_PP_MAP(ENUM_CASE_HANDLE, (MLUOP_STATUS_ENUM_LIST)); }
  return "MLUOP_STATUS_UNKNOWN";
}

const char* MLUOP_WIN_API mluOpGetNameOfDataType(mluOpDataType_t dtype) {
  switch (dtype) {
    MLUOP_PP_MAP(ENUM_CASE_NO_PREFIX_HANDLE,
                 (MLUOP_DATA_TYPE_ENUM_NO_PREFIX_LIST));
  }
  return "DTYPE_INVALID";
}

const char* MLUOP_WIN_API
mluOpGetNameOfTensorLayout(mluOpTensorLayout_t layout) {
  switch (layout) {
    MLUOP_PP_MAP(ENUM_CASE_NO_PREFIX_HANDLE,
                 (MLUOP_TENSOR_LAYOUT_ENUM_NO_PREFIX_LIST));
  }
  return "LAYOUT_ARRAY";
}

namespace mluop {

std::string MLUOP_WIN_API MLUOP_ATTRIBUTE_FLATTEN getNameOfDataType(mluOpDataType_t dtype) {  // NOLINT
  return mluOpGetNameOfDataType(dtype);
}

std::string MLUOP_WIN_API MLUOP_ATTRIBUTE_FLATTEN getNameOfTensorLayout(mluOpTensorLayout_t layout) {  // NOLINT
  return mluOpGetNameOfTensorLayout(layout);
}

size_t getSizeOfDataType(mluOpDataType_t dtype) {
  switch (dtype) {
    default: {
      return 0;
    }
    case MLUOP_DTYPE_BOOL:
    case MLUOP_DTYPE_INT8:
    case MLUOP_DTYPE_UINT8: {
      return 1;
    }
    case MLUOP_DTYPE_INT16:
    case MLUOP_DTYPE_UINT16:
    case MLUOP_DTYPE_HALF:
    case MLUOP_DTYPE_BFLOAT16: {
      return 2;
    }
    case MLUOP_DTYPE_INT31:
    case MLUOP_DTYPE_INT32:
    case MLUOP_DTYPE_UINT32:
    case MLUOP_DTYPE_FLOAT:
    case MLUOP_DTYPE_COMPLEX_HALF: {
      return 4;
    }
    case MLUOP_DTYPE_UINT64:
    case MLUOP_DTYPE_INT64:
    case MLUOP_DTYPE_DOUBLE:
    case MLUOP_DTYPE_COMPLEX_FLOAT: {
      return 8;
    }
  }
}

}  // namespace mluop
