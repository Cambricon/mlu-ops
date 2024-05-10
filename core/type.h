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
#ifndef CORE_TYPE_H_
#define CORE_TYPE_H_

#include <string>
#include "core/logging.h"
#include "mlu_op.h"

namespace mluop {
// This function is used to get high 32bit and low 32bit of param value.
// The hardware hasn't support 8 bytes operation, so if the sizeof(dtype) is 8
// bytes, sometimes we need to separate 8bytes to two 4bytes. Example:
// uint32_t high_value = 0, low_value = 0;
// if (getSizeOfDataType(dtype) == sizeof(int64_t)) {
//   getLowAndHighValueFrom64Bits(*(int64_t*)padding_value_ptr, &high_value,
//   &low_value);
// }
template <typename T>
static mluOpStatus_t getLowAndHighValueFrom64Bits(T value, uint32_t* high,
                                                  uint32_t* low) {
  if (sizeof(T) != sizeof(int64_t)) {
    VLOG(5)
        << "getLowAndHighValueFrom64Bits() only supports 64 bits data type.";
    return MLUOP_STATUS_INTERNAL_ERROR;
  }
  uint64_t temp = *(uint64_t*)&value;
  // get the high 32bit value
  *high = temp >> 32;
  // get the low 32bit value
  *low = temp;
  return MLUOP_STATUS_SUCCESS;
}

static inline size_t MLUOP_WIN_API getSizeOfDataType(mluOpDataType_t dtype) {
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

std::string MLUOP_WIN_API getNameOfDataType(mluOpDataType_t dtype);  // NOLINT
std::string MLUOP_WIN_API getNameOfTensorLayout(mluOpTensorLayout_t layout);  // NOLINT

}  // namespace mluop

extern "C" {
// TODO(None) consider move into mlu_op.h
const char* MLUOP_WIN_API mluOpGetNameOfDataType(mluOpDataType_t dtype);
const char* MLUOP_WIN_API
mluOpGetNameOfTensorLayout(mluOpTensorLayout_t layout);
}

#endif  // CORE_TYPE_H_
