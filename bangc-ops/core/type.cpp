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
#include "core/type.h"
#define to_string(a) #a

namespace mluop {
size_t getSizeOfDataType(mluOpDataType_t dtype) {
  switch (dtype) {
    case MLUOP_DTYPE_BOOL:
    case MLUOP_DTYPE_INT8:
    case MLUOP_DTYPE_UINT8: {
      return 1;
    }
    case MLUOP_DTYPE_INT16:
    case MLUOP_DTYPE_UINT16:
    case MLUOP_DTYPE_HALF: {
      return 2;
    }
    case MLUOP_DTYPE_INT32:
    case MLUOP_DTYPE_UINT32:
    case MLUOP_DTYPE_FLOAT:
    case MLUOP_DTYPE_COMPLEX_HALF: {
      return 4;
    }
    case MLUOP_DTYPE_INT64:
    case MLUOP_DTYPE_UINT64:
    case MLUOP_DTYPE_DOUBLE:
    case MLUOP_DTYPE_COMPLEX_FLOAT: {
      return 8;
    }
    default: {
      return 0;
    }
  }
}

std::string getNameOfDataType(const mluOpDataType_t dtype) {
  std::string dtype_name;
  switch (dtype) {
    case MLUOP_DTYPE_BOOL: {
      dtype_name = to_string(DTYPE_BOOL);
    } break;
    case MLUOP_DTYPE_INT8: {
      dtype_name = to_string(DTYPE_INT8);
    } break;
    case MLUOP_DTYPE_UINT8: {
      dtype_name = to_string(DTYPE_UINT8);
    } break;
    case MLUOP_DTYPE_INT16: {
      dtype_name = to_string(DTYPE_INT16);
    } break;
    case MLUOP_DTYPE_UINT16: {
      dtype_name = to_string(DTYPE_UINT16);
    } break;
    case MLUOP_DTYPE_INT32: {
      dtype_name = to_string(DTYPE_INT32);
    } break;
    case MLUOP_DTYPE_UINT32: {
      dtype_name = to_string(DTYPE_UINT32);
    } break;
    case MLUOP_DTYPE_INT64: {
      dtype_name = to_string(DTYPE_INT64);
    } break;
    case MLUOP_DTYPE_UINT64: {
      dtype_name = to_string(DTYPE_UINT64);
    } break;
    case MLUOP_DTYPE_HALF: {
      dtype_name = to_string(DTYPE_HALF);
    } break;
    case MLUOP_DTYPE_FLOAT: {
      dtype_name = to_string(DTYPE_FLOAT);
    } break;
    case MLUOP_DTYPE_DOUBLE: {
      dtype_name = to_string(DTYPE_DOUBLE);
    } break;
    case MLUOP_DTYPE_COMPLEX_HALF: {
      dtype_name = to_string(DTYPE_COMPLEX_HALF);
    } break;
    case MLUOP_DTYPE_COMPLEX_FLOAT: {
      dtype_name = to_string(DTYPE_COMPLEX_FLOAT);
    } break;
    default: {
      dtype_name = "DTYPE_INVALID";
    } break;
  }
  return dtype_name;
}

std::string getNameOfTensorLayout(const mluOpTensorLayout_t layout) {
  std::string layout_name;
  switch (layout) {
    case MLUOP_LAYOUT_NCHW: {
      layout_name = to_string(LAYOUT_NCHW);
    } break;
    case MLUOP_LAYOUT_NHWC: {
      layout_name = to_string(LAYOUT_NHWC);
    } break;
    case MLUOP_LAYOUT_HWCN: {
      layout_name = to_string(LAYOUT_HWCN);
    } break;
    case MLUOP_LAYOUT_NDHWC: {
      layout_name = to_string(LAYOUT_NDHWC);
    } break;
    case MLUOP_LAYOUT_ARRAY: {
      layout_name = to_string(LAYOUT_ARRAY);
    } break;
    case MLUOP_LAYOUT_NCDHW: {
      layout_name = to_string(LAYOUT_NCDHW);
    } break;
    case MLUOP_LAYOUT_TNC: {
      layout_name = to_string(LAYOUT_TNC);
    } break;
    case MLUOP_LAYOUT_NTC: {
      layout_name = to_string(LAYOUT_NTC);
    } break;
    case MLUOP_LAYOUT_NLC: {
      layout_name = to_string(LAYOUT_NLC);
    } break;
    case MLUOP_LAYOUT_NC: {
      layout_name = to_string(LAYOUT_NC);
    } break;
    default: {
      layout_name = "LAYOUT_ARRAY";
      break;
    }
  }
  return layout_name;
}
}  // namespace mluop
