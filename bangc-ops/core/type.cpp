/*************************************************************************
 * Copyright (C) 2021 by Cambricon, Inc. All rights reserved.
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
    case MLUOP_DTYPE_HALF: {
      return 2;
    }
    case MLUOP_DTYPE_INT31:
    case MLUOP_DTYPE_INT32:
    case MLUOP_DTYPE_FLOAT: {
      return 4;
    }
    case MLUOP_DTYPE_INT64: {
      return 8;
    }
  }
}

cnrtDataType_t toCnrtDataType(mluOpDataType_t dtype) {
  switch (dtype) {
    default: {
      return cnrtInvalid;
    }
    case MLUOP_DTYPE_HALF: {
      return cnrtFloat16;
    }
    case MLUOP_DTYPE_FLOAT: {
      return cnrtFloat32;
    }
    case MLUOP_DTYPE_INT8: {
      return cnrtInt8;
    }
    case MLUOP_DTYPE_INT16: {
      return cnrtInt16;
    }
    case MLUOP_DTYPE_INT32: {
      return cnrtInt32;
    }
    case MLUOP_DTYPE_UINT8: {
      return cnrtUInt8;
    }
    case MLUOP_DTYPE_BOOL: {
      return cnrtBool;
    }
  }
}
