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

#include <string>
#include <stdexcept>
#include "mlu_op.h"
#include "core/logging.h"

const char *mluOpGetErrorString(mluOpStatus_t status) {
  CHECK_GE(status, 0);

  switch (status) {
    default: {
      return (char *)"MLUOP_STATUS_UNKNOWN";
    }
    case MLUOP_STATUS_SUCCESS: {
      return (char *)"MLUOP_STATUS_SUCCESS";
    }
    case MLUOP_STATUS_NOT_INITIALIZED: {
      return (char *)"MLUOP_STATUS_NOT_INITIALIZED";
    }
    case MLUOP_STATUS_ALLOC_FAILED: {
      return (char *)"MLUOP_STATUS_ALLOC_FAILED";
    }
    case MLUOP_STATUS_BAD_PARAM: {
      return (char *)"MLUOP_STATUS_BAD_PARAM";
    }
    case MLUOP_STATUS_INTERNAL_ERROR: {
      return (char *)"MLUOP_STATUS_INTERNAL_ERROR";
    }
    case MLUOP_STATUS_ARCH_MISMATCH: {
      return (char *)"MLUOP_STATUS_MISMATCH";
    }
    case MLUOP_STATUS_EXECUTION_FAILED: {
      return (char *)"MLUOP_STATUS_EXECUTION_FAILED";
    }
    case MLUOP_STATUS_NOT_SUPPORTED: {
      return (char *)"MLUOP_STATUS_NOT_SUPPORTED";
    }
    case MLUOP_STATUS_NUMERICAL_OVERFLOW: {
      return (char *)"MLUOP_STATUS_NUMERICAL_OVERFLOW";
    }
  }
}

void mluOpCheck(mluOpStatus_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {
    std::string error = "\"" + std::string(mluOpGetErrorString(result)) +
                        " in " + std::string(func) + "\"";
    LOG(ERROR) << error;
    throw std::runtime_error(error);
  }
}
