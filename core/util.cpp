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

#include "core/logging.h"
#include "core/util.h"
#include "mlu_op.h"

void mluOpCheck(mluOpStatus_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {
    std::string error = "\"" + std::string(mluOpGetErrorString(result)) +
                        " in " + std::string(func) + "\"";
    LOG(ERROR) << error;
    // TODO(liuduanhui): Remove error throwing in c library in future.
    //                   MLUOP_CHECK should not be used in host side code.
    //                   And now it is only used in gtest code.
    throw std::runtime_error(error);
  }
}

bool isStrideTensor(const int dim, const int64_t *dims,
                    const int64_t *strides) {
  int64_t stride_base = 1;

  for (int i = dim - 1; i >= 0; i--) {
    if (dims[i] != 1 && strides[i] != stride_base) {
      return true;
    }

    stride_base *= dims[i];
  }

  return false;
}
