/*************************************************************************
 * Copyright (C) [2026] by Cambricon, Inc.
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

#include "mlu_op.h"

#include <string>

#include "core/log_core/log_engine.h"

extern "C" {

mluOpStatus_t MLUOP_WIN_API mluOpSetMaxLogLevel(MluOpLogLevel level) {
  mluop::log::LogEngine::setMaxLogLevel(static_cast<int>(level));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetLogColorPrint(int color) {
  mluop::log::LogEngine::instance().setColorPrint(color != 0);
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetLogFile(const char* file_name) {
  if (file_name == nullptr) {
    // 空指针：关闭日志落盘。
    mluop::log::LogEngine::instance().disableLogFile();
    return MLUOP_STATUS_SUCCESS;
  }
  if (file_name[0] == '\0') {
    // 空串：非法参数。
    return MLUOP_STATUS_BAD_PARAM;
  }
  mluop::log::LogEngine::instance().setLogFile(std::string(file_name));
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetLogCallback(MluOpLogCallback callback) {
  mluop::log::LogEngine::instance().setLogCallback(
      reinterpret_cast<mluop::log::LogCallback>(callback));
  return MLUOP_STATUS_SUCCESS;
}

}  // extern "C"
