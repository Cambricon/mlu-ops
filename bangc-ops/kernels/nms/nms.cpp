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
#include "kernels/kernel_wrapper/wrapper.h"

mluOpStatus_t MLUOP_WIN_API
mluOpNms(mluOpHandle_t handle, const mluOpNmsDescriptor_t nms_desc,
         const mluOpTensorDescriptor_t boxes_desc, const void *boxes,
         const mluOpTensorDescriptor_t confidence_desc, const void *confidence,
         void *workspace, size_t workspace_size,
         const mluOpTensorDescriptor_t output_desc, void *output,
         void *output_size) {
  nmsWrapper wrapper;
  mluOpStatus_t ret = wrapper.invoke(
      handle, nms_desc, boxes_desc, boxes, confidence_desc, confidence,
      workspace, workspace_size, output_desc, output, output_size);
  return ret;
}
