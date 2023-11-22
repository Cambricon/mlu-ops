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
#ifndef KERNELS_UTILS_CNNL_HELPER_H_
#define KERNELS_UTILS_CNNL_HELPER_H_

#include <iostream>

#include "core/logging.h"
#include "mlu_op.h"
#include "cnnl.h"

// Checks returned status(with propose_status)
// and if not equal returns status(ret_status)
#define CHECK_FUNC_RETURN(ret, propose_status, message, ret_status)  \
  {                                                                  \
    if (ret != propose_status) {                                     \
      LOG(ERROR)<< "MLUOPS STATUS CHECK:"<< message;                 \
      return ret_status;                                             \
    }                                                                \
  }

// Call CNNL API
#define CALL_CNNL(cnnl_call_ret)                                        \
  {                                                                     \
    mluOpStatus_t ret = mluOpStatus_t(cnnl_call_ret);                   \
    if (ret != MLUOP_STATUS_SUCCESS) {                                  \
      LOG(ERROR)<< "CNNL_HELPER: Internal cnnl api call error accured.";\
      return ret;                                                       \
    }                                                                   \
  }

// TensorDescriptor convert
cnnlStatus_t mluOpConvertDescriptor(mluOpTensorDescriptor_t desc,
                                    cnnlTensorDescriptor_t _desc);

cnnlStatus_t mluOpConvertHandle(mluOpHandle_t handle, cnnlHandle_t _handle);

// Pointer type force convert
template <typename STYPE, typename DTYPE>
DTYPE mluOpPointerForceConvert(STYPE ptr);

// TensorDescriptor
#define CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR(desc, _desc)                     \
  cnnlTensorDescriptor_t _desc;                                                \
  {                                                                            \
    if (desc != NULL) {                                                        \
      cnnlStatus_t ret = cnnlCreateTensorDescriptor(&_desc);                   \
      if (ret != CNNL_STATUS_SUCCESS) {                                        \
        LOG(ERROR)<< "CNNL_HELPER: CNNL creates tensor descriptor failed.";    \
        return MLUOP_STATUS_INTERNAL_ERROR;                                    \
      }                                                                        \
      ret = mluOpConvertDescriptor(desc, _desc);                               \
      if (ret != CNNL_STATUS_SUCCESS) {                                        \
        LOG(ERROR)<< "CNNL_HELPER: Internal convert tensor descriptor failed.";\
        return MLUOP_STATUS_INTERNAL_ERROR;                                    \
      }                                                                        \
    } else {                                                                   \
      _desc = NULL;                                                            \
    }                                                                          \
  }

#define CREATE_AND_SET_CNNL_TENSOR_DESCRIPTOR_V2(desc, _desc)                  \
  {                                                                            \
    cnnlStatus_t ret = cnnlCreateTensorDescriptor(&_desc);                     \
    if (ret != CNNL_STATUS_SUCCESS) {                                          \
      LOG(ERROR)<< "CNNL_HELPER: CNNL creates tensor descriptor failed.";      \
      return MLUOP_STATUS_INTERNAL_ERROR;                                      \
    }                                                                          \
    ret = mluOpConvertDescriptor(desc, _desc);                                 \
    if (ret != CNNL_STATUS_SUCCESS) {                                          \
      LOG(ERROR)<< "CNNL_HELPER: Internal convert tensor descriptor failed.";  \
      return MLUOP_STATUS_INTERNAL_ERROR;                                      \
    }                                                                          \
  }

#define DESTROY_CNNL_TENSOR_DESCRIPTOR(_desc)                                  \
  {                                                                            \
    if (_desc != NULL) {                                                       \
      cnnlStatus_t ret = cnnlDestroyTensorDescriptor(_desc);                   \
      if (ret != CNNL_STATUS_SUCCESS) {                                        \
        LOG(ERROR)<< "CNNL_HELPER: CNNL destroy tensor descriptor failed.";    \
        return MLUOP_STATUS_INTERNAL_ERROR;                                    \
      }                                                                        \
    }                                                                          \
  }

// Handle
#define CREATE_AND_SET_CNNL_HANDLE(handle, _handle)                  \
  cnnlHandle_t _handle;                                              \
  {                                                                  \
    if (handle != NULL) {                                            \
      cnnlStatus_t ret = cnnlCreate(&_handle);                       \
      if (ret != CNNL_STATUS_SUCCESS) {                              \
        LOG(ERROR)<< "CNNL_HELPER: CNNL create handle failed.";      \
        return MLUOP_STATUS_INTERNAL_ERROR;                          \
      }                                                              \
      ret = mluOpConvertHandle(handle, _handle);                     \
      if (ret != CNNL_STATUS_SUCCESS) {                              \
        LOG(ERROR)<< "CNNL_HELPER: Internal convert handle failed."; \
        return MLUOP_STATUS_INTERNAL_ERROR;                          \
      }                                                              \
    }                                                                \
  }

#define DESTROY_CNNL_HANDLE(_handle)                                  \
  {                                                                   \
    if (_handle != NULL) {                                            \
      cnnlStatus_t ret = cnnlSetQueue(_handle, nullptr);              \
      if (ret != CNNL_STATUS_SUCCESS) {                               \
        LOG(ERROR)<< "CNNL_HELPER: Internal set handle queue failed.";\
        return MLUOP_STATUS_INTERNAL_ERROR;                           \
      }                                                               \
      ret = cnnlDestroy(_handle);                                     \
      if (ret = CNNL_STATUS_SUCCESS) {                                \
        LOG(ERROR)<< "CNNL_HELPER: Internal destroy handle failed.";  \
        return MLUOP_STATUS_INTERNAL_ERROR;                           \
      }                                                               \
    }                                                                 \
  }

#endif  // KERNELS_UTILS_CNNL_HELPER_H_
