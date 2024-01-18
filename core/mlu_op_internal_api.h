/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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
#pragma once

#include <stdint.h>
#include <stdlib.h>

#include "mlu_op.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
enum mluOpInternalEventType : uint32_t;
#else
typedef uint32_t mluOpInternalEventType;
#endif

#define MLUOP_EVENT_CNRT_INVOKE_KERNEL ((mluOpInternalEventType)0x2)
#define MLUOP_EVENT_MLUOP_API ((mluOpInternalEventType)0x1000)

struct mluOpSubscriberStruct {
  uint32_t idx[4];
  mluOpInternalEventType event_type;
  void *internal_fields;
};

typedef struct mluOpSubscriberStruct mluOpSubscriber_t;

// XXX ABI may not be stable
struct mluOpEventParamCnrtInvokeKernel {
  const void *kernel;
  cnrtDim3_t dim;
  cnrtFunctionType_t ktype;
  void **args;
};

typedef void (*mluOpInternalHandler_t)(const void *, void *);

MLUOP_WIN_API mluOpStatus_t mluOpInternalSubscribe(
    mluOpInternalEventType event_type, mluOpInternalHandler_t handler,
    void *usr, mluOpSubscriber_t *subscriber);
MLUOP_WIN_API mluOpStatus_t
mluOpInternalUnsubscribe(mluOpSubscriber_t subscriber);

MLUOP_WIN_API mluOpStatus_t mluOpInternalGetKernelName(
    const void *kernel, const char **name, int *);  // api is not stable yet

MLUOP_WIN_API const char *mluOpInternalGetApiNameById(int id);
MLUOP_WIN_API int mluOpInternalGetApiNameNumber();

MLUOP_WIN_API const char *mluOpInternalGetCommitId();
MLUOP_WIN_API const char *mluOpInternalGetBranchInfo();

#ifdef __cplusplus
}  // extern "C"
#endif
