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
#ifndef KERNELS_ADAMW_ADAMW_H_
#define KERNELS_ADAMW_ADAMW_H_

#include "mlu_op.h"

struct mluOpAdamWStruct {
  float weight_decay = 0;
  float grad_scale = 1;
  bool use_nesterov = false;
};

mluOpStatus_t MLUOP_WIN_API KernelApplyAdamW(
    const cnrtDim3_t k_dim, const cnrtFunctionType_t k_type,
    const cnrtQueue_t queue, void *param, void *param_h, void *grad,
    void *momentum, void *velocity, float lr, float beta1, float beta2,
    float bias1, float bias2, float epsilon, float weight_decay, float scale,
    bool use_nesterov, size_t size);

#endif  // KERNELS_ADAMW_ADAMW_H_
