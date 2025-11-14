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

#ifndef KERNELS_ADAM_W_LITE_ADAM_W_LITE_H_
#define KERNELS_ADAM_W_LITE_ADAM_W_LITE_H_

#include "bangc_helper_dtype.h"
#include "bangc_kernels.h"

NAMESPACE_BANGC_KERNELS_BEGIN

// Group: AdamW
/*!
 * @brief Updates each attribute by using AdamW.
 *
 * @param[in] queue
 * A pointer to the cnrtQueue struct holding the information about a queue.
 * @param[in] lr
 * A hyperparameter representing the learning rate.
 * @param[in] beta1
 * A hyperparameter for updating momentum.
 * @param[in] beta2
 * A hyperparameter for updating velocity.
 * @param[in] bias1
 * A hyperparameter for updating param.
 * @param[in] bias2
 * A hyperparameter for updating param.
 * @param[in] epsilon
 * A fraction that prevents the denominator from being zero.
 * @param[in] weight_decay
 * A hyperparameter representing weight decay.
 * @param[in] scale
 * A hyperparameter of a shrinking gradient.
 * @param[in] use_nesterov
 * A parameter that determines whether to use the NAG algorithm.
 * @param[in] need_remove_nan
 * A parameter that determines whether to remove nan output.
 * @param[in] size
 * A parameter that represents the amount of data for the parameter.
 * @param[in] param_h
 * Pointer to the MLU memory that stores the param_h tensor.
 * @param[in] grad
 * Pointer to the MLU memory that stores the grad tensor.
 * @param[in] param
 * Pointer to the MLU memory that stores the grad tensor.
 * @param[in] momentum
 * Pointer to the MLU memory that stores the grad tensor.
 * @param[in] velocity
 * Pointer to the MLU memory that stores the grad tensor.
 * @par Return
 * - ::BANGC_KERNELS_STATUS_SUCCESS
 *
 * @par Data Type
 * - The supported data types of input and output tensors are as follows:
 *   - param_h tensor: bfloat16
 *   - grad tensor: bfloat16
 *   - param tensor: float
 *   - momentum tensor: float
 *   - velocity tensor: float
 *
 * @par Data Layout
 * - The supported data layouts of \b param tensor, \b param_h tensor, \b
 *   momentum tensor, \b velocity tensor, and \b grad tensor are as follows:
 *   - param tensor: ARRAY
 *   - param_h tensor: ARRAY
 *   - momentum tensor: ARRAY
 *   - velocity tensor: ARRAY
 *   - grad tensor: ARRAY
 *
 * @par Scale Limitation
 * - None.
 *
 * @par API Dependency
 * - None.
 *
 * @par Note
 * - None.
 *
 * @par Example
 * - None.
 *
 * @par Reference
 * - https://github.com/OpenBMB/BMTrain/blob/6abcf772aa1e120192f7656e55c4adbcde53c886/csrc/cuda/adam_cuda.cu
 */
template <typename T>
bangcKernelsStatus_t BANGC_KERNELS_WIN_API
mluAdamW(const cnrtQueue_t queue, const float lr, const float beta1,
         const float beta2, const float bias1, const float bias2,
         const float epsilon, const float weight_decay, const float scale,
         const bool use_nesterov, const bool need_remove_nan, const size_t size,
         T *param_h, T *grad, float *param, float *momentum, float *velocity);

NAMESPACE_BANGC_KERNELS_END

#endif  // KERNELS_ADAM_W_LITE_ADAM_W_LITE_H_
