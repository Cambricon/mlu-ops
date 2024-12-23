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
#ifndef BANGC_KERNELS_H_
#define BANGC_KERNELS_H_

#ifndef NAMESPACE_BANGC_KERNELS_BEGIN
#define NAMESPACE_BANGC_KERNELS_BEGIN namespace bangc_kernels {
#endif

NAMESPACE_BANGC_KERNELS_BEGIN

#ifndef BANGC_KERNELS_WIN_API
#ifdef _WIN32
#define BANGC_KERNELS_WIN_API __stdcall
#else
#define BANGC_KERNELS_WIN_API
#endif
#endif

typedef enum {
  BANGC_KERNELS_STATUS_SUCCESS =
      0, /*!< The operation is successfully completed. */
  BANGC_KERNELS_STATUS_ALLOC_FAILED = 1,
  /*!< This error occurs when the resource allocation fails, which is usually
     caused by failing to call cnMallocHost due to exceeded memory usage. Make
     sure that the memory allocated previously is deallocated as much as
     possible. */
  BANGC_KERNELS_STATUS_BAD_PARAM = 2,
  /*!< Invalid value or parameters are passed to the function, including data
     type, layout, dimensions, etc. */
  BANGC_KERNELS_STATUS_INTERNAL_ERROR = 3,
  /*!< An error occurs inside of the function, which may indicate an internal
     error or bug in the library. This error is usually caused by failing to
     call cnrtMemcpyAsync. Check whether the memory passed to the function is
     deallocated before the completion of the routine. */
  BANGC_KERNELS_STATUS_ARCH_MISMATCH = 4,
  /*!< Invalid MLU device which is not supported by current function. */
  BANGC_KERNELS_STATUS_EXECUTION_FAILED = 5,
  /*!< An error occurs when the function fails to be executed on MLU device due
     to multiple reasons. You can check whether the hardware environment, driver
     version and other prerequisite libraries are correctly installed. */
  BANGC_KERNELS_STATUS_NOT_SUPPORTED = 6,
  /*!< An error occurs when the requested functionality is not supported in this
     version but would be supported in the future. */
  BANGC_KERNELS_STATUS_NUMERICAL_OVERFLOW = 7,
  /*!< A numerical overflow occurs when executing the function, which is usually
     due to large scale or inappropriate range of value of input tensor. */
} bangcKernelsStatus_t;

template <typename T>
bangcKernelsStatus_t BANGC_KERNELS_WIN_API mluAdamW(const cnrtQueue_t queue,
                                                    const float lr,
                                                    const float beta1,
                                                    const float beta2,
                                                    const float bias1,
                                                    const float bias2,
                                                    const float epsilon,
                                                    const float weight_decay,
                                                    const float scale,
                                                    const bool use_nesterov,
                                                    const size_t size,
                                                    T *param_h,
                                                    T *grad,
                                                    void *param,
                                                    void *momentum,
                                                    void *velocity);

#ifndef NAMESPACE_BANGC_KERNELS_END
#define NAMESPACE_BANGC_KERNELS_END }
#endif

NAMESPACE_BANGC_KERNELS_END

#endif  // BANGC_KERNELS_H_
