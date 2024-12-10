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

#include "kernels/kernel.h"

#define SCATTER_GATHER_PARAMS                                        \
  T *dst, const T *src, const uint32_t *offset, const uint8_t *mask, \
      const uint32_t transfer_size, const mluMemcpyDirection_t dir,  \
      const uint32_t stride, const uint32_t data_num

#if __BANG_ARCH__ > 592
#define MLUOP_SCATTER_GATHER(func, is_scatter)                               \
  template <typename T>                                                      \
  __mlu_func__ void __mluop_##func(SCATTER_GATHER_PARAMS) {                  \
    if (data_num <= UINT16_MAX) {                                            \
      if (mask) {                                                            \
        __##func(dst, src, offset, (const void *)mask, transfer_size, dir,   \
                 stride, data_num);                                          \
      } else {                                                               \
        __##func(dst, src, offset, transfer_size, dir, stride, data_num);    \
      }                                                                      \
    } else {                                                                 \
      uint16_t data_num_new = PAD_DOWN(UINT16_MAX, 64);                      \
      uint32_t remain = data_num % data_num_new;                             \
      uint32_t repeat = data_num / data_num_new + uint32_t(remain > 0);      \
      uint32_t dst_offset = is_scatter ? 0 : data_num_new;                   \
      uint32_t src_offset = is_scatter ? data_num_new : 0;                   \
                                                                             \
      for (uint32_t i = 0; i <= repeat; ++i) {                               \
        const uint16_t data_num_loop = i < repeat ? data_num_new : remain;   \
        if (mask) {                                                          \
          __##func(dst + i * dst_offset, src + i * src_offset,               \
                   mask + i * (data_num_new / 8), offset + i * data_num_new, \
                   transfer_size, dir, stride, data_num_loop);               \
        } else {                                                             \
          __##func(dst + i * dst_offset, src + i * src_offset,               \
                   offset + i * data_num_new, transfer_size, dir, stride,    \
                   data_num_loop);                                           \
        }                                                                    \
      }                                                                      \
    }                                                                        \
  }

/* __mlu_op_scatter
 * __mlu_op_scatter_async
 * __mlu_op_gather
 * __mlu_op_gather_async
 */
MLUOP_SCATTER_GATHER(gather_async, false)
MLUOP_SCATTER_GATHER(gather, false)
MLUOP_SCATTER_GATHER(scatter_async, true)
MLUOP_SCATTER_GATHER(scatter, true)

#elif __BANG_ARCH__ == 592
#define MLUOP_SCATTER_GATHER(func)                                            \
  template <typename T>                                                       \
  __mlu_func__ void __mluop_##func(SCATTER_GATHER_PARAMS) {                    \
    if (mask) {                                                               \
      __##func(dst, src, offset, mask, transfer_size, dir, stride, data_num); \
    } else {                                                                  \
      __##func(dst, src, offset, transfer_size, dir, stride, data_num);       \
    }                                                                         \
  }

MLUOP_SCATTER_GATHER(gather_async)
MLUOP_SCATTER_GATHER(gather)
MLUOP_SCATTER_GATHER(scatter_async)
MLUOP_SCATTER_GATHER(scatter)

#endif  // __BANG_ARCH__ > 592
