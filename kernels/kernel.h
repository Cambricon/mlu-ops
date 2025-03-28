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
#ifndef KERNELS_KERNEL_H_
#define KERNELS_KERNEL_H_

#if defined(__BANG__)
#include <mlu.h>
#endif  // defined(__BANG__)

/******************************************************************************
 * Macros for mluop kernels
 ******************************************************************************/
// in future, can be "__BANG_ARCH__ == 592 || __BANG_ARCH__ == xxx || ...)"
#define ARCH_SUPPORT_LARGE_TENSOR (__BANG_ARCH__ >= 592)

#define MAX_WRAM_SIZE (__MLU_WRAM_SIZE__ * 1024)
#define WRAM_LT_STRIDE (__MLU_WRAM_SIZE__ * 1024 / 64)

#if (__BANG_ARCH__ == 290) && defined(CONV_WARM_UP)
#undef MAX_WRAM_SIZE
#define MAX_WRAM_SIZE (__MLU_WRAM_SIZE__ * 1024 - 8 * 1024)
#endif

// only support when __BANG_ARCH__ > 300
#if (__BANG_ARCH__ > 300)
#define WRAM_LT_MAP16_STRIDE (__MLU_WRAM_SIZE__ * 1024 / 16)
#endif

#define DDR_ALIGN_MAP3 (1024 * 16)  // 16KB
#define NFU_ALIGN_SIZE 128          // Byte
#define WRAM_ALIGN_SIZE 64
#define LT_NUM 64
#define COMPUTE_COUNT_ALIGN 64  // elem_count must be divisible by 64

#if __BANG_ARCH__ == 322
#define CORE_DIM 1
#else
#define CORE_DIM 4
#endif
#define CLUSTER_DIM_OF_BLOCK 0
#define CLUSTER_DIM_OF_UNION1 1
#define CLUSTER_DIM_OF_UNION2 2
#define CLUSTER_DIM_OF_UNION4 4
#define CLUSTER_DIM_OF_UNION8 8
#define CLUSTER_DIM_OF_UNION16 16

#define REM_FOR_STACK (128 * 1024)           // 128KB reserved for cncc
#define THRESHOLD_SIZE_OF_UNION (64 * 1024)  // Split NRAM to 6 * 64KB

#ifdef __BANG_ARCH__
#define MAX_NRAM_SIZE (__MLU_NRAM_SIZE__ * 1024 - REM_FOR_STACK)
#if __BANG_ARCH__ > 592
#define MAX_UNIFIED_NRAM_SIZE (MAX_NRAM_SIZE + MAX_WRAM_SIZE)
#else
#define MAX_UNIFIED_NRAM_SIZE MAX_NRAM_SIZE
#endif
#if __MLU_SRAM_SIZE__ == 0
#define MAX_SRAM_SIZE 0
#else
#define MAX_SRAM_SIZE (__MLU_SRAM_SIZE__ * 1024 - REM_FOR_STACK)
#endif
#else                                // __BANG_ARCH__
#define MAX_NRAM_SIZE (384 * 1024)   // 384KB, initialization value
#define MAX_UNIFIED_NRAM_SIZE MAX_NRAM_SIZE   // 384KB, initialization value
#define MAX_SRAM_SIZE (1920 * 1024)  // 1920KB,initialization value
#endif                               // __BANG_ARCH__

#if __BANG_ARCH__ == 372
#define sizeof(T) (uint32_t(sizeof(T)))
#endif  // __BANG_ARCH__ == 372

#ifndef PAD_UP
#define PAD_UP(x, y) (((x) / (y) + (int)((x) % (y) > 0)) * (y))
#endif

#ifndef PAD_DOWN
#define PAD_DOWN(x, y) (((x) / (y)) * (y))
#endif

#ifndef DIV_UP
#define DIV_UP(x, y) ((x) % (y) > 0 ? ((x) / (y) + 1) : ((x) / (y)))
#endif

#ifndef DIV_DN
#define DIV_DN(x, y) ((x) / (y))
#endif

#define CEIL_ALIGN(x, align) (((x) + (align)-1) / (align) * (align))
#define FLOOR_ALIGN(x, align) ((x) / (align) * (align))

#ifdef ROUND_HALF_UP
#define HALF2INT8(dst, src, count, pos)                           \
  {                                                               \
    __bang_add_scalar(src, src, (half)(powf(2, pos - 1)), count); \
    __bang_half2int8_dn(dst, src, count, pos);                    \
  }
#define HALF2INT16(dst, src, count, pos)                          \
  {                                                               \
    __bang_add_scalar(src, src, (half)(powf(2, pos - 1)), count); \
    __bang_half2int16_dn(dst, src, count, pos);                   \
  }
#define FLOAT2INT8(dst, src, count, pos)                  \
  {                                                       \
    __bang_add_scalar(src, src, powf(2, pos - 1), count); \
    __bang_float2int8_dn(dst, src, count, pos);           \
  }
#define FLOAT2INT16(dst, src, count, pos)                 \
  {                                                       \
    __bang_add_scalar(src, src, powf(2, pos - 1), count); \
    __bang_float2int16_dn(dst, src, count, pos);          \
  }
#else
#define HALF2INT8(dst, src, count, pos) \
  { __bang_half2int8_rd(dst, src, count, pos); }
#define HALF2INT16(dst, src, count, pos) \
  { __bang_half2int16_rd(dst, src, count, pos); }
#define FLOAT2INT8(dst, src, count, pos) \
  { __bang_float2int8_rd(dst, src, count, pos); }
#define FLOAT2INT16(dst, src, count, pos) \
  { __bang_float2int16_rd(dst, src, count, pos); }
#endif

// maximum integer that can be represented by float
#define MAX_INT2FLOAT_EXACT (powf(2, 24))
#define NEG_MAX_INT2FLOAT_EXACT (-powf(2, 24))

#endif  // KERNELS_KERNEL_H_
