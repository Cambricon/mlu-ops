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

// public functions are stored in this file
#ifndef KERNELS_UTILS_COMMON_H_
#define KERNELS_UTILS_COMMON_H_

#include <type_traits>

#include "float.h"
#include "kernels/kernel.h"

#define HALFMAX 65504

template <typename T>
__mlu_func__ bool __mluop_is_float() {
  return false;
}

template <>
__mlu_func__ bool __mluop_is_float<float>() {
  return true;
}

template <typename T>
__mlu_func__ bool __mluop_is_half() {
  return false;
}

template <>
__mlu_func__ bool __mluop_is_half<half>() {
  return true;
}

template <typename T>
__mlu_func__ inline T __mluop_min(T a, T b) {
  return a < b ? a : b;
}

template <typename T>
__mlu_func__ inline T __mluop_max(T a, T b) {
  return a > b ? a : b;
}

/******************************************************************************
 * MLUOP FUNC: __mluop_float2half
 * param 'dst' is the destination pointer in NRAM.
 * param 'src' is the source pointer in NRAM.
 * param 'src_count' is the src element count.
 * Note:
 *      The rounding mode on MLU200 is rd, on MLU300 is rn.
 ******************************************************************************/
__mlu_func__ inline void __mluop_float2half(half *dst, float *src,
                                            int src_count) {
#if __BANG_ARCH__ >= 300
  __bang_float2half_rn(dst, src, src_count);
#else
  __bang_float2half_rd(dst, src, src_count);
#endif
}

__mlu_func__ inline half __mluop_float2half(float a) {
#if __BANG_ARCH__ >= 300
  return __float2half_rn(a);
#else
  return __float2half_rd(a);
#endif
}

/******************************************************************************
 * MLUOP FUNC: __mluop_div
 * param 'nram_dst' is the nram destination address, which supports half or
 * float data type.  
 * param 'nram_src0' is the nram source address, which has the same data
 * type as nram_dst.  
 * param 'nram_src1' is the nram source address, which has the same data
 * type as nram_dst.  
 * param 'nram_addition' is the nram addition address.
 * Pass NULL if the data type of nram_src is float and architecture >= 300,
 * otherwise the space size is at least twice as much as nram_src.
 * param 'deal_num' is the num of input data.
 *
 * remarks:
 * 1. nram_dst and nram_src can not be homologous operand if architecture <
 * 300.  
 * 2. On MLU2XX, nram_src1(dividend) must be positive due to limitations
 * of bang_active_reciphp.
*******************************************************************************/
template <typename T>
__mlu_func__ void __mluop_div(T *nram_dst, T *nram_src0, T *nram_src1,
                              T *nram_addition, int is_high_precision,
                              int deal_num) {
  if (sizeof(T) == sizeof(float)) {
#if (__BANG_ARCH__ >= 300) && (__BANG_ARCH__ != 372)
    __bang_div((float *)nram_dst, (float *)nram_src0, (float *)nram_src1,
               deal_num);
#else
    __bang_recip((float *)nram_dst, (float *)nram_src1, deal_num);
    __bang_mul((float *)nram_dst, (float *)nram_src0, (float *)nram_dst,
               deal_num);
#endif
  } else if (sizeof(T) == sizeof(half)) {
#if (__BANG_ARCH__ >= 300) && (__BANG_ARCH__ != 372)
    __bang_div((half *)nram_dst, (half *)nram_src0, (half *)nram_src1,
               deal_num);
#else
    if (is_high_precision) {
#if __BANG_ARCH__ == 372
      __bang_half2float((float *)nram_addition, (half *)nram_src1, deal_num);
      __bang_recip((float *)nram_addition, (float *)nram_addition, deal_num);
      __mluop_float2half((half *)nram_src1, (float *)nram_addition, deal_num);
      __bang_mul((half *)nram_dst, (half *)nram_src0, (half *)nram_src1,
                 deal_num);
#else
      __bang_half2float((float *)nram_addition, (half *)nram_src1, deal_num);
      __bang_recip((float *)nram_addition, (float *)nram_addition, deal_num);
      __mluop_float2half((half *)nram_src1, (float *)nram_addition, deal_num);
      __bang_mul((half *)nram_dst, (half *)nram_src0, (half *)nram_src1,
                 deal_num);
#endif
    } else {
      __bang_active_reciphp((T *)nram_dst, (T *)nram_src1, deal_num);
      __bang_mul((T *)nram_dst, (T *)nram_src0, (T *)nram_dst, deal_num);
    }
#endif
  } else {
    return;
  }
}

/*******************************************************************************
 * MLUOPS FUNC: __mluop_recip
 * param 'nram_dst' is the nram destination address, which supports half or
 * float data type. param 'nram_src' is the nram source address, which has the
 * same data type as nram_dst. param 'nram_addition' is the nram addition
 * address. Pass NULL if the data type of nram_src is float, otherwise the space
 * size is at least twice as much as nram_src. param 'is_high_precision' is the
 * precision flag. param 'deal_num' is the num of input data. remarks:
 *   1. nram_dst and nram_src can be homologous operand.
 *   2. On MLU2XX, input must be in the range [0.00391, 2e6] for float and
 * [0.00391, 65504] for half. Please refer to bangC Developer Guide for detailed
 * information.
 ******************************************************************************/
template <typename T>
__mlu_func__ void __mluop_recip(T *nram_dst, T *nram_src, void *nram_addition,
                                const bool is_high_precision,
                                const uint32_t deal_num) {
  if (sizeof(T) == sizeof(float)) {
#if __BANG_ARCH__ >= 300
    __bang_recip((float *)nram_dst, (float *)nram_src, deal_num);
#else
    __bang_active_reciphp((float *)nram_dst, (float *)nram_src, deal_num);
#endif
  } else if (sizeof(T) == sizeof(half)) {
#if __BANG_ARCH__ >= 300
    __bang_half2float((float *)nram_addition, (half *)nram_src, deal_num);
    __bang_recip((float *)nram_addition, (float *)nram_addition, deal_num);
    __bang_float2half_rn((half *)nram_dst, (float *)nram_addition, deal_num);
#else
    if (is_high_precision) {
      __bang_half2float((float *)nram_addition, (half *)nram_src, deal_num);
      __bang_active_reciphp((float *)nram_addition, (float *)nram_addition,
                            deal_num);
      __bang_float2half_rd((half *)nram_dst, (float *)nram_addition, deal_num);
    } else {
      __bang_active_reciphp((half *)nram_dst, (half *)nram_src, deal_num);
    }
#endif
  } else {
    return;
  }
}

/******************************************************************************
 * MLUOPS FUNC: __mluop_exp
 * param 'nram_dst' is the nram destination address, which supports half or
 * float data type. param 'nram_src' is the nram source address, which has the
 * same data type as nram_dst. param 'nram_addition' is the nram addition
 * address. Pass NULL if the data type of nram_src is float, otherwise the space
 * size is at least twice as much as nram_src. param 'is_high_precision' is the
 * precision flag. param 'deal_num' is the num of input data. remarks: nram_dst
 * and nram_src can be homologous operand.
 ******************************************************************************/
template <typename T>
__mlu_func__ void __mluop_exp(T *nram_dst, T *nram_src, void *nram_addition,
                              const int is_high_precision, const int deal_num) {
  if (sizeof(T) == sizeof(float)) {
#if __BANG_ARCH__ >= 300
    int x2d = 0x3fb8aa3b;
    float log2e = *(float *)&x2d;
    __bang_mul_scalar((float *)nram_dst, (float *)nram_src, (float)log2e,
                      deal_num);
    __bang_pow2((float *)nram_dst, (float *)nram_dst, deal_num);
#else
    __bang_active_exphp((float *)nram_dst, (float *)nram_src, deal_num);
#endif
  } else if (sizeof(T) == sizeof(half)) {
#if __BANG_ARCH__ >= 300
    int x2d = 0x3fb8aa3b;
    float log2e = *(float *)&x2d;
    __bang_half2float((float *)nram_addition, (half *)nram_src, deal_num);
    __bang_mul_scalar((float *)nram_addition, (float *)nram_addition,
                      (float)log2e, deal_num);
    __bang_pow2((float *)nram_addition, (float *)nram_addition, deal_num);
    __bang_float2half_rn((half *)nram_dst, (float *)nram_addition, deal_num);
#else
    if (is_high_precision) {
      __bang_half2float((float *)nram_addition, (half *)nram_src, deal_num);
      __bang_active_exphp((float *)nram_addition, (float *)nram_addition,
                          deal_num);
      __bang_float2half_rd((half *)nram_dst, (float *)nram_addition, deal_num);
    } else {
      __bang_active_exphp((half *)nram_dst, (half *)nram_src, deal_num);
    }
#endif
  } else {
    return;
  }
}

/******************************************************************************
 * MLUOPS FUNC: __mluop_log
 * param 'nram_dst' is the nram destination address, which supports half or
 * float data type.
 * param 'nram_src' is the nram source address, which has the same data type
 * as nram_dst.
 * param 'nram_addition' is the nram addition address. Pass NULL if the data
 * type of nram_src is float, otherwise the space size is at least twice as
 * much as nram_src.
 * param 'is_high_precision' is the precision flag.
 * param 'deal_num' is the num of input data.
 * remarks:
 *   nram_dst and nram_src can be homologous operand.
 ******************************************************************************/
template <typename T>
__mlu_func__ void __mluop_log(T *nram_dst, T *nram_src, void *nram_addition,
                              int is_high_precision, int deal_num) {
  if (sizeof(T) == sizeof(float)) {
    int x2d = 0x3f317217;
    float rlog2e = *(float *)&x2d;
    __bang_log((float *)nram_dst, (float *)nram_src, deal_num);
    __bang_mul_scalar((float *)nram_dst, (float *)nram_dst, (float)rlog2e,
                      deal_num);
  } else if (sizeof(T) == sizeof(half)) {
    int x2d = 0x3f317217;
    float rlog2e = *(float *)&x2d;
    __bang_half2float((float *)nram_addition, (half *)nram_src, deal_num);
    __bang_log((float *)nram_addition, (float *)nram_addition, deal_num);
    __mluop_float2half((half *)nram_dst, (float *)nram_addition, deal_num);
    __bang_mul_scalar((half *)nram_dst, (half *)nram_dst, (half)rlog2e,
                      deal_num);

  } else {
    return;
  }
}

/******************************************************************************
 * MLUOPS FUNC: __mluop_sigmoid
 * param 'nram_dst' is the nram destination address, which supports half or
 * float data type. param 'nram_src' is the nram source address, which has the
 * same data type as nram_dst. param 'nram_addition' is the nram addition
 * address. Pass NULL if the data type of nram_src is float, otherwise the space
 * size is at least twice as much as nram_src. param 'is_high_precision' is the
 * precision flag. param 'deal_num' is the num of input data. remarks: nram_dst
 * and nram_src can be homologous operand.
 ******************************************************************************/
template <typename T>
__mlu_func__ void __mluop_sigmoid(T *nram_dst, T *nram_src, void *nram_addition,
                                  const int is_high_precision,
                                  const int deal_num) {
  if (sizeof(T) == sizeof(float)) {
#if __BANG_ARCH__ >= 300
    __bang_mul_scalar((float *)nram_dst, (float *)nram_src, (float)-1.0,
                      deal_num);
    __mluop_exp((float *)nram_dst, (float *)nram_dst, NULL, 0, deal_num);
    __bang_add_scalar((float *)nram_dst, (float *)nram_dst, (float)1.0,
                      deal_num);
    __mluop_recip((float *)nram_dst, (float *)nram_dst, NULL, 0, deal_num);
#else
    __bang_active_sigmoid((float *)nram_dst, (float *)nram_src, deal_num);
#endif
  } else if (sizeof(T) == sizeof(half)) {
#if __BANG_ARCH__ >= 300
    __bang_half2float((float *)nram_addition, (half *)nram_src, deal_num);
    __bang_mul_scalar((float *)nram_addition, (float *)nram_addition,
                      (float)-1.0, deal_num);
    __mluop_exp((float *)nram_addition, (float *)nram_addition, NULL, 0,
                deal_num);
    __bang_add_scalar((float *)nram_addition, (float *)nram_addition,
                      (float)1.0, deal_num);
    __mluop_recip((float *)nram_dst, (float *)nram_addition, NULL, 0, deal_num);
    __bang_float2half_rn((half *)nram_dst, (float *)nram_dst, deal_num);
#else
    if (is_high_precision) {
      __bang_half2float((float *)nram_addition, (half *)nram_src, deal_num);
      __bang_active_sigmoid((float *)nram_addition, (float *)nram_addition,
                            deal_num);
      __bang_float2half_rd((half *)nram_dst, (float *)nram_addition, deal_num);
    } else {
      __bang_active_sigmoid((half *)nram_dst, (half *)nram_src, deal_num);
    }
#endif
  } else {
    return;
  }
}

/******************************************************************************
 * MLUOPS FUNC: __mluop_recursive_sum_pool
 * param 'dst' is the src and dst nram addr
 * param 'low_dim' is the number of low dim
 * param 'high_dim' is the number of high dim
 * param 'kernel_limit' is the high_dim of sumpool per time
 ******************************************************************************/
template <typename T>
__mlu_func__ void __mluop_recursive_sum_pool(T *dst, int low_dim, int high_dim,
                                             int kernel_limit) {
  for (; high_dim > 1;) {
    int repeat_s = high_dim / kernel_limit;
    int remain_s = high_dim % kernel_limit;
    if (remain_s) {
      __bang_sumpool((T *)dst, (T *)dst, low_dim, 1, remain_s, 1, remain_s, 1,
                     1);
    }
    if (repeat_s) {
      __bang_sumpool((T *)dst + (remain_s > 0 ? low_dim : 0),
                     (T *)dst + remain_s * low_dim, low_dim,
                     kernel_limit * repeat_s, 1, kernel_limit, 1, 1,
                     kernel_limit);
    }
    high_dim = repeat_s + static_cast<int>(remain_s > 0);
  }
  return;
}

/*****************************************************************************
 * MLUOPS FUNC: __mluop_int322float
 * param 'dst' is the destination pointer in NRAM, same memory space as src
 * required in NRAM
 * param 'dst_addition' is the addition workspace of dst, requiring the same
 * amount of space as dst in NRAM
 * param 'src' is the source pointer in NRAM
 * param 'src_addition' is the addition workspace of src, requiring only 128B
 * space in NRAM
 * param 'src_count' is the src element count
 * Notes:
 *   the sapces pointed by dst and src can not overlap
 *   src_count*sizeof(float) should be divisible by 128
 *   src input must be in range of [-2^23, 2^23-1] for MLU270 and MLU290
 *****************************************************************************/
__mlu_func__ void __mluop_int322float(float *dst, float *dst_addition,
                                      int32_t *src, float *src_addition,
                                      int32_t src_count) {
#if __BANG_ARCH__ >= 300
  __bang_int322float((float *)dst, (int32_t *)src, src_count, 0);
#else
  // get sign bit
  int32_t seg_elem_count = 32;  // 128/sizeof(float) = 32
  int32_t float_size = 4;       // sizeof(float) = 4
  int32_t align_128 = 128;
  float move_23bit = 8388608.0;
  // 0x80000000 = 1,000000000,0000000000000000000000000000
  __bang_write_value((unsigned *)src_addition, seg_elem_count,
                     (unsigned)0x80000000);
  __bang_cycle_band((char *)dst_addition, (char *)src, (char *)src_addition,
                    src_count * float_size, align_128);
  // get 1 or 0 from sign bit
  // judge is Odd
  __bang_write_value((unsigned *)src_addition, seg_elem_count,
                     (unsigned)0x00000001);
  __bang_cycle_bor((char *)dst_addition, (char *)dst_addition,
                   (char *)src_addition, src_count * float_size, align_128);
  __bang_write_value((unsigned *)src_addition, seg_elem_count,
                     (unsigned)0x80000001);
  __bang_cycle_eq(dst_addition, dst_addition, src_addition, src_count,
                  seg_elem_count);
  // minus xor, positive num invariant
  __bang_write_value((unsigned *)src_addition, seg_elem_count,
                     (unsigned)0xffffffff);
  __bang_cycle_mul(dst, dst_addition, src_addition, src_count, seg_elem_count);
  __bang_bxor((char *)dst, (char *)src, (char *)dst, src_count * float_size);
  // convert int32 to float32
  __bang_write_value((unsigned *)src_addition, seg_elem_count,
                     (unsigned)0x7fffff);
  __bang_cycle_band((char *)dst, (char *)dst, (char *)src_addition,
                    src_count * float_size, align_128);
  __bang_write_value((unsigned *)src_addition, seg_elem_count,
                     (unsigned)0x4b000000);
  __bang_cycle_bor((char *)dst, (char *)dst, (char *)src_addition,
                   src_count * float_size, align_128);
  __bang_sub_scalar(dst, dst, move_23bit, src_count);
  // add one
  __bang_add(dst, dst, dst_addition, src_count);
  // set sign for float32
  __bang_write_value((unsigned *)src_addition, seg_elem_count,
                     (unsigned)0xffffffff);
  __bang_cycle_mul(dst_addition, dst_addition, src_addition, src_count,
                   seg_elem_count);

  // fix on MLU300
  __bang_write_value((unsigned *)src_addition, seg_elem_count,
                     (unsigned)0x00000001);
  __bang_cycle_add(dst_addition, dst_addition, src_addition, src_count,
                   seg_elem_count);
  // end fix

  __bang_write_value((unsigned *)src_addition, seg_elem_count,
                     (unsigned)0x80000000);
  __bang_cycle_band((char *)dst_addition, (char *)dst_addition,
                    (char *)src_addition, src_count * float_size, align_128);
  __bang_bor((char *)dst, (char *)dst, (char *)dst_addition,
             src_count * float_size);
#endif
}

/*****************************************************************************
 * MLUOPS FUNC: __mluop_float2int32
 * param 'dst' is the destination pointer in NRAM, same memory space as src
 * required in NRAM
 * param 'dst_addition' is the addition workspace of dst, requiring the same
 * amount of space as dst in NRAM
 * param 'src' is the source pointer in NRAM
 * param 'src_addition' is the addition workspace of src, requiring only 128B
 * space in NRAM
 * param 'src_count' is the src element count
 * Notes:
 *   the sapces pointed by dst and src can not overlap
 *   src_count*sizeof(float) should be divisible by 128
 *   src input must be in range of [-2^23, 2^23-1] for MLU270 and MLU290
 *****************************************************************************/
__mlu_func__ void __mluop_float2int32(int32_t *dst, float *dst_addition,
                                      float *src, float *src_addition,
                                      int32_t src_count) {
#if __BANG_ARCH__ >= 322
  __bang_float2int32_tz((int32_t *)dst, (float *)src, src_count, 0);
#else
  // sign ===> src_addition
  // dst=-1.0 : when src[i] is a negative number
  // dst=+1.0 : when src[i] is a positive number
  int32_t floatDchar = sizeof(float) / sizeof(char);
  __bang_active_sign((float *)dst, src, src_count);
  // dst_addition = abs(src)
  __bang_mul(dst_addition, src, (float *)dst, src_count);
  // if dst_addition < 1.0, then src_addition + 1. to fix add error
  __bang_write_value((float *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     1.0f);
  __bang_cycle_lt(dst_addition, dst_addition, (float *)src_addition, src_count,
                  NFU_ALIGN_SIZE / sizeof(float));
  __bang_add_tz((float *)dst, (float *)dst, (float *)dst_addition, src_count);
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     0xbf800000);
  // set negative flag -1.0 = 0xbf80000
  __bang_cycle_eq(
      (float *)dst, (float *)dst, (float *)src_addition, src_count,
      NFU_ALIGN_SIZE / sizeof(float));  // to mask all src in [x < -1.0]
  __bang_active_abs(dst_addition, src, src_count);
  __bang_write_value((float *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     8388608.0f);
  // mask shift move 23
  __bang_cycle_add_tz(
      dst_addition, dst_addition, src_addition, src_count,
      NFU_ALIGN_SIZE / sizeof(float));  // right shift move 23bit
  // dst=1.0, when src < -1.0
  // dst=0.0, when src >=-1.0
  __bang_sub(dst_addition, dst_addition, (float *)dst, src_count);
  // to fix max value
  __bang_mul_scalar((float *)dst, (float *)dst, 16777215.0, src_count);
  __bang_bxor((char *)dst_addition, (char *)dst_addition, (char *)dst,
              src_count * floatDchar);
  // get log 23bit
  __bang_write_value((unsigned *)src_addition, NFU_ALIGN_SIZE / sizeof(float),
                     (unsigned)0x007fffff);
  // mask low 23bit is 1
  __bang_cycle_band((char *)dst_addition, (char *)dst_addition,
                    (char *)src_addition, src_count * floatDchar,
                    NFU_ALIGN_SIZE / sizeof(char));

  __bang_write_value(src_addition, NFU_ALIGN_SIZE / sizeof(float), 0x3f800000);
  __bang_cycle_and((float *)dst, (float *)dst, src_addition, src_count,
                   NFU_ALIGN_SIZE / sizeof(float));
  // src or dst_addition
  __bang_bor((char *)dst_addition, (char *)dst, (char *)dst_addition,
             src_count * floatDchar);
  __bang_mul_scalar((float *)dst, (float *)dst, -2.0, src_count);
  __bang_bor((char *)dst, (char *)dst, (char *)dst_addition,
             src_count * floatDchar);
#endif
}

__mlu_func__ void pvLock() {
#if __BANG_ARCH__ == 270
  if (__is_ipu()) {
    __bang_lock(0, 0);
  }
#endif
}

__mlu_func__ void pvUnlock() {
#if __BANG_ARCH__ == 270
  if (__is_ipu()) {
    __bang_unlock(0, 0);
  }
#endif
}

/******************************************************************************
 * MLUOPS FUNC: __mluop_load_str_2D
 * param 'size' is the getC size.
 * param 'seg_num' is the loop times.
 * param 'dst_str' is nram stride, c_align on onchip.
 * param 'src_str' is gdram stride, as usual is equal to c_unalign.
 * Note:
 *      The data between 'size' and 'dst_str' in every seg_num
 *      may be contaminated.
 ******************************************************************************/
template <typename T>
__mlu_func__ void __mluop_load_str_2D(T *dst, T *src, int size, int dst_str,
                                      int src_str, int seg_num) {
  if (dst_str == src_str && size == src_str) {
    __memcpy(dst, src, src_str * seg_num * sizeof(T), GDRAM2NRAM);
  } else if ((size == src_str || src_str <= dst_str) &&
             src_str * sizeof(T) <= 512) {  // IO efficiency is best when
                                            // datasize gather than 512bytes
    T *tmp = (T *)dst + (dst_str - src_str) * seg_num;
    __memcpy(tmp, src, (src_str * (seg_num - 1) + size) * sizeof(T),
             GDRAM2NRAM);
    __memcpy(dst, tmp, size * sizeof(T), NRAM2NRAM, dst_str * sizeof(T),
             src_str * sizeof(T), seg_num - 1);
  } else {
    __memcpy(dst, src, size * sizeof(T), GDRAM2NRAM, dst_str * sizeof(T),
             src_str * sizeof(T), seg_num - 1);
  }
}

/******************************************************************************
 * MLUOPS FUNC: __mluop_load_str_3D
 * param 'size' is the getC size.
 * param 'seg_num_in' is the in loop times.
 * param 'seg_num_out' is the out loop times.
 * param 'dst_str_in' is nram in stride.
 * param 'dst_str_out' is nram out stride.
 * param 'src_str_in' is gdram in stride.
 * param 'src_str_out' is gdram out stride.
 ******************************************************************************/
template <typename T>
__mlu_func__ void __mluop_load_str_3D(T *dst, T *src, int size, int seg_num_in,
                                      int seg_num_out, int dst_str_in,
                                      int dst_str_out, int src_str_in,
                                      int src_str_out) {
  T *tmp_dst = dst;
  T *tmp_src = src;
  for (int i = 0; i < seg_num_out; ++i) {
    __mluop_load_str_2D(tmp_dst, tmp_src, size, dst_str_in, src_str_in,
                        seg_num_in);
    tmp_src = (T *)tmp_src + src_str_out;
    tmp_dst = (T *)tmp_dst + dst_str_out;
  }
}

/******************************************************************************
 * MLUOPS FUNC: __mluop_store_str_2D
 * param 'size' is the getC size.
 * param 'seg_num' is the loop times.
 * param 'dst_str' is gdram stride, c_align on onchip.
 * param 'src_str' is nram stride, as usual is equal to c_unalign.
 * Note:
 *      If the data to be stored will reuse later,
 *      don't use this function, use MEMCPY instead.
 ******************************************************************************/
template <typename T>
__mlu_func__ void __mluop_store_str_2D(T *dst, T *src, int size, int seg_num,
                                       int dst_str, int src_str) {
  if ((size == dst_str && dst_str <= src_str) &&
      dst_str * sizeof(T) <=
          512) {  // IO efficiency is best when datasize gather than 512bytes
    if (dst_str != src_str) {
      __memcpy(src, src, size * sizeof(T), NRAM2NRAM, dst_str * sizeof(T),
               src_str * sizeof(T), seg_num - 1);
    }
    __memcpy(dst, src, size * seg_num * sizeof(T), NRAM2GDRAM);
  } else {
    __memcpy(dst, src, size * sizeof(T), NRAM2GDRAM, dst_str * sizeof(T),
             src_str * sizeof(T), seg_num - 1);
  }
}

/******************************************************************************
 * MLUOPS FUNC: __mluop_store_str_3D
 * param 'size' is the getC size.
 * param 'seg_num_in' is the in loop times.
 * param 'seg_num_out' is the out loop times.
 * param 'dst_str_in' is gdram in stride.
 * param 'dst_str_out' is gdram out stride.
 * param 'src_str_in' is nram in stride.
 * param 'src_str_out' is nram out stride.
 * Note:
 *      If the data to be stored will reuse later,
 *      don't use this function, use MEMCPY instead.
 ******************************************************************************/
template <typename T>
__mlu_func__ void __mluop_store_str_3D(T *dst, T *src, int size, int seg_num_in,
                                       int seg_num_out, int dst_str_in,
                                       int dst_str_out, int src_str_in,
                                       int src_str_out) {
  T *tmp_dst = dst;
  T *tmp_src = src;
  for (int i = 0; i < seg_num_out; ++i) {
    __mluop_store_str_2D(tmp_dst, tmp_src, size, seg_num_in, dst_str_in,
                         src_str_in);
    tmp_src = (T *)tmp_src + src_str_out;
    tmp_dst = (T *)tmp_dst + dst_str_out;
  }
}

/*******************************************************************************
 * MLUOPS FUNC: __mluop_get_stage_indices_tfuse
 * param 'dst_nram' is nram space for store result
 * param 'length' is the continuous indices length
 * Note:
 *      Get [0, length-1] stage indices in nram on mlu590 mlu300
 *      and other platform which support tfuse instruction.
 *      length not need to be aligned any number.
 *      dst_nram only support nram.
 * ****************************************************************************/
__mlu_func__ void __mluop_get_stage_indices_tfuse(int *dst_nram, int length) {
#if (__BANG_ARCH__ == 372 || __BANG_ARCH__ == 592)
  int align_num = 128;
  int repeat = (int)(logf(length / align_num) / logf(2));
  int remain = length / align_num - powf(2, repeat);
  int global_remain = length % align_num;
  int count = 1;
  for (int i = 0; i < align_num; i++) {
    dst_nram[i] = i;
    if (i == length - 1) {
      return;
    }
  }
  for (int i = 0; i < repeat; i++) {
    __asm__ volatile(
        "fuse.nram.u32 [%[dst_nram]], %[once_process_num], "
        "[%[src_nram]], .add(%[region_length]); \n\t" ::[dst_nram] "r"(
            dst_nram + count * align_num),
        [ src_nram ] "r"(dst_nram), [ once_process_num ] "r"(count * align_num),
        [ region_length ] "r"(count * align_num));
    count *= 2;
  }
  if (remain > 0) {
    __asm__ volatile(
        "fuse.nram.u32 [%[dst_nram]], %[once_process_num], "
        "[%[src_nram]], .add(%[region_length]); \n\t" ::[dst_nram] "r"(
            dst_nram + count * align_num),
        [ src_nram ] "r"(dst_nram),
        [ once_process_num ] "r"(remain * align_num),
        [ region_length ] "r"(count * align_num));
  }
  if (global_remain > 0) {
    __asm__ volatile(
        "fuse.nram.u32 [%[dst_nram]], %[once_process_num], "
        "[%[src_nram]], .add(%[region_length]); \n\t" ::[dst_nram] "r"(
            dst_nram + count * align_num + remain * align_num),
        [ src_nram ] "r"(dst_nram), [ once_process_num ] "r"(global_remain),
        [ region_length ] "r"(count * align_num + remain * align_num));
  }
#endif
}

/***************************************************************************
 * MLUOPS FUNC: __mluop_get_indices.
 * param "dst" is needed for holding the final result.
 * param "start_index" is the smallest integer to be generated.
 * param "len" is the total number of integers to be generated.
 * Note:
 *      Get [start_index, len-1] stage indices in nram on mlu590 mlu300
 *      and other platform which support necessary instruction.
 *      len not need to be aligned any number.
 *      dst only support nram.
 *      This funciton currently only supports float type indices.
 * *************************************************************************/
__mlu_vector__ void __mluop_get_indices(float *dst, float start_index,
                                        uint32_t len) {
  vv_float r_out, r_dim;
  unsigned BlockDim = __vv_get_length() / sizeof(float);
  __asm__ volatile("index.vvr.f32 %[dst], %[base], 1;\n\t"
                   : [ dst ] "+r"(r_out)
                   : [ base ] "r"(start_index));
  __vv_move(r_dim, BlockDim);
  int repeat = DIV_UP(len, BlockDim);
  for (int iter = 0; iter < repeat; iter++) {
    __vv_store(dst + iter * BlockDim, r_out);
    __vv_add(r_out, r_out, r_dim);
  }
}

#endif  // KERNELS_UTILS_COMMON_H_
