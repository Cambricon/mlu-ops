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

#include <algorithm>
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
__mlu_func__ T __mluop_min(T a, T b) {
  return a < b ? a : b;
}

template <typename T>
__mlu_func__ T __mluop_max(T a, T b) {
  return a > b ? a : b;
}

/******************************************************************************
 * MLUOP FUNC: __mluop_float2half
 * param 'dst' is the destination pointer in NRAM.
 * param 'src' is the source pointer in NRAM.
 * param 'src_count' is the src element count.
 * Note:
 *      The rounding mode on MLU300 is rn.
 ******************************************************************************/
__mlu_func__ void __mluop_float2half(half *dst, float *src, int src_count) {
  __bang_float2half_rn(dst, src, src_count);
}

__mlu_func__ half __mluop_float2half(float a) { return __float2half_rn(a); }

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
    __bang_recip((float *)nram_dst, (float *)nram_src, deal_num);
  } else if (sizeof(T) == sizeof(half)) {
    __bang_half2float((float *)nram_addition, (half *)nram_src, deal_num);
    __bang_recip((float *)nram_addition, (float *)nram_addition, deal_num);
    __bang_float2half_rn((half *)nram_dst, (float *)nram_addition, deal_num);
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
    int x2d = 0x3fb8aa3b;
    float log2e = *(float *)&x2d;
    __bang_mul_scalar((float *)nram_dst, (float *)nram_src, (float)log2e,
                      deal_num);
    __bang_pow2((float *)nram_dst, (float *)nram_dst, deal_num);
  } else if (sizeof(T) == sizeof(half)) {
    int x2d = 0x3fb8aa3b;
    float log2e = *(float *)&x2d;
    __bang_half2float((float *)nram_addition, (half *)nram_src, deal_num);
    __bang_mul_scalar((float *)nram_addition, (float *)nram_addition,
                      (float)log2e, deal_num);
    __bang_pow2((float *)nram_addition, (float *)nram_addition, deal_num);
    __bang_float2half_rn((half *)nram_dst, (float *)nram_addition, deal_num);
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
    __bang_log2((float *)nram_dst, (float *)nram_src, deal_num);
    __bang_mul_scalar((float *)nram_dst, (float *)nram_dst, (float)rlog2e,
                      deal_num);
  } else if (sizeof(T) == sizeof(half)) {
    int x2d = 0x3f317217;
    float rlog2e = *(float *)&x2d;
    __bang_half2float((float *)nram_addition, (half *)nram_src, deal_num);
    __bang_log2((float *)nram_addition, (float *)nram_addition, deal_num);
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
    __bang_mul_scalar((float *)nram_dst, (float *)nram_src, (float)-1.0,
                      deal_num);
    __mluop_exp((float *)nram_dst, (float *)nram_dst, NULL, 0, deal_num);
    __bang_add_scalar((float *)nram_dst, (float *)nram_dst, (float)1.0,
                      deal_num);
    __mluop_recip((float *)nram_dst, (float *)nram_dst, NULL, 0, deal_num);
  } else if (sizeof(T) == sizeof(half)) {
    __bang_half2float((float *)nram_addition, (half *)nram_src, deal_num);
    __bang_mul_scalar((float *)nram_addition, (float *)nram_addition,
                      (float)-1.0, deal_num);
    __mluop_exp((float *)nram_addition, (float *)nram_addition, NULL, 0,
                deal_num);
    __bang_add_scalar((float *)nram_addition, (float *)nram_addition,
                      (float)1.0, deal_num);
    __mluop_recip((float *)nram_dst, (float *)nram_addition, NULL, 0, deal_num);
    __bang_float2half_rn((half *)nram_dst, (float *)nram_dst, deal_num);
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

template <typename T>
__mlu_func__ void __mlu_op_arange_base_(T *dst_nram, uint32_t numel,
                                        T start_index, T step) {
  for (uint32_t i = 0; i < numel; i++) {
    dst_nram[i] = start_index + i * step;
  }
}

#define MLUOP_ARANGE_VV_IMPL(VVType, vv_num, dst_nram, start_index, step) \
  do {                                                                    \
    VVType vv_index[8];                                                   \
    __vv_index(vv_index[0], start_index, step);                           \
    __vv_add(vv_index[1], vv_index[0], 1 * vv_num * step);                \
    __vv_add(vv_index[2], vv_index[0], 2 * vv_num * step);                \
    __vv_add(vv_index[3], vv_index[0], 3 * vv_num * step);                \
    __vv_add(vv_index[4], vv_index[0], 4 * vv_num * step);                \
    __vv_add(vv_index[5], vv_index[0], 5 * vv_num * step);                \
    __vv_add(vv_index[6], vv_index[0], 6 * vv_num * step);                \
    __vv_add(vv_index[7], vv_index[0], 7 * vv_num * step);                \
    __vv_store(dst_nram, vv_index[0], vv_num);                            \
    __vv_store(dst_nram + vv_num, vv_index[1], vv_num);                   \
    __vv_store(dst_nram + 2 * vv_num, vv_index[2], vv_num);               \
    __vv_store(dst_nram + 3 * vv_num, vv_index[3], vv_num);               \
    __vv_store(dst_nram + 4 * vv_num, vv_index[4], vv_num);               \
    __vv_store(dst_nram + 5 * vv_num, vv_index[5], vv_num);               \
    __vv_store(dst_nram + 6 * vv_num, vv_index[6], vv_num);               \
    __vv_store(dst_nram + 7 * vv_num, vv_index[7], vv_num);               \
  } while (false)

template <typename T>
__mlu_vector__ void __mlu_op_arange_vv_(T *dst_nram, T start_index, T step) {
#if 592 < _BANG_ARCH_
  static_assert(
      (std::is_same<T, float>::value || std::is_same<T, half>::value ||
       std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value),
      "__mlu_op_arange_vv type error!");
#else  // #if 592 < _BANG_ARCH_
  static_assert(
      (std::is_same<T, float>::value || std::is_same<T, half>::value ||
       std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
       std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value),
      "__mlu_op_arange_vv type error!");
#endif

  const uint32_t vv_num = __vv_get_length() / sizeof(T);

#if _BANG_ARCH_ <= 592
  if constexpr(std::is_same<T, uint32_t>::value) {
    MLUOP_ARANGE_VV_IMPL(vv_uint32, vv_num, dst_nram, start_index, step);
  } else if constexpr(std::is_same<T, int32_t>::value) {
    MLUOP_ARANGE_VV_IMPL(vv_int32, vv_num, dst_nram, start_index, step);
  }
#endif  // if _BANG_ARCH_ <= 592
  if constexpr(std::is_same<T, uint16_t>::value) {
    MLUOP_ARANGE_VV_IMPL(vv_uint16, vv_num, dst_nram, start_index, step);
  } else if constexpr(std::is_same<T, int16_t>::value) {
    MLUOP_ARANGE_VV_IMPL(vv_int16, vv_num, dst_nram, start_index, step);
  } else if constexpr(std::is_same<T, float>::value) {
    MLUOP_ARANGE_VV_IMPL(vv_float, vv_num, dst_nram, start_index, step);
  } else if constexpr(std::is_same<T, half>::value) {
    MLUOP_ARANGE_VV_IMPL(vv_half, vv_num, dst_nram, start_index, step);
  }
  return;
}

#if 592 < _BANG_ARCH_
template <typename T>
__mlu_func__ void __mlu_op_gen_integer_incr_seq_(T *dst_nram,
                                                 uint32_t elem_count,
                                                 T start = 0, T step = 1) {
  static_assert(
      (std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
       std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>),
      "__mlu_op_gen_integer_incr_seq type error!");
  if (std::is_same<T, uint32_t>::value) {
    __bang_incseq(reinterpret_cast<int32_t *>(dst_nram), elem_count);
  } else if (std::is_same<T, uint64_t>::value) {
    __bang_incseq(reinterpret_cast<int64_t *>(dst_nram), elem_count);
  } else {
    __bang_incseq(dst_nram, elem_count);
  }

  if (start != 0) {
    if (std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value) {
      if (step != 1) {
        __bang_mul_scalar(dst_nram, dst_nram, step, elem_count);
      }
      __bang_add_scalar(dst_nram, dst_nram, start, elem_count);
    } else {
      __bang_fusion(FUSION_FMA, dst_nram, dst_nram, step, start, elem_count);
    }
  }
}
#endif  // if 592 < _BANG_ARCH_

#define u32_sizeof(T) ((uint32_t)sizeof(T))

template <typename T>
__mlu_func__ void __mlu_op_arange_by_expand_(T *dst_nram, uint32_t numel,
                                             T start_index = 0, T step = 1) {
#if 592 < _BANG_ARCH_
  static_assert(
      (std::is_same<T, float>::value || std::is_same<T, half>::value ||
       std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value),
      "__mlu_op_arange_by_expand type error!");
#else   // if 592 < _BANG_ARCH_
  static_assert(
      (std::is_same<T, float>::value || std::is_same<T, half>::value ||
       std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
       std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
       std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value),
      "__mlu_op_arange_by_expand type error!");
#endif  // if 592 < _BANG_ARCH_

  // using AluGenSize = std::integral_constant<uint32_t, NFU_ALIGN_SIZE>;
  using GuGenSize = std::integral_constant<uint32_t, 2048>;
  uint32_t gu_gen_num = GuGenSize::value / u32_sizeof(T);
  uint32_t alu_gen_num = NFU_ALIGN_SIZE / u32_sizeof(T);
  uint32_t base_num = alu_gen_num;
#if _BANG_ARCH_ <= 592
  if (std::is_same<T, uint64_t>::value || std::is_same<T, int64_t>::value) {
    const uint32_t prologue_num = std::min(numel, base_num);
    __mlu_op_arange_base_(dst_nram, prologue_num, start_index, step);

    if (numel <= base_num) {
      return;
    }
  } else {
    if (numel <= gu_gen_num) {
      const uint32_t prologue_num = std::min(numel, base_num);
      __mlu_op_arange_base_(dst_nram, prologue_num, start_index, step);

      if (numel <= base_num) {
        return;
      }
    } else {
      __mlu_op_arange_vv_(dst_nram, start_index, step);
      base_num = gu_gen_num;
    }
  }
#else
  if (numel <= gu_gen_num) {
    const uint32_t prologue_num = std::min(numel, base_num);
    __mlu_op_arange_base_(dst_nram, prologue_num, start_index, step);

    if (numel <= base_num) {
      return;
    }
  } else {
    __mlu_op_arange_vv_(dst_nram, start_index, step);
    base_num = gu_gen_num;
  }
#endif
  // base_num = 2^exp
  uint32_t exp = 0;
  asm volatile("findlast1.gpr.b32 %[dst], %[src];\n\t"
               : [ dst ] "+&r"(exp)
               : [ src ] "r"(base_num));
  // numel = count * base_num + remain
  const uint32_t segnum = numel >> exp;
  // count = 2^repeat
  uint32_t repeat = 0;
  asm volatile("findlast1.gpr.b32 %[dst], %[src];\n\t"
               : [ dst ] "+&r"(repeat)
               : [ src ] "r"(segnum));
  uint32_t count = 1;
  for (uint32_t i = 0; i < repeat; ++i) {
    __bang_add_scalar(dst_nram + count * base_num, dst_nram,
                      count * base_num * step, count * base_num);
    count *= 2;
  }

  const uint32_t remain = numel - count * base_num;
  if (0 < remain) {
    __bang_add_scalar(dst_nram + count * base_num, dst_nram,
                      count * base_num * step, remain);
  }
}
/***************************************************************************

    CNNL FUNC: __mlu_op_gen_stage_index.
    param "dst_nram" is a nram pointer to the generated result.
    param "numel" is the element number of to be generated.
    param "start_index" is the starting value for the set of points. Default: 0.
    param "step" is the gap between each pair of adjacent points points.
   Default: 1. dst_addition. remarks: Detailed introduction for reference
    http://wiki.cambricon.com/pages/viewpage.action?pageId=119467501.
    int64_t and uint64_t types are under-optimized and can be improved with GU.
    *************************************************************************/

template <typename T>
__mlu_func__ void __mlu_op_gen_stage_index(T *dst_nram, uint32_t numel,
                                           T start_index = 0, T step = 1) {
#if 592 < _BANG_ARCH_
  if (std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
      std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value) {
    __mlu_op_gen_integer_incr_seq_(dst_nram, numel, start_index, step);
  } else {
    __mlu_op_arange_by_expand_(dst_nram, numel, start_index, step);
  }
#else
  __mlu_op_arange_by_expand_(dst_nram, numel, start_index, step);
#endif
}

#endif  // KERNELS_UTILS_COMMON_H_
