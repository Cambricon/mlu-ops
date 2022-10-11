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
#ifndef KERNELS_UTILS__COMMON_H_
#define KERNELS_UTILS__COMMON_H_

#include <type_traits>
#include "float.h"
#include "kernels/kernel.h"

#define HALFMAX 65504

/******************************************************************************************
 * MLUOPS FUNC: computeRecip
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
 ******************************************************************************************/
template <typename T>
static __mlu_func__ void computeRecip(T *nram_dst, T *nram_src,
                                      void *nram_addition,
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
 * MLUOPS FUNC: computeExp
 * param 'nram_dst' is the nram destination address, which supports half or
 * float data type. param 'nram_src' is the nram source address, which has the
 * same data type as nram_dst. param 'nram_addition' is the nram addition
 * address. Pass NULL if the data type of nram_src is float, otherwise the space
 * size is at least twice as much as nram_src. param 'is_high_precision' is the
 * precision flag. param 'deal_num' is the num of input data. remarks: nram_dst
 * and nram_src can be homologous operand.
 ******************************************************************************/
template <typename T>
static __mlu_func__ void computeExp(T *nram_dst, T *nram_src,
                                    void *nram_addition,
                                    const int is_high_precision,
                                    const int deal_num) {
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
 * MLUOPS FUNC: computeSigmoid
 * param 'nram_dst' is the nram destination address, which supports half or
 * float data type. param 'nram_src' is the nram source address, which has the
 * same data type as nram_dst. param 'nram_addition' is the nram addition
 * address. Pass NULL if the data type of nram_src is float, otherwise the space
 * size is at least twice as much as nram_src. param 'is_high_precision' is the
 * precision flag. param 'deal_num' is the num of input data. remarks: nram_dst
 * and nram_src can be homologous operand.
 ******************************************************************************************/
template <typename T>
static __mlu_func__ void computeSigmoid(T *nram_dst, T *nram_src,
                                        void *nram_addition,
                                        const int is_high_precision,
                                        const int deal_num) {
  if (sizeof(T) == sizeof(float)) {
#if __BANG_ARCH__ >= 300
    __bang_mul_scalar((float *)nram_src, (float *)nram_src, (float)-1.0,
                      deal_num);
    computeExp((float *)nram_src, (float *)nram_src, NULL, 0, deal_num);
    __bang_add_scalar((float *)nram_src, (float *)nram_src, (float)1.0,
                      deal_num);
    computeRecip((float *)nram_dst, (float *)nram_src, NULL, 0, deal_num);
#else
    __bang_active_sigmoid((float *)nram_dst, (float *)nram_src, deal_num);
#endif
  } else if (sizeof(T) == sizeof(half)) {
#if __BANG_ARCH__ >= 300
    __bang_half2float((float *)nram_addition, (half *)nram_src, deal_num);
    __bang_mul_scalar((float *)nram_addition, (float *)nram_addition,
                      (float)-1.0, deal_num);
    computeExp((float *)nram_addition, (float *)nram_addition, NULL, 0,
               deal_num);
    __bang_add_scalar((float *)nram_addition, (float *)nram_addition,
                      (float)1.0, deal_num);
    computeRecip((float *)nram_dst, (float *)nram_addition, NULL, 0, deal_num);
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

#endif  // KERNELS_UTILS__COMMON_H_
