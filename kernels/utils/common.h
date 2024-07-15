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


#define STEP 1
#define VEC_LEN (32 * 1024)
#define VEC_BYTES (VEC_LEN * 4)

__nram__ uint32_t ptxm_rsqrt_table[128][3] = {
{0x3fffffb, 0xfffb, 0x2f2},
{0x3f817ab, 0xfa18, 0x2d5},
{0x3f05d8d, 0xf46f, 0x2ba},
{0x3e8cfe1, 0xeefb, 0x2a0},
{0x3e16d06, 0xe9bb, 0x288},
{0x3da336b, 0xe4ab, 0x271},
{0x3d32197, 0xdfc9, 0x25b},
{0x3cc3623, 0xdb13, 0x246},
{0x3c56fb9, 0xd687, 0x232},
{0x3becd10, 0xd222, 0x21f},
{0x3b84cf2, 0xcde4, 0x20e},
{0x3b1ee3d, 0xc9c9, 0x1fc},
{0x3abafd0, 0xc5d1, 0x1ec},
{0x3a590a4, 0xc1fa, 0x1dc},
{0x39f8fb1, 0xbe42, 0x1cd},
{0x399ac06, 0xbaa8, 0x1be},
{0x393e4b4, 0xb72b, 0x1b1},
{0x38e38e1, 0xb3ca, 0x1a4},
{0x388a7ae, 0xb082, 0x197},
{0x3833053, 0xad54, 0x18b},
{0x37dd208, 0xaa3e, 0x17f},
{0x3788c0f, 0xa73f, 0x174},
{0x3735db3, 0xa457, 0x16a},
{0x36e4647, 0xa184, 0x160},
{0x3694523, 0x9ec5, 0x156},
{0x36459a9, 0x9c1a, 0x14c},
{0x35f8339, 0x9982, 0x143},
{0x35ac142, 0x96fc, 0x13a},
{0x3561334, 0x9487, 0x131},
{0x3517886, 0x9224, 0x12a},
{0x34cf0b2, 0x8fd0, 0x122},
{0x3487b3e, 0x8d8d, 0x11b},
{0x34417ac, 0x8b58, 0x113},
{0x33fc585, 0x8932, 0x10c},
{0x33b8454, 0x871a, 0x106},
{0x33753b2, 0x850f, 0x0ff},
{0x3333331, 0x8311, 0x0f8},
{0x32f226a, 0x8120, 0x0f2},
{0x32b20fc, 0x7f3b, 0x0ec},
{0x3272e83, 0x7d62, 0x0e7},
{0x3234aaa, 0x7b94, 0x0e1},
{0x31f7510, 0x79d1, 0x0dc},
{0x31bad65, 0x7819, 0x0d7},
{0x317f353, 0x766b, 0x0d2},
{0x3144689, 0x74c7, 0x0cd},
{0x310a6b8, 0x732d, 0x0c9},
{0x30d1398, 0x719c, 0x0c4},
{0x3098cda, 0x7013, 0x0bf},
{0x306123c, 0x6e94, 0x0bb},
{0x302a375, 0x6d1d, 0x0b7},
{0x2ff4045, 0x6bae, 0x0b3},
{0x2fbe86c, 0x6a48, 0x0b0},
{0x2f89bab, 0x68e9, 0x0ac},
{0x2f559c3, 0x6791, 0x0a8},
{0x2f2227b, 0x6641, 0x0a5},
{0x2eef598, 0x64f8, 0x0a2},
{0x2ebd2e6, 0x63b5, 0x09e},
{0x2e8ba2d, 0x6279, 0x09a},
{0x2e5ab38, 0x6144, 0x097},
{0x2e2a5d3, 0x6015, 0x094},
{0x2dfa9ce, 0x5eec, 0x091},
{0x2dcb6f6, 0x5dc9, 0x08f},
{0x2d9cd24, 0x5cac, 0x08c},
{0x2d6ec23, 0x5b94, 0x089},
{0x2d413c7, 0xb501, 0x215},
{0x2ce7c60, 0xb0d8, 0x201},
{0x2c905a4, 0xacd7, 0x1ed},
{0x2c3ae53, 0xa8fc, 0x1db},
{0x2be7548, 0xa546, 0x1cb},
{0x2b9596a, 0xa1b1, 0x1ba},
{0x2b459af, 0x9e3e, 0x1ab},
{0x2af7512, 0x9ae9, 0x19c},
{0x2aaaaa8, 0x97b2, 0x18e},
{0x2a5f985, 0x9496, 0x180},
{0x2a160d3, 0x9196, 0x173},
{0x29cdfbb, 0x8eaf, 0x167},
{0x298757b, 0x8be1, 0x15c},
{0x2942150, 0x8929, 0x150},
{0x28fe285, 0x8688, 0x146},
{0x28bb873, 0x83fd, 0x13c},
{0x287a269, 0x8185, 0x132},
{0x2839fcf, 0x7f21, 0x129},
{0x27fb00e, 0x7cd0, 0x120},
{0x27bd28f, 0x7a90, 0x117},
{0x27806c7, 0x7861, 0x10f},
{0x2744c35, 0x7643, 0x107},
{0x270a255, 0x7435, 0x100},
{0x26d08ad, 0x7235, 0x0f8},
{0x2697ec5, 0x7044, 0x0f1},
{0x266042c, 0x6e61, 0x0eb},
{0x2629877, 0x6c8c, 0x0e5},
{0x25f3b3e, 0x6ac3, 0x0de},
{0x25bec15, 0x6907, 0x0d9},
{0x258aaa4, 0x6756, 0x0d2},
{0x2557685, 0x65b1, 0x0cd},
{0x2524f65, 0x6417, 0x0c7},
{0x24f34e5, 0x6288, 0x0c3},
{0x24c26ba, 0x6103, 0x0be},
{0x2492491, 0x5f88, 0x0b9},
{0x2462e18, 0x5e16, 0x0b4},
{0x2434308, 0x5cae, 0x0b0},
{0x2406317, 0x5b4e, 0x0ab},
{0x23d8e00, 0x59f7, 0x0a7},
{0x23ac380, 0x58a9, 0x0a3},
{0x2380352, 0x5762, 0x09f},
{0x2354d3b, 0x5623, 0x09b},
{0x232a0fb, 0x54ec, 0x098},
{0x22ffe5b, 0x53bc, 0x094},
{0x22d651c, 0x5293, 0x091},
{0x22ad50c, 0x5171, 0x08e},
{0x2284df5, 0x5055, 0x08a},
{0x225cf9d, 0x4f40, 0x088},
{0x22359d9, 0x4e31, 0x085},
{0x220ec77, 0x4d28, 0x082},
{0x21e8748, 0x4c24, 0x07e},
{0x21c2a1b, 0x4b27, 0x07c},
{0x219d4c4, 0x4a2f, 0x07a},
{0x217871d, 0x493c, 0x077},
{0x21540f4, 0x484e, 0x075},
{0x213022a, 0x4765, 0x072},
{0x210ca91, 0x4681, 0x070},
{0x20e9a06, 0x45a2, 0x06e},
{0x20c7065, 0x44c7, 0x06b},
{0x20a4d86, 0x43f1, 0x069},
{0x2083148, 0x431f, 0x067},
{0x2061b87, 0x4251, 0x065},
{0x2040c25, 0x4187, 0x063},
{0x2020305, 0x40c2, 0x061}
};

__mlu_func__ int my_ilogbf(float x) {
  int xi = *((int*)(&x)) & 0x7fffffff;
  if (xi < 0x00800000) {
    int y = -126;
    for (xi <<= 8; xi > 0; xi <<= 1, y--) {}
    return y;
  } else {
    return (xi >> 23) - 127;
  }
}

__mlu_func__ void add_i64(uint32_t xh, uint32_t xl, uint32_t yh, uint32_t yl, uint32_t* zh, uint32_t* zl) {
  uint32_t tmp = xl < yl ? xl : yl;
  *zl = xl + yl;
  *zh = xh + yh;
  if (*zl < tmp) {
    *zh = *zh + 1;
  }
}

__mlu_func__ void sub_i64(uint32_t xh, uint32_t xl, uint32_t yh, uint32_t yl, uint32_t* zh, uint32_t* zl) {
  bool c = xl < yl;
  *zl = xl - yl;
  *zh = xh - yh;
  if (c) {
    *zh = *zh - 1;
  }
}

__mlu_func__ void sll_i64(uint32_t xh, uint32_t xl, uint32_t y, uint32_t* zh, uint32_t* zl) {
  uint32_t tmp = xl >> (32 - y);
  *zl = xl << y;
  *zh = tmp | (xh << y);
}

__mlu_func__ uint32_t ppm_antidiagonal(uint32_t x) {
  uint32_t y = x & 1;
  y |= ((x & 2) << 1);
  y |= ((x & 4) << 2);
  y |= ((x & 8) << 3);
  y |= ((x & 16) << 4);
  y |= ((x & 32) << 5);
  y |= ((x & 64) << 6);
  y |= ((x & 128) << 7);
  y |= ((x & 256) << 8);
  y |= ((x & 512) << 9);
  return y;
}

__mlu_func__ uint32_t ppm_row(uint32_t x, int i) {
  if (!((x >> i) & 1)) return 0;
  x <<= i + 1;
  x &= ~((1 << (2 * i + 2)) - 1);
  return x;
}

__mlu_func__ void ptxm_square_approx(uint32_t c2, uint32_t xl, uint32_t* yhi, uint32_t* ylo) {
  uint32_t error_lo = ppm_antidiagonal(xl);
  uint32_t error_hi = 0;
  for (int i = 0; i < 9; i++) {
    uint32_t tmp = ppm_row(xl, i) & 0x7ffff;
    add_i64(error_hi, error_lo, 0, tmp, &error_hi, &error_lo);
  }
  uint32_t exact_lo = xl * xl;
  uint32_t exact_hi = __cn_scalar_mulh_u32(xl, xl);
  uint32_t subhi, sublo;
  sub_i64(exact_hi, exact_lo, error_hi, error_lo, &subhi, &sublo);
  *ylo = c2 * sublo;
  *yhi = __cn_scalar_mulh_u32(c2, sublo) + c2 * subhi;
}

__mlu_func__ float rsqrt_nv(float x) {
  int x_log2 = 0;
  uint32_t xi = *((uint32_t*)(&x)), ptxm_nan = 0x7fffffff, pinf = 0x7f800000, ninf = 0xff800000U;
  if (xi == 0x80000000U) return *((float*)(&ninf));
  if (xi == 0) return *((float*)(&pinf));
  if (xi == pinf) return 0.0f;
  if (xi > pinf) return *((float*)(&ptxm_nan));
  if (xi < 0x00800000) {
    x_log2 = -24;
    x *= 0x1p24f;  // 2^24
  }
  x_log2 += my_ilogbf(x);
  uint32_t x_bits = *((uint32_t*)(&x));
  uint32_t xh = (x_bits >> 17) & 0x3f;
  uint32_t xl = x_bits & 0x1ffff;
  if ((x_log2 & 1) == 1) {
    x_log2 -= 1;
    xh += 64;
  } else if ((x_bits & 0x7fffff) == 0) {
    return __cn_scalar_scalbn_f32(1.0f, -x_log2 >> 1);
  }
  const uint32_t *const c = ptxm_rsqrt_table[xh];
  uint32_t c0_term_lo = c[0];
  uint32_t c0_term_hi = 0;
  uint32_t c1_term_lo = c[1];
  uint32_t c1_term_hi = __cn_scalar_mulh_u32(c1_term_lo, xl);
  c1_term_lo *= xl;
  uint32_t c2_term_lo, c2_term_hi;
  ptxm_square_approx(c[2], xl, &c2_term_hi, &c2_term_lo);
  sll_i64(c0_term_hi, c0_term_lo, 31, &c0_term_hi, &c0_term_lo);
  sll_i64(c1_term_hi, c1_term_lo, 17, &c1_term_hi, &c1_term_lo);
  uint32_t sum_lo, sum_hi;
  sub_i64(c0_term_hi, c0_term_lo, c1_term_hi, c1_term_lo, &sum_hi, &sum_lo);
  add_i64(sum_hi, sum_lo, c2_term_hi, c2_term_lo, &sum_hi, &sum_lo);
  add_i64(sum_hi, sum_lo, 1, 0xFFFE0000U, &sum_hi, &sum_lo);
  sll_i64(sum_hi, sum_lo, 1, &sum_hi, &sum_lo);
  uint32_t r_frac = (sum_hi >> 2) & 0x7fffff;
  uint32_t r_bits = 0x3F000000 | r_frac;
  return __cn_scalar_scalbn_f32(*((float*)(&r_bits)), -x_log2 >> 1);
}


__mlu_func__ float approx_fma_f32_2(float a, float b, float C) {
  int ahi2 = *((int*)&a) & 0xfffff000;
  float ahi = *((float*)&ahi2);
  int bhi2 = *((int*)&b) & 0xfffff000;
  float bhi = *((float*)&bhi2);
  float alo = a - ahi, blo = b - bhi;
  return ahi*bhi + C + alo*blo + ahi*blo + alo*bhi;
}

__mlu_func__ float sqrt_ulp0(float x) {
  uint32_t xi = *((uint32_t*)&x), Cspe = 0x5f7fffff;
  if (xi == 0x7f7ffffe) return *((float*)&Cspe);
  if (xi == 0x7f800000 || xi == 0x80000000U) return x;
  if (xi < 0x00800000) return __cn_scalar_sqrt_f32(x);
  float R0 = __cn_scalar_rsqrt_f32(x);
  float R2 = R0 * x;
  R0 = R0 * 0.5f;
  float R3 = approx_fma_f32_2(-R2, R2, x);
  R0 = R3 * R0 + R2;
  return R0;
}

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
 *      The rounding mode on MLU200 is rd, on MLU300 is rn.
 ******************************************************************************/
__mlu_func__ void __mluop_float2half(half *dst, float *src,
                                            int src_count) {
#if __BANG_ARCH__ >= 300
  __bang_float2half_rn(dst, src, src_count);
#else
  __bang_float2half_rd(dst, src, src_count);
#endif
}

__mlu_func__ half __mluop_float2half(float a) {
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
