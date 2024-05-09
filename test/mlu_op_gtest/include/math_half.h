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

#include <cnrt.h>

#if !defined(__aarch64__) && defined(__x86_64__)
#include <immintrin.h>
#endif

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <string>


/* NOTE: should not include mluop_test pb and other mluop_gtest specified header
 * (executor, evaluator) */

// XXX (jiaminghao): consider move template implementation into separated files
// to speedup compilation (At present it is unnecessary)

#undef REG_MAP
#define REG_MAP(x) #x, x

namespace mluoptest {

#undef ALGO_HALF_TO_FLOAT_MAP
#define ALGO_HALF_TO_FLOAT_MAP(XX) \
  XX(SOPA)                         \
  XX(MLUOPGTEST2)                  \
  XX(CNRT)                         \
  XX(CPU_INTRINSIC)                \
  XX(LOOKUP_TABLE)

#undef TYPE_
#define TYPE_(x) x,

enum AlgoHalfToFloat : uint32_t {
  ALGO_HALF_TO_FLOAT_MAP(TYPE_) INVALID  // always the last one
};
#undef STR_TYPE_
#define STR_TYPE_(x) {x, #x},
const std::unordered_map<uint32_t, const std::string> AlgoHalfToFloatStr = {
    ALGO_HALF_TO_FLOAT_MAP(STR_TYPE_)};

// ref: sopa/core/src/util/type_converter.cpp
template <AlgoHalfToFloat algo = AlgoHalfToFloat::SOPA>
static float cvtHalfToFloatImpl(uint16_t src) {
  if (sizeof(int16_t) == 2) {
    int re = src;
    float f = 0.;
    int sign = (re >> 15) ? (-1) : 1;
    int exp = (re >> 10) & 0x1f;
    int eff = re & 0x3ff;
    float half_max = 65504.;
    float half_min = -65504.;  // or to be defined as infinity
    if (exp == 0x1f && eff) {
      // when half is nan, float also return nan, reserve sign bit
      int tmp = (sign < 0) ? 0xffffffff : 0x7fffffff;
      return *(float *)&tmp;
    } else if (exp == 0x1f && sign == 1) {
      // add upper bound of half. 0x7bff： 0 11110 1111111111 =  65504
      return half_max;
    } else if (exp == 0x1f && sign == -1) {
      // add lower bound of half. 0xfbff： 1 11110 1111111111 = -65504
      return half_min;
    }
    if (exp > 0) {
      exp -= 15;
      eff = eff | 0x400;
    } else {
      exp = -14;
    }
    int sft;
    sft = exp - 10;
    if (sft < 0) {
      f = (float)sign * eff / (1 << (-sft));
    } else {
      f = ((float)sign) * (1 << sft) * eff;
    }
    return f;
  } else if (sizeof(int16_t) == 4) {
    // using float
    return src;
  }
}

// refactor based on SOPA version, with speed up
template <>
float cvtHalfToFloatImpl<AlgoHalfToFloat::MLUOPGTEST2>(uint16_t src) {
  typedef struct {
    uint16_t fraction : 10;
    uint8_t exponent : 5;
    uint8_t sign : 1;
  } Float16Bit_t;

  typedef union {
    uint16_t data;
    Float16Bit_t bits;
  } Float16Container_t;
  static_assert(sizeof(Float16Bit_t) == sizeof(uint16_t),
                "Float16Bit should be 16 bit");
  static_assert(sizeof(Float16Container_t) == sizeof(uint16_t),
                "Float16Bit should be 16 bit");

#if 0
  constexpr float half_max = INFINITY;
  constexpr float half_min = -INFINITY;
#else
  constexpr float half_max = 65504.f;
  constexpr float half_min = -65504.f;
#endif

  Float16Container_t value{src};

  uint8_t exp = value.bits.exponent;
  uint16_t fraction = value.bits.fraction;

  const float jmp_nan_inf_table[2][2] = {{std::nanf(""), -std::nanf("")},
                                         {half_max, half_min}};
  if (exp == 0x1f) {
    return jmp_nan_inf_table[fraction == 0][value.bits.sign];
  }

  constexpr float jump_sign_table[2] = {1.f, -1.f};
  float signf = jump_sign_table[value.bits.sign];

  if (exp == 0) {
    return signf * fraction / (1 << 24);  // 24: x / 1024 / (1<<14)
  }
  int exp_offset = static_cast<int>(exp) - 15;
  float ratio = (exp_offset >= 0 ? signf * (1 << exp_offset)
                                 : (signf / (1 << -exp_offset)));
  return ratio * (0x400u | fraction) / 0x400u;
}

template <>
float cvtHalfToFloatImpl<AlgoHalfToFloat::CNRT>(uint16_t src) {
  float dst = 0;
  cnrtCastDataType_V2(&src, cnrtHalf, &dst, cnrtFloat, 1, NULL,
                      cnrtRounding_rm);
  return dst;
}

#if defined(__aarch64__)

template <>
__attribute__((always_inline)) inline float
cvtHalfToFloatImpl<AlgoHalfToFloat::CPU_INTRINSIC>(uint16_t src) {
  using float16 = __fp16;  // XXX _Float16 not work I dont know why
  return static_cast<float>(*reinterpret_cast<float16 *>(&src));
}

#elif defined(__x86_64__)
// ref: github.com/microsoft/DirectXMath/Extensions/DirectXMathF16C.h
template <>
__attribute__((always_inline)) inline float
cvtHalfToFloatImpl<AlgoHalfToFloat::CPU_INTRINSIC>(uint16_t src) {
  // NOTE for llvm/clang, may use `__fp16` directly
  return _mm_cvtss_f32(
      _mm_cvtph_ps(_mm_cvtsi32_si128(static_cast<unsigned int>(src))));
}

#else  // !__aarch64__  && !__x86_64__
#warning "unsupported architecture, fallback default implementation"
#endif  // defined(__aarch64__)

template <>
__attribute__((always_inline)) inline float
cvtHalfToFloatImpl<AlgoHalfToFloat::LOOKUP_TABLE>(uint16_t src) {
#if 0
  extern const std::array<float, UINT16_MAX+1> MappingHalfToFloat;
  return MappingHalfToFloat[src];
#else
  extern const std::array<uint32_t, UINT16_MAX + 1> MappingHalfToFloat;
  float ret;
  memcpy(&ret, &(MappingHalfToFloat[src]), sizeof(float));
  return ret;
//  return reinterpret_cast<float *>(
//      const_cast<uint32_t *>(MappingHalfToFloat.data())
//      )[src];
#endif
}

}  // namespace mluoptest
