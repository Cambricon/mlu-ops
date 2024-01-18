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

#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>
#include <iomanip>
#include <sstream>
#include <memory>
#include <vector>

#include "cnrt.h"

#include "gtest/gtest.h"
#include "tools.h"
#include "variable.h"
#include "math_half.h"

template <typename T>
std::string to_hex_str(T input) {
  std::stringstream oss;
  if (sizeof(T) == 1) {
    oss << "0x" << std::hex << (uint32_t)(*(uint8_t *)(&input));
  } else if (sizeof(T) == 2) {
    oss << "0x" << std::hex << *((uint16_t *)(&input));
  } else if (sizeof(T) == 4) {
    oss << "0x" << std::hex << *((uint32_t *)(&input));
  } else {
    oss << "0x" << std::hex << input;
  }
  return oss.str();
}

namespace {
using mluoptest::AlgoHalfToFloat;
using mluoptest::AlgoHalfToFloatStr;
using mluoptest::global_var;
// Test function inside mluop_gtest source code itself
TEST(DISABLED_ArrayCastHalfToFloatSelfTest, TEST) {
  auto algo = global_var.half2float_algo_;
  constexpr size_t len = UINT16_MAX + 1;
  std::vector<uint16_t> src(len);
  std::vector<float> dst_base(len);
  std::vector<float> dst_compare(len);

  auto eq = [algo](uint16_t *src, float *dst_compare, float *dst_base,
                   size_t len) {
    memset(dst_compare, 0, len);
    cnrtCastDataType_V2(src, cnrtHalf, dst_base, cnrtFloat, len, NULL,
                        cnrtRounding_rm);
    mluoptest::arrayCastHalfToFloat(dst_compare,
                                    reinterpret_cast<int16_t *>(src), len);
    for (size_t i = 0; i < len; i++) {
      if (isnanf(dst_base[i])) {
        EXPECT_EQ(isnanf(dst_compare[i]), isnanf(dst_base[i]))
            << "src[" << i << "] should be nan";
      } else {
        if (algo == AlgoHalfToFloat::SOPA ||
            algo == AlgoHalfToFloat::MLUOPGTEST2) {
          if (isinff(dst_base[i])) {
            if (dst_base[i] > 0) {
              EXPECT_EQ(65504.f, dst_compare[i])
                  << "src[" << i << "] should be 65504.f";
            } else {
              EXPECT_EQ(-65504.f, dst_compare[i])
                  << "src[" << i << "] should be -65504.f";
            }
          } else {
            EXPECT_EQ(dst_compare[i], dst_base[i])
                << "src[" << i << "]=" << to_hex_str(src[i])
                << " conversion failed. should be " << to_hex_str(dst_base[i])
                << ", not " << to_hex_str(dst_compare[i]);
          }
        } else {
          EXPECT_EQ(dst_compare[i], dst_base[i])
              << "src[" << i << "]=" << to_hex_str(src[i])
              << " conversion failed. should be " << to_hex_str(dst_base[i])
              << ", not " << to_hex_str(dst_compare[i]);
        }
      }
    }
  };

  for (size_t i = 0; i < len; i++) {
    src[i] = static_cast<uint16_t>(i);
  }
  eq(src.data(), dst_compare.data(), dst_base.data(), len);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint16_t> dist(0, UINT16_MAX);
  for (size_t i = 0; i < len; i++) {
    src[i] = dist(gen);
  }
  size_t pos = 1;
  eq(src.data(), dst_compare.data(), dst_base.data(), pos);
  eq(src.data() + 1, dst_compare.data(), dst_base.data(), pos);
  pos = 2;
  eq(src.data() + 1, dst_compare.data(), dst_base.data(), pos);
  pos = 5;
  eq(src.data() + 1, dst_compare.data(), dst_base.data(), pos);
  pos = 7;
  eq(src.data() + 1, dst_compare.data(), dst_base.data(), pos);
  eq(src.data() + 8, dst_compare.data(), dst_base.data(), pos);
  pos = 129;
  eq(src.data() + 6, dst_compare.data(), dst_base.data(), pos);
  pos = 259;
  eq(src.data() + 13, dst_compare.data(), dst_base.data(), pos);
  pos = 519;
  eq(src.data() + 30, dst_compare.data(), dst_base.data(), pos);
  eq(src.data(), dst_compare.data(), dst_base.data(), len - 1);

  // delete [] src;
  // delete [] dst_base;
  // delete [] dst_compare;
}
}  // namespace
