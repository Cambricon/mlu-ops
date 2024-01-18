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

/* NOTE: should not include mluop_test pb nor other mluop_gtest specified header
 * (executor, evaluator) */

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "math_half.h"

auto code_template = R"(
// AUTO GENERATED
#include <array>
#include <cstdint>
#include <cmath>
namespace mluoptest {
  union _IntAsFloat {
    uint32_t i;
    float f;
  };
  // __attribute__((always_inline)) inline float _cvt(uint32_t src) {
  //   return *reinterpret_cast<float *>(const_cast<uint32_t *>(&src));
  // }
  extern const std::array<uint32_t, UINT16_MAX+1> MappingHalfToFloat = {
  %s
  };
}
)";

int main(int argc, char *argv[]) {
  (void)argc, (void)argv;  // At present, argc and argv not used yet
  char filename[] = "half2float_table.cpp";
  FILE *fp = fopen(filename, "w");
  if (fp == NULL) {
    perror("create file failed\n");
    return -1;
  }

  auto _half2float =
      mluoptest::cvtHalfToFloatImpl<mluoptest::AlgoHalfToFloat::CPU_INTRINSIC>;

  std::ostringstream ofs;
  uint16_t inp = 0;
  for (int i = 0; i <= UINT16_MAX; i++) {
    float out = _half2float(inp++);
    // I just want to record hex representation, may be changed in the future
    // (cuz compile time cost)
#if 0
    ofs << "  _IntAsFloat{"
        << std::showbase << std::hex
        << *reinterpret_cast<uint32_t *>(&out) << "}.f";
#else
    ofs << "  " << std::showbase << std::hex
        << *reinterpret_cast<uint32_t *>(&out) << "";
#endif
    ofs << ",  // " << i << "\n";
  }
  assert(inp == 0);
  auto ofs_str = ofs.str();
  fprintf(fp, code_template, ofs_str.c_str());
  return 0;
}
