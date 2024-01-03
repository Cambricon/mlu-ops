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
#ifndef TEST_MLU_OP_GTEST_INCLUDE_CPU_DTYPE_H_
#define TEST_MLU_OP_GTEST_INCLUDE_CPU_DTYPE_H_

#include <type_traits>
#include "Eigen/Core"
#include "mlu_op.h"

#define CPU_DTYPE(DTYPE) GTEST_DTYPE<DTYPE>::type

namespace mluoptest {

using Eigen::bfloat16;
using Eigen::half;

template <mluOpDataType_t dtype>
struct GTEST_DTYPE {
  using type = float;
};

template <>
struct GTEST_DTYPE<MLUOP_DTYPE_FLOAT> {
  using type = float;
};

template <>
struct GTEST_DTYPE<MLUOP_DTYPE_HALF> {
  using type = half;
};

template <>
struct GTEST_DTYPE<MLUOP_DTYPE_BFLOAT16> {
  using type = bfloat16;
};

template <>
struct GTEST_DTYPE<MLUOP_DTYPE_DOUBLE> {
  using type = double;
};

template <>
struct GTEST_DTYPE<MLUOP_DTYPE_INT8> {
  using type = int8_t;
};

template <>
struct GTEST_DTYPE<MLUOP_DTYPE_INT16> {
  using type = int16_t;
};

template <>
struct GTEST_DTYPE<MLUOP_DTYPE_INT32> {
  using type = int32_t;
};

template <>
struct GTEST_DTYPE<MLUOP_DTYPE_INT64> {
  using type = int64_t;
};

template <>
struct GTEST_DTYPE<MLUOP_DTYPE_UINT8> {
  using type = uint8_t;
};

template <>
struct GTEST_DTYPE<MLUOP_DTYPE_UINT16> {
  using type = uint16_t;
};

template <>
struct GTEST_DTYPE<MLUOP_DTYPE_UINT32> {
  using type = uint32_t;
};

template <>
struct GTEST_DTYPE<MLUOP_DTYPE_UINT64> {
  using type = uint64_t;
};

template <>
struct GTEST_DTYPE<MLUOP_DTYPE_BOOL> {
  using type = bool;
};

template <>
struct GTEST_DTYPE<MLUOP_DTYPE_COMPLEX_HALF> {
  using type = std::complex<half>;
};

template <>
struct GTEST_DTYPE<MLUOP_DTYPE_COMPLEX_FLOAT> {
  using type = std::complex<float>;
};

template <>
struct GTEST_DTYPE<MLUOP_DTYPE_INT31> {
  // only for main process, op cannot use it
  using type = int32_t;
};
}  // namespace mluoptest
#endif  // TEST_MLU_OP_GTEST_INCLUDE_CPU_DTYPE_H_
