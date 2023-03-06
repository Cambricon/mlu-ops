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
#ifndef CORE_TOOL_H_
#define CORE_TOOL_H_

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "core/logging.h"
#include "core/type.h"
#include "mlu_op.h"


namespace mluop {
// The API is used for no scling factor quantization.
mluOpStatus_t getPosition(float *input, size_t num, mluOpDataType_t datatype,
                          int *position);

// The API is used for scaling factor quantization.
mluOpStatus_t getPositionAndScale(float *input, size_t num,
                                  mluOpDataType_t datatype, int *position,
                                  float *scale);
// The API is used for asymmetrical quantization.
mluOpStatus_t getPositionScaleAndOffset(float *input, size_t num,
                                        mluOpDataType_t datatype, int *position,
                                        float *scale, int *offset);
mluOpStatus_t castDtypeToBitwidth(mluOpDataType_t quantize_dtype,
                                  int *bitwidth);

int16_t castFloat32ToHalf(float src);
float castHalfToFloat32(int16_t src);
size_t getMemorySize(const void *ptr);
mluOpStatus_t checkMemorySize(mluOpTensorDescriptor_t tensor, const void *ptr);

inline bool isTensorDimsEqual(mluOpTensorDescriptor_t a,
                              mluOpTensorDescriptor_t b) {
  int a_dim;
  int b_dim;
  int a_dims[MLUOP_DIM_MAX];
  int b_dims[MLUOP_DIM_MAX];
  mluOpGetTensorDescriptor(a, nullptr, nullptr, &a_dim, a_dims);
  mluOpGetTensorDescriptor(b, nullptr, nullptr, &b_dim, b_dims);

  if (a_dim == b_dim) {
    if (0 == memcmp(a_dims, b_dims, a_dim * sizeof(int))) {
      return true;
    }
  }
  return false;
}

template <typename T>
inline bool isTwoArraysEqual(T *a, T *b, int num) {
  if (0 == memcmp(a, b, num * sizeof(T))) {
    return true;
  }
  return false;
}

int mkdirIfNotExist(const char *pathname);
int mkdirRecursive(const char *pathname);
uint64_t getUintEnvVar(const std::string &str, uint64_t default_para = 0);
std::string getStringEnvVar(const std::string &str,
                            std::string default_para = "");
bool getBoolEnvVar(const std::string &str, bool default_para = false);


template <typename FixedType>
mluOpStatus_t castFloat32ToFixed(
    const float *src, FixedType *dst, const size_t num, const int position = 0,
    const float scale = 1.0, const int offset = 0,
    mluOpQuantizeRoundMode_t round_mode = MLUOP_ROUND_HALF_OFF_ZERO) {
  PARAM_CHECK("[castFloat32ToFixed]", src != NULL);
  PARAM_CHECK("[castFloat32ToFixed]", dst != NULL);
  PARAM_CHECK("[castFloat32ToFixed]", num > 0);

  const float max = pow(2, sizeof(FixedType) * 8 - 1) + (-1);
  const float min = pow(2, sizeof(FixedType) * 8 - 1) * (-1);
  for (size_t i = 0; i < num; ++i) {
    float res =
        static_cast<float>((src[i] * scale / pow(2, position) + offset));
    if (res > max) {
      res = max;
    } else if (res < min) {
      res = min;
    }
    if (round_mode == MLUOP_ROUND_HALF_OFF_ZERO) {
      dst[i] = static_cast<FixedType>(round(res));
    } else if (round_mode == MLUOP_ROUND_HALF_TO_EVEN) {
      dst[i] = static_cast<FixedType>(rint(res));
    } else if (round_mode == MLUOP_ROUND_HALF_UP) {
      dst[i] = static_cast<FixedType>(floor(res + 0.5));
    }
  }
  return MLUOP_STATUS_SUCCESS;
}

template <typename FixedType>
mluOpStatus_t castFixedToFloat32(const FixedType *src, float *dst,
                                 const size_t num, const int position = 0,
                                 const float scale = 1.0,
                                 const int offset = 0) {
  PARAM_CHECK("[castFixedToFloat32]", src != NULL);
  PARAM_CHECK("[castFixedToFloat32]", dst != NULL);
  PARAM_CHECK("[castFixedToFloat32]", num > 0);

  for (size_t i = 0; i < num; ++i) {
    dst[i] = (static_cast<float>(src[i]) - offset) * pow(2, position) / scale;
  }
  return MLUOP_STATUS_SUCCESS;
}
}  // namespace mluop

#endif  // CORE_TOOL_H_
