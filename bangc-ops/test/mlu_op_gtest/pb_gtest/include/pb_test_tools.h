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
#ifndef TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_TOOLS_H_
#define TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_TOOLS_H_

#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <random>
#include <thread>  //NOLINT
#include <vector>
#include <limits>
#include <unordered_map>
#include "core/tensor.h"
#include "evaluator.h"
#include "gtest/gtest.h"
#include "mlu_op_test.pb.h"
#include "mlu_op.h"
#include "pb_test_tools.h"

namespace mluoptest {

// for debug
void saveDataToFile(const std::string &file, float *data, size_t count);
void saveDataToFile(const std::string &file, void *data, mluOpDataType_t dtype,
                    size_t count);
void readDataFromFile(const std::string &file, float *data, size_t count);
void saveHexDataToFile(const std::string &file, void *data,
                       mluOpDataType_t dtype, size_t count);

// include stride
// if no stride this count == shape count
size_t shapeStrideCount(const Shape *shape);
// not include stride
size_t shapeElementCount(const Shape *shape);

cnrtDataType_t cvtMluOpDtypeToCnrt(mluOpDataType_t dtype);
mluOpDataType_t cvtProtoDtypeToMluOp(DataType dtype);
mluOpTensorLayout_t cvtProtoLayoutToMluOp(TensorLayout order);
int16_t cvtFloatToHalf(float x);
float cvtHalfToFloat(int16_t);

cnrtRet_t wrapRtConvertFloatToHalf(uint16_t *f16, float d);
cnrtRet_t wrapRtConvertHalfToFloat(float *d, uint16_t f16);

void arrayCastFloatToHalf(int16_t *dst, float *src, size_t num);
void arrayCastHalfToFloat(float *dst, int16_t *src, size_t num);
void arrayCastFloatAndNormal(void *src_data, mluOpDataType_t src_dtype,
                             void *dst_data, mluOpDataType_t dst_dtype,
                             size_t num);
void arrayCastHalfToInt8or16HalfUp(void *dst, int16_t *src, int pos, size_t num,
                                   int int8or16);
bool getEnv(const std::string &env, bool default_ret);
int getEnvInt(const std::string &env, int default_ret);
size_t proc_usage_peak();
std::unordered_map<std::string, std::vector<std::string>> readFileByLine(
    const std::string &file);
// half mult
int float_mult(int in_a, int in_b, int float_16or32, int round_mode,
               int ieee754);

// half add
int float_add(int in_a, int in_b, int float_16or32, int round_mode,
              int add_or_sub, int ieee754);

bool hasRandomBound(const RandomData *random_param);
// force fix to float
template <typename FixedType>
void forceFixToFloat(FixedType *src, float *dst, const size_t num) {
  for (size_t i = 0; i < num; ++i) {
    dst[i] = float(int(src[i]));
  }
}

typedef enum {
  ROUND_MODE_TO_ZERO,
  ROUND_MODE_OFF_ZERO,
  ROUND_MODE_UP,
  ROUND_MODE_DOWN,
  ROUND_MODE_NEAREST_OFF_ZERO,
  ROUND_MODE_NEAREST_EVEN,
  ROUND_MODE_MATH,
  ROUND_MODE_NO,
} round_mode_t;

// half -> int8/int16
template <typename FixedType>
void cvtHalfToFixed(const float *src, FixedType *dst, const size_t num,
                    const int position = 0, const float scale = 1.0,
                    const int offset = 0) {
  const float max = pow(2, sizeof(FixedType) * 8 - 1) + (-1);
  const float min = pow(2, sizeof(FixedType) * 8 - 1) * (-1);
  for (size_t i = 0; i < num; ++i) {
    int16_t res = float_mult(cvtFloatToHalf(src[i]), cvtFloatToHalf(scale), 0,
                             ROUND_MODE_NEAREST_EVEN, 1);
    // use 10 because half exponend width only 5 bit
    int pos_tmp = position >= 0 ? 10 : -10;
    float tmp = powf(2, -pos_tmp);
    int16_t tmp_half = cvtFloatToHalf(tmp);
    int16_t offset_half = cvtFloatToHalf(offset);
    for (int cycle = 0; cycle < position / pos_tmp; ++cycle) {
      res = float_mult(res, tmp_half, 0, ROUND_MODE_NEAREST_EVEN, 1);
    }
    if (position % pos_tmp) {
      tmp = pow(2, -(position % pos_tmp));
      int16_t tmp_half = cvtFloatToHalf(tmp);
      res = float_mult(res, tmp_half, 0, ROUND_MODE_NEAREST_EVEN, 1);
    }
    res = float_add(res, offset_half, 0, ROUND_MODE_NEAREST_EVEN, 0, 1);
    float res1 = cvtHalfToFloat(res);
    if (res1 > max) {
      res1 = max;
    } else if (res1 < min) {
      res1 = min;
    }
    dst[i] = static_cast<FixedType>(round(res1));
  }
}

template <typename T>
void generateRandomData(T *data, size_t count, const RandomData *random_param,
                        DataType dtype) {
  // round to int
  // if convert_dtype == true, round(float) to int,
  // else don't round, int is qint
  bool convert_dtype =
      random_param->has_convert_dtype() ? random_param->convert_dtype() : false;
  int seed = 23;
  if (random_param->has_seed()) {
    seed = random_param->seed();
  }
  // generate random data
  std::default_random_engine re(seed);  // re for random engine
  bool is_lower_equal_upper = false;

  if (random_param->distribution() == mluoptest::UNIFORM) {
    T lower = 1.;
    T upper = -1.;
    if (random_param->has_lower_bound_double()) {
      lower = (T)random_param->lower_bound_double();
      upper = (T)random_param->upper_bound_double();
    } else {
      lower = (T)random_param->lower_bound();
      upper = (T)random_param->upper_bound();
    }
    if (lower == upper) {
      is_lower_equal_upper = true;
      for (size_t i = 0; i < count; ++i) {
        data[i] = lower;
      }
    } else {
      // uniform_real_distribution is [lower, upper)
      upper = std::nexttoward(upper, -std::numeric_limits<T>::infinity());
      std::uniform_real_distribution<T> dis(lower, upper);
      for (size_t i = 0; i < count; ++i) {
        data[i] = dis(re);
      }
    }
  } else if (random_param->distribution() == mluoptest::GAUSSIAN) {
    T mu = 0;
    T sigma = 1;
    if (random_param->has_mu_double()) {
      mu = (T)random_param->mu_double();
      sigma = (T)random_param->sigma_double();
    } else {
      mu = (T)random_param->mu();
      sigma = (T)random_param->sigma();
    }
    // uniform_real_distribution is [lower, upper)
    std::normal_distribution<T> dis(mu, sigma);
    for (size_t i = 0; i < count; ++i) {
      data[i] = dis(re);
    }
  }

  // reset data by dtype
  switch (dtype) {
    case DTYPE_HALF:
    case DTYPE_FLOAT:
    case DTYPE_DOUBLE:
    case DTYPE_COMPLEX_HALF:
    case DTYPE_COMPLEX_FLOAT:
      break;
    case DTYPE_INT8:
    case DTYPE_INT16: {
      if (convert_dtype) {
        // if convert_dtype == true, round(float) to int,
        // else don't round, int is qint
        for (size_t i = 0; i < count; ++i) {
          int x = std::floor(data[i]);
          data[i] = x;
        }
      }
    }; break;
    case DTYPE_UINT8:
    case DTYPE_UINT16:
    case DTYPE_UINT32: {
      for (size_t i = 0; i < count; ++i) {
        uint32_t x = std::floor(data[i]);
        data[i] = x;
      }
    }; break;
    case DTYPE_INT32: {
      for (size_t i = 0; i < count; ++i) {
        int x = std::floor(data[i]);
        data[i] = x;
      }
    }; break;
    case DTYPE_INT64: {
      for (size_t i = 0; i < count; ++i) {
        int64_t x = std::floor(data[i]);
        data[i] = x;
      }
    }; break;
    case DTYPE_UINT64: {
      for (size_t i = 0; i < count; ++i) {
        uint64_t x = std::floor(std::abs(data[i]));
        data[i] = x;
      }
    }; break;
    case DTYPE_BOOL: {
      if (!hasRandomBound(random_param)) {
        LOG(ERROR) << "Generate bool data should use uniform distribution.";
        throw std::invalid_argument(std::string(__FILE__) + " +" +
                                    std::to_string(__LINE__));
      }
      if (is_lower_equal_upper) {
        for (size_t i = 0; i < count; ++i) {
          data[i] = (data[i] > 0) ? 1.0f : 0.0f;
        }
        break;
      }
      T mid = 0;
      if (random_param->has_upper_bound_double()) {
        mid = (T)(random_param->upper_bound_double() +
                  random_param->lower_bound_double()) /
              2;
      } else {
        mid =
            (T)(random_param->upper_bound() + random_param->lower_bound()) / 2;
      }
      for (size_t i = 0; i < count; ++i) {
        data[i] = (data[i] < mid) ? 0.0f : 1.0f;
      }
    }; break;
    default:
      LOG(ERROR) << "Generate random data failed. ";
      throw std::invalid_argument(std::string(__FILE__) + " +" +
                                  std::to_string(__LINE__));
  }
}

template <typename T>
inline void hash_combine(size_t &seed, T value) {
  seed ^= std::hash<T>()(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
}  // namespace mluoptest

#endif  // TEST_MLU_OP_GTEST_PB_GTEST_INCLUDE_TOOLS_H_
