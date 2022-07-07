/*************************************************************************
 * Copyright (C) 2021 by Cambricon, Inc. All rights reserved.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef TEST_MLU_OP_GTEST_INCLUDE_TOOLS_H_
#define TEST_MLU_OP_GTEST_INCLUDE_TOOLS_H_

#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <random>
#include <thread> 
#include <vector>
#include <unordered_map>
#include "gtest/gtest.h"
#include "mlu_op_test.pb.h"
#include "mlu_op.h"
#include "core/tensor.h"
#include "evaluator.h"

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define GTEST_CHECK(condition, ...)                                            \
  if (unlikely(!(condition))) {                                                \
    ADD_FAILURE() << "Check failed: " #condition ". " #__VA_ARGS__;            \
    throw std::invalid_argument(std::string(__FILE__) + " +" +                 \
                                std::to_string(__LINE__));                     \
  }

#define GTEST_WARNING(condition, ...)                                          \
  if (unlikely(!(condition))) {                                                \
    LOG(WARNING) << "Check failed: " #condition ". " #__VA_ARGS__;             \
  }

namespace mluoptest {

  // for debug
  void saveDataToFile(const std::string &file, float *data, size_t count);
  void saveDataToFile(const std::string &file,
                      float *data,
                      mluOpDataType_t dtype,
                      size_t count);
  void readDataFromFile(const std::string &file, float *data, size_t count);
  void saveHexDataToFile(const std::string &file,
                         void *data,
                         mluOpDataType_t dtype,
                         size_t count);
  void generateRandomData(float *data,
                          size_t count,
                          const RandomData *random_data,
                          DataType dtype);

  // include stride
  // if no stride this count == shape count
  size_t shapeStrideCount(const Shape *shape);
  // not include stride
  size_t shapeElementCount(const Shape *shape);

  cnrtDataType_t cvtMluOpDtypeToCnrt(mluOpDataType_t dtype);
  mluOpDataType_t cvtProtoDtypeToMluOp(DataType dtype);
  mluOpTensorLayout_t cvtProtoLayoutToMluOp(TensorLayout order);
  int16_t cvtFloatToHalf(float x);
  int64_t cvtFloatToInt64(float x);
  float cvtHalfToFloat(int16_t);

  void arrayCastFloatToHalf(int16_t *dst, float *src, int num);
  void arrayCastFloatToInt64(int64_t *dst, float *src, int num);
  void arrayCastHalfToFloat(float *dst, int16_t *src, int num);
  void arrayCastFloatAndNormal(void *src_data,
                               mluOpDataType_t src_dtype,
                               void *dst_data,
                               mluOpDataType_t dst_dtype,
                               int num);
  void arrayCastHalfToInt8or16HalfUp(
      void *dst, int16_t *src, int pos, int num, int int8or16);
  uint64_t GenNumberOfFixedWidth(uint64_t a, int witdh);

  bool getEnv(const std::string &env, bool default_ret);
  size_t proc_usage_peak();
  std::unordered_map<std::string, std::vector<std::string>>
      readFileByLine(const std::string &file);
  // half mult
  int float_mult(
      int in_a, int in_b, int float_16or32, int round_mode, int ieee754);

  // half add
  int float_add(int in_a,
                int in_b,
                int float_16or32,
                int round_mode,
                int add_or_sub,
                int ieee754);

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
  void cvtHalfToFixed(const float *src,
                      FixedType *dst,
                      const size_t num,
                      const int position = 0,
                      const float scale  = 1.0,
                      const int offset   = 0) {
    const float max = pow(2, sizeof(FixedType) * 8 - 1) + (-1);
    const float min = pow(2, sizeof(FixedType) * 8 - 1) * (-1);
    for (size_t i = 0; i < num; ++i) {
      int16_t res = float_mult(cvtFloatToHalf(src[i]),
                               cvtFloatToHalf(scale),
                               0,
                               ROUND_MODE_NEAREST_EVEN,
                               1);
      // use 10 because half exponend width only 5 bit
      int pos_tmp         = position >= 0 ? 10 : -10;
      float tmp           = powf(2, -pos_tmp);
      int16_t tmp_half    = cvtFloatToHalf(tmp);
      int16_t offset_half = cvtFloatToHalf(offset);
      for (int cycle = 0; cycle < position / pos_tmp; ++cycle) {
        res = float_mult(res, tmp_half, 0, ROUND_MODE_NEAREST_EVEN, 1);
      }
      if (position % pos_tmp) {
        tmp              = pow(2, -(position % pos_tmp));
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

} // namespace mluoptest

#endif // TEST_MLU_OP_GTEST_INCLUDE_TOOLS_H_
