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

#include "pb_test_tools.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace mluoptest {

cnrtRet_t wrapRtConvertFloatToHalf(uint16_t *f16, float d) {
  return cnrtCastDataType_V2(&d, cnrtFloat, f16, cnrtHalf, 1, NULL,
                             cnrtRounding_rm);
}

cnrtRet_t wrapRtConvertHalfToFloat(float *d, uint16_t f16) {
  return cnrtCastDataType_V2(&f16, cnrtHalf, d, cnrtFloat, 1, NULL,
                             cnrtRounding_rm);
}

size_t shapeStrideCount(const Shape *shape) {
  if (shape->dims_size() == 0) {
    return 0;
  }
  size_t total = 1;
  if (shape->dim_stride_size() == 0) {
    for (int i = 0; i < shape->dims_size(); ++i) {
      total *= (size_t)shape->dims(i);
    }
  } else {
    if (shape->dims_size() != shape->dim_stride_size()) {
      LOG(ERROR) << "[GTEST] prototxt reading error! The dimensions size of"
                 << " tensor (which is " << shape->dims_size()
                 << ") is not equal to the dimensions size of it's"
                 << " strides (which is " << shape->dim_stride_size() << ").";
      GTEST_CHECK(shape->dim_stride_size() == shape->dims_size());
      return total;
    }
    for (int i = 0; i < shape->dims_size(); ++i) {
      if (shape->dims(i) == 0) {
        total = 0;
        break;
      }
      total += (size_t)(shape->dims(i) - 1) * shape->dim_stride(i);
    }
  }
  return total;
}

size_t shapeElementCount(const Shape *shape) {
  if (shape->dims_size() == 0) {
    return 0;
  }
  size_t total = 1;
  for (int i = 0; i < shape->dims_size(); ++i) {
    total *= shape->dims(i);
  }
  return total;
}

void saveDataToFile(const std::string &file, void *data, mluOpDataType_t dtype,
                    size_t count) {
  std::ostringstream oss;
  oss << std::this_thread::get_id();
  VLOG(4) << "Save data to file: " << file;
  std::ofstream fout(file + "_" + oss.str(), std::ios::out);
  switch (dtype) {
    case MLUOP_DTYPE_DOUBLE: {
      for (auto i = 0; i < count; ++i) {
        fout << ((double *)data)[i] << std::endl;
      }
    } break;
    case MLUOP_DTYPE_COMPLEX_FLOAT: {
      for (auto i = 0; i < 2 * count; ++i) {
        fout << ((float *)data)[i] << std::endl;
      }
    } break;
    default: {
      for (auto i = 0; i < count; ++i) {
        fout << ((float *)data)[i] << std::endl;
      }
    }
  }
  fout.close();
}

// this is kept for compatibility in case called from user's gtest code
void saveDataToFile(const std::string &file, float *data, size_t count) {
  std::ostringstream oss;
  oss << std::this_thread::get_id();
  VLOG(4) << "Save data to file: " << file;
  std::ofstream fout(file + "_" + oss.str(), std::ios::out);
  for (int i = 0; i < count; ++i) {
    fout << data[i] << std::endl;
  }
  fout.close();
}

void readDataFromFile(const std::string &file, float *data, size_t count) {
  VLOG(4) << "Read data from file: " << file;
  std::ifstream fin(file, std::ios::in);
  for (int i = 0; i < count; ++i) {
    std::string line;
    getline(fin, line);
    if (line.empty()) {
      LOG(ERROR) << "Data in " << file << " not enough, at least " << count;
      throw std::invalid_argument(std::string(__FILE__) + " +" +
                                  std::to_string(__LINE__));
    }
    data[i] = stof(line);
  }
  fin.close();
}

void saveHexDataToFile(const std::string &file, void *data,
                       mluOpDataType_t dtype, size_t count) {
  VLOG(4) << "Save data to file: " << file;
  std::ofstream fout(file, std::ios::out);
  switch (dtype) {
    case MLUOP_DTYPE_COMPLEX_HALF: {
      for (int i = 0; i < 2 * count; i += 2) {
        fout << "real_hex: " << std::setw(10) << std::hex
             << ((int16_t *)data)[i] << std::setw(20)
             << "real_dec: " << std::setw(10) << std::dec
             << cvtHalfToFloat(((int16_t *)data)[i]) << std::endl;
        fout << "imag_hex: " << std::setw(10) << std::hex
             << ((int16_t *)data)[i + 1] << std::setw(20)
             << "imag_dec: " << std::setw(10) << std::dec
             << cvtHalfToFloat(((int16_t *)data)[i + 1]) << std::endl;
      }
    } break;
    case MLUOP_DTYPE_COMPLEX_FLOAT: {
      for (int i = 0; i < 2 * count; i += 2) {
        fout << "real_hex: " << std::setw(10) << std::hex
             << ((int32_t *)data)[i] << std::setw(20)
             << "real_dec: " << std::setw(10) << std::dec << ((float *)data)[i]
             << std::endl;
        fout << "imag_hex: " << std::setw(10) << std::hex
             << ((int32_t *)data)[i + 1] << std::setw(20)
             << "imag_dec: " << std::setw(10) << std::dec
             << ((float *)data)[i + 1] << std::endl;
      }
    } break;
    case MLUOP_DTYPE_HALF: {
      for (int i = 0; i < count; ++i) {
        fout << "hex: " << std::setw(10) << std::hex << ((int16_t *)data)[i]
             << std::setw(20) << "dec: " << std::setw(10) << std::dec
             << cvtHalfToFloat(((int16_t *)data)[i]) << std::endl;
      }
    } break;
    case MLUOP_DTYPE_FLOAT: {
      for (int i = 0; i < count; ++i) {
        fout << "hex: " << std::setw(10) << std::hex << ((int32_t *)data)[i]
             << std::setw(20) << "dec: " << std::setw(10) << std::dec
             << ((float *)data)[i] << std::endl;
      }
    } break;
    case MLUOP_DTYPE_DOUBLE: {
      for (int i = 0; i < count; ++i) {
        fout << "hex: " << std::setw(10) << std::hex << ((int64_t *)data)[i]
             << std::setw(20) << "dec: " << std::setw(10) << std::dec
             << ((double *)data)[i] << std::endl;
      }
    } break;
    case MLUOP_DTYPE_INT8: {
      for (int i = 0; i < count; ++i) {
        fout << "hex: " << std::setw(10) << std::hex
             << (int32_t)((int8_t *)data)[i] << std::setw(20)
             << "dec: " << std::setw(10) << std::dec
             << (int32_t)((int8_t *)data)[i]  // don't show char
             << std::endl;
      }
    } break;
    case MLUOP_DTYPE_UINT8: {
      for (int i = 0; i < count; ++i) {
        fout << "hex: " << std::setw(10) << std::hex
             << (uint32_t)((uint8_t *)data)[i] << std::setw(20)
             << "dec: " << std::setw(10) << std::dec
             << (uint32_t)((uint8_t *)data)[i]  // not char
             << std::endl;
      }
    } break;
    case MLUOP_DTYPE_INT16: {
      for (int i = 0; i < count; ++i) {
        fout << "hex: " << std::setw(10) << std::hex << ((int16_t *)data)[i]
             << std::setw(20) << "dec: " << std::setw(10) << std::dec
             << ((int16_t *)data)[i] << std::endl;
      }
    } break;
    case MLUOP_DTYPE_UINT16: {
      for (int i = 0; i < count; ++i) {
        fout << "hex: " << std::setw(10) << std::hex << ((uint16_t *)data)[i]
             << std::setw(20) << "dec: " << std::setw(10) << std::dec
             << ((uint16_t *)data)[i] << std::endl;
      }
    } break;
    case MLUOP_DTYPE_INT32: {
      for (int i = 0; i < count; ++i) {
        fout << "hex: " << std::setw(10) << std::hex << ((int32_t *)data)[i]
             << std::setw(20) << "dec: " << std::setw(10) << std::dec
             << ((int32_t *)data)[i] << std::endl;
      }
    } break;
    case MLUOP_DTYPE_UINT32: {
      for (int i = 0; i < count; ++i) {
        fout << "hex: " << std::setw(10) << std::hex << ((uint32_t *)data)[i]
             << std::setw(20) << "dec: " << std::setw(10) << std::dec
             << ((uint32_t *)data)[i] << std::endl;
      }
    } break;
    case MLUOP_DTYPE_INT64: {
      for (int i = 0; i < count; ++i) {
        fout << "hex: " << std::setw(10) << std::hex << ((int64_t *)data)[i]
             << std::setw(20) << "dec: " << std::setw(10) << std::dec
             << ((int64_t *)data)[i] << std::endl;
      }
    } break;
    case MLUOP_DTYPE_UINT64: {
      for (int i = 0; i < count; ++i) {
        fout << "hex: " << std::setw(10) << std::hex << ((uint64_t *)data)[i]
             << std::setw(20) << "dec: " << std::setw(10) << std::dec
             << ((uint64_t *)data)[i] << std::endl;
      }
    } break;
    case MLUOP_DTYPE_BOOL: {
      for (int i = 0; i < count; ++i) {
        fout << "hex: " << std::setw(10) << std::hex
             << (int32_t)((bool *)data)[i] << std::setw(20)
             << "dec: " << std::setw(10) << std::dec
             << (int32_t)((bool *)data)[i] << std::endl;
      }
    } break;

    default: {
      VLOG(4) << "Unsupported dtype " << mluop::getNameOfDataType(dtype);
    } break;
  }
  fout.close();
}

cnrtDataType_t cvtMluOpDtypeToCnrt(mluOpDataType_t dtype) {
  switch (dtype) {
    case MLUOP_DTYPE_HALF:
      return CNRT_FLOAT16;
    case MLUOP_DTYPE_FLOAT:
      return CNRT_FLOAT32;
    case MLUOP_DTYPE_DOUBLE:
      return CNRT_FLOAT64;
    case MLUOP_DTYPE_INT8:
      return CNRT_INT8;
    case MLUOP_DTYPE_INT16:
      return CNRT_INT16;
    case MLUOP_DTYPE_INT32:
      return CNRT_INT32;
    case MLUOP_DTYPE_INT64:
      return CNRT_INT64;
    case MLUOP_DTYPE_BOOL:
      return CNRT_BOOL;
    case MLUOP_DTYPE_UINT8:
      return CNRT_UINT8;
    case MLUOP_DTYPE_UINT16:
      return CNRT_UINT16;
    case MLUOP_DTYPE_UINT32:
      return CNRT_UINT32;
    default:
      LOG(ERROR) << "NOT support this dtype yet";
      throw std::invalid_argument(std::string(__FILE__) + " +" +
                                  std::to_string(__LINE__));
  }
}

mluOpDataType_t cvtProtoDtypeToMluOp(DataType dtype) {
  switch (dtype) {
    case DTYPE_HALF:
      return MLUOP_DTYPE_HALF;
    case DTYPE_FLOAT:
      return MLUOP_DTYPE_FLOAT;
    case DTYPE_DOUBLE:
      return MLUOP_DTYPE_DOUBLE;
    case DTYPE_COMPLEX_HALF:
      return MLUOP_DTYPE_COMPLEX_HALF;
    case DTYPE_COMPLEX_FLOAT:
      return MLUOP_DTYPE_COMPLEX_FLOAT;
    case DTYPE_INT8:
      return MLUOP_DTYPE_INT8;
    case DTYPE_INT16:
      return MLUOP_DTYPE_INT16;
    case DTYPE_INT32:
      return MLUOP_DTYPE_INT32;
    case DTYPE_INT64:
      return MLUOP_DTYPE_INT64;
    case DTYPE_BOOL:
      return MLUOP_DTYPE_BOOL;
    case DTYPE_UINT8:
      return MLUOP_DTYPE_UINT8;
    case DTYPE_UINT16:
      return MLUOP_DTYPE_UINT16;
    case DTYPE_UINT32:
      return MLUOP_DTYPE_UINT32;
    case DTYPE_UINT64:
      return MLUOP_DTYPE_UINT64;
    default:
      LOG(ERROR) << "NOT support this dtype yet";
      throw std::invalid_argument(std::string(__FILE__) + " +" +
                                  std::to_string(__LINE__));
  }
}

mluOpTensorLayout_t cvtProtoLayoutToMluOp(TensorLayout order) {
  switch (order) {
    case LAYOUT_ARRAY:
      return MLUOP_LAYOUT_ARRAY;
    case LAYOUT_NCHW:
      return MLUOP_LAYOUT_NCHW;
    case LAYOUT_NHWC:
      return MLUOP_LAYOUT_NHWC;
    case LAYOUT_HWCN:
      return MLUOP_LAYOUT_HWCN;
    case LAYOUT_NDHWC:
      return MLUOP_LAYOUT_NDHWC;
    case LAYOUT_NCDHW:
      return MLUOP_LAYOUT_NCDHW;
    case LAYOUT_TNC:
      return MLUOP_LAYOUT_TNC;
    case LAYOUT_NTC:
      return MLUOP_LAYOUT_NTC;
    case LAYOUT_NLC:
      return MLUOP_LAYOUT_NLC;
    case LAYOUT_NC:
      return MLUOP_LAYOUT_NC;
    default:
      LOG(ERROR) << "Don't support this layout.";
      throw std::invalid_argument(std::string(__FILE__) + " +" +
                                  std::to_string(__LINE__));
  }
}

// ref: sopa/core/src/util/type_converter.cpp
int16_t cvtFloatToHalf(float x) {
  const int fs_shift = 31;
  const int fe_shift = 23;
  const int fe_mark = 0xff;
  const int hs_shift = 15;
  const int he_shift = 10;
  int *in1 = (int *)&x;
  int in = *in1;
  int sign = in >> fs_shift;
  int exp = ((in >> fe_shift) & fe_mark) - 127;
  int denorm = 0;
  int eff;
  int g = 0;  // for round
  int gr_last = 0;
  int gr_first = 0;
  int g_last = 0;
  if ((exp == 128) && (in & 0x7fffff)) {  // NaN
    exp = 0x1f - 15;
    eff = 0x200;
  } else if (exp >= 16) {
    exp = 0xf;
    eff = 0x3ff;
  } else if (exp >= -14) {
    gr_last = in & (0xfff);
    gr_first = (in >> 12) & 1;
    g_last = (in >> 13) & 1;
    g = ((gr_first && gr_last) || (gr_first && g_last));
    eff = (in >> 13) & 0x3ff;
  } else if (exp >= -24) {
    g = (((in & 0x7fffff) | 0x800000) >> (-exp - 2)) & 1;
    eff = (((in & 0x7fffff) | 0x800000) >> (-exp - 1)) & 0x3ff;
    denorm = 1;
    exp = 0;
  } else {
    exp = 0;
    denorm = 1;
    eff = (in & 0x7fffffff) ? 1 : 0;
  }
  eff += g;  // round
  exp = (denorm == 1) ? exp : (exp + 15);
  int result = (sign << hs_shift) + (exp << he_shift) + eff;
  return result;
}

// ref: sopa/core/src/util/type_converter.cpp
float cvtHalfToFloat(int16_t src) {
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

bool getEnv(const std::string &env, bool default_ret) {
  char *env_temp = getenv(env.c_str());
  if (env_temp != NULL) {
    if (strcmp(env_temp, "ON") == 0 || strcmp(env_temp, "1") == 0) {
      return true;
    } else if (strcmp(env_temp, "OFF") == 0 || strcmp(env_temp, "0") == 0) {
      return false;
    } else {
      return default_ret;
    }
  } else {
    return default_ret;
  }
}

int getEnvInt(const std::string &env, int default_ret) {
  char *env_temp = std::getenv(env.c_str());
  if (env_temp) {
    return std::atoi(env_temp);
  }
  return default_ret;
}

bool hasRandomBound(const RandomData *random_param) {
  bool has_float_bound = false;
  bool has_double_bound = false;
  if (random_param->has_lower_bound() && random_param->has_upper_bound()) {
    has_float_bound = true;
  }
  if (random_param->has_lower_bound_double() &&
      random_param->has_upper_bound_double()) {
    has_double_bound = true;
  }
  return has_float_bound || has_double_bound;
}

size_t proc_usage_peak() {
  auto pid = getpid();
  std::string name = "/proc/" + std::to_string(pid) + "/status";
  std::ifstream fin(name, std::ios::in);
  if (!fin.is_open()) {
    LOG(WARNING) << "MLUOPGTEST: failed open " << name << "\n";
    return 0;
  }
  std::string line;
  while (!fin.eof()) {
    getline(fin, line);
    if (line.find("VmPeak:") != std::string::npos) {
      try {
        // remove space
        auto it = std::remove(line.begin(), line.end(), ' ');
        line.erase(it, line.end());

        auto end = line.rfind("kB");
        auto start = line.find(":") + 1;
        auto kb_str = line.substr(start, end - start);
        // cvt to digit
        return std::stoul(kb_str) * 1024;
      } catch (std::exception &e) {
        LOG(ERROR) << "MLUOPGTEST: grep number in " << line << " failed, "
                   << e.what();
        return 0;
      }
    }
  }
  return 0;
}

void arrayCastFloatToHalf(int16_t *dst, float *src, size_t num) {
  for (int i = 0; i < num; ++i) {
    dst[i] = cvtFloatToHalf(src[i]);
  }
}

void arrayCastHalfToFloat(float *dst, int16_t *src, size_t num) {
  for (int i = 0; i < num; ++i) {
    dst[i] = cvtHalfToFloat(src[i]);
  }
}

// support uint8, uint16, uint32, uint64, int32, int64
template <typename T1, typename T2>
void arrayCastFloatAndNormal(void *dst, void *src, size_t num) {
  for (int i = 0; i < num; ++i) {
    ((T2 *)dst)[i] = (T2)((T1 *)src)[i];
  }
}

// support uint8, uint16, uint32, uint64, int8, int16, int32, int64, bool
void arrayCastFloatAndNormal(void *src_data, mluOpDataType_t src_dtype,
                             void *dst_data, mluOpDataType_t dst_dtype,
                             size_t num) {
  if (src_dtype == MLUOP_DTYPE_FLOAT && dst_dtype == MLUOP_DTYPE_UINT8) {
    arrayCastFloatAndNormal<float, uint8_t>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_FLOAT &&
             dst_dtype == MLUOP_DTYPE_UINT16) {
    arrayCastFloatAndNormal<float, uint16_t>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_FLOAT &&
             dst_dtype == MLUOP_DTYPE_UINT32) {
    arrayCastFloatAndNormal<float, uint32_t>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_FLOAT &&
             dst_dtype == MLUOP_DTYPE_UINT64) {
    arrayCastFloatAndNormal<float, uint64_t>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_FLOAT && dst_dtype == MLUOP_DTYPE_INT8) {
    arrayCastFloatAndNormal<float, int8_t>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_FLOAT && dst_dtype == MLUOP_DTYPE_INT16) {
    arrayCastFloatAndNormal<float, int16_t>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_FLOAT && dst_dtype == MLUOP_DTYPE_INT32) {
    arrayCastFloatAndNormal<float, int32_t>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_FLOAT && dst_dtype == MLUOP_DTYPE_INT64) {
    arrayCastFloatAndNormal<float, int64_t>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_FLOAT && dst_dtype == MLUOP_DTYPE_BOOL) {
    arrayCastFloatAndNormal<float, bool>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_FLOAT && dst_dtype == MLUOP_DTYPE_HALF) {
    arrayCastFloatToHalf((int16_t *)dst_data, (float *)src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_UINT8 && dst_dtype == MLUOP_DTYPE_FLOAT) {
    arrayCastFloatAndNormal<uint8_t, float>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_UINT16 &&
             dst_dtype == MLUOP_DTYPE_FLOAT) {
    arrayCastFloatAndNormal<uint16_t, float>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_UINT32 &&
             dst_dtype == MLUOP_DTYPE_FLOAT) {
    arrayCastFloatAndNormal<uint32_t, float>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_UINT64 &&
             dst_dtype == MLUOP_DTYPE_FLOAT) {
    arrayCastFloatAndNormal<uint64_t, float>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_INT8 && dst_dtype == MLUOP_DTYPE_FLOAT) {
    arrayCastFloatAndNormal<int8_t, float>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_INT16 && dst_dtype == MLUOP_DTYPE_FLOAT) {
    arrayCastFloatAndNormal<int16_t, float>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_INT32 && dst_dtype == MLUOP_DTYPE_FLOAT) {
    arrayCastFloatAndNormal<int32_t, float>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_INT64 && dst_dtype == MLUOP_DTYPE_FLOAT) {
    arrayCastFloatAndNormal<int64_t, float>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_BOOL && dst_dtype == MLUOP_DTYPE_FLOAT) {
    arrayCastFloatAndNormal<bool, float>(dst_data, src_data, num);
  } else if (src_dtype == MLUOP_DTYPE_HALF && dst_dtype == MLUOP_DTYPE_FLOAT) {
    arrayCastHalfToFloat((float *)dst_data, (int16_t *)src_data, num);
  } else if ((src_dtype == MLUOP_DTYPE_COMPLEX_HALF &&
              dst_dtype == MLUOP_DTYPE_FLOAT) ||
             (src_dtype == MLUOP_DTYPE_COMPLEX_HALF &&
              dst_dtype == MLUOP_DTYPE_COMPLEX_FLOAT)) {
    arrayCastHalfToFloat((float *)dst_data, (int16_t *)src_data, 2 * num);
  } else if ((src_dtype == MLUOP_DTYPE_FLOAT &&
              dst_dtype == MLUOP_DTYPE_COMPLEX_HALF) ||
             (src_dtype == MLUOP_DTYPE_COMPLEX_FLOAT &&
              dst_dtype == MLUOP_DTYPE_COMPLEX_HALF)) {
    arrayCastFloatToHalf((int16_t *)dst_data, (float *)src_data, 2 * num);
  } else {
    LOG(ERROR) << "MLUOPGTEST: arrayCastFloatAndNormal not supported! the "
                  "src_dtype is "
               << src_dtype << ", the dst_dype is " << dst_dtype << ".";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
  }
}

// read info from file, return a map, key is name, value is ops
// eg: key: black_list_zero_input, value: quantize, pad, xx, ...
std::unordered_map<std::string, std::vector<std::string>> readFileByLine(
    const std::string &file) {
  std::unordered_map<std::string, std::vector<std::string>> map_info;
  std::string line;
  std::ifstream fin(file, std::ios::in);
  if (!fin) {
    LOG(ERROR) << "MLUOPGTEST: " << __func__ << " failed on file " << file;
    return map_info;
  } else {
    std::string key_str = "";
    while (getline(fin, line)) {
      auto key_pos_begin = line.find("[");
      if (key_pos_begin != std::string::npos) {
        auto key_pos_end = line.find("]");
        auto key =
            line.substr(key_pos_begin + 1, key_pos_end - key_pos_begin - 1);
        key_str = key;
      } else {
        map_info[key_str].emplace_back(line);
      }
    }
  }
  return map_info;
}

uint64_t GenNumberOfFixedWidth(uint64_t a, int width) {
  uint64_t mask = 0;
  uint64_t index = 1;
  for (int i = 0; i < width; i++) {
    mask |= uint64_t(index << i);
  }
  return (a & mask);
}

void arrayCastHalfToInt8or16HalfUp(void *dst, int16_t *src, int pos, size_t num,
                                   int int8or16) {
  for (int i = 0; i < num; ++i) {
    int8_t res = 0;
    int16_t src_int16 = src[i];

    float offset_f = powf(2, pos - 1);
    int16_t offset_half = cvtFloatToHalf(offset_f);

    src_int16 =
        float_add(src_int16, offset_half, 0, ROUND_MODE_NEAREST_EVEN, 0, 1);

    int exp = GenNumberOfFixedWidth(src_int16 >> 10, 5);
    int eff = (src_int16 & 0x3ff);

    if (pos > 0) {
      if (exp > pos) {
        exp -= pos;
        src_int16 = (src_int16 & 0x83ff) | (exp << 10);
      } else {
        if (exp == 0) {
          exp = 0;
          eff = eff >> pos;
          src_int16 = (src_int16 & 0x8000) | eff;
        } else {
          exp = 0;
          eff = eff | 0x400;
          eff = eff >> (pos - exp + 1);
          src_int16 = (src_int16 & 0x8000) | eff;
        }
      }
    } else if (pos < 0) {
      int pos_inv = pos * -1;
      if (exp == 0) {
        eff = eff << pos_inv;
        for (int i = pos_inv; i > 0; i--) {
          if (eff & (1 << (9 + i))) {
            exp = i;
            break;
          }
        }
        eff = eff & 0x3ff;
        src_int16 = (src_int16 & 0x8000) | eff | (exp << 10);
      } else {
        exp += pos_inv;
        if (exp >= 0x1f) {
          src_int16 = (src_int16 & 0x8000) | 0x7aff;
        } else {
          src_int16 = (src_int16 & 0x83ff) | (exp << 10);
        }
      }
    }

    float src_f = cvtHalfToFloat(src_int16);
    int dst_int32 = floor(src_f);
    if (int8or16) {
      if (dst_int32 > 127) {
        dst_int32 = 127;
      }
      if (dst_int32 < -128) {
        dst_int32 = -128;
      }
      int8_t *dstInt8 = (int8_t *)dst;
      dstInt8[i] = dst_int32;
    } else {
      if (dst_int32 > 32767) {
        dst_int32 = 32767;
      }
      if (dst_int32 < -32768) {
        dst_int32 = -32768;
      }
      int16_t *dstInt16 = (int16_t *)dst;
      dstInt16[i] = dst_int32;
    }
  }
}

int float_number_is_nan_inf(int data_width, int float_number) {
  int sign, exp, eff;
  if (data_width == 16) {
    sign = ((float_number >> 15) & 0x1);
    exp = ((float_number >> 10) & 0x1f);
    eff = (float_number & 0x3ff);
    if (exp == 0x1f) {
      if (eff) {
        return sign ? -1 : 1;  // +-NAN
      } else {
        return sign ? -2 : 2;  // +-INF
      }
    } else {
      return 0;
    }
  } else if (data_width == 32) {
    sign = ((float_number >> 31) & 0x1);
    exp = ((float_number >> 23) & 0xff);
    eff = (float_number & 0x7fffff);
    if (exp == 0xff) {
      if (eff) {
        return sign ? -1 : 1;  // +-NAN
      } else {
        return sign ? -2 : 2;  // +-INF
      }
      return 0;

    } else {
      return 0;
    }
  } else {
    LOG(ERROR) << "Don't support this data_width.";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
    return 0;
  }
}

int float_add_regular(int in_a, int in_b, int float_16or32, int round_mode,
                      int add_or_sub, int &up, int &down) {
  up = 0;
  down = 0;
  // parse number:
  int number_bw = float_16or32 ? 32 : 16;
  int exp_bw = float_16or32 ? 8 : 5;
  int eff_bw = float_16or32 ? 23 : 10;
  int sign_a = (in_a >> (number_bw - 1)) & 0x1;
  int exp_a = GenNumberOfFixedWidth(in_a >> eff_bw, exp_bw);
  int eff_a = GenNumberOfFixedWidth(in_a, eff_bw);
  int sign_b = (in_b >> (number_bw - 1)) & 0x1;
  int exp_b = GenNumberOfFixedWidth(in_b >> eff_bw, exp_bw);
  int eff_b = GenNumberOfFixedWidth(in_b, eff_bw);
  // unusual number treatment:
  eff_a =
      ((exp_a == 0 || exp_a == (pow(2, exp_bw) - 1)) ? eff_a
                                                     : (eff_a | (1 << eff_bw)))
      << 3;
  eff_b =
      ((exp_b == 0 || exp_b == (pow(2, exp_bw) - 1)) ? eff_b
                                                     : (eff_b | (1 << eff_bw)))
      << 3;
  exp_a = (exp_a == 0 && eff_a != 0) ? (exp_a + 1) : exp_a;
  exp_b = (exp_b == 0 && eff_b != 0) ? (exp_b + 1) : exp_b;
  // put larger one in a:
  int change_pos = 0;
  if ((exp_b > exp_a) || (exp_b == exp_a && eff_b > eff_a)) {
    int temp_sign = sign_b;
    int temp_exp = exp_b;
    int temp_eff = eff_b;
    sign_b = sign_a;
    exp_b = exp_a;
    eff_b = eff_a;
    sign_a = temp_sign;
    exp_a = temp_exp;
    eff_a = temp_eff;
    change_pos = 1;
  }
  // eff shift:
  int sticky_bit = (exp_a - exp_b >= 32)
                       ? (eff_b != 0)
                       : (GenNumberOfFixedWidth(eff_b, exp_a - exp_b) != 0);
  eff_b = ((exp_a - exp_b >= 32) ? 0 : (eff_b >> (exp_a - exp_b))) | sticky_bit;
  // eff add or sub:
  int eff_res =
      ((sign_a == sign_b) && add_or_sub) || ((sign_a != sign_b) && !add_or_sub)
          ? (eff_a - eff_b)
          : (eff_a + eff_b);
  int exp_res = exp_a;
  int sign_res = (change_pos && add_or_sub) ? (!sign_a) : sign_a;
  // eff normalize:
  int drop_bit = 0;
  int drop_highest_bit = 0;
  int drop_else_bit = 0;
  if (eff_res >= pow(2, eff_bw + 4)) {
    if ((eff_res & 0x1) != 0) {
      drop_bit = 1;
      drop_else_bit = 1;
    }
    eff_res = eff_res >> 1;
    exp_res += 1;
  } else {
    while ((eff_res < pow(2, eff_bw + 3)) && (exp_res > 1)) {
      eff_res = eff_res << 1;
      exp_res -= 1;
    }
  }
  // final res:
  if (exp_res >= pow(2, exp_bw) - 1) {
    eff_res = 0xfffffff;
    exp_res = pow(2, exp_bw) - 2;
    up = 1;
  }
  if (((eff_res < pow(2, eff_bw + 3)) || eff_res == 0) &&
      (exp_res == 1)) {  // DENORM
    exp_res = 0;
  }
  if ((eff_res & 0x7) != 0) {
    drop_bit = 1;
  }
  if ((eff_res & 0x4) != 0) {
    drop_highest_bit = 1;
  }
  if ((eff_res & 0x3) != 0) {
    drop_else_bit = 1;
  }
  eff_res = (GenNumberOfFixedWidth(eff_res, eff_bw + 3) >> 3);
  int res = ((sign_res << (number_bw - 1)) | (exp_res << eff_bw) | eff_res);
  // round:
  // if (round_mode == ROUND_MODE_TO_ZERO) {
  // }
  if (round_mode == ROUND_MODE_OFF_ZERO) {
    if (drop_bit) {
      res += 1;
    }
  }
  if (round_mode == ROUND_MODE_UP) {
    if (drop_bit && !sign_res) {
      res += 1;
    }
  }
  if (round_mode == ROUND_MODE_DOWN) {
    if (drop_bit && sign_res) {
      res += 1;
    }
  }
  if (round_mode == ROUND_MODE_NEAREST_OFF_ZERO) {
    if (drop_bit && drop_highest_bit) {
      res += 1;
    }
  }
  if (round_mode == ROUND_MODE_NEAREST_EVEN) {
    if (drop_bit && drop_highest_bit && !drop_else_bit) {
      if (res & 0x1) {
        res += 1;
      }
    } else if (drop_bit && drop_highest_bit) {
      res += 1;
    }
  }
  return res;
}

int float_add_up_down(int in_a, int in_b, int float_16or32, int round_mode,
                      int add_or_sub, int ieee754, int &up, int &down) {
  if ((float_16or32 == 0) && (add_or_sub == 0)) {  // float16 add
    int sign_a = (in_a >> 15) & 0x1;
    int sign_b = (in_b >> 15) & 0x1;
    int eff_a = in_a & 0x3ff;
    int eff_b = in_b & 0x3ff;

    if (ieee754) {  // ieee754 fp16 add
      // exception treatment:
      if ((float_number_is_nan_inf(16, in_a) == 1) ||
          (float_number_is_nan_inf(16, in_a) == -1) ||
          (float_number_is_nan_inf(16, in_b) == 1) ||
          (float_number_is_nan_inf(16, in_b) == -1)) {
        // one is NAN
        return 0x7c01;
      } else if (((float_number_is_nan_inf(16, in_a) == 2) &&
                  (float_number_is_nan_inf(16, in_b) == -2)) ||
                 ((float_number_is_nan_inf(16, in_a) == -2) &&
                  (float_number_is_nan_inf(16, in_b) ==
                   2))) {  // one is +INF, the other -INF
        return 0x7c01;
      } else if (((float_number_is_nan_inf(16, in_a) == 2) &&
                  (float_number_is_nan_inf(16, in_b) == 2)) ||
                 ((float_number_is_nan_inf(16, in_a) == -2) &&
                  (float_number_is_nan_inf(16, in_b) ==
                   -2))) {  // both +INF or both -INF
        return ((sign_a << 15) | 0x7c00);
      } else if ((float_number_is_nan_inf(16, in_a) == 2) ||
                 (float_number_is_nan_inf(16, in_a) ==
                  -2)) {  // a is INF, sign = sign_a
        return ((sign_a << 15) | 0x7c00);
      } else if ((float_number_is_nan_inf(16, in_b) == 2) ||
                 (float_number_is_nan_inf(16, in_b) ==
                  -2)) {  // b is INF, sign = sign_b
        return ((sign_b << 15) | 0x7c00);
      } else if ((((in_a & 0xffff) == 0x0) && ((in_b & 0xffff) == 0x0)) ||
                 (((in_a & 0xffff) == 0x8000) && ((in_b & 0xffff) == 0x8000))) {
        // both +0 or both -0
        return ((sign_a << 15) | 0x0);
      } else if ((((in_a & 0xffff) == 0x0) && ((in_b & 0xffff) == 0x8000)) ||
                 (((in_a & 0xffff) == 0x8000) && ((in_b & 0xffff) == 0x0))) {
        // one is +0, the other -0
        if (round_mode == ROUND_MODE_DOWN) {
          return 0x8000;
        } else {
          return 0x0;
        }
      } else if ((in_a & 0x7fff) == 0x0) {  // a is 0
        return (in_b & 0xffff);
      } else if ((in_b & 0x7fff) == 0x0) {  // b is 0
        return (in_a & 0xffff);
      } else if (((in_a & 0x7fff) == (in_b & 0x7fff)) && (sign_a != sign_b)) {
        if (round_mode == ROUND_MODE_DOWN) {
          return 0x8000;
        } else {
          return 0x0;
        }
        // regular treatment:
      } else {
        int temp = float_add_regular(in_a, in_b, 0, round_mode, 0, up, down);
        // according to DW, the result can be INF
        /*
           if (temp == 0xfc00){
           temp = 0xfbff;
           }
           if (temp == 0x7c00){
           temp = 0x7bff;
           }
        */
        return temp;
      }       // ieee754 fp16 add
    } else {  // not ieee754 fp16 add
      if ((float_number_is_nan_inf(16, in_a) != 0) &&
          (float_number_is_nan_inf(16, in_b) != 0)) {
        if (eff_a > eff_b) {
          return ((sign_a << 15) | 0x7bff);
        } else if (eff_a < eff_b) {
          return ((sign_b << 15) | 0x7bff);
        } else {
          return (sign_a == sign_b) ? ((sign_a << 15) | 0x7bff) : 0x7bff;
        }
      } else if (float_number_is_nan_inf(16, in_a) != 0) {
        return (sign_a << 15) | 0x7bff;
      } else if (float_number_is_nan_inf(16, in_b) != 0) {
        return (sign_b << 15) | 0x7bff;
      } else if (((in_a & 0x7fff) == 0) && ((in_b & 0x7fff) == 0)) {
        if ((sign_a == 1) && (sign_b == 1)) {
          return 0x8000;
        } else {
          return 0x0;
        }
      } else if ((in_a & 0x7fff) == 0) {
        return in_b & 0xffff;
      } else if ((in_b & 0x7fff) == 0) {
        return in_a & 0xffff;
      } else if (((in_a & 0x7fff) == (in_b & 0x7fff)) && (sign_a != sign_b)) {
        return 0x0;
      } else {
        int temp = float_add_regular(in_a, in_b, 0, round_mode, 0, up, down);
        if (temp == 0xfc00) {
          temp = 0xfbff;
        }
        if (temp == 0x7c00) {
          temp = 0x7bff;
        }
        return temp;
      }
    }  // not ieee754 fp16 add
    // fp16 add
  } else {
    LOG(ERROR) << "CPU float add only support half add now.";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
    return -1;
  }
}

int float_add(int in_a, int in_b, int float_16or32, int round_mode,
              int add_or_sub, int ieee754) {
  int up;
  int down;
  return float_add_up_down(in_a, in_b, float_16or32, round_mode, add_or_sub,
                           ieee754, up, down);
}

int float_mult_regular(int in_a, int in_b, int float_16or32, int round_mode,
                       int &up, int &down) {
  up = 0;
  down = 0;
  // parse number:
  int number_bw = float_16or32 ? 32 : 16;
  int exp_bw = float_16or32 ? 8 : 5;
  int eff_bw = float_16or32 ? 23 : 10;
  int sign_a = (in_a >> (number_bw - 1)) & 0x1;
  int exp_a = GenNumberOfFixedWidth(in_a >> eff_bw, exp_bw);
  int eff_a = GenNumberOfFixedWidth(in_a, eff_bw);
  int sign_b = (in_b >> (number_bw - 1)) & 0x1;
  int exp_b = GenNumberOfFixedWidth(in_b >> eff_bw, exp_bw);
  int eff_b = GenNumberOfFixedWidth(in_b, eff_bw);
  // unusual number treatment:
  eff_a = (exp_a == 0 || exp_a == (pow(2, exp_bw) - 1))
              ? eff_a
              : (eff_a | (1 << eff_bw));
  // INF and NAN won't happen here
  eff_b = (exp_b == 0 || exp_b == (pow(2, exp_bw) - 1))
              ? eff_b
              : (eff_b | (1 << eff_bw));
  exp_a = (exp_a == 0 && eff_a != 0) ? (exp_a + 1) : exp_a;
  exp_b = (exp_b == 0 && eff_b != 0) ? (exp_b + 1) : exp_b;
  int exp_offset = float_16or32 ? 0x7f : 0xf;
  // mult:
  int sign_res = sign_a ^ sign_b;
  uint64_t eff_res = uint64_t(eff_a) * uint64_t(eff_b);
  int exp_res = ((eff_res == 0) ? 0 : (exp_a + exp_b - exp_offset));
  // eff_res == 0 won't happen here(if eff_res == 0 then eff_a/b == 0 then
  // exp_a/b == 0, it is 0) eff normalize:
  int drop_bit = 0;
  int drop_highest_bit = 0;
  int drop_else_bit = 0;
  if ((eff_res >> (eff_bw * 2 + 1)) != 0) {
    if ((eff_res & 0x1) != 0) {
      // (1)drop judge of right shift by one(48bit -> 47bit):
      drop_bit = 1;
      drop_else_bit = 1;
    }
    eff_res = eff_res >> 1;  // eff_res has been 47-bit
    exp_res += 1;
  } else {
    // put msb at the 47-bit, need not drop judge:
    while ((eff_res < pow(2, eff_bw * 2)) && (exp_res > 1)) {
      eff_res = eff_res << 1;
      exp_res -= 1;
    }
  }
  // final res:
  if (exp_res >= pow(2, exp_bw) - 1) {  // saturate
    eff_res = 0xffffffffffffULL;
    exp_res = pow(2, exp_bw) - 2;
    up = 1;
  } else if (exp_res <= 0) {  // DENORM
    // (2)drop judge of DENORM shift(right shift until exp_res == 1):
    if (((1 - exp_res) >= 64) && (eff_res != 0)) {
      drop_bit = 1;
      drop_else_bit = 1;
    } else if (((1 - exp_res) >= 32) &&
               (((eff_res & 0xffffffff) != 0) ||
                (GenNumberOfFixedWidth((eff_res >> 32) & 0xffffffff,
                                       1 - exp_res - 32) != 0))) {
      drop_bit = 1;
      drop_else_bit = 1;
    } else if (((1 - exp_res) < 32) &&
               (GenNumberOfFixedWidth(eff_res & 0xffffffff, 1 - exp_res) !=
                0)) {
      drop_bit = 1;
      drop_else_bit = 1;
    }
    eff_res = ((1 - exp_res) >= 64) ? 0 : (eff_res >> (1 - exp_res));
    // "right shift count cannot >= width of type"
    exp_res = 1;  // DENORM
  }
  if ((exp_res == 1) && (eff_res < pow(2, eff_bw * 2) || eff_res == 0)) {
    // result from right shift of DENORM mode
    exp_res = 0;
  }
  // (3)drop judge of right shift eff_bw:
  if (GenNumberOfFixedWidth(eff_res & 0xffffffff, eff_bw) != 0) {
    drop_bit = 1;
    if (GenNumberOfFixedWidth(eff_res & 0xffffffff, eff_bw - 1) != 0) {
      drop_else_bit = 1;
    }
    if (((eff_res >> (eff_bw - 1)) & 0x1) != 0) {
      drop_highest_bit = 1;
    }
  }
  eff_res = GenNumberOfFixedWidth(eff_res >> eff_bw, eff_bw);
  int res = (sign_res << (number_bw - 1)) | (exp_res << eff_bw) | eff_res;
  if (((res & int(pow(2, number_bw - 1) - 1)) == 0) && (drop_bit)) {
    down = 1;
  }
  // round:
  // if (round_mode == ROUND_MODE_TO_ZERO) {
  // }
  if (round_mode == ROUND_MODE_OFF_ZERO) {
    if (drop_bit) {
      res += 1;
    }
  }
  if (round_mode == ROUND_MODE_UP) {
    if (drop_bit && !sign_res) {
      res += 1;
    }
  }
  if (round_mode == ROUND_MODE_DOWN) {
    if (drop_bit && sign_res) {
      res += 1;
    }
  }
  if (round_mode == ROUND_MODE_NEAREST_OFF_ZERO) {
    if (drop_bit && drop_highest_bit) {
      res += 1;
    }
  }
  if (round_mode == ROUND_MODE_NEAREST_EVEN) {
    if (drop_bit && drop_highest_bit && !drop_else_bit) {
      if (res & 0x1) {
        res += 1;
      }
    } else if (drop_bit && drop_highest_bit) {
      res += 1;
    }
  }
  return res;
}

int float_mult_up_down(int in_a, int in_b, int float_16or32, int round_mode,
                       int ieee754, int &up, int &down) {
  if (float_16or32 == 0) {  // fp16 mult
    int sign_a = (in_a >> 15) & 0x1;
    int sign_b = (in_b >> 15) & 0x1;

    if (ieee754) {  // ieee754 fp16 mult
      // exception treatment:
      if ((float_number_is_nan_inf(16, in_a) == 1) ||
          (float_number_is_nan_inf(16, in_a) == -1) ||
          (float_number_is_nan_inf(16, in_b) == 1) ||
          (float_number_is_nan_inf(16, in_b) == -1)) {
        // one is NAN
        return 0x7c01;
      } else if (((in_a & 0x7fff) == 0x0) &&
                 ((float_number_is_nan_inf(16, in_b) == 2) ||
                  (float_number_is_nan_inf(16, in_b) ==
                   -2))) {  // a is 0, b is INF
        return 0x7c01;
      } else if (((in_b & 0x7fff) == 0x0) &&
                 ((float_number_is_nan_inf(16, in_a) == 2) ||
                  (float_number_is_nan_inf(16, in_a) ==
                   -2))) {  // a is INF, b is 0
        return 0x7c01;
      } else if ((float_number_is_nan_inf(16, in_a) == 2) ||
                 (float_number_is_nan_inf(16, in_a) == -2) ||
                 (float_number_is_nan_inf(16, in_b) == 2) ||
                 (float_number_is_nan_inf(16, in_b) == -2)) {  // one is INF
        return (((sign_a ^ sign_b) << 15) | 0x7c00);
      } else if (((in_a & 0x7fff) == 0x0) ||
                 ((in_b & 0x7fff) == 0x0)) {  // one is 0
        return (((sign_a ^ sign_b) << 15) | 0x0);
      } else {  // regular treatment:
        int temp = float_mult_regular(in_a, in_b, 0, round_mode, up, down);
        /*
           if (temp == 0x7c00){
           temp = 0x7bff;
           }
           else if (temp == 0xfc00){
           temp = 0xfbff;
           }
        */
        return temp;
      }
      // ieee754 fp16 mult
    } else {  // not ieee754 fp16 mult
      if (((in_a & 0x7fff) == 0x0) || ((in_b & 0x7fff) == 0x0)) {
        return ((sign_a == sign_b) ? 0x0 : 0x8000);
      } else if ((float_number_is_nan_inf(16, in_a) != 0) ||
                 (float_number_is_nan_inf(16, in_b) != 0)) {
        return ((sign_a == sign_b) ? 0x7bff : 0xfbff);
      } else {
        int temp =
            float_mult_regular(in_a, in_b, float_16or32, round_mode, up, down);
        if (temp == 0x7c00) {
          temp = 0x7bff;
        } else if (temp == 0xfc00) {
          temp = 0xfbff;
        }
        return temp;
      }
    }  // not ieee754 fp16 mult
    // fp16 mult
  } else {
    LOG(ERROR) << "CPU float mult only support half now.";
    throw std::invalid_argument(std::string(__FILE__) + " +" +
                                std::to_string(__LINE__));
    return -1;
  }
}

int float_mult(int in_a, int in_b, int float_16or32, int round_mode,
               int ieee754) {
  int up;
  int down;
  return float_mult_up_down(in_a, in_b, float_16or32, round_mode, ieee754, up,
                            down);
}

}  // namespace mluoptest
