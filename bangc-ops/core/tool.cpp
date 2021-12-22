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
#include <core/tool.h>

#define INT31_BITWIDTH 31
#define INT16_BITWIDTH 16

mluOpStatus_t castFloat32ToInt31(float *src, size_t num, void *dst) {
  if (src == NULL) {
    LOG(ERROR) << "[castFloat32ToInt31]:The pointer of src is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (dst == NULL) {
    LOG(ERROR) << "[castFloat32ToInt31]:The pointer of dst is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (num == 0) {
    LOG(ERROR) << "[castFloat32ToInt31]:Intput num is wrong, it must be greater than 0.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  int position = 0;
  int var = std::pow(2, INT31_BITWIDTH - 1) - 1;
  float temp = 0.0f;
  float temp_high = 0.0f;
  float temp_low = 0.0f;

  // get absmax of the float data
  float absmax = std::fabs(src[0]);
  for (size_t i = 0; i < num; ++i) {
    if (std::fabs(src[i]) > absmax)
      absmax = std::fabs(src[i]);
  }

  // Formula: int31 , position = floor(log2(absmax) - 29))
  if (absmax == 0) {
    position = 0;
  } else {
    position = static_cast<int>(std::floor(std::log2(absmax)) - 29);
  }

  if (absmax == 0) {
    for (size_t i = 0; i < num; ++i) {
      // low int16 data
      ((int16_t *)dst)[i] = 0;
      // high int16 data
      ((int16_t *)dst)[i + num] = 0;
    }
  } else {
    // Formula: f = (high * 2^15 + low) * 2^position.
    var = std::pow(2, INT16_BITWIDTH - 1);
    for (size_t i = 0; i < num; ++i) {
      temp = src[i] / (std::pow(2, position));
      temp = (temp >= 0) ? (temp + 0.5f) : (temp - 0.5f);

      // high int16 data
      temp_high = temp / var;
      ((int16_t *)dst)[i + num] = static_cast<int16_t>(temp_high);
      // low int16 data
      temp_low = temp - ((int16_t *)dst)[i + num] * var;
      ((int16_t *)dst)[i] = static_cast<int16_t>(temp_low);
    }
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t getPosition(float *input, size_t num, mluOpDataType_t datatype, int *position) {
  if (input == NULL) {
    LOG(ERROR) << "[getPosition]:The pointer of input is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (position == NULL) {
    LOG(ERROR) << "[getPosition]:The pointer of position is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (num == 0) {
    LOG(ERROR) << "[getPosition]:Input num is wrong, it must be greater than 0.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  int bitwidth = 8;
  if (datatype == MLUOP_DTYPE_INT8) {
    bitwidth = 8;
  } else if (datatype == MLUOP_DTYPE_INT16) {
    bitwidth = 16;
  } else if (datatype == MLUOP_DTYPE_INT31) {
    bitwidth = 31;
  } else {
    LOG(ERROR) << "[getPosition]:Input data type is not supported.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // Formula: position = floor(log2(absmax) - (bitwidth - 2)))
  float absmax = std::fabs(input[0]);
  for (size_t index = 0; index < num; ++index) {
    if (std::fabs(input[index]) > absmax)
      absmax = std::fabs(input[index]);
  }

  if (absmax == 0) {
    *position = 0;
  } else {
    *position = static_cast<int>(std::floor(std::log2(absmax)) - (bitwidth - 2));
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t getPositionAndScale(float *input,
                                  size_t num,
                                  mluOpDataType_t datatype,
                                  int *position,
                                  float *scale) {
  if (input == NULL) {
    LOG(ERROR) << "[getPositionAndScale]:The pointer of input is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (num == 0) {
    LOG(ERROR) << "[getPositionAndScale]:Input num is 0, it must be greater than 0.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (position == NULL) {
    LOG(ERROR) << "[getPositionAndScale]:The pointer of position is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (scale == NULL) {
    LOG(ERROR) << "[getPositionAndScale]:The pointer of scale is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  int bitwidth = 8;
  if (datatype == MLUOP_DTYPE_INT8) {
    bitwidth = 8;
  } else if (datatype == MLUOP_DTYPE_INT16) {
    bitwidth = 16;
  } else if (datatype == MLUOP_DTYPE_INT31) {
    bitwidth = 31;
  } else {
    LOG(ERROR) << "[getPositionAndScale]:Input data type is not supported.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  int scale_var = std::pow(2, bitwidth - 1) - 1;
  float max_data = std::fabs(input[0]);
  for (size_t index = 0; index < num; ++index) {
    if (std::fabs(input[index]) > max_data)
      max_data = std::fabs(input[index]);
  }
  if (max_data == 0) {
    *position = 0;
    *scale = 1.0;
  } else if (bitwidth != 31) {
    *position = static_cast<int>(std::floor(std::log2(max_data)) - (bitwidth - 2));
    *scale = static_cast<float>(std::pow(2, *position) * scale_var / max_data);
  } else {
    *position = static_cast<int>(std::floor(std::log2(max_data)) - (bitwidth - 2));
    *scale = 1.0;
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t getPositionScaleAndOffset(float *input,
                                        size_t num,
                                        mluOpDataType_t datatype,
                                        int *position,
                                        float *scale,
                                        int *offset) {
  if (input == NULL) {
    LOG(ERROR) << "[getPositionScaleAndOffset]:The pointer of input is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (num == 0) {
    LOG(ERROR) << "[getPositionScaleAndOffset]:Input num is 0, it must be greater than 0.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (position == NULL) {
    LOG(ERROR) << "[getPositionScaleAndOffset]:The pointer of position is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (scale == NULL) {
    LOG(ERROR) << "[getPositionScaleAndOffset]:The pointer of scale is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (offset == NULL) {
    LOG(ERROR) << "[getPositionScaleAndOffset]:The pointer of offset is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  int bitwidth = 8;
  if (datatype == MLUOP_DTYPE_INT8) {
    bitwidth = 8;
  } else if (datatype == MLUOP_DTYPE_INT16) {
    bitwidth = 16;
  } else {
    LOG(ERROR) << "[getPositionScaleAndOffset]:input data type is not supported.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  float max_data = input[0];
  float min_data = input[0];
  for (size_t i = 0; i < num; ++i) {
    max_data = max_data > input[i] ? max_data : input[i];
    min_data = min_data < input[i] ? min_data : input[i];
  }

  max_data = max_data > 0 ? max_data : 0;
  min_data = min_data < 0 ? min_data : 0;

  if (max_data == min_data) {
    *position = 0;
    *scale = 1;
    *offset = 0;
  } else {
    *position = (int)(floorf(log2f(max_data - min_data)) - (bitwidth - 1));
    *scale = powf(2, *position) * (powf(2, bitwidth) - 1) / (max_data - min_data);
    *offset = (int)roundf(-powf(2, bitwidth - 1) -
                          min_data * (powf(2, bitwidth) - 1) / (max_data - min_data));
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t castInt31ToFloat32(void *src, float *dst, size_t num, int position) {
  if (src == NULL) {
    LOG(ERROR) << "[castInt31ToFloat32]:The pointer of src is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (dst == NULL) {
    LOG(ERROR) << "[castInt31ToFloat32]:The pointer of dst is NULL.";
    return MLUOP_STATUS_BAD_PARAM;
  }
  if (num == 0) {
    LOG(ERROR) << "[castInt31ToFloat32]:Intput num is wrong, it must be greater than 0.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // Formula: f = (high * 2^15 + low) * 2^position.
  int16_t *low = (int16_t *)src;
  int16_t *high = (int16_t *)(low + num);
  float tmp = 0.0f;
  for (size_t i = 0; i < num; i++) {
    tmp = high[i] * std::pow(2, INT16_BITWIDTH - 1);
    tmp = tmp + low[i];
    dst[i] = tmp * std::pow(2, position);
  }

  return MLUOP_STATUS_SUCCESS;
}

int16_t castFloat32ToHalf(float src) {
  /**
   * @desc:
   *  convert a number form `float32` to `int16_t(float16)`.
   * @param:
   *  a nubmer of type `float32`
   * @return:
   *  number of `int16_t`.
   * **/
  const int fs_shift = 31;
  const int fe_shift = 23;
  const int fe_mark = 0xff;
  const int hs_shift = 15;
  const int he_shift = 10;
  int *in1 = (int *)&src;
  int in = *in1;
  int sign = in >> fs_shift;
  int exp = ((in >> fe_shift) & fe_mark) - 127;
  int denorm = 0;
  int eff = 0;
  int g = 0;  // for round
  if (exp >= 16) {
    exp = 0xf;
    eff = 0x3ff;
  } else if (exp >= -14) {
    g = (in >> 12) & 1;
    eff = (in >> 13) & 0x3ff;
  } else if (exp >= -24) {
    g = (((in & 0x7fffff) | 0x800000) >> (-exp - 2)) & 1;
    eff = (((in & 0x7fffff) | 0x800000) >> (-exp - 1)) & 0x3ff;
    denorm = 1;
    exp = 0;
  } else {
    exp = 0;
    denorm = 1;
    eff = in ? 1 : 0;
  }
  eff += g;  // round
  exp = (denorm == 1) ? exp : (exp + 15);
  int result = (sign << hs_shift) + (exp << he_shift) + eff;
  return result;
}
#undef INT31_BITWIDTH
#undef INT16_BITWIDTH
