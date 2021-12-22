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
#ifndef CORE_TOOL_H_
#define CORE_TOOL_H_

#include <stdint.h>
#include <cmath>
#include "core/logging.h"
/**
 * @brief cast float32 data to int31 data
 *
 * @param[in]
 *        src. a pointer to float32 data
 * @param[in]
 *        num. the number of float32 data
 * @param[out]
 *        dst. a pointer to int31 data
 * @return MLUOP_STATUS_SUCCESS if success,
 *         otherwise the error code is returned.
 */
mluOpStatus_t castFloat32ToInt31(float *src, size_t num, void *dst);

// The API is used for no scling factor quantization.
mluOpStatus_t getPosition(float *input, size_t num, mluOpDataType_t datatype, int *position);

// The API is used for scaling factor quantization.
mluOpStatus_t getPositionAndScale(float *input,
                                  size_t num,
                                  mluOpDataType_t datatype,
                                  int *position,
                                  float *scale);
// The API is used for asymmetrical quantization.
mluOpStatus_t getPositionScaleAndOffset(float *input,
                                        size_t num,
                                        mluOpDataType_t datatype,
                                        int *position,
                                        float *scale,
                                        int *offset);

/**
 * @brief cast int31 data to float32 data
 *
 * @param[in]
 *        src. a pointer to int31 data
 * @param[out]
 *        dst. a pointer to float data
 * @param[in]
 *        num. the number of float32 data
 * @param[in]
 *         position. the position of quantify param
 * @return MLUOP_STATUS_SUCCESS if success,
 *         otherwise the error code is returned.
 */

mluOpStatus_t castInt31ToFloat32(void *src, float *dst, size_t num, int position);

int16_t castFloat32ToHalf(float src);

/**
 * @brief Casts data from float32 to int8/int16. If you would like to
 *        cast from float32 to int31, please call castFloat32ToInt31
 *        rather than this function.
 *
 * @param[in] src
 *   Input. Pointer to float32 data.
 * @param[out] dst
 *   Output. Pointer to int8/int16 data.
 * @param[in] num
 *   Input. The length of float32 data.
 * @param[in] position
 *   Input. The position factor for quantization.
 * @param[in] scale
 *   Input. The scale factor for quantization.
 * @param[in] offset
 *   Input. The offset factor for quantization.
 * @return MLUOP_STATUS_SUCCESS if success,
 *         otherwise the error code is returned.
 */
template <typename FixedType>
mluOpStatus_t castFloat32ToFixed(const float *src,
                                 FixedType *dst,
                                 const size_t num,
                                 const int position = 0,
                                 const float scale  = 1.0,
                                 const int offset   = 0) {
  PARAM_CHECK("[castFloat32ToFixed]", src != NULL);
  PARAM_CHECK("[castFloat32ToFixed]", dst != NULL);
  PARAM_CHECK("[castFloat32ToFixed]", num > 0);

  const float max = pow(2, sizeof(FixedType) * 8 - 1) + (-1);
  const float min = pow(2, sizeof(FixedType) * 8 - 1) * (-1);
  for (size_t i = 0; i < num; ++i) {
    float res = static_cast<float>((src[i] * scale / pow(2, position) + offset));
    if (res > max) {
      res = max;
    } else if (res < min) {
      res = min;
    }
    dst[i] = static_cast<FixedType>(round(res));
  }
  return MLUOP_STATUS_SUCCESS;
}

/**
 * @brief Casts data from int8/int16 to float32. If you would like to
 *        cast from int31 to float32, please call castInt31ToFloat32
 *        rather than this function.
 *
 * @param[in] src
 *   Input. Pointer to int8/int16 data.
 * @param[out] dst
 *   Output. Pointer to int8/int16 data.
 * @param[in] num
 *   Input. The length of float32 data.
 * @param[in] position
 *   Input. The position factor for quantization.
 * @param[in] scale
 *   Input. The scale factor for quantization.
 * @param[in] offset
 *   Input. The offset factor for quantization.
 * @return MLUOP_STATUS_SUCCESS if success,
 *         otherwise the error code is returned.
 */
template <typename FixedType>
mluOpStatus_t castFixedToFloat32(const FixedType *src,
                                 float *dst,
                                 const size_t num,
                                 const int position = 0,
                                 const float scale  = 1.0,
                                 const int offset   = 0) {
  PARAM_CHECK("[castFixedToFloat32]", src != NULL);
  PARAM_CHECK("[castFixedToFloat32]", dst != NULL);
  PARAM_CHECK("[castFixedToFloat32]", num > 0);

  for (size_t i = 0; i < num; ++i) {
    dst[i] = (static_cast<float>(src[i]) - offset) * pow(2, position) / scale;
  }
  return MLUOP_STATUS_SUCCESS;
}

#endif  // CORE_TOOL_H_
