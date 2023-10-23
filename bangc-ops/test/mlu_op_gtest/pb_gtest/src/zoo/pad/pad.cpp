/*************************************************************************
 * Copyright (C) [2023] by Cambricon, Inc.
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
#include "pad.h"
#include <algorithm>
#include <string>

namespace mluoptest {

cnrtDataType_V2_t cvtMluOpDtypeToCnrt_V2(mluOpDataType_t dtype) {
  switch (dtype) {
    case MLUOP_DTYPE_HALF:
      return cnrtHalf;
    case MLUOP_DTYPE_FLOAT:
      return cnrtFloat;
    case MLUOP_DTYPE_DOUBLE:
      return cnrtDouble;
    case MLUOP_DTYPE_INT8:
      return cnrtChar;
    case MLUOP_DTYPE_INT16:
      return cnrtShort;
    case MLUOP_DTYPE_INT32:
      return cnrtInt;
    case MLUOP_DTYPE_INT64:
      return cnrtLonglong;
    case MLUOP_DTYPE_BOOL:
      return cnrtBoolean;
    case MLUOP_DTYPE_UINT8:
      return cnrtUchar;
    case MLUOP_DTYPE_UINT16:
      return cnrtUshort;
    case MLUOP_DTYPE_UINT32:
      return cnrtUint;
    case MLUOP_DTYPE_UINT64:
      return cnrtUlonglong;
    default:
      LOG(ERROR) << "NOT support this dtype yet";
  }
}

void PadExecutor::paramCheck() {
  if (!parser_->getProtoNode()->has_pad_param()) {
    LOG(ERROR) << "Lose pad param. ";
  }
  if (parser_->getInputNum() != 2) {
    LOG(ERROR) << "pad input number is wrong. ";
  }
  if (parser_->getOutputNum() != 1) {
    LOG(ERROR) << "pad output number is wrong. ";
  }
}

void PadExecutor::compute() {
  VLOG(4) << "PadExecutor compute ";
  if (!parser_->getProtoNode()->has_pad_param()) {
    LOG(ERROR) << "Lose pad param. ";
  }
  float proto_padding_value =
      parser_->getProtoNode()->pad_param().padding_value();
  std::string padding_value_string_ =
      parser_->getProtoNode()->pad_param().padding_value_hex();
  char *stop;
  host_padding_value_ = strtoul(padding_value_string_.data(), &stop, 16);
  if (*stop != '\0') {
    GTEST_CHECK(
        0,
        "mluOpPad: invalid padding value hex, type in hex type. (0x1234abcd)");
  }

  auto input_desc = tensor_desc_[0].tensor;
  auto output_desc = tensor_desc_[2].tensor;
  auto input_dev = data_vector_[0].device_ptr;
  auto paddings_host = data_vector_[1].host_ptr;
  auto output_dev = data_vector_[2].device_ptr;
  if (parser_->getProtoNode()->pad_param().padding_param_size() > 0) {
    int padding_num = parser_->getProtoNode()->pad_param().padding_param_size();
    for (int i = 0; i < padding_num; i++) {
      VLOG(4) << "padding param:"
              << parser_->getProtoNode()->pad_param().padding_param(i);
      *((int *)paddings_host + i) =
          parser_->getProtoNode()->pad_param().padding_param(i);
    }
    for (int i = padding_num; i < input_desc->dim * 2; i++) {
      *((int *)paddings_host + i) = 0;
    }
  }
  mluOpDataType_t data_type = output_desc->dtype;
  void *host_padding_value_ptr;
  cnrtDataType_V2_t cnrt_dtype = cvtMluOpDtypeToCnrt_V2(data_type);

  if (proto_padding_value != 0.0) {
    VLOG(4) << "pad from fp32 value";
    fp32_padding_value_ = proto_padding_value;

    VLOG(4) << "padding value from gtest: " << proto_padding_value;
    host_padding_value_ptr =
        (void *)cpu_runtime_.allocate(mluOpDataTypeBytes(data_type));
    if (data_type == MLUOP_DTYPE_INT8) {
      *(int8_t *)host_padding_value_ptr = (int8_t)proto_padding_value;
    } else if (data_type == MLUOP_DTYPE_UINT8) {
      *(uint8_t *)host_padding_value_ptr = (uint8_t)proto_padding_value;
    } else if (data_type == MLUOP_DTYPE_BOOL) {
      *(bool *)host_padding_value_ptr = (bool)proto_padding_value;
    } else if (data_type == MLUOP_DTYPE_INT16) {
      *(int16_t *)host_padding_value_ptr = (int16_t)proto_padding_value;
    } else if (data_type == MLUOP_DTYPE_UINT16) {
      *(uint16_t *)host_padding_value_ptr = (uint16_t)proto_padding_value;
    } else if (data_type == MLUOP_DTYPE_INT32) {
      *(int32_t *)host_padding_value_ptr = (int32_t)proto_padding_value;
    } else if (data_type == MLUOP_DTYPE_UINT32) {
      *(uint32_t *)host_padding_value_ptr = (uint32_t)proto_padding_value;
    } else if (data_type == MLUOP_DTYPE_UINT64) {
      *(uint64_t *)host_padding_value_ptr = (uint64_t)proto_padding_value;
    } else if (data_type == MLUOP_DTYPE_INT64) {
      *(int64_t *)host_padding_value_ptr = (int64_t)proto_padding_value;
    } else {
      cnrtCastDataType_V2(&proto_padding_value, cnrtFloat,
                          host_padding_value_ptr, cnrt_dtype, 1, nullptr,
                          cnrtRounding_rn);
    }
  } else {  // value from hex
    VLOG(4) << "value from hex";
    float float_padding_value;
    host_padding_value_ptr = &host_padding_value_;
    if (data_type == MLUOP_DTYPE_HALF) {
      uint16_t fp16_padding_value = *(uint16_t *)&host_padding_value_;
      wrapRtConvertHalfToFloat(&fp32_padding_value_, fp16_padding_value);
    } else if (data_type == MLUOP_DTYPE_BOOL) {
      fp32_padding_value_ = (bool)((uint8_t)host_padding_value_);
    } else if (data_type == MLUOP_DTYPE_INT8) {
      fp32_padding_value_ = (int8_t)host_padding_value_;
    } else if (data_type == MLUOP_DTYPE_UINT8) {
      fp32_padding_value_ = (uint8_t)host_padding_value_;
    } else if (data_type == MLUOP_DTYPE_INT16) {
      fp32_padding_value_ = (int16_t)host_padding_value_;
    } else if (data_type == MLUOP_DTYPE_UINT16) {
      fp32_padding_value_ = (uint16_t)host_padding_value_;
    } else if (data_type == MLUOP_DTYPE_INT32) {
      fp32_padding_value_ = (int32_t)host_padding_value_;
    } else if (data_type == MLUOP_DTYPE_UINT32) {
      fp32_padding_value_ = (uint32_t)host_padding_value_;
    } else if (data_type == MLUOP_DTYPE_UINT64) {
      fp32_padding_value_ = (uint64_t)host_padding_value_;
    } else {
      cnrtCastDataType_V2(&host_padding_value_, cnrt_dtype,
                          &fp32_padding_value_, cnrtFloat, 1, nullptr,
                          cnrtRounding_rn);
    }
  }
  VLOG(4) << GREEN << "padding value in fp32: " << fp32_padding_value_;
  VLOG(4) << GREEN << "padding value in hex: " << std::hex
          << host_padding_value_;
  VLOG(4) << "call mluOp mluOpPad()";
  interface_timer_.start();
  MLUOP_CHECK(mluOpPad(handle_, input_desc, input_dev, paddings_host,
                       host_padding_value_ptr, output_desc, output_dev));
  interface_timer_.stop();
}

void PadExecutor::padPerLayer(float *input, int64_t in_idx, float *output,
                              int64_t out_idx, int64_t *paddings,
                              int64_t *shape, int dims, int64_t cnt) {
  const int64_t idx = dims - cnt - 1;
  int64_t padding_start = paddings[idx * 2 + 0];
  int64_t padding_end = paddings[idx * 2 + 1];
  int64_t src_start_coor = std::max(-padding_start, int64_t(0));
  int64_t src_end_coor =
      padding_end > 0 ? shape[idx] : shape[idx] + padding_end;
  int64_t dst_start_coor = std::max(padding_start, int64_t(0));
  int64_t dst_end_coor = dst_start_coor + (src_end_coor - src_start_coor);
  if (cnt == 0) {
    in_idx += src_start_coor;
    out_idx += dst_start_coor;
    for (int64_t i = 0; i < src_end_coor - src_start_coor; i++) {
      output[out_idx + i] = input[in_idx + i];
    }
    in_idx -= src_start_coor;
    out_idx -= dst_start_coor;
  } else {
    for (int64_t dim_i = 0; dim_i < src_end_coor - src_start_coor; dim_i++) {
      in_idx += (dim_i + src_start_coor) * origin_[dims - cnt];
      out_idx += (dim_i + dst_start_coor) * current_[dims - cnt];
      padPerLayer(input, in_idx, output, out_idx, paddings, shape, dims,
                  cnt - 1);
      in_idx -= (dim_i + src_start_coor) * origin_[dims - cnt];
      out_idx -= (dim_i + dst_start_coor) * current_[dims - cnt];
    }
  }
}

void PadExecutor::padCpu(float *input, float *output, int64_t *paddings,
                         int64_t *shape, int dims, float padding_value) {
  int64_t ont = 1, cnt = 1;
  for (int i = dims - 1; i >= 0; i--) {
    ont *= shape[i] + paddings[2 * i] + paddings[2 * i + 1];
    current_[i] = ont;
    cnt *= shape[i];
    origin_[i] = cnt;
  }
  for (int64_t i = 0; i < ont; i++) {
    output[i] = padding_value;
  }
  padPerLayer(input, 0, output, 0, paddings, shape, dims, dims - 1);
}

void PadExecutor::cpuCompute() {
  assert(parser_->getInputNum() == 2);
  assert(parser_->getOutputNum() == 1);
  float padding_value = parser_->getProtoNode()->pad_param().padding_value();

  auto count1 = parser_->getInputDataCount(0);         // the input tensor
  auto count2 = parser_->getInputDataCount(1);         // the padding parameters
  auto output_count = parser_->getOutputDataCount(0);  // the output tensor
  auto tensor_a = tensor_desc_[0].tensor;              // input
  int dims = tensor_a->dim;
  // n in paddings[n, 2] equals to input dimensions
  GTEST_CHECK(count2 == dims * 2,
              "[mluOpPad]: the number of padding parameters is invalid.");
  float *input = (float *)cpu_runtime_.allocate(count1 * sizeof(float));
  int64_t *padding = (int64_t *)cpu_runtime_.allocate(count2 * sizeof(int64_t));
  int64_t *input_shape =
      (int64_t *)cpu_runtime_.allocate(dims * sizeof(int64_t));
  for (int i = 0; i < dims; i++) {
    input_shape[i] = tensor_a->dims[i];
  }
  for (int64_t i = 0; i < count1; i++) {
    input[i] = cpu_fp32_input_[0][i];
  }
  for (int i = 0; i < count2; i++) {
    padding[i] = cpu_fp32_input_[1][i];
  }
  if (parser_->getProtoNode()->pad_param().padding_param_size() > 0) {
    int64_t padding_num =
        parser_->getProtoNode()->pad_param().padding_param_size();
    for (int64_t i = 0; i < padding_num; i++) {
      *((int64_t *)padding + i) =
          parser_->getProtoNode()->pad_param().padding_param(i);
    }
  }
  mluOpDataType_t dtype = tensor_desc_[0].tensor->dtype;
  if (dtype == MLUOP_DTYPE_HALF) {
    uint16_t temp;
    wrapRtConvertFloatToHalf(&temp, padding_value);
    wrapRtConvertHalfToFloat(&padding_value, temp);
  } else if (dtype == MLUOP_DTYPE_BOOL) {
    padding_value = (bool)padding_value;
  }
  padCpu(input, cpu_fp32_output_[0], padding, input_shape, dims,
         fp32_padding_value_);
  cpu_runtime_.deallocate(input);
  cpu_runtime_.deallocate(padding);
  cpu_runtime_.deallocate(input_shape);
}

int64_t PadExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->getOutputDataCount(0);
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

int64_t PadExecutor::getTheoryIoSize() {
  int64_t total_size_intput = 1;
  for (int i = 0; i < data_vector_[0].shape.size(); i++) {
    total_size_intput *=
        data_vector_[0].shape[i] +
        std::min(((int32_t *)data_vector_[1].host_ptr)[2 * i], (int)0) +
        std::min(((int32_t *)data_vector_[1].host_ptr)[2 * i + 1], (int)0);
  }
  total_size_intput =
      total_size_intput * data_vector_[0].size / data_vector_[0].count;
  int64_t total_size_output = data_vector_[2].size;
  return total_size_intput + total_size_output;
}
}  // namespace mluoptest
