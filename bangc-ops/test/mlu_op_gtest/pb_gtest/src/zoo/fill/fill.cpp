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
#include <string>

#include "fill.h"
namespace mluoptest {

void FillExecutor::paramCheck() {
  if (parser_->getInputNum() != 1) {
    LOG(ERROR) << "FillExecutor: input number is wrong. ";
  }

  GTEST_CHECK(parser_->node()->has_fill_param(), "Lose fill param.");
}

void FillExecutor::compute() {
  VLOG(4) << "fill executor compute ";
  float value = parser_->getProtoNode()->fill_param().value();
  float proto_value = parser_->getProtoNode()->fill_param().value();
  std::string value_string_ = parser_->getProtoNode()->fill_param().value_hex();
  char *stop;
  host_value_ = strtoul(value_string_.data(), &stop, 16);
  if (*stop != '\0') {
    GTEST_CHECK(0,
                "mluOpFill: invalid value hex, type in hex type. (0x1234abcd)");
  }

  mluOpPointerMode_t mode =
      (mluOpPointerMode_t)parser_->getProtoNode()->fill_param().mode();
  cnrtQuantizedParam_t quant = nullptr;
  auto tensor_input = tensor_desc_[0].tensor;
  mluOpDataType_t data_type = tensor_input->dtype;
  void *host_value_ptr =
      static_cast<void *>(cpu_runtime_.allocate(sizeof(uint64_t)));
  memcpy(host_value_ptr, &host_value_, sizeof(uint64_t));
  if (proto_value != 0.0) {
    fp32_value_ = proto_value;
    cnrtDataType cnrt_dtype;
    cnrtQuantizedParam_t quant = nullptr;
    switch (data_type) {
      case MLUOP_DTYPE_FLOAT:
        cnrt_dtype = CNRT_FLOAT32;
        break;
      case MLUOP_DTYPE_HALF:
        cnrt_dtype = CNRT_FLOAT16;
        break;
      case MLUOP_DTYPE_INT32:
      case MLUOP_DTYPE_UINT32:
        cnrtCreateQuantizedParam(&quant, 0, 1.0, 0);
        proto_value = (int)proto_value;
        cnrt_dtype = CNRT_INT32;
        break;
      case MLUOP_DTYPE_INT64:
      case MLUOP_DTYPE_UINT64:
        proto_value = (int64_t)proto_value;
        cnrt_dtype = CNRT_INT64;
        break;
      case MLUOP_DTYPE_INT16:
      case MLUOP_DTYPE_UINT16:
        proto_value = (int)proto_value;
        cnrtCreateQuantizedParam(&quant, 0, 1.0, 0);
        cnrt_dtype = CNRT_INT16;
        break;
      case MLUOP_DTYPE_BOOL:
        proto_value = (bool)proto_value;
        cnrtCreateQuantizedParam(&quant, 0, 1.0, 0);
        cnrt_dtype = CNRT_INT8;
        break;
      case MLUOP_DTYPE_INT8:
        proto_value = (int)proto_value;
        cnrtCreateQuantizedParam(&quant, 0, 1.0, 0);
        cnrt_dtype = CNRT_INT8;
        break;
      case MLUOP_DTYPE_UINT8:
        proto_value = (int)proto_value;
        cnrtCreateQuantizedParam(&quant, 0, 1.0, 0);
        cnrt_dtype = CNRT_INT32;
        break;
      default: {
        cnrt_dtype = CNRT_INVALID;
        LOG(ERROR) << "[mluOpFill]: unsupported data type.";
        break;
      }
    }
    VLOG(4) << "value from gtest: " << proto_value;
    if (data_type == MLUOP_DTYPE_UINT16) {
      *(uint16_t *)host_value_ptr = (uint16_t)proto_value;
    } else if (data_type == MLUOP_DTYPE_UINT32) {
      *(uint32_t *)host_value_ptr = (uint32_t)proto_value;
    } else if (data_type == MLUOP_DTYPE_UINT64) {
      *(uint64_t *)host_value_ptr = (uint64_t)proto_value;
    } else if (data_type == MLUOP_DTYPE_INT64) {
      *(int64_t *)host_value_ptr = (int64_t)proto_value;
    } else {
      cnrtCastDataType(&proto_value, CNRT_FLOAT32, host_value_ptr, cnrt_dtype,
                       1, quant);
    }
    if (quant != nullptr) {
      cnrtDestroyQuantizedParam(quant);
    }
  } else if (host_value_ != 0X0) {
    float float_value;
    cnrtDataType cnrt_dtype;
    switch (data_type) {
      case MLUOP_DTYPE_FLOAT:
        cnrt_dtype = CNRT_FLOAT32;
        break;
      case MLUOP_DTYPE_HALF:
        cnrt_dtype = CNRT_FLOAT16;
        break;
      case MLUOP_DTYPE_INT32:
        cnrt_dtype = CNRT_INT32;
        break;
      case MLUOP_DTYPE_UINT32:
        cnrt_dtype = CNRT_UINT32;
        break;
      case MLUOP_DTYPE_INT64:
      case MLUOP_DTYPE_UINT64:
        cnrt_dtype = CNRT_INT64;
        break;
      case MLUOP_DTYPE_INT16:
        cnrt_dtype = CNRT_INT16;
        break;
      case MLUOP_DTYPE_UINT16:
        cnrt_dtype = CNRT_UINT16;
        break;
      case MLUOP_DTYPE_BOOL:
      case MLUOP_DTYPE_INT8:
        cnrt_dtype = CNRT_INT8;
        break;
      case MLUOP_DTYPE_UINT8:
        cnrt_dtype = CNRT_UINT8;
        break;
      default: {
        cnrt_dtype = CNRT_INVALID;
        LOG(ERROR) << "[mluOpFill]: invalid data type.";
      }
    }
    if (data_type == MLUOP_DTYPE_HALF) {
      uint16_t fp16_value = *(uint16_t *)&host_value_;
      // wrapRtConvertHalfToFloat(&fp32_value_, fp16_value);
      cnrtCastDataType_V2(&fp16_value, cnrtHalf, &fp32_value_, cnrtFloat, 1,
                          NULL, cnrtRounding_rm);
    } else if (data_type == MLUOP_DTYPE_UINT16) {
      fp32_value_ = (uint16_t)host_value_;
    } else if (data_type == MLUOP_DTYPE_UINT32) {
      fp32_value_ = (uint32_t)host_value_;
    } else if (data_type == MLUOP_DTYPE_UINT64) {
      fp32_value_ = (uint64_t)host_value_;
    } else if (data_type == MLUOP_DTYPE_BOOL) {
      fp32_value_ = (bool)((uint8_t)host_value_);
      cnrtQuantizedParam_t quant2 = nullptr;
      cnrtCreateQuantizedParam(&quant2, 0, 1.0, 0);
      cnrtCastDataType(&fp32_value_, CNRT_FLOAT32, host_value_ptr, CNRT_INT8, 1,
                       quant2);
      cnrtDestroyQuantizedParam(quant2);
    } else {
      cnrtQuantizedParam_t quant_param = nullptr;
      if (cnrt_dtype != CNRT_UINT8) {
        cnrtCastDataType(&host_value_, cnrt_dtype, &fp32_value_, CNRT_FLOAT32,
                         1, quant_param);
      } else {
        cnrtCreateQuantizedParam(&quant_param, 0, 1.0, 0);
        cnrtCastDataType(&host_value_, CNRT_INT32, &fp32_value_, CNRT_FLOAT32,
                         1, quant_param);
        memcpy(host_value_ptr, &host_value_, sizeof(uint64_t));
        fp32_value_ = *(uint8_t *)host_value_ptr;
      }
      cnrtDestroyQuantizedParam(quant_param);
    }
  }
  VLOG(4) << GREEN << "value in fp32: " << fp32_value_;
  VLOG(4) << GREEN << "value in hex: " << std::hex << host_value_;
  VLOG(4) << "call mluOp fill()";
  interface_timer_.start();
  auto value_device = data_vector_[0].device_ptr;
  auto dev_out = data_vector_[1].device_ptr;
  if (mode == MLUOP_POINTER_MODE_DEVICE) {
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtMemcpy(value_device, host_value_ptr,
                                               mluOpDataTypeBytes(data_type),
                                               CNRT_MEM_TRANS_DIR_HOST2DEV));
    MLUOP_CHECK(mluOpFill_v3(handle_, MLUOP_POINTER_MODE_DEVICE, value_device,
                          tensor_input, dev_out));
  } else {
    MLUOP_CHECK(mluOpFill_v3(handle_, MLUOP_POINTER_MODE_HOST, host_value_ptr,
                          tensor_input, dev_out));
  }
  interface_timer_.stop();
}

void FillExecutor::cpuCompute() {
  float value = parser_->node()->fill_param().value();
  auto count1 = parser_->input(0)->shape_count;
  auto dtype = parser_->input(0)->dtype;

  float temp = value;
  if (dtype == MLUOP_DTYPE_HALF) {
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtCastDataType(&value, CNRT_FLOAT32,
                                                     &temp, CNRT_FLOAT16, 1,
                                                     nullptr));
    GTEST_CHECK(CNRT_RET_SUCCESS == cnrtCastDataType(&temp, CNRT_FLOAT16,
                                                     &value, CNRT_FLOAT32, 1,
                                                     nullptr));
  }
  if (dtype == MLUOP_DTYPE_BOOL) {
    value = (float)((bool)((uint8_t)temp));
  } else if (dtype == MLUOP_DTYPE_INT8) {
    value = (float)((int8_t)temp);
  } else if (dtype == MLUOP_DTYPE_UINT8) {
    value = (float)((uint8_t)temp);
  } else if (dtype == MLUOP_DTYPE_INT16) {
    value = (float)((int16_t)temp);
  } else if (dtype == MLUOP_DTYPE_UINT16) {
    value = (float)((uint16_t)temp);
  } else if (dtype == MLUOP_DTYPE_INT32) {
    value = (float)((int32_t)temp);
  } else if (dtype == MLUOP_DTYPE_UINT32) {
    value = (float)((uint32_t)temp);
  } else if (dtype == MLUOP_DTYPE_INT64) {
    value = (float)((int64_t)temp);
  } else if (dtype == MLUOP_DTYPE_UINT64) {
    value = (float)((uint64_t)temp);
  }

  for (int i = 0; i < count1; ++i) {
    if (host_value_ != 0X0)
      cpu_fp32_output_[0][i] = fp32_value_;
    else
      cpu_fp32_output_[0][i] = value;
  }
}

int64_t FillExecutor::getTheoryOps() {
  int64_t theory_ops = parser_->output(0)->total_count;
  VLOG(4) << "getTheoryOps: " << theory_ops << " ops";
  return theory_ops;
}

int64_t FillExecutor::getTheoryIoSize() {
  auto dtype = parser_->output(0)->dtype;
  int64_t theory_ios =
      parser_->output(0)->total_count * mluop::getSizeOfDataType(dtype);
  VLOG(4) << "getTheoryIOs: " << theory_ios << " bytes";
  return theory_ios;
}
}  // namespace mluoptest
