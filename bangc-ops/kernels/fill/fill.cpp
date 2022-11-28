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
#include <cmath>
#include <limits>

#include "fill_mlu.h"
#include "core/gen_case.h"
#include "core/logging.h"
#include "core/runtime/device.h"
#include "core/tensor.h"
#include "core/type.h"

static void cvtINT64ToSaturation(mluOpDataType_t k_dtype, float value,
                                 int64_t &value_int64, uint64_t &value_uint64) {
  float uint64_max = (float)std::numeric_limits<uint64_t>::max();
  float int64_max = (float)std::numeric_limits<int64_t>::max();
  float int64_min = (float)std::numeric_limits<int64_t>::min();
  if (k_dtype == MLUOP_DTYPE_INT64 &&
      (value > int64_max || value < int64_min)) {
    if (value > 0) {
      value_int64 = std::numeric_limits<int64_t>::max();
    } else {
      value_int64 = std::numeric_limits<int64_t>::min();
    }
  }

  if (k_dtype == MLUOP_DTYPE_UINT64 && value > uint64_max) {
    value_uint64 = std::numeric_limits<uint64_t>::max();
  }
}

static bool policyFunc(mluOpHandle_t handle, size_t total_size,
                       cnrtFunctionType_t *type, cnrtDim3_t *k_dim) {
  const int union_number = mluop::runtime::getClusterLimitCapability(handle);
  const int core_dim = mluop::runtime::getCoreNumOfEachUnionCapability(handle);

  size_t sram_size = 1024;
  if (handle->arch >= 590) {
    sram_size = 20 * 1024;
  } else if (handle->arch >= 300) {
    sram_size = 18 * 1024;
  } else if (handle->arch >= 200) {
    sram_size = 32 * 1024;
  }

  int cluster_required = total_size / sram_size;
  cluster_required = cluster_required == 0 ? 1 : cluster_required;
  const int cluster_dim = (int)std::fmin(
      std::exp2(std::ceil(std::log2(cluster_required))), union_number);

  *type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = core_dim;
  k_dim->y = cluster_dim;
  k_dim->z = 1;
  return true;
}

mluOpStatus_t MLUOP_WIN_API mluOpFill(mluOpHandle_t handle,
                                      const mluOpPointerMode_t pointer_mode,
                                      const void *value,
                                      const mluOpTensorDescriptor_t output_desc,
                                      void *output) {
  PARAM_CHECK("[mluOpFill]", handle != NULL);
  PARAM_CHECK("[mluOpFill]", output_desc != NULL);

  mluOpDataType_t output_dtype = output_desc->dtype;
  if (output_dtype != MLUOP_DTYPE_BOOL && output_dtype != MLUOP_DTYPE_INT8 &&
      output_dtype != MLUOP_DTYPE_UINT8 && output_dtype != MLUOP_DTYPE_INT16 &&
      output_dtype != MLUOP_DTYPE_INT32 && output_dtype != MLUOP_DTYPE_HALF &&
      output_dtype != MLUOP_DTYPE_FLOAT && output_dtype != MLUOP_DTYPE_INT64 &&
      output_dtype != MLUOP_DTYPE_UINT64 &&
      output_dtype != MLUOP_DTYPE_UINT16 &&
      output_dtype != MLUOP_DTYPE_UINT32) {
    LOG(ERROR)
        << "[mluOpFill] output_desc only support "
           "bool/int8/uint8/int16/int32/int64/half/"
        << "float/uint64/uint32/uint16 type. Current output data type is "
        << getNameOfDataType(output_dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  if (mluOpGetTensorElementNum(output_desc) == 0) {
    VLOG(5) << "mluOpFill skip zero element tensor.";
    return MLUOP_STATUS_SUCCESS;
  }

  PARAM_CHECK("[mluOpFill]", output != NULL);
  PARAM_CHECK("[mluOpFill]", value != NULL);

  // generate mluOpFill prototxt start!
  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("fill");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", NULL, output_desc, 10, 1);
    GEN_CASE_DATA(false, "output", output, output_desc, 0, 0);
    if (pointer_mode == MLUOP_POINTER_MODE_DEVICE) {
      size_t output_dtype_size = getSizeOfDataType(output_desc->dtype);
      void *value_host = malloc(output_dtype_size);
      cnrtMemcpyAsync(value_host, const_cast<void *>(value), output_dtype_size,
                      handle->queue, CNRT_MEM_TRANS_DIR_DEV2HOST);
      cnrtQueueSync(handle->queue);
      char value_str[21];
      snprintf(value_str, sizeof(value_str), "\"0X%lx\"",
               *(uint64_t *)value_host);
      GEN_CASE_OP_PARAM_SINGLE(0, "fill", "value_hex", value_str);
      GEN_CASE_OP_PARAM_SINGLE(1, "fill", "version", 3);
      GEN_CASE_OP_PARAM_SINGLE(2, "fill", "mode", std::to_string(pointer_mode));
      GEN_CASE_TEST_PARAM_NEW(false, false, true, 0, 0, 0);
      free(value_host);
    } else {
      char value_str[21];
      snprintf(value_str, sizeof(value_str), "\"0X%lx\"", *(uint64_t *)value);
      GEN_CASE_OP_PARAM_SINGLE(0, "fill", "value_hex", value_str);
      GEN_CASE_OP_PARAM_SINGLE(1, "fill", "version", 3);
      GEN_CASE_OP_PARAM_SINGLE(2, "fill", "mode", std::to_string(pointer_mode));
      GEN_CASE_TEST_PARAM_NEW(false, false, true, 0, 0, 0);
    }
  }
  // generate mluOpFill prototxt end!

  mluOpDataType_t k_datatype = output_dtype;
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  size_t output_num = mluOpGetTensorElementNum(output_desc);
  size_t output_size = output_num * sizeof(k_datatype);

  // choose best task dimension
  if (policyFunc(handle, output_size, &k_type, &k_dim) != true) {
    LOG(ERROR) << "[mluOpFill] policyFunc is not executed successfully.";
    return MLUOP_STATUS_BAD_PARAM;
  }

  bool if_stride_kernel = false;
  if (strideCaseWithNotConsistentDense(1, output_desc)) {
    if_stride_kernel = true;
  }

  if (pointer_mode == MLUOP_POINTER_MODE_DEVICE) {
    if (if_stride_kernel) {
      PARAM_CHECK("[mluOpFill]", output_desc->dim <= MLUOP_DIM_MAX);
      TensorShape output_shape;
      getTensorShape(output_desc, &output_shape);
      VLOG(5) << "mluOpFill:Launch Kernel "
                 "MLUUnion1KernelFillDeviceValueWithStride<<<Union"
              << k_type / CORE_DIM << "," << k_dim.x << ", " << k_dim.y << ", "
              << k_dim.z << ">>>"
              << " CORE_DIM : " << CORE_DIM;
      KERNEL_CHECK((mluOpUnion1KernelFillDeviceValueWithStride(
          k_dim, k_type, handle->queue, k_datatype, output, output_shape,
          output_num, value)));
    } else {
      VLOG(5)
          << "mluOpFill:Launch Kernel MLUUnion1KernelFillDeviceValue<<<Union"
          << k_type / CORE_DIM << "," << k_dim.x << ", " << k_dim.y << ", "
          << k_dim.z << ">>>"
          << " CORE_DIM : " << CORE_DIM;
      KERNEL_CHECK((mluOpUnion1KernelFillDeviceValue(
          k_dim, k_type, handle->queue, k_datatype, output, output_num,
          value)));
    }
  } else {
    if (if_stride_kernel) {
      PARAM_CHECK("[mluOpFill]", output_desc->dim <= MLUOP_DIM_MAX);
      TensorShape output_shape;
      getTensorShape(output_desc, &output_shape);
      uint32_t value_high = 0, value_low = 0;
      uint64_t host_value = *(uint64_t *)value;
      getLowAndHighValueFrom64Bits(host_value, &value_high, &value_low);
      VLOG(5) << "mluOpFill:Launch Kernel "
                 "MLUUnion1KernelFillHostValueWithStride<<<Union"
              << k_type / CORE_DIM << "," << k_dim.x << ", " << k_dim.y << ", "
              << k_dim.z << ">>>"
              << " CORE_DIM : " << CORE_DIM;
      KERNEL_CHECK((mluOpUnion1KernelFillHostValueWithStride(
          k_dim, k_type, handle->queue, k_datatype, output, output_shape,
          output_num, (uint32_t)host_value, value_high, value_low)));
    } else {
      uint32_t value_high = 0, value_low = 0;
      uint64_t host_value = *(uint64_t *)value;

      getLowAndHighValueFrom64Bits(host_value, &value_high, &value_low);
      VLOG(5) << "mluOpFill:Launch Kernel MLUUnion1KernelFillHostValue<<<Union"
              << k_type / CORE_DIM << "," << k_dim.x << ", " << k_dim.y << ", "
              << k_dim.z << ">>>"
              << " CORE_DIM : " << CORE_DIM;
      KERNEL_CHECK((mluOpUnion1KernelFillHostValue(
          k_dim, k_type, handle->queue, k_datatype, output, output_num,
          (uint32_t)host_value, value_high, value_low)));
    }
  }
  GEN_CASE_END();
  return MLUOP_STATUS_SUCCESS;
}
