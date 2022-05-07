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
#include <iomanip>
#include <algorithm>
#include "core/tensor.h"
#include "core/logging.h"
#include "core/type.h"

/* mluOpTensorStruct */

mluOpStatus_t mluOpTensorStruct::tensorDimN(size_t &dim_out) {
  size_t index;
  switch (layout) {
    case MLUOP_LAYOUT_NCHW:
    case MLUOP_LAYOUT_NHWC:
    case MLUOP_LAYOUT_NDHWC:
      index = 0;
      break;
    case MLUOP_LAYOUT_HWCN:
      index = 3;
      break;
    default:
      LOG(ERROR) << "mluOpTensorStruct::tensorDimN, "
                 << "illegal layout in descriptor: " << layout;
      return MLUOP_STATUS_BAD_PARAM;
  }
  if (index > dim) {
    LOG(ERROR) << "mluOpTensorStruct::tensorDimN, "
               << "mismatch layout and dimension. layout: " << layout;
    return MLUOP_STATUS_NOT_INITIALIZED;
  }
  dim_out = dims[index];
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpTensorStruct::tensorDimC(size_t &dim_out) {
  size_t index;
  switch (layout) {
    case MLUOP_LAYOUT_NCHW:
      index = 1;
      break;
    case MLUOP_LAYOUT_NHWC:
      index = 3;
      break;
    case MLUOP_LAYOUT_NDHWC:
      index = 4;
      break;
    case MLUOP_LAYOUT_HWCN:
      index = 2;
      break;
    default:
      LOG(ERROR) << "mluOpTensorStruct::tensorDimC, "
                 << "illegal layout in descriptor: " << layout;
      return MLUOP_STATUS_BAD_PARAM;
  }
  if (index > dim) {
    LOG(ERROR) << "mluOpTensorStruct::tensorDimC, "
               << "mismatch layout and dimension. layout: " << layout;
    return MLUOP_STATUS_NOT_INITIALIZED;
  }
  dim_out = dims[index];
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpTensorStruct::tensorDimH(size_t &dim_out) {
  size_t index;
  switch (layout) {
    case MLUOP_LAYOUT_NCHW:
      index = 2;
      break;
    case MLUOP_LAYOUT_NHWC:
      index = 1;
      break;
    case MLUOP_LAYOUT_NDHWC:
      index = 2;
      break;
    case MLUOP_LAYOUT_HWCN:
      index = 0;
      break;
    default:
      LOG(ERROR) << "mluOpTensorStruct::tensorDimH, "
                 << "illegal layout in descriptor: " << layout;
      return MLUOP_STATUS_BAD_PARAM;
  }
  if (index > dim) {
    LOG(ERROR) << "mluOpTensorStruct::tensorDimH, "
               << "mismatch layout and dimension. layout: " << layout;
    return MLUOP_STATUS_NOT_INITIALIZED;
  }
  dim_out = dims[index];
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpTensorStruct::tensorDimW(size_t &dim_out) {
  size_t index;
  switch (layout) {
    case MLUOP_LAYOUT_NCHW:
      index = 3;
      break;
    case MLUOP_LAYOUT_NHWC:
      index = 2;
      break;
    case MLUOP_LAYOUT_NDHWC:
      index = 3;
      break;
    case MLUOP_LAYOUT_HWCN:
      index = 1;
      break;
    default:
      LOG(ERROR) << "mluOpTensorStruct::tensorDimW, "
                 << "illegal layout in descriptor: " << layout;
      return MLUOP_STATUS_BAD_PARAM;
  }
  if (index > dim) {
    LOG(ERROR) << "mluOpTensorStruct::tensorDimW, "
               << "mismatch layout and dimension. layout: " << layout;
    return MLUOP_STATUS_NOT_INITIALIZED;
  }
  dim_out = dims[index];
  return MLUOP_STATUS_SUCCESS;
}

/* MLUOP interface */
mluOpStatus_t mluOpCreateTensorDescriptor(mluOpTensorDescriptor_t *desc) {
  PARAM_CHECK("[mluOpCreateTensorDescriptor]", desc != NULL);
  mluOpTensorStruct *ts = new mluOpTensorStruct();
  *desc = ts;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpSetTensorDescriptor(mluOpTensorDescriptor_t desc,
                                       mluOpTensorLayout_t layout,
                                       mluOpDataType_t dtype, int dimNb,
                                       const int dimSize[]) {
  PARAM_CHECK("[mluOpSetTensorDescriptor]", desc != NULL);
  PARAM_CHECK("[mluOpSetTensorDescriptor]", dimNb > 0);
  PARAM_CHECK("[mluOpSetTensorDescriptor]", dimSize != NULL);
  PARAM_CHECK("[mluOpSetTensorDescriptor]", layout >= 0);
  PARAM_CHECK("[mluOpSetTensorDescriptor]", dtype >= 0);

  desc->dim = dimNb;
  desc->dtype = dtype;
  desc->layout = layout;

  if (MLUOP_PREDICT_FALSE(dimNb > MLUOP_DIM_MAX)) {
    desc->larger_dims = new int[dimNb];
    desc->larger_strides = new int[dimNb];
    desc->dims = desc->larger_dims;
    desc->strides = desc->larger_strides;
  } else {
    desc->dims = desc->normal_dims;
    desc->strides = desc->normal_strides;
  }
  memcpy(desc->dims, dimSize, dimNb * sizeof(int));

  // infer strides of dimNb dimensions and compute total_num & total_size
  int strideBase = 1;
  for (int i = dimNb - 1; i >= 0; --i) {
    desc->strides[i] = strideBase;
    strideBase *= desc->dims[i];
  }
  desc->total_element_num = strideBase;
  desc->total_tensor_size = desc->total_element_num * getSizeOfDataType(dtype);
  // judge int overflow situation
  int total_size = desc->total_tensor_size;
  if (total_size != 0) {
    int last_num = total_size / getSizeOfDataType(dtype);
    for (int i = 0; i < dimNb - 1; ++i) {
      last_num = last_num / dimSize[i];
    }
    if (last_num < dimSize[dimNb - 1]) {
      std::stringstream tensor_info;
      tensor_info << "dims:(";
      for (int i = 0; i < dimNb - 1; ++i) {
        tensor_info << dimSize[i] << ", ";
      }

      tensor_info << dimSize[dimNb - 1]
                  << "), data width:" << getSizeOfDataType(dtype) << ".";
      LOG(WARNING) << "[mluOpSetTensorDescriptor]: overflow tensor size with "
                      "type 'int'. Currently, mluOp supports tensor size "
                      "smaller than 2^31, now tensor "
                   << tensor_info.str();
    }
  }
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpResetTensorDescriptor(mluOpTensorDescriptor_t desc) {
  PARAM_CHECK("[mluOpResetTensorDescriptor]", desc != NULL);
  desc->reset();

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpSetTensorDescriptorEx(mluOpTensorDescriptor_t desc,
                                         mluOpTensorLayout_t layout,
                                         mluOpDataType_t dtype, int dimNb,
                                         const int dimSize[],
                                         const int dimStride[]) {
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", desc != NULL);
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", dimSize != NULL);
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", dimStride != NULL);
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", layout >= 0);
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", dtype >= 0);
  PARAM_CHECK("[mluOpSetTensorDescriptorEx]", dimNb > 0);

  desc->dim = dimNb;
  memcpy(desc->dims, dimSize, dimNb * sizeof(int));
  memcpy(desc->strides, dimStride, dimNb * sizeof(int));

  // assign total_element_num and total_tensor_size
  desc->total_element_num = 1;
  for (int i = 0; i < dimNb; ++i) {
    desc->total_element_num *= dimSize[i];
  }
  desc->total_tensor_size = desc->total_element_num * getSizeOfDataType(dtype);

  desc->dtype = dtype;
  desc->layout = layout;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpSetTensorDescriptorOnchipDataType(
    mluOpTensorDescriptor_t desc, mluOpDataType_t onchip_dtype) {
  PARAM_CHECK("[mluOpSetTensorDescriptorOnchipDataType]", desc != NULL);

  desc->onchip_dtype = onchip_dtype;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpSetTensorDescriptorPosition(mluOpTensorDescriptor_t desc,
                                               int position) {
  PARAM_CHECK("[mluOpSetTensorDescriptorPosition]", desc != NULL);

  desc->position = position;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpSetTensorDescriptorPositionAndScale(
    mluOpTensorDescriptor_t desc, int position, float scale) {
  PARAM_CHECK("[mluOpSetTensorDescriptorPositionAndScale]", desc != NULL);

  desc->position = position;
  desc->scale = scale;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpSetTensorDescriptorPositionScaleAndOffset(
    mluOpTensorDescriptor_t desc, int position, float scale, int offset) {
  PARAM_CHECK("[mluOpSetTensorDescriptorPositionScaleAndOffset]", desc != NULL);

  desc->position = position;
  desc->scale = scale;
  desc->offset = offset;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetTensorDescriptorEx(const mluOpTensorDescriptor_t desc,
                                         mluOpTensorLayout_t *layout,
                                         mluOpDataType_t *dtype, int *dimNb,
                                         int dimSize[], int dimStride[]) {
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", desc != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", layout != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", dtype != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", dimNb != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", dimSize != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorEx]", dimStride != NULL);

  *layout = desc->layout;
  *dtype = desc->dtype;
  *dimNb = desc->dim;
  for (int i = 0; i < *dimNb; ++i) {
    dimSize[i] = desc->dims[i];
    dimStride[i] = desc->strides[i];
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetTensorDescriptor(const mluOpTensorDescriptor_t desc,
                                       mluOpTensorLayout_t *layout,
                                       mluOpDataType_t *dtype, int *dimNb,
                                       int dimSize[]) {
  PARAM_CHECK("[mluOpGetTensorDescriptor]", desc != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptor]", layout != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptor]", dtype != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptor]", dimNb != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptor]", dimSize != NULL);

  *layout = desc->layout;
  *dtype = desc->dtype;
  *dimNb = desc->dim;
  for (int i = 0; i < *dimNb; ++i) {
    dimSize[i] = desc->dims[i];
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetTensorDescriptorOnchipDataType(
    const mluOpTensorDescriptor_t desc, mluOpDataType_t *onchip_dtype) {
  PARAM_CHECK("[mluOpGetTensorDescriptorOnchipDataType]", desc != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorOnchipDataType]", onchip_dtype != NULL);

  *onchip_dtype = desc->onchip_dtype;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetTensorDescriptorPosition(
    const mluOpTensorDescriptor_t desc, int *position) {
  PARAM_CHECK("[mluOpGetTensorDescriptorPosition]", desc != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorPosition]", position != NULL);

  *position = desc->position;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetTensorDescriptorPositionAndScale(
    const mluOpTensorDescriptor_t desc, int *position, float *scale) {
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionAndScale]", desc != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionAndScale]", position != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionAndScale]", scale != NULL);

  *position = desc->position;
  *scale = desc->scale;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpGetTensorDescriptorPositionScaleAndOffset(
    const mluOpTensorDescriptor_t desc, int *position, float *scale,
    int *offset) {
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionScaleAndOffset]", desc != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionScaleAndOffset]",
              position != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionScaleAndOffset]",
              scale != NULL);
  PARAM_CHECK("[mluOpGetTensorDescriptorPositionScaleAndOffset]",
              offset != NULL);

  *position = desc->position;
  *scale = desc->scale;
  *offset = desc->offset;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t mluOpDestroyTensorDescriptor(mluOpTensorDescriptor_t desc) {
  PARAM_CHECK("[mluOpDestroyTensorDescriptor]", desc != NULL);
  desc->reset();
  delete desc;
  return MLUOP_STATUS_SUCCESS;
}

// usr interface.
uint64_t mluOpGetTensorElementNum(const mluOpTensorDescriptor_t desc) {
  CHECK(desc != NULL);
  uint64_t tensor_num = 1;
  auto return_status = desc->tensorElementsNumber(tensor_num);
  return tensor_num;
}
